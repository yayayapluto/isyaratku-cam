"""Skrip pengukuran offline (BUKAN kode inti aplikasi).

Mengukur akurasi model A-Z pada dataset VOC berlabel
`data/bisindo_rhio/{train,test}` dengan pipeline yang sama seperti produksi:
crop bbox VOC -> letterbox PADDING -> preprocess(INPUT_SIZE) -> model.

Kenapa ada: menghindari angka yang dikutip dari ingatan. Semua klaim akurasi
di docs/ harus bisa direproduksi lewat:
    python scripts/eval_offline.py

Exit code:
    0  lolos  (akurasi >= 0.68 DAN forward rata-rata <= 0.080 s)
    1  gagal
"""

from __future__ import annotations

import os
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import recognizer  # noqa: E402
from hand_detect import PADDING, crop_hand  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "bisindo_rhio")
SPLITS = ("train", "test")

ACC_MIN = 0.68       # 68% (terukur 260/pad10 = 68,85%)
FORWARD_MAX = 0.080  # detik per forward (terukur 16 core = 60,8 ms)


def load_dataset() -> list[tuple[str, np.ndarray, str]]:
    """Kembalikan daftar (nama_file, frame_bgr, huruf_benar).

    Frame dimuat penuh seperti webcam; crop mengikuti anotasi VOC supaya
    jalur uji identik dengan produksi (crop_hand dari landmark)."""
    items: list[tuple[str, np.ndarray, str]] = []
    for split in SPLITS:
        folder = os.path.join(DATA_DIR, split)
        if not os.path.isdir(folder):
            continue
        for xml_name in sorted(os.listdir(folder)):
            if not xml_name.endswith(".xml"):
                continue
            xml_path = os.path.join(folder, xml_name)
            jpg_path = os.path.join(folder, xml_name[:-4] + ".jpg")
            if not os.path.exists(jpg_path):
                continue
            root_el = ET.parse(xml_path).getroot()
            obj = root_el.find("object")
            if obj is None:
                continue
            label = obj.find("name").text.strip().upper()
            box = obj.find("bndbox")
            x0 = int(box.find("xmin").text)
            y0 = int(box.find("ymin").text)
            x1 = int(box.find("xmax").text)
            y1 = int(box.find("ymax").text)
            frame = cv2.imread(jpg_path)
            if frame is None:
                continue
            # landmark sintetis dari sudut bbox: crop_hand() memakai
            # koordinat relatif, jadi strelets: 0..1.
            h, w = frame.shape[:2]
            landmarks = [
                (x0 / w, y0 / h), (x1 / w, y0 / h),
                (x1 / w, y1 / h), (x0 / w, y1 / h),
            ]
            items.append((xml_name, frame, label, landmarks))
    return items


def evaluate(samples, model, labels_index) -> tuple[float, float, Counter]:
    """Inferensi semua sampel. Return (akurasi, forward_rata_s, confusion)."""
    hit = 0
    total_ms = 0.0
    confusion: Counter = Counter()
    with torch.no_grad():
        for _, frame, label, landmarks in samples:
            cropped = crop_hand(frame, landmarks)
            if cropped is None:
                # bbox terlalu dekat tepi / crop invalid: hitung sebagai salah
                # supaya jumlah sampel evaluasi tetap = jumlah file.
                confusion[(label, "<crop=None>")] += 1
                continue
            start = time.perf_counter()
            probs = torch.softmax(model(recognizer.preprocess(cropped)), 1)[0]
            total_ms += (time.perf_counter() - start) * 1000.0
            pred = labels_index[int(probs.argmax())]
            if pred == label:
                hit += 1
            else:
                confusion[(label, pred)] += 1
    total = len(samples)
    return hit / total, (total_ms / total) / 1000.0, confusion


def main() -> int:
    samples = load_dataset()
    if not samples:
        print("dataset tidak ditemukan / kosong:", DATA_DIR)
        return 1
    model, labels = recognizer.build_model()
    # urutan label indeks -> dict untuk O(1) lookup prediksi
    accuracy, forward_s, confusion = evaluate(samples, model, labels)

    print(f"sampel        : {len(samples)} citra")
    print(f"ukuran input  : {recognizer.INPUT_SIZE[0]}x{recognizer.INPUT_SIZE[1]}"
          f" (PADDING={PADDING} px)")
    print(f"akurasi       : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"forward rata  : {forward_s * 1000:.1f} ms")
    print(f"threads torch : {torch.get_num_threads()}")
    print()
    print("Confusion teratas (benar -> prediksi model):")
    for (true, pred), count in confusion.most_common(15):
        print(f"  {true:>2} -> {pred:<2} : {count}")

    ok = accuracy >= ACC_MIN and forward_s <= FORWARD_MAX
    print()
    print(f"LOLOS ({accuracy:.2%} >= {ACC_MIN:.0%} & "
          f"{forward_s * 1000:.0f} ms <= {FORWARD_MAX * 1000:.0f} ms)" if ok else
          f"GAGAL (akurasi {accuracy:.2%} / forward "
          f"{forward_s * 1000:.0f} ms)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
