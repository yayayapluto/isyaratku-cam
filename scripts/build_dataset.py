"""Rakit dataset fine-tuning BISINDO dari beberapa sumber.

Setiap sumber = <path>:<format>, format ada dua:

  voc     bbox dimuat dari anotasi Pascal VOC; kedua layout diterima:
          <root>/collectedimages/{train,test}/ ATAU <root>/{train,test}/
          (*.jpg + *.xml). Bila keduanya ada, tiap direktori split dibaca
          sekali — folder yang sama tidak dibaca dua kali.

Citra keluar selalu HAND CROP: potong bbox, letterbox PADDING, resize
260x260 (jalur sama dengan produksi, lihat docs/MODEL_SELECTION.md §3).

    python scripts/build_dataset.py \
        --sources data/bisindo_rhio:voc data/bisindo_user:folders \
        --out data/finetune --val-split 0.2 --seed 42

Pemisahan train/val menentukan (stratified per huruf) supaya hasil build bisa
direproduksi — seed yang sama selalu memberi pembagian yang sama.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from typing import Optional

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # import hand_detect/logs dari akar repo

import logs
from hand_detect import HandDetector, crop_hand
from recognizer import INPUT_SIZE

LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
_VOC_SPLITS = ("train", "test")
_LOG = logs.get_logger()


def _reset_out(out: str) -> None:
    """Buang train/val lama supaya sisa crop build sebelumnya tidak ikut
    terpakai (LetterDataset membaca semua isi folder)."""
    for split in ("train", "val"):
        shutil.rmtree(os.path.join(out, split), ignore_errors=True)


def _landmarks_from_bbox(
    x0: int, y0: int, x1: int, y1: int, w: int, h: int
) -> list[tuple[float, float]]:
    """Sudut bbox VOC jadi landmark sintetis 0..1: crop_hand memakai koordinat
    relatif, jadi strelet: 0..1 tidak perlu tahu ukuran asli kotak di frame."""
    return [(x0 / w, y0 / h), (x1 / w, y0 / h),
            (x1 / w, y1 / h), (x0 / w, y1 / h)]


def _voc_base_dirs(root: str) -> list[str]:
    """Layout VOC yang mungkin (urut prioritas: yang pertama menang).

      1. <root>/collectedimages/<split>   (layout VOC asli)
      2. <root>/<split>                   (layout hasil pindah di repo)

    Hanya SATU base per root yang dipakai: bila base pertama ada dan
    berisi anotasi .xml, base kedua diabaikan — supaya sumber yang sama
    tidak ter-stage dua kali sebagai dataset terpisah.
    """
    dirs: list[str] = []
    for base in (os.path.join(root, "collectedimages"), root):
        found = False
        for split in _VOC_SPLITS:
            folder = os.path.join(base, split)
            if not os.path.isdir(folder):
                continue
            if not any(n.endswith(".xml") for n in os.listdir(folder)):
                continue
            found = True
            real = os.path.realpath(folder)
            if real not in dirs:
                dirs.append(real)
        if found:
            break
    return dirs


def _voc_candidate_dirs(root: str) -> list[str]:
    """Folder VOC layak dibaca (alias pendek `_voc_base_dirs`)."""
    return _voc_base_dirs(root)


def _read_voc_annotations(
    items: list[tuple[str, str, np.ndarray]], folder: str
) -> int:
    """Baca satu folder VOC -> items; balikin jumlah pasangan jpg+xml valid."""
    copy_skipped = False
    staged = 0
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".xml"):
            continue
        if " - Copy" in name:  # salinan manual Finder/Explorer: byte-identik
            if not copy_skipped:
                _LOG.info("salinan ' - Copy' dilewati: %s", folder)
                copy_skipped = True
            continue
        jpg = os.path.join(folder, name[:-4] + ".jpg")
        if not os.path.exists(jpg):
            _LOG.warning("xml tanpa citra: %s", jpg)
            continue
        frame = cv2.imread(jpg)
        if frame is None:
            _LOG.warning("citra tidak terbaca: %s", jpg)
            continue
        tree = ET.parse(os.path.join(folder, name))
        obj = tree.getroot().find("object")
        if obj is None:
            _LOG.warning("xml tanpa object: %s", name)
            continue
        letter = (obj.findtext("name") or "").strip().upper()
        box = obj.find("bndbox")
        if letter not in LETTERS or box is None:
            _LOG.warning("anotasi tidak valid (label/bbox): %s", name)
            continue
        h, w = frame.shape[:2]
        items.append((letter, jpg, _landmarks_from_bbox(
            int(box.findtext("xmin")), int(box.findtext("ymin")),
            int(box.findtext("xmax")), int(box.findtext("ymax")), w, h
        )))
        staged += 1
    return staged


def read_voc_source(root: str) -> list[tuple[str, str, np.ndarray]]:
    """Baca sumber voc -> daftar (huruf, path_jpg, landmarks)."""
    items: list[tuple[str, str, np.ndarray]] = []
    for folder in _voc_candidate_dirs(root):
        _read_voc_annotations(items, folder)
    return items


def read_folders_source(root: str) -> list[tuple[str, str, None]]:
    """Baca sumber folders -> daftar (huruf, path_jpg, None=bbox dari MediaPipe)."""
    items: list[tuple[str, str, None]] = []
    for name in sorted(os.listdir(root)):
        letter_dir = os.path.join(root, name)
        if name not in LETTERS or not os.path.isdir(letter_dir):
            continue
        for file_name in sorted(os.listdir(letter_dir)):
            if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                items.append((name, os.path.join(letter_dir, file_name), None))
    return items


def load_detector() -> Optional[HandDetector]:
    """HandDetector hanya dimuat kalau ada sumber folders: satu landmarker
    untuk seluruh build jauh lebih murah daripada per-sumber."""
    try:
        return HandDetector()
    except Exception as exc:  # mediapipe/path bermasalah
        _LOG.error("detektor tangan gagal disiapkan: %s", exc)
        return None


def build_crop(
    frame: np.ndarray, landmarks: Optional[list[tuple[float, float]]],
    detector: Optional[HandDetector]
) -> Optional[np.ndarray]:
    """Crop 260x260: dari landmark bbox/voc, atau deteksi MediaPipe bila None."""
    if landmarks is None:
        if detector is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected = detector.detect(rgb)
        if detected is None:
            return None
        landmarks = detected
    crop = crop_hand(frame, landmarks)
    if crop is None:
        return None
    return cv2.resize(crop, INPUT_SIZE, interpolation=cv2.INTER_AREA)


def save_sample(
    out_split: str, letter: str, index: int, image: np.ndarray
) -> str:
    letter_dir = os.path.join(out_split, letter)
    os.makedirs(letter_dir, exist_ok=True)
    path = os.path.join(letter_dir, f"{index:04d}.jpg")
    cv2.imwrite(path, image)
    return path


def parse_sources(values: list[str]) -> list[tuple[str, str]]:
    """'data/x:voc' -> (data/x, voc). Format wajib voc|folders."""
    sources: list[tuple[str, str]] = []
    for value in values:
        path, sep, fmt = value.rpartition(":")
        fmt = fmt.lower().strip()
        if not sep or fmt not in ("voc", "folders"):
            raise SystemExit(
                f"ERROR: format sumber tidak valid: {value!r} "
                "(pakai <path>:voc atau <path>:folders)"
            )
        if not os.path.isdir(path):
            raise SystemExit(f"ERROR: folder sumber tidak ada: {path}")
        sources.append((path, fmt))
    return sources


def build(
    sources: list[tuple[str, str]], out: str, val_split: float, seed: int
) -> int:
    detector = load_detector() if any(f == "folders" for _, f in sources) else None
    _reset_out(out)
    by_letter: dict[str, list[tuple[str, np.ndarray]]] = {}
    seen_hashes: set[str] = set()  # md5 crop, lintas sumber: pertama menang

    for root, fmt in sources:
        staged: list[tuple[str, str, object]] = (
            read_voc_source(root) if fmt == "voc" else read_folders_source(root)
        )
        skipped = 0
        for letter, jpg_path, landmarks in staged:
            frame = cv2.imread(jpg_path)
            if frame is None:
                _LOG.warning("citra tidak terbaca: %s", jpg_path)
                skipped += 1
                continue
            image = build_crop(frame, landmarks, detector)
            if image is None:
                _LOG.warning(
                    "tanpa tangan terdeteksi & tanpa anotasi bbox: %s", jpg_path
                )
                skipped += 1
                continue
            digest = hashlib.md5(image.tobytes()).hexdigest()
            if digest in seen_hashes:
                _LOG.warning("crop duplikat lintas sumber, dilewati: %s", jpg_path)
                skipped += 1
                continue
            seen_hashes.add(digest)
            by_letter.setdefault(letter, []).append((jpg_path, image))
        print(f"sumber {root} ({fmt}): {len(staged) - skipped} citra, "
              f"{skipped} dilewati")

    rng = random.Random(seed)
    rows: list[tuple[str, int, int]] = []
    for letter in LETTERS:
        items = by_letter.get(letter, [])
        if not items:
            continue
        rng.shuffle(items)
        n_val = int(round(len(items) * val_split))
        n_train = len(items) - n_val
        for i, (_, image) in enumerate(items):
            split = "val" if i >= n_train else "train"
            save_sample(os.path.join(out, split), letter, i + 1, image)
        rows.append((letter, n_train, n_val))

    print()
    print(f"{'huruf':<6}{'train':>7}{'val':>7}")
    for letter, n_train, n_val in rows:
        print(f"{letter:<6}{n_train:>7}{n_val:>7}")
    total = sum(r[1] + r[2] for r in rows)
    print(f"{'total':<6}{sum(r[1] for r in rows):>7}{sum(r[2] for r in rows):>7}")
    print(f"\n{total} crop disimpan ke {out} "
          f"(train/val, val-split={val_split}, seed={seed})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rakit dataset fine-tuning BISINDO dari sumber voc/folders."
    )
    parser.add_argument("--sources", nargs="+", required=True,
                        help="<path>:<format>, format voc|folders")
    parser.add_argument("--out", default="data/finetune",
                        help="folder keluaran (train/<letter>, val/<letter>)")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="porsi validasi per huruf (0..1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed acak supaya pembagian bisa direproduksi")
    args = parser.parse_args()
    if not 0.0 <= args.val_split < 1.0:
        parser.error("--val-split harus 0 <= x < 1")
    logs.setup_logging()
    sources = parse_sources(args.sources)
    return build(sources, args.out, args.val_split, args.seed)


if __name__ == "__main__":
    sys.exit(main())
