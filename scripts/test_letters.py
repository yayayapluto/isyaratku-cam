"""M2: ukur akurasi model A-Z di webcam nyata, satu huruf per giliran.

Untuk tiap huruf A-Z: user memosisikan tangan lalu menekan Enter; script
menangkap beberapa frame, menjalankan model dengan smoothing 4-dari-5 yang
SAMA dengan produk, lalu mencatat hasil.

Dua kondisi diukur sekaligus (A/B di kamera & cahaya yang sama):
  --full-frame : model makan frame penuh (baseline lama)
  default      : model makan crop tangan + letterbox (seperti worker.py)

Tanpa baseline full-frame, hasil buruk tidak bisa diatribusi ke crop.

Output: docs/results_m2.json (data mentah, jangan diedit manual).
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import torch

import recognizer
from hand_detect import HandDetector, crop_hand

FRAMES_PER_LETTER = 20
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "results_m2.json")
FULL_FRAME = "--full-frame" in sys.argv


def predict(model, frame: torch.Tensor, labels: list[str]) -> tuple[str, float]:
    with torch.no_grad():
        probs = torch.softmax(model(frame), dim=1)[0]
    idx = int(probs.argmax())
    return labels[idx], float(probs.max())


def measure_letter(model, labels, detector, cap, letter: str) -> dict | None:
    """Satu huruf: tangkap 20 frame, kembalikan ringkasan. None kalau tak ada tangan."""
    for _ in range(5):  # buang frame lama, stabilkan auto-exposure
        cap.read()
    smoother = recognizer.Smoother()  # policy identik produk
    confidences: list[float] = []
    raw: list[str] = []
    stable_hits: list[str] = []
    for _ in range(FRAMES_PER_LETTER):
        ok, frame = cap.read()
        if not ok:
            continue
        # TIDAK di-flip: worker mengirim crop tak dicermin, dan A/B terukur
        # (docs/results_m_flip.json) menunjukkan asli > flip 4,6 poin.
        if FULL_FRAME:
            tensor = recognizer.preprocess(frame)
        else:
            marks = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if marks is None:
                continue  # frame tanpa tangan tidak dihitung
            crop = crop_hand(frame, marks)
            if crop is None:
                continue  # tangan terlalu kecil/jauh
            tensor = recognizer.preprocess(crop)
        label, conf = predict(model, tensor, labels)
        confidences.append(conf)
        raw.append(label)
        stable = smoother.update(label)
        if stable:
            stable_hits.append(stable)
        cv2.putText(frame, f"target={letter} now={label}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imshow("IsyaratKu Cam M2", frame)
        cv2.waitKey(1)

    if not raw:
        return None
    # prediksi stabil = mayoritas dari label yang lolos gate 4-dari-5
    stable = max(set(stable_hits), key=stable_hits.count) if stable_hits else \
        max(set(raw), key=raw.count)
    return {
        "target": letter,
        "prediksi_stabil": stable,
        "benar": stable == letter,
        "confidence_rata": round(sum(confidences) / len(confidences), 3),
        "kondisi": "full-frame" if FULL_FRAME else "crop+letterbox",
        "frame_terukur": len(raw),
        "semua_prediksi": raw,
    }


def main() -> int:
    model, labels = recognizer.build_model()
    detector = None if FULL_FRAME else HandDetector()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("webcam tidak terbuka")
        return 1

    mode = "FRAME PENUH (baseline)" if FULL_FRAME else "CROP + LETTERBOX"
    results: dict[str, dict] = {}
    print("=== Pengukuran akurasi A-Z ===")
    print(f"Kondisi input: {mode}")
    print("Untuk tiap huruf: posisikan tangan, Enter untuk tangkap "
          f"({FRAMES_PER_LETTER} frame). q + Enter untuk berhenti.")

    for letter in labels:
        try:
            answer = input(f"\nHuruf {letter}: [Enter] tangkap, [q] stop: ").strip().lower()
        except EOFError:
            break
        if answer == "q":
            break
        record = measure_letter(model, labels, detector, cap, letter)
        if record is None:
            print(f"  -> {letter}: LEWAT (tangan tidak terdeteksi, 0 frame terukur)")
            continue
        results[letter] = record
        print(f"  -> {letter}: {'BENAR' if record['benar'] else 'SALAH'} "
              f"(stabil={record['prediksi_stabil']}, "
              f"conf={record['confidence_rata']:.2f})")

    cap.release()
    cv2.destroyAllWindows()

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    if not results:
        print("tidak ada data")
        return 0
    benar = sum(r["benar"] for r in results.values())
    print(f"\nAkurasi ({mode}): {benar}/{len(results)} = "
          f"{benar / len(results):.0%}  -> {OUT}")
    if ragged := {k: v["prediksi_stabil"] for k, v in results.items()
                  if not v["benar"]}:
        print(f"Salah: {ragged}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
