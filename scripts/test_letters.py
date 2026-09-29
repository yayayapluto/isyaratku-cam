"""M2: ukur akurasi model A-Z di webcam nyata, satu huruf per giliran.

Untuk tiap huruf A-Z: user memosisikan tangan lalu menekan Enter; script
menangkap beberapa frame, menjalankan model dengan smoothing 4-dari-5, lalu
mencatat hasil. Output: docs/results_m2.json (data mentah, jangan diedit manual).

Jalankan: python scripts/test_letters.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch

import recognizer
from hand_detect import HandDetector, crop_hand

FRAMES_PER_LETTER = 20
SMOOTH_WINDOW = 5
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "results_m2.json")


def predict(model, frame: np.ndarray, labels: list[str]) -> tuple[str, float]:
    with torch.no_grad():
        probs = torch.softmax(model(recognizer.preprocess(frame)), dim=1)[0]
    idx = int(probs.argmax())
    return labels[idx], float(probs.max())


def main() -> int:
    model, labels = recognizer.build_model()
    detector = HandDetector()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("webcam tidak terbuka")
        return 1

    results = {}
    print("=== Pengukuran akurasi A-Z ===")
    print("Untuk tiap huruf: posisikan tangan di kotak, tekan Enter saat stabil.")
    print("q + Enter kapan saja untuk berhenti lebih awal.")

    for letter in labels:
        try:
            answer = input(
                f"\nHuruf {letter}: posisikan tangan, Enter untuk tangkap "
                f"({FRAMES_PER_LETTER} frame), atau q untuk stop: "
            ).strip().lower()
        except EOFError:
            break
        if answer == "q":
            break
        for _ in range(5):  # buang frame lama, stabilkan auto-exposure
            cap.read()
        votes: deque[str] = deque(maxlen=SMOOTH_WINDOW)
        confidences = []
        frames = []
        for _ in range(FRAMES_PER_LETTER):
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            marks = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if marks is None:
                continue  # frame tanpa tangan tidak dihitung
            label, conf = predict(model, crop_hand(frame, marks), labels)
            votes.append(label)
            confidences.append(conf)
            frames.append(label)
            smooth = max(set(votes), key=votes.count)
            cv2.putText(frame, f"target={letter} stabil={smooth}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.imshow("IsyaratKu Cam M2", frame)
            cv2.waitKey(1)
        if not frames:
            print(f"  -> {letter}: LEWAT (tangan tidak terdeteksi)")
            continue
        stable = max(set(frames), key=frames.count)
        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
        correct = stable == letter
        results[letter] = {
            "target": letter,
            "prediksi_stabil": stable,
            "benar": correct,
            "confidence_rata": round(mean_conf, 3),
            "semua_prediksi": frames,
        }
        print(f"  -> {letter}: {'BENAR' if correct else 'SALAH'} "
              f"(stabil={stable}, conf={mean_conf:.2f})")

    cap.release()
    cv2.destroyAllWindows()

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    if results:
        acc = sum(r["benar"] for r in results.values()) / len(results)
        print(f"\nAkurasi: {sum(r['benar'] for r in results.values())}/{len(results)}"
              f" = {acc:.0%}  -> {OUT}")
    else:
        print("tidak ada data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
