"""M2: uji model A-Z di webcam nyata.

Model card TIDAK menyebutkan ukuran input, normalisasi, urutan label. Script
ini memakai nilai konvensi EfficientNet (224x224, mean/std ImageNet, urutan
A-Z) dan menampilkan top-3 tiap 10 frame agar bisa dikoreksi manual kalau
hasilnya aneh.
"""

from __future__ import annotations

import os
import sys
from collections import deque

import cv2
import torch

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
from recognizer import Smoother, build_model, preprocess  # noqa: E402



def main() -> int:
    try:
        model, labels = build_model()
    except FileNotFoundError as exc:
        print(f"model tidak tersedia: {exc}")
        return 1
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("webcam tidak terbuka")
        return 1

    print("tekan q untuk berhenti; tunjukkan huruf ke webcam")
    smoother = Smoother()
    frame_no = 0
    smooth = ""
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        # TIDAK di-flip: sama dengan pipeline produk (lihat catatan di
        # scripts/test_letters.py dan docs/results_m_flip.json).
        with torch.no_grad():
            logits = model(preprocess(frame))
            probs = torch.softmax(logits, dim=1)[0]
        top3 = torch.topk(probs, 3)
        top_labels = [labels[i] for i in top3.indices.tolist()]
        stable = smoother.update(top_labels[0])
        if stable:
            smooth = stable

        frame_no += 1
        if frame_no % 10 == 0:
            conf = ", ".join(
                f"{l}={p:.2f}" for l, p in zip(top_labels, top3.values.tolist())
            )
            print(f"frame {frame_no}: stabil={smooth} | {conf}")

        cv2.putText(frame, f"{smooth} {probs.max():.2f}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imshow("IsyaratKu Cam M2", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
