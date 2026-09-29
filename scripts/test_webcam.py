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
import numpy as np
import torch
import torchvision.models as models
from torch import nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "bisindo_alphabet",
                          "efficientnet_bisindo_sign_language.pth")
# [ASUMSI] standar ImageNet; divalidasi manual saat tes
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SIZE = (224, 224)  # (width, height)


def build_model() -> tuple[nn.Module, list[str]]:
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = models.efficientnet_b3(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4), nn.Linear(1536, 512), nn.SiLU(),
        nn.Dropout(p=0.3), nn.Linear(512, 26),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    labels = [None] * len(ckpt["class_to_idx"])
    for label, idx in ckpt["class_to_idx"].items():
        labels[idx] = label
    assert None not in labels, "class_to_idx tidak kontinyu"
    return model, labels


def preprocess(frame: np.ndarray) -> torch.Tensor:
    img = cv2.resize(frame, SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)


def main() -> int:
    if not os.path.exists(MODEL_PATH):
        print(f"model tidak ada: {MODEL_PATH}")
        return 1
    model, labels = build_model()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("webcam tidak terbuka")
        return 1

    print("tekan q untuk berhenti; tunjukkan huruf ke webcam")
    history: deque[str] = deque(maxlen=5)
    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        with torch.no_grad():
            logits = model(preprocess(frame))
            probs = torch.softmax(logits, dim=1)[0]
        top3 = torch.topk(probs, 3)
        top_labels = [labels[i] for i in top3.indices.tolist()]
        history.append(top_labels[0])
        smooth = max(set(history), key=history.count)

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
