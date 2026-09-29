"""M3: pemuat model A–Z + smoothing prediksi.

Satu sumber kebenaran untuk cara model dimuat; script uji dan worker keduanya
mengimpor dari sini (jangan duplikasi loader).
"""

from __future__ import annotations

import os
from collections import Counter, deque
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.models as torchvision_models
from torch import nn

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "models", "bisindo_alphabet",
                          "efficientnet_bisindo_sign_language.pth")

# [ASUMSI] ukuran input & normalisasi ImageNet — model card diam.
# Divalidasi manual saat pengukuran A-Z (docs/MODEL_SELECTION.md).
INPUT_SIZE = (224, 224)  # (width, height)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_model() -> tuple[nn.Module, list[str]]:
    """Muhat bobot. Kembalikan (model eval, label urut indeks)."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"bobot tidak ada: {MODEL_PATH}")
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = torchvision_models.efficientnet_b3(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4), nn.Linear(1536, 512), nn.SiLU(),
        nn.Dropout(p=0.3), nn.Linear(512, 26),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    labels: list[Optional[str]] = [None] * len(ckpt["class_to_idx"])
    for label, idx in ckpt["class_to_idx"].items():
        labels[idx] = label
    assert None not in labels, "class_to_idx tidak kontinyu"
    return model, [str(x) for x in labels]


def preprocess(frame: np.ndarray) -> torch.Tensor:
    img = cv2.resize(frame, INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)


class Smoother:
    """N-dari-5: label harus muncul >= 4 kali dari 5 frame agar stabil."""

    def __init__(self, window: int = 5, needed: int = 4) -> None:
        self._history: deque[str] = deque(maxlen=window)
        self._needed = needed

    def update(self, label: str) -> Optional[str]:
        self._history.append(label)
        if len(self._history) < self._history.maxlen:
            return None
        label, count = Counter(self._history).most_common(1)[0]
        return label if count >= self._needed else None

    def reset(self) -> None:
        self._history.clear()
