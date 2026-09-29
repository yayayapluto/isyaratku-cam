"""Deteksi tangan + crop bbox (MediaPipe Tasks API).

MediaPipe 1.0.1 tidak lagi punya `solutions.Hands`; pakai Tasks
HandLandmarker. Crop = bbox landmark + padding (TECH_SPEC §4.2).
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL = os.path.join(ROOT, "models", "hand", "hand_landmarker.task")
PADDING = 20  # px, sesuai TECH_SPEC §4.2
MIN_CROP = 32  # px; crop lebih kecil dari ini tidak dipakai untuk prediksi

_landmarker = None


class HandDetector:
    """Deteksi tangan; kumpulkan landmark x/y (0..1) per tangan.

    num_hands=2: BISINDO memakai bentuk kedua tangan dalam sebagian huruf,
    jadi bbox harus mencakup keduanya (lihat docs/MODEL_SELECTION.md).
    """

    def __init__(self, num_hands: int = 2) -> None:
        global _landmarker
        if _landmarker is None:
            if not os.path.exists(HAND_MODEL):
                raise FileNotFoundError(
                    f"model tangan tidak ada: {HAND_MODEL} "
                    "(jalankan scripts/download_models.sh)"
                )
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core import base_options

            options = vision.HandLandmarkerOptions(
                base_options=base_options.BaseOptions(
                    model_asset_path=HAND_MODEL
                ),
                num_hands=num_hands,
            )
            _landmarker = vision.HandLandmarker.create_from_options(options)
        self._landmarker = _landmarker

    def detect(self, frame_rgb: np.ndarray) -> Optional[list[tuple[float, float]]]:
        """Landmark SEMUA tangan yang terlihat sebagai [(x, y)] 0..1, atau None."""
        import mediapipe as mp

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect(image)
        if not result.hand_landmarks:
            return None
        return [(lm.x, lm.y) for hand in result.hand_landmarks for lm in hand]


def crop_hand(
    frame_bgr: np.ndarray, landmarks: list[tuple[float, float]]
) -> Optional[np.ndarray]:
    """Crop bbox tangan + padding, lalu letterbox ke persegi.

    Letterbox: rasio aspek asli dipertahankan (padding tepi), supaya tangan
    tinggi tidak gepeng saat di-resize 224x224 di recognizer.preprocess.
    Kembalikan None kalau hasil crop jauh lebih kecil dari MIN_CROP.
    """
    h, w = frame_bgr.shape[:2]
    xs = [x * w for x, _ in landmarks]
    ys = [y * h for _, y in landmarks]
    x0 = max(0, int(min(xs)) - PADDING)
    y0 = max(0, int(min(ys)) - PADDING)
    x1 = min(w, int(max(xs)) + PADDING)
    y1 = min(h, int(max(ys)) + PADDING)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = frame_bgr[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    if max(ch, cw) < MIN_CROP:
        return None  # terlalu jauh dari kamera: prediksi hasil upscale tak berguna

    side = max(ch, cw)
    out = np.zeros((side, side, 3), dtype=frame_bgr.dtype)
    out[(side - ch) // 2:(side - ch) // 2 + ch,
        (side - cw) // 2:(side - cw) // 2 + cw] = crop
    return out


if __name__ == "__main__":
    import cv2

    detector = HandDetector()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    print("tunjukkan tangan, q untuk keluar")
    frames_with_hand = 0
    total = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        marks = detector.detect(rgb)
        total += 1
        if marks:
            frames_with_hand += 1
            crop = crop_hand(frame, marks)
            if crop is not None:
                cv2.imshow("crop (letterbox)", crop)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"tangan terdeteksi {frames_with_hand}/{total} frame")
    assert frames_with_hand > 0, "MediaPipe tidak menemukan tangan sama sekali"
    print("hand detector self-check OK")
