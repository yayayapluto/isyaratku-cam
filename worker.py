"""Walking skeleton: webcam -> teks overlay -> kamera virtual.

M1 (BUILD_ORDER.md): membuktikan pyvirtualcam + capture hidup di mesin ini.
Belum ada MediaPipe atau model. Backend kamera virtual: unitycapture lebih
dulu, fallback OBS Virtual Camera (karena Unity Capture DirectShow filter tidak
selalu terpasang).
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np
import pyvirtualcam

WIDTH, HEIGHT, FPS = 1280, 720, 20
# pyvirtualcam mengiklankan 1280x720, tapi OBS Virtual Camera mennegosiasikan
# 640x480 saat dibaca ulang (terukur). Frame di-resize ke WIDTHxHEIGHT.
# urutan backend: Unity Capture (DirectShow filter) lalu OBS Virtual Camera
BACKEND_ORDER = ("unitycapture", "obs")


def open_vcam(preferred: Optional[str] = None) -> tuple:
    """Buka kamera virtual. Kembalikan (camera, backend_name)."""
    order = (preferred,) if preferred else BACKEND_ORDER
    errors = []
    for name in order:
        try:
            cam = pyvirtualcam.Camera(
                width=WIDTH, height=HEIGHT, fps=FPS, backend=name
            )
            return cam, name
        except Exception as exc:  # backend tidak terpasang / device lemah
            errors.append(f"{name}: {exc}")
    raise RuntimeError(" | ".join(errors))


def open_webcam() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("webcam tidak ditemukan")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    return cap


def draw_text(frame: np.ndarray, text: str) -> np.ndarray:
    cv2.rectangle(frame, (0, 0), (WIDTH, 70), (20, 20, 20), -1)
    cv2.putText(
        frame, text, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
        (0, 255, 0), 3, cv2.LINE_AA,
    )
    return frame


def main() -> None:
    try:
        cam, backend = open_vcam()
    except RuntimeError as exc:
        print(f"ERROR kamera virtual: {exc}")
        return
    cap = open_webcam()
    print(f"kamera virtual: {cam.device} (backend={backend}) — Ctrl+C untuk stop")

    start = time.time()
    frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (WIDTH, HEIGHT))
            fps_now = frames / (time.time() - start) if time.time() > start else 0
            draw_text(frame, f"IsyaratKu Cam M1 | {backend} | {fps_now:4.1f} FPS")
            cam.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cam.sleep_until_next_frame()
            frames += 1
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cam.close()
        print(f"selesai, {frames} frame")


if __name__ == "__main__":
    main()
