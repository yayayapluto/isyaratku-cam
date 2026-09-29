"""Bukti M1: worker streaming -> OBS Virtual Camera dibaca ulang sebagai input.

Menjalankan worker di background, lalu membaca device virtual dan memeriksa
teks overlay muncul (area header bukan hitam polos).
"""

from __future__ import annotations

import subprocess
import sys
import time

import cv2
import numpy as np

def _spawn_worker() -> subprocess.Popen:
    """Jalankan worker dengan kata overlay dipaksa non-kosong.

    Tanpa tangan, `word` kosong dan overlay memang tidak digambar. Properti
    `text` dipaksa mengembalikan kata uji, kalau tidak gate selalu baca 0."""
    code = (
        "import text_pipeline;"
        "text_pipeline.TextPipeline.text = property(lambda self: 'HALO');"
        "import worker;"
        "worker.Worker(on_status=None, enable_tts=False).run()"
    )
    return subprocess.Popen([sys.executable, "-c", code], cwd=".")


WORKER = _spawn_worker()


def green_subtitle_pixels(frame: np.ndarray) -> int:
    """Jumlah pixel teks subtitle hijau di strip bawah frame.

    Mask dominan hijau (bukan >150 absolut) supaya tahan anti-aliasing di
    atas latar webcam."""
    footer = frame[-140:]
    return int(
        ((footer[:, :, 1].astype(int) - footer[:, :, 0].astype(int) > 40) &
         (footer[:, :, 1].astype(int) - footer[:, :, 2].astype(int) > 40)).sum()
    )


def find_virtual_device() -> tuple[int, np.ndarray]:
    """Device yang benar-benar memuat overlay, bukan sekadar bisa dibaca."""
    for idx in range(1, 7):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None and green_subtitle_pixels(frame) > 50:
            return idx, frame
    raise RuntimeError("device virtual dengan overlay tidak ditemukan")


try:
    # torch + mediapipe + webcam perlu waktu; coba berapa kali sampai streaming
    deadline = time.time() + 90
    device, first = None, None
    while time.time() < deadline:
        try:
            device, first = find_virtual_device()
            break
        except RuntimeError:
            time.sleep(2)
    if device is None:
        raise RuntimeError("device virtual tidak mulai streaming dalam 90 detik")
    h, w = first.shape[:2]
    print(f"device virtual: index={device} ukuran={w}x{h}")

    cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)
    time.sleep(1.5)
    best = None
    for _ in range(5):
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        best = frame
    cap.release()

    if best is None:
        raise RuntimeError("tidak bisa membaca frame dari device virtual")

    green = green_subtitle_pixels(best)
    # ambang skala lebar frame: 640x480 mengukur ~414, 1280x720 ~900+
    threshold = max(50, int(best.shape[1] * 0.4))
    print(f"pixel hijau di area subtitle: {green} (ambang {threshold})")
    assert green > threshold, "teks subtitle hijau tidak terdeteksi"
    print("M1 OK: stream virtual memuat subtitle film di bawah frame")
finally:
    WORKER.terminate()
    try:
        WORKER.wait(timeout=5)
    except subprocess.TimeoutExpired:
        WORKER.kill()
