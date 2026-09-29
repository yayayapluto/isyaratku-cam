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

WORKER = subprocess.Popen([sys.executable, "worker.py"], cwd=".")


def find_virtual_device() -> tuple[int, int]:
    """Cari device non-webcam (indeks > 0) yang bisa dibaca."""
    for idx in range(1, 7):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            return idx, frame.shape[1], frame.shape[0]
    raise RuntimeError("device virtual tidak ditemukan")


try:
    time.sleep(4)  # tunggu worker mulai streaming
    device, w, h = find_virtual_device()
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

    # header frame: strip 70px atas ditutup rectangle gelap + teks hijau
    header = best[:70]
    green = int(
        ((header[:, :, 1] > 150) & (header[:, :, 0] < 120) & (header[:, :, 2] < 120)).sum()
    )
    print(f"pixel hijau di header: {green}")
    assert green > 500, "teks overlay hijau tidak terdeteksi"
    print("M1 OK: stream virtual memuat overlay teks")
finally:
    WORKER.terminate()
    try:
        WORKER.wait(timeout=5)
    except subprocess.TimeoutExpired:
        WORKER.kill()
