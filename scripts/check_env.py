"""Cek lingkungan sebelum demo (AC-06 / AC-07).

Periksa: (1) device audio virtual VB-Cable, (2) file model A-Z, (3) voice Piper,
(4) espeak-ng, (5) backend kamera virtual. Exit 0 = semua prasyarat demo ada.
"""

from __future__ import annotations

import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ok = True


def check(label: str, condition: bool, hint: str) -> None:
    global ok
    mark = "OK  " if condition else "MISS"
    print(f"[{mark}] {label}" + ("" if condition else f" — {hint}"))
    ok = ok and condition


def cable_device() -> tuple[bool, str]:
    try:
        import sounddevice as sd
    except ImportError:
        return False, "sounddevice belum terpasang"
    outs = [
        d for d in sd.query_devices()
        if "CABLE" in d["name"] and d["max_output_channels"] > 0
    ]
    if not outs:
        return False, "pasang VB-Cable: https://vb-audio.com/Cable/"
    return True, outs[0]["name"]


def model_files() -> tuple[bool, str]:
    d = os.path.join(ROOT, "models", "bisindo_alphabet")
    if not os.path.isdir(d):
        return False, "jalankan scripts/download_models.sh"
    if not any(f.endswith((".pt", ".pth", ".onnx")) for f in os.listdir(d)):
        return False, "folder model kosong — jalankan scripts/download_models.sh"
    return True, d


def voice_files() -> tuple[bool, str]:
    d = os.path.join(ROOT, "models", "voices")
    if not os.path.isdir(d) or not any(f.endswith(".onnx") for f in os.listdir(d)):
        return False, "voice Piper hilang — jalankan scripts/download_models.sh"
    return True, d


def hand_files() -> tuple[bool, str]:
    p = os.path.join(ROOT, "models", "hand", "hand_landmarker.task")
    return os.path.isfile(p), "jalankan scripts/download_models.sh"


def espeak() -> tuple[bool, str]:
    # cek persis seperti tts.py: PATH dulu, lalu kopi lokal tools/espeak-ng/
    if shutil.which("espeak-ng"):
        return True, "tersedia di PATH"
    local = os.path.join(
        ROOT, "tools", "espeak-ng", "eSpeak NG", "espeak-ng.exe"
    )
    if os.path.isfile(local):
        return True, f"tersedia lokal ({os.path.relpath(local, ROOT)})"
    return False, "extract MSI espeak-ng ke tools/espeak-ng/ (lihat README)"


def vcam_backend() -> tuple[bool, str]:
    try:
        import pyvirtualcam
        for backend in ("obs", "unitycapture"):
            try:
                with pyvirtualcam.Camera(width=160, height=120, fps=10, backend=backend):
                    return True, f"{backend} aktif"
            except Exception:
                continue
        return False, 'hidupkan "Start Virtual Camera" di OBS'
    except ImportError:
        return False, "pyvirtualcam belum terpasang"


def main() -> int:
    for label, fn in (
        ("VB-Cable (CABLE Output/Input)", cable_device),
        ("Model A-Z", model_files),
        ("Model tangan (MediaPipe)", hand_files),
        ("Voice Piper", voice_files),
        ("espeak-ng", espeak),
        ("Kamera virtual", vcam_backend),
    ):
        good, detail = fn()
        check(label, good, detail)
    print("SEMUA SIAP" if ok else "ADA YANG KURANG")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
