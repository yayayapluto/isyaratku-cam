"""Bukti M5: TTS Piper nyata terdengar di jalur VB-Cable.

Rekam dari "CABLE Output" (input device) sementara TTS memutar teks, lalu
periksa RMS di atas ambang. Exit 0 = suara benar-benar masuk kabel virtual.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import sounddevice as sd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


TEXT = "halo, ini tes suara Isyarat Ku Cam"
DURATION = 4.0


def find_cable_input() -> int:
    api_names = [a["name"] for a in sd.query_hostapis()]
    wasapi = api_names.index("Windows WASAPI") if "Windows WASAPI" in api_names else None
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and "CABLE Output" in d["name"]:
            if wasapi is not None and d["hostapi"] == wasapi:
                return i
    raise RuntimeError("CABLE Output tidak ditemukan sebagai input device")


def main() -> int:
    env = os.environ.copy()
    env["PATH"] = os.path.abspath(
        os.path.join("tools", "espeak-ng", "eSpeak NG")
    ) + os.pathsep + env.get("PATH", "")

    in_dev = find_cable_input()
    rate = int(sd.query_devices(in_dev)["default_samplerate"])
    print(f"rekam dari CABLE Output idx={in_dev} rate={rate}")

    worker = subprocess.Popen([sys.executable, "tts.py"], env=env)
    import tts

    recording = sd.rec(
        int(DURATION * rate), samplerate=rate, channels=1, dtype="int16",
        device=in_dev, blocking=False,
    )
    try:
        tts.speak(TEXT)
    finally:
        sd.wait()
        try:
            worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            worker.kill()

    samples = recording[:, 0].astype(np.float32)
    rms = float(np.sqrt(np.mean(samples**2)))
    peak = int(np.max(np.abs(samples)))
    print(f"RMS={rms:.1f} peak={peak}")
    assert rms > 100, f"suara tidak masuk VB-Cable (RMS={rms:.1f})"
    print("M5 OK: TTS masuk VB-Cable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
