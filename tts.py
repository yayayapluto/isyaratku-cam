"""TTS offline: Piper (ONNX) -> perangkat audio virtual VB-Cable.

Cari device "CABLE Input" berdasarkan nama; error jelas kalau tidak ada.
Pilih host API WASAPI (48 kHz, 2 kanal) karena MME/DirectSound mendaftar
entri ganda untuk perangkat yang sama.
"""

from __future__ import annotations

import os
import shutil
import threading
from typing import Optional

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "models", "voices")
VOICE = "id_ID-news_tts-medium"
CABLE_HINT = "CABLE"

_voice = None
_lock = threading.Lock()


class TtsUnavailable(RuntimeError):
    """Device/model/espeak tidak siap."""


def _ensure_espeak() -> None:
    """Cari espeak-ng: PATH dulu, lalu kopi lokal tools/espeak-ng/ (extract MSI)."""
    if shutil.which("espeak-ng"):
        return
    bundled = os.path.join(ROOT, "tools", "espeak-ng", "eSpeak NG")
    exe = os.path.join(bundled, "espeak-ng.exe")
    if not os.path.exists(exe):
        raise TtsUnavailable(
            "espeak-ng tidak ditemukan (fonemisasi Piper id_ID) — pasang sistem-wide"
            " atau extract MSI ke tools/espeak-ng/"
        )
    os.environ["PATH"] = bundled + os.pathsep + os.environ.get("PATH", "")


def _load_voice():
    import piper  # import lokal; modul berat

    _ensure_espeak()
    onnx = os.path.join(MODEL_DIR, f"{VOICE}.onnx")
    if not os.path.exists(onnx):
        raise TtsUnavailable(f"voice tidak ditemukan: {onnx}")
    return piper.PiperVoice.load(onnx)


def cable_output_device() -> int:
    """Indeks sounddevice untuk playback ke VB-Cable (CABLE Input)."""
    import sounddevice as sd

    api_names = [a["name"] for a in sd.query_hostapis()]
    wasapi = api_names.index("Windows WASAPI") if "Windows WASAPI" in api_names else None
    best: Optional[int] = None
    for i, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] <= 0 or CABLE_HINT not in d["name"]:
            continue
        if wasapi is not None and d["hostapi"] == wasapi and d["max_output_channels"] == 2:
            return i  # WASAPI stereo = paling bisa diandalkan
        best = best if best is not None else i
    if best is None:
        raise TtsUnavailable("VB-Cable tidak ditemukan (pasang vb-audio.com/Cable)")
    return best


def speak(text: str) -> None:
    """Ucapkan teks. Dipanggil dari thread kerja, non-blocking terhadap GUI."""
    text = text.strip()
    if not text:
        return
    global _voice
    with _lock:
        if _voice is None:
            _voice = _load_voice()
        chunks = list(_voice.synthesize(text))
    if not chunks:
        return
    audio = np.concatenate([c.audio_int16_array for c in chunks]).astype(np.float32)
    src_rate = chunks[0].sample_rate

    import sounddevice as sd

    device = cable_output_device()
    dst_rate = int(sd.query_devices(device)["default_samplerate"])
    if dst_rate != src_rate:
        # ponytail: resample linear (np.interp), tambah soxr bila kualitas kurang
        n = int(len(audio) * dst_rate / src_rate)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio
        )
    sd.play(
        np.clip(audio, -32768, 32767).astype(np.int16),
        samplerate=dst_rate, device=device, blocking=True,
    )


if __name__ == "__main__":
    try:
        speak("halo, ini tes suara Isyarat Ku Cam")
        print("TTS OK")
    except TtsUnavailable as exc:
        print(f"TTS TIDAK SIAP: {exc}")
