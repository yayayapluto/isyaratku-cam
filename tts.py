"""TTS offline: Piper (ONNX) -> perangkat audio virtual VB-Cable.

Cari device "CABLE Input" berdasarkan nama; error jelas kalau tidak ada.
Pilih host API WASAPI (48 kHz, 2 kanal) karena MME/DirectSound mendaftar
entri ganda untuk perangkat yang sama.
"""

from __future__ import annotations
import os
import queue
import time
import shutil
import sys
import threading

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

_queue: Optional["queue.Queue[str]"] = None
DEBUG_MONITOR = "--debug" in sys.argv   # speaker nyata, bukan CABLE Input

def _ensure_worker() -> None:
    """Buat antrean + thread daemon TTS sekali saja (dipakai prewarm/enqueue)."""
    global _queue
    if _queue is None:
        _queue = queue.Queue()
        threading.Thread(target=_tts_worker, daemon=True,
                         name="isyaratku-tts").start()


def prewarm() -> None:
    """Muat voice SEBELUM loop video: pemanggilan pertama ~1,7 s, dan di luar
    loop itu tak terlihat sebagai freeze. Aman dipanggil berulang (no-op)."""
    _ensure_worker()
    global _voice
    with _lock:
        if _voice is None:
            _voice = _load_voice()


def speak_async(text: str) -> None:
    """Antre ucapan; thread daemon memulainya. Tidak pernah menahan loop video:
    worker video tetap mengirim frame selagi audio diputar terpisah."""
    _ensure_worker()
    # Antrean dibiarkan panjang: kata yang dibuang di tengah jalan terdengar
    # terpotong. Backlog dibersihkan hanya kalau sudah terlalu jauh (>6),
    # supaya ucapan tetap natural walau user berhenti agak terlambat.
    while _queue.qsize() > 6:
        _queue.get_nowait()
        _queue.task_done()
    _queue.put(text)


_last_error: str = ""   # kegagalan terakhir; dibaca worker untuk status debug


def last_error() -> str:
    """Pesan kegagalan TTS terakhir (kosong = sukses)."""
    return _last_error


def _tts_worker() -> None:
    global _last_error
    while True:
        text = _queue.get()
        try:
            _emit(text)
            _last_error = ""
        except Exception as exc:
            # Semua kegagalan dilaporkan: TTS yang mati tanpa pesan tak
            # bisa dibedakan dari "tidak ada yang diucapkan". Ditulis ke
            # _last_error juga supaya jendela debug tak mengklaim "aktif".
            _last_error = f"{type(exc).__name__}: {exc}"
            print(f"TTS dilewati: {_last_error}")
        _queue.task_done()


def _emit(text: str) -> None:
    """Ucapkan teks, blocking (thread khusus TTS, bukan thread video)."""
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

    # Mode debug: keluarkan ke speaker nyata supaya bisa didengar saat
    # kembangkan (CABLE Input cuma didengar aplikasi seperti Zoom/Meet).
    # Indeks default diresolve dulu: query_devices(None) mengembalikan
    # seluruh tabel device, bukan info default output.

    device = sd.default.device[1] if DEBUG_MONITOR else cable_output_device()
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


def speak(text: str) -> None:
    """Blocking — untuk skrip verifikasi (verify_m5) yang memanggil langsung."""
    _emit(text)


def drain(timeout: float = 10.0) -> None:
    """Tunggu semua antrean selesai (dipakai BERHENTI di GUI).

    Batas waktu di sisi pemanggil: kata yang sudah masuk tetap diputar sampai
    habis selama waktu cukup; kalau kebot terlampaui, sisa tetap diantre dan
    daemon melanjutkan memutarnya.
    """
    if _queue is None:
        return
    selesai = threading.Event()

    def _join() -> None:
        _queue.join()
        selesai.set()

    threading.Thread(target=_join, daemon=True, name="isyaratku-drain").start()
    selesai.wait(timeout)


if __name__ == "__main__":
    try:
        speak("halo, ini tes suara Isyarat Ku Cam")
        print("TTS OK")
    except TtsUnavailable as exc:
        print(f"TTS TIDAK SIAP: {exc}")
