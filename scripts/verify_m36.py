"""Bukti M3 (deteksi+klasifikasi) dan M4 (pipeline): worker end-to-end tanpa GUI.

Gate diukur dari sisi penulis, bukan dari "device ketemu atau tidak":
1. worker benar-benar mulai (status "berjalan"),
2. frame benar-benar terkirim ke kamera virtual (sent_frames >= 10),
3. MediaPipe benar-benar menemukan tangan (hand_frames > 0),
4. model benar-benar mengklasifikasi crop tangan (inferred_frames > 0).

TTS sengaja dimatikan; suara nyata diuji terpisah di scripts/verify_m5.py.

Jalankan: python scripts/verify_m36.py
"""

from __future__ import annotations

import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import worker as worker_mod

MIN_SENT = 10
TIMEOUT = 60  # detik


def main() -> int:
    statuses: list[str] = []
    w = worker_mod.Worker(on_status=statuses.append, enable_tts=False)

    t = threading.Thread(target=w.run, daemon=True)
    t.start()

    deadline = time.time() + TIMEOUT
    while time.time() < deadline:
        if any("berjalan" in s for s in statuses):
            break
        if w.tts_error or statuses and "gagal" in (statuses[-1] or ""):
            break
        time.sleep(0.2)

    started = any("berjalan" in s for s in statuses)
    if started:
        # biarkan loop berjalan sampai gate terpenuhi atau batas waktu
        end = time.time() + TIMEOUT
        while time.time() < end:
            if w.sent_frames >= MIN_SENT and w.hand_frames > 0:
                break
            time.sleep(0.2)

    w.stop()
    t.join(timeout=8)

    results = {
        "worker mulai": started,
        f"frame terkirim >= {MIN_SENT}": w.sent_frames >= MIN_SENT,
        "MediaPipe menemukan tangan": w.hand_frames > 0,
        "model mengklasifikasi crop": w.inferred_frames > 0,
    }
    for name, ok in results.items():
        print(f"{'OK  ' if ok else 'GAGAL'} {name}")
    print(
        f"sent={w.sent_frames} hand={w.hand_frames} inferred={w.inferred_frames} "
        f"letter_terakhir={w.last_letter}"
    )

    failed = [k for k, ok in results.items() if not ok]
    if failed:
        print(f"GAGAL: {', '.join(failed)}")
        print(f"status worker: {statuses[-3:]}")
        return 1
    print("M3-M4 OK: deteksi tangan, klasifikasi crop, dan stream kamera virtual aktif")
    return 0


if __name__ == "__main__":
    sys.exit(main())
