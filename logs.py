"""Logging stdlib: log biasa (console + file) + log debug (file).

Satu file, satu tanggung jawab. Bukan framework: modul `logging` bawaan
Python, tanpa dependency baru, tanpa config UI, tanpa rotasi.

Handler dipasang di logger "isyaratku" (bukan root) supaya level DEBUG bisa
disaring per handler: console & file biasa INFO, file debug DEBUG. Kalau root
di-set INFO, record DEBUG dibuang SEBELUM sampai handler dan file debug selalu
kosong. Menggunakan satu logger bernama juga menjaga catatan dari library
(qt, mediapipe) keluar dari file debug aplikasi.

- logs/isyaratku.log       : level INFO, console + file
- logs/isyaratku-debug.log : level DEBUG, file saja (diagnostik per frame)

Pemakaian:
    from logs import setup_logging, get_logger, stage_timer
    setup_logging(debug=True)          # sekali, di awal program
    log = get_logger()
    log.info("siap")
    with stage_timer("t_model"):
        ...                            # durasi masuk log debug
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "isyaratku.log")
DEBUG_FILE = os.path.join(LOG_DIR, "isyaratku-debug.log")
LOGGER_NAME = "isyaratku"

_FMT = "%(asctime)s %(levelname)s %(name)s %(message)s"
_READY = False


def setup_logging(debug: bool = False) -> None:
    """Pasang handler sekali (idempoten). Logger aplikasi level DEBUG; setiap
    handler menyaring sendiri: console INFO (DEBUG kalau debug=True), file
    biasa INFO, file debug selalu DEBUG."""
    global _READY
    if _READY:
        return
    os.makedirs(LOG_DIR, exist_ok=True)
    log = logging.getLogger(LOGGER_NAME)
    log.setLevel(logging.DEBUG)
    log.propagate = False   # handler kita sudah cukup; root dibiarkan sendiri
    formatter = logging.Formatter(_FMT)

    # Console: INFO biasa, DEBUG hanya saat mode debug diminta.
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if debug else logging.INFO)
    console.setFormatter(formatter)
    log.addHandler(console)

    # File biasa: append, utf-8, INFO.
    plain = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    plain.setLevel(logging.INFO)
    plain.setFormatter(formatter)
    log.addHandler(plain)

    # File debug: append, utf-8, DEBUG — SELALU diisi walau console normal,
    # supaya diagnosis per-frame tetap bisa dibaca sesudah kejadian.
    debug_file = logging.FileHandler(DEBUG_FILE, mode="a", encoding="utf-8")
    debug_file.setLevel(logging.DEBUG)
    debug_file.setFormatter(formatter)
    log.addHandler(debug_file)

    _READY = True


def get_logger() -> logging.Logger:
    """Logger aplikasi. Aman walau setup_logging belum jalan: tanpa handler
    -> tidak error, hanya tidak ada keluaran."""
    return logging.getLogger(LOGGER_NAME)


@contextmanager
def stage_timer(stage: str):
    """Catat durasi satu tahap ke log debug: `<stage> dur_ms=12.34`."""
    start = time.perf_counter()
    try:
        yield
    finally:
        ms = (time.perf_counter() - start) * 1000.0
        get_logger().debug("%s dur_ms=%.2f", stage, ms)
