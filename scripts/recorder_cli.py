"""Rekam data latih BISINDO tambahan dari webcam pengguna.

Tujuan: menambah variasi tangan pengguna sendiri ke dataset VOC yang sudah
ada, sehingga fine-tuning tidak hanya cocok ke satu persona.

    python scripts/recorder_cli.py --out data/bisindo_user --letter A --count 100
    python scripts/recorder_cli.py --out data/bisindo_user --all

Frame disimpan mentah (belum di-crop): pemotongan & resize 260x260 dikerjakan
scripts/build_dataset.py supaya satu sumber citra cukup untuk semua eksperimen.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from typing import Optional

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # import hand_detect/logs dari akar repo

import logs
from hand_detect import HandDetector

LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
_INDEX_RE = re.compile(r"^([A-Z])_(\d+)\.jpg$", re.IGNORECASE)
_LOG = logs.get_logger()


def open_webcam(camera: int) -> cv2.VideoCapture:
    """Backend capture: MSMF dulu, DSHOW cadangan.

    Urutan diambil dari worker.open_webcam (terukur: MSMF ~22 frame/detik,
    DSHOW ~7). DSHOW hanya menyelamatkan webcam lama tanpa pipeline MSMF."""
    for api in (cv2.CAP_MSMF, cv2.CAP_DSHOW):
        cap = cv2.VideoCapture(camera, api)
        if cap.isOpened():
            return cap
        cap.release()
    raise RuntimeError("webcam tidak ditemukan")


def next_index(letter_dir: str, letter: str) -> int:
    """Nomor berilanjut berikutnya = max(n) dari berkas yang sudah ada + 1,
    supaya rekaman lama tidak pernah ditimpa."""
    top = 0
    if os.path.isdir(letter_dir):
        for name in os.listdir(letter_dir):
            m = _INDEX_RE.match(name)
            if m and m.group(1).upper() == letter.upper():
                top = max(top, int(m.group(2)))
    return top + 1


def save_frame(letter_dir: str, letter: str, frame: np.ndarray) -> str:
    path = os.path.join(letter_dir, f"{letter}_{next_index(letter_dir, letter)}.jpg")
    cv2.imwrite(path, frame)
    return path


def draw_preview(
    frame: np.ndarray,
    letter: str,
    saved: int,
    target: Optional[int],
    landmarks: Optional[list[tuple[float, float]]],
) -> None:
    """Overlay huruf sasaran + kotak tangan. Kotak dari bbox landmark apa
    adanya (bukan crop_hand) supaya yang digambar benar-benar wilayah yang
    dilihat pengguna, bukan hasil letterbox."""
    h, w = frame.shape[:2]
    if landmarks:
        xs = [x for x, _ in landmarks]
        ys = [y for _, x in landmarks]
        x0 = max(0, int(min(xs) * w))
        y0 = max(0, int(min(ys) * h))
        x1 = min(w, int(max(xs) * w))
        y1 = min(h, int(max(ys) * h))
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 200, 0), 2)
        cv2.putText(frame, "tangan terdeteksi", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
    else:
        cv2.putText(frame, "tangan tidak terlihat", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 220), 2)

    cv2.putText(frame, letter, (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 5)
    counter = f"tersimpan: {saved}" + (f"/{target}" if target else "")
    cv2.putText(frame, counter, (10, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "SPACE = simpan   ESC/Q = keluar", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


def record(args: argparse.Namespace) -> int:
    cap = open_webcam(args.camera)
    detector = HandDetector()
    letters = LETTERS if args.all_letters else [args.letter]
    window = ("Rekam BISINDO - SEMUA HURUF" if args.all_letters
              else "Rekam BISINDO - " + args.letter)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    total_saved = 0
    warned_no_hand = False
    last_save = 0.0
    try:
        for letter in letters:
            letter_dir = os.path.join(args.out, letter)
            os.makedirs(letter_dir, exist_ok=True)
            saved = 0
            print(f"[{letter}] arahkan tangan lalu tekan SPACE (ESC/Q untuk keluar)")
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("gagal membaca frame dari webcam")
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                landmarks = detector.detect(rgb)
                draw_preview(frame, letter, saved, args.count, landmarks)
                cv2.imshow(window, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    print(f"selesai huruf {letter}: {saved} frame -> {letter_dir}")
                    print(f"selesai: {total_saved + saved} frame -> {args.out}")
                    return 0
                if key != 32:  # SPACE
                    continue
                # tahan SPACE jangan jadi banjir frame; jeda minimum antar simpan
                if time.time() - last_save < args.interval:
                    continue
                if landmarks is None and not warned_no_hand:
                    warned_no_hand = True
                    _LOG.warning(
                        "frame disimpan tanpa tangan terdeteksi (%s)", letter_dir
                    )
                path = save_frame(letter_dir, letter, frame)
                last_save = time.time()
                saved += 1
                print(f"tersimpan {path}")
                if args.count and saved >= args.count:
                    break
            print(f"selesai huruf {letter}: {saved} frame -> {letter_dir}")
            total_saved += saved
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"selesai: {total_saved} frame -> {args.out}")
    return 0


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rekam citra latih BISINDO A-Z dari webcam."
    )
    parser.add_argument("--out", default="data/bisindo_user",
                        help="folder keluaran; dibuat per huruf: <out>/<letter>/")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--letter", type=str.upper,
                       help="satu huruf A-Z yang direkam")
    group.add_argument("--all", dest="all_letters", action="store_true",
                       help="lewatkan A-Z berturut-turut")
    parser.add_argument("--count", type=int, default=0,
                        help="berhenti otomatis setelah n frame per huruf (0 = tanpa batas)")
    parser.add_argument("--camera", type=int, default=0, help="indeks kamera")
    parser.add_argument("--interval", type=float, default=0.2,
                        help="jeda minimum antar simpan dalam detik")
    args = parser.parse_args(argv)
    if args.letter and args.letter not in LETTERS:
        parser.error(f"--letter harus A-Z, dapat: {args.letter}")
    return args


def main() -> int:
    args = parse_args()
    logs.setup_logging()
    try:
        return record(args)
    except KeyboardInterrupt:
        print("dibatalkan")
        return 1
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
