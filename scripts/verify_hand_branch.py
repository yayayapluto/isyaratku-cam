"""Verifikasi branch tangan + panel debug — tanpa webcam/device.

Memutar ulang frame VOC berlabel melalui jalur worker SUNGGUHAN
(detect → crop → preprocess → model → smoother → pipeline), mencatat latensi
ke `logs/isyaratku-debug.log`, dan menyimpan ketiga kanvas debug ke
`logs/panel_{main,crop,status}.png`.

Angka yang DIKELUARKAN script ini:
- kandidat/dtk — emisi kandidat per frame (tangan terlihat + model jalan)
- huruf/dtk    — huruf yang benar-benar diterima ke buffer kata

Catatan: fps no-hand (jalur deteksi cepat, tanpa crop/model) diukur
TERPISAH dan dicatat di `docs/MODEL_SELECTION.md`; script ini TIDAK
mengeluarkan angka itu.

Pemakaian:
    python scripts/verify_hand_branch.py [--seconds 12] [--out logs]
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import tempfile
import threading
import time
import types

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logs  # noqa: E402
import hand_detect  # noqa: E402
import worker as wk  # noqa: E402


def _release_runtime(holder: dict, w, sink: dict, frames: list) -> None:
    """Lepas device + model SEKARANG, selagi torch/mediapipe masih hidup.

    Interpreter free-threaded (Python 3.14t) bisa fatal saat shutdown
    (exit 5, "Windows fatal exception: access violation") kalau delegate
    TFLite XNNPACK mediapipe atau VideoCapture baru dibuang ketika module
    sudah diturunkan. Release eksplisit di sini membuat perilaku itu
    deterministik: semua objek native sudah bersih sebelum turun.
    """
    cap = holder.get("cap")
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    landmarker = getattr(hand_detect, "_landmarker", None)
    if landmarker is not None:
        try:
            landmarker.close()
        except Exception:
            pass
        hand_detect._landmarker = None
    cv2.destroyAllWindows()
    # Putus referensi supaya refcount tensor/module turun sekarang, bukan
    # sewaktu-waktu selama GC shutdown.
    for attr in ("_model", "_smoother", "_pipeline", "on_candidate", "_preview"):
        try:
            setattr(w, attr, None)
        except Exception:
            pass
    sink.clear()
    del frames[:]
    gc.collect()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES_DIR = os.path.join(ROOT, "data", "bisindo_rhio", "train")

# 40 frame ditulis berulang: 800 frame @ 25 fps ~ 32 s video, cukup untuk
# --seconds 12 walau worker baru mulai setelah model + torch siap.
_REPLAY_PASSES = 20


class FakeCam:
    """Pengganti kamera virtual: tidak perlu Unity Capture/OBS."""

    def __init__(self) -> None:
        self.frames = 0

    def send(self, arr) -> None:
        self.frames += 1

    def sleep_until_next_frame(self) -> None:
        time.sleep(0.01)

    def close(self) -> None:
        pass


def temp_video_webcam(frames, fps: float, tmpdir: str) -> tuple:
    """Tulis frame VOC ke MP4 sementara, lalu buka dengan VideoCapture NYATA.

    Worker memakai API VideoCapture asli: cap.get(CAP_PROP_FRAME_WIDTH/HEIGHT)
    dan cap.release(), jadi pengganti harus objek asli juga. Namespace boneka
    dengan get() -> konstan tak mungkin bikin resolusi video (640x480) cocok,
    dan atribut cv2.VideoCapture tak bisa di-patch per instance — subclass
    LoopCap dipakai supaya read() balik ke frame 0 saat video habis.

    Video ditulis 40 frame x 20 pass (~32 s @ 25 fps) supaya lebih panjang
    dari --seconds; worker tak perlu pernah kehabisan frame.

    Cap disimpan di _CAP_HOLDER supaya main() bisa release-nya SEKARANG
    (selagi torch/mediapipe masih hidup), bukan lewat atexit yang baru
    jalan saat interpreter sudah setengah turun.
    """
    fd, path = tempfile.mkstemp(suffix=".mp4", dir=tmpdir)
    os.close(fd)

    def _cleanup() -> None:
        try:
            os.remove(path)
        except OSError:
            pass

    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        _cleanup()
        raise RuntimeError(f"VideoWriter gagal dibuka untuk {path}")
    try:
        for _ in range(_REPLAY_PASSES):
            for img in frames:
                vw.write(img)
    finally:
        vw.release()

    class LoopCap(cv2.VideoCapture):
        """VideoCapture sungguhan yang balik ke frame 0 saat video habis."""

        def read(self):
            ok, img = cv2.VideoCapture.read(self)
            if not ok:
                cv2.VideoCapture.set(self, cv2.CAP_PROP_POS_FRAMES, 0)
                ok, img = cv2.VideoCapture.read(self)
            return ok, img

    cap = LoopCap(path)
    if not cap.isOpened():
        _cleanup()
        raise RuntimeError(f"VideoCapture gagal dibuka untuk {path}")
    _CAP_HOLDER["cap"] = cap
    return cap, w, h


# Cap VideoCapture terakhir yang dibuat temp_video_webcam(); dipakai main()
# untuk release eksplisit sebelum interpreter turun.
_CAP_HOLDER: dict = {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "logs"))
    ap.add_argument("--fps", type=float, default=25.0,
                    help="kecepatan replay webcam palsu")
    args = ap.parse_args()

    logs.setup_logging(debug=True)
    os.makedirs(args.out, exist_ok=True)

    # Dataset VOC tidak masuk git (data/ lokal). Skrip permanen TIDAK BOLEH
    # merah di mesin tanpa dataset: cukup lapor SKIP dan keluar 0.
    if not os.path.isdir(FRAMES_DIR):
        print("SKIP: dataset VOC tidak ada (data/ lokal, tidak masuk git)")
        return 0
    names = [n for n in sorted(os.listdir(FRAMES_DIR)) if n.endswith(".jpg")]
    sampled = names[:: max(1, len(names) // 40)][:40]
    frames = []
    for name in sampled:
        img = cv2.imread(os.path.join(FRAMES_DIR, name))
        if img is not None:
            frames.append(img)
    if not frames:
        print("SKIP: dataset VOC tidak ada (data/ lokal, tidak masuk git)")
        return 0
    print(f"frame VOC dimuat: {len(frames)} @ {args.fps:.0f} fps replay"
          f" (~{args.fps * args.seconds:.0f} frame di jendela {args.seconds:.0f}s)")

    with tempfile.TemporaryDirectory(prefix="isyaratku_verify_") as tmpdir:
        wk.open_webcam = lambda: temp_video_webcam(frames, args.fps, tmpdir)
        wk.open_vcam = lambda w, h: (FakeCam(), "fake")

        stats = {"candidates": 0}
        sink: dict[str, object] = {}

        w = wk.Worker(
            enable_tts=False,
            debug=True,
            on_candidate=lambda l: stats.__setitem__(
                "candidates", stats["candidates"] + 1),
        )
        # _preview versi worker sudah menerima _sink; kirim supaya kanvas frame
        # terakhir (dengan crop nyata) tersimpan untuk PNG tanpa perlu jendela.
        w._preview = types.MethodType(
            lambda self, *a, **kw: wk.Worker._preview(self, *a, _sink=sink,
                                                      **kw), w)

        # Thread non-daemon: thread yang masih hidup saat shutdown bikin
        # interpreter free-threaded fatal (exit 5, access violation) saat
        # DLL torch/mediapipe diturunkan. join di bawah menunggu sampai
        # selesai — kalau macet, gagal terang (lebih baik daripada sleep).
        t = threading.Thread(target=w.run)
        t.start()
        time.sleep(args.seconds)
        w.stop()
        t.join(timeout=15)
        if t.is_alive():
            raise RuntimeError("worker thread belum berhenti setelah 15 s")

        letters = len(w._pipeline._letters) if w._pipeline else 0
        candidates = stats["candidates"]
        print(f"RESULT sent={w.sent_frames} hand={w.hand_frames} "
              f"inferred={w.inferred_frames} kandidat={candidates} "
              f"huruf={letters} huruf_per_s={letters / args.seconds:.1f} "
              f"fps_replay={args.fps:.0f} last={w.last_letter}")

        # Toleran: satu frame tangan saja sudah membuktikan branch jalan.
        # jumlah huruf diterima TIDAK diasertakan — itu butuh run panjang
        # dan model nyata.
        assert w.hand_frames > 0, "branch tangan tidak terpicu"
        assert w.inferred_frames > 0, "inferensi tidak jalan"

        counts: dict[str, int] = {}
        debug_log = os.path.join(args.out, "isyaratku-debug.log")
        with open(debug_log, encoding="utf-8") as fh:
            for line in fh:
                for st in ("t_crop", "t_model", "t_smooth"):
                    if f"{st} dur_ms" in line:
                        counts[st] = counts.get(st, 0) + 1
        assert all(counts.get(st, 0) > 0
                   for st in ("t_crop", "t_model", "t_smooth")), counts
        print("stage counts:", counts)

        expected = {"main": (640, 480), "status": (480, 240)}
        saved = []
        for key in ("main", "crop", "status"):
            arr = sink.get(key)
            if arr is None:
                print(f"gagal: kanvas {key} tidak pernah dirender")
                return 1
            h, wd = arr.shape[:2]
            if key in expected:
                assert (wd, h) == expected[key], (
                    f"kanvas {key} {wd}x{h}, mau {expected[key]}")
            else:
                # crop di-upscale maksimal ke 240px (worker.py); dulu 290px
                # lolos karena tak ada assertion bentuk.
                assert wd <= 240 and h <= 240, f"crop {wd}x{h} > 240px"
                print(f"crop aktual: {wd}x{h}px (<=240)")
            path = os.path.join(args.out, f"panel_{key}.png")
            ok = cv2.imwrite(path, arr)
            assert ok, f"gagal menyimpan {path}"
            print(f"panel_{key}.png -> {path} {wd}x{h}")
            saved.append(path)
        print("saved:", ", ".join(saved))

        # Release SEKARANG, selagi torch/mediapipe/vc masih hidup dan dir
        # sementara masih ada — bukan lewat teardown interpreter.
        _release_runtime(_CAP_HOLDER, w, sink, frames)
    return 0


if __name__ == "__main__":
    sys.exit(main())
