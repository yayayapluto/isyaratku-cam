"""Worker: webcam -> pengenalan -> teks -> kamera virtual + TTS.

Satu thread kerja (dipanggil dari QThread di app.py, atau langsung lewat
main() untuk smoke test). Setiap frame: model -> smoothing 4-dari-5 ->
pipeline teks -> overlay -> pyvirtualcam. Frame tidak di-flip: output harus
sama dengan gambar asli webcam (tanpa efek cermin).
"""

from __future__ import annotations

import sys
import time
from typing import Callable, Optional

import cv2
import numpy as np
import pyvirtualcam
import torch

import recognizer
import text_pipeline
import tts
from hand_detect import HandDetector, crop_hand

FPS = 20
TTS_DRAIN_MAX = 10.0   # detik; drain cukup ~4 kata (~0,98 s/kata)
BACKEND_ORDER = ("unitycapture", "obs")  # Unity Capture utama, OBS cadangan


def open_vcam(width: int, height: int, preferred: Optional[str] = None) -> tuple:
    """Buka kamera virtual di resolusi frame webcam. (camera, backend_name)."""
    order = (preferred,) if preferred else BACKEND_ORDER
    errors: list[str] = []
    for name in order:
        try:
            cam = pyvirtualcam.Camera(
                width=width, height=height, fps=FPS, backend=name
            )
            return cam, name
        except Exception as exc:  # backend tidak terpasang / device lemah
            # Kumpulkan untuk diagnosa di stderr, JANGAN masuk ke pesan yang
            # dilihat user: teks itu sudah ditetapkan jadi bahasa perbaikan.
            errors.append(f"{name}: {exc}")
    if errors:
        print(f"[vcam] gagal: {' | '.join(errors)}", file=sys.stderr)
    raise RuntimeError(
        "Kamera virtual tidak ditemukan. Jalankan Install.bat Unity Capture "
        "sebagai Administrator, lalu restart aplikasi. Cadangan: pasang OBS."
    )


def open_webcam() -> tuple[cv2.VideoCapture, int, int]:
    """Buka webcam di resolusi native — TIDAK diminta/diresize ke 720p:
    upscale 640x480->1280x720 menurunkan akurasi deteksi tangan (ukur:
    3/62 native vs 0/60 setelah upscale) dan buang CPU."""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("webcam tidak ditemukan")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, w, h


def _wrap_lines(text: str, max_w: int, scale: float, thick: int) -> list[str]:
    """Potong teks jadi baris selebar max_w (greedy, per kata)."""
    lines: list[str] = []
    for para in text.split("\n"):
        cur = ""
        for word in para.split():
            cand = f"{cur} {word}".strip()
            tw, _ = cv2.getTextSize(cand, cv2.FONT_HERSHEY_SIMPLEX, scale,
                                   thick)[0]
            if cur and tw > max_w:
                lines.append(cur)
                cur = word
            else:
                cur = cand
        lines.append(cur)
    return [ln for ln in lines if ln]


def draw_text(frame: np.ndarray, text: str, font_size: int = 22) -> np.ndarray:
    """Subtitle: teks putih + halo hitam tipis, wrap baris, bergantung bawah.

    Sesuai referensi: tanpa band gelap, teks langsung di atas gambar.
    OpenCV 5 membatasi ketebalan stroke putText (th 10 == th 3), jadi halo
    dibuat lewat mask + dilate — bukan stroke lebih tebal."""
    if not text.strip():
        return frame
    h, w = frame.shape[:2]
    scale = font_size / 20.0
    thick = max(1, int(round(font_size / 7)))

    lines = _wrap_lines(text, w - 20, scale, thick)
    mask = np.zeros((h, w), np.uint8)
    (_, th), _ = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    step = int(th * 1.45)
    base_y = max(th + 8, h - int(h * 0.08))
    for i, line in enumerate(reversed(lines)):
        y = base_y - i * step
        if y < th + 8:
            break  # ruang habis: kalimat sangat panjang, baris atas dipotong
        x = max(10, (w - cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX,
                                        scale, thick)[0][0]) // 2)
        cv2.putText(mask, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    255, thick, cv2.LINE_AA)

    # mask teks -> dilate = halo, tempel hitam di bawah glyph putih
    halo = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    frame[halo > 0] = (0, 0, 0)
    frame[mask > 0] = (255, 255, 255)
    return frame


class Worker:
    """Loop utama.

    Semua callback dipanggil dari worker thread — GUI wajib memakainya lewat
    signal Qt, tidak boleh langsung menyentuh widget:
      on_status(teks)      overlay/kalimat berubah
      on_candidate(huruf)  huruf kandidat terbaru (belum tentu stabil)
      on_spoken(kalimat)   kalimat selesai (buffer dibersihkan)
      on_health(dict)      kesehatan device: virtual_cam/virtual_mic/model
    """

    def __init__(
        self,
        on_status: Optional[Callable[[str], None]] = None,
        enable_tts: bool = True,
        font_size: int = 22,
        debug: bool = False,
        on_candidate: Optional[Callable[[str], None]] = None,
        on_spoken: Optional[Callable[[str], None]] = None,
        on_health: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.on_status = on_status
        self.on_candidate = on_candidate
        self.on_spoken = on_spoken
        self.on_health = on_health
        self.enable_tts = enable_tts
        self.font_size = font_size
        self.debug = debug
        self.running = False
        self._model: Optional[tuple] = None
        self._pipeline: Optional[text_pipeline.TextPipeline] = None
        self.tts_error: Optional[str] = None
        # bukti terukur (dipakai verify headless; bukan telemetri)
        self.sent_frames = 0
        self.hand_frames = 0
        self.inferred_frames = 0
        self.last_letter: Optional[str] = None
        self.backend: Optional[str] = None

    def stop(self) -> None:
        self.running = False

    def _preview(
        self,
        frame: np.ndarray,
        marks,
        text: str,
        crop=None,
        letter: str = "",
        conf: float = 0.0,
    ) -> None:
        """Mode debug saja: DUA jendela.

        1. kamera penuh — landmark, bbox, panel status, overlay teks.
        2. crop — citra PERSIS yang masuk model (sudah mirror), di-upscale.

        Tampilan jendela 1 dibalik supaya terasa seperti cermin (tangan kanan
        user muncul di kanan jendela). Landmark & bbox DIPETIK lewat (1-x):
        keduanya diukur pada frame tak dibalik, jadi tanpa pemetaan itu jatuh
        di sisi yang salah. Crop yang masuk model tetap dibalik di tempat lain
        (pemanggil), jendela 2 menampilkannya apa adanya.
        """
        if not self.debug:
            return
        show = cv2.flip(frame, 1)
        h, w = show.shape[:2]
        for x, y in marks if marks else []:
            cv2.circle(show, (int((1 - x) * w), int(y * h)), 3,
                       (0, 255, 255), -1)
        if marks:
            xs = [(1 - x) * w for x, _ in marks]
            ys = [y * h for _, y in marks]
            cv2.rectangle(show, (int(min(xs)), int(min(ys))),
                          (int(max(xs)), int(max(ys))), (255, 0, 255), 2)
        crop_note = "-" if crop is None else f"{crop.shape[1]}x{crop.shape[0]}px"
        # tts_error di atas hanya dari pemeriksaan sebelum loop. Kegagalan di
        # tengah jalan (device hilang, synth rusak) dilaporkan lewat
        # tts.last_error() — tanpa ini baris ini bisa bilang "aktif" padahal
        # audionya mati.
        tts_note = "TTS: mati" if not self.enable_tts else (
            f"TTS: error ({self.tts_error or tts.last_error()})"
            if (self.tts_error or tts.last_error()) else "TTS: aktif")
        line = (f"tangan: {'ya' if marks else 'tidak'} | "
               f"crop: {crop_note} | {tts_note}")
        (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        # Panel tepi-atas: latar putih penuh lebar, teks hitam. Lebar = panjang
        # teks s/d batas lebar frame, jadi tak pernah terpotong.
        px, py = 10, 24
        panel_w = min(w - 2 * px, tw + 20)
        cv2.rectangle(show, (px - 6, py - 22), (px + panel_w, py + 10),
                      (255, 255, 255), -1)
        cv2.putText(show, line, (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 2, cv2.LINE_AA)
        # overlay huruf gaya sama dengan vcam, tapi ukurannya dibatasi:
        # jendela debug 640x480, dan draw_text menskalakan tinggi baris dari
        # font_size — 32px di situ menabrak tepi bawah.
        draw_text(show, text, min(self.font_size, 18))
        cv2.imshow("IsyaratKu debug - kamera (tekan q untuk tutup)", show)

        # Jendela 2: crop yang DIBALIK (sama dengan input model), di-upscale
        # ke 240px supaya bentuk tangan terbaca walau crop aslinya kecil.
        if crop is None:
            view = np.zeros((240, 240, 3), np.uint8)
            cv2.putText(view, "tidak ada crop", (16, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            view = cv2.flip(crop, 1)
            side = max(view.shape[:2])
            if side < 240:
                view = cv2.resize(
                    view, (int(round(view.shape[1] * 240 / side)),
                           int(round(view.shape[0] * 240 / side))),
                    interpolation=cv2.INTER_NEAREST)
        if letter:
            cv2.putText(view, f"{letter} {conf:.2f}", (8, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                        cv2.LINE_AA)
        cv2.imshow("IsyaratKu debug - crop (tekan q untuk tutup)", view)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.running = False


    def _status(self, text: str) -> None:
        if self.on_status:
            self.on_status(text)

    def _flush(self) -> None:
        """Jalankan timer pipeline; laporkan kalimat yang selesai."""
        flushed = self._pipeline.tick()
        if flushed:
            if self.on_spoken:
                self.on_spoken(flushed)
            self._status(flushed)

    def _health(self, backend: Optional[str] = None) -> None:
        """Lapor kesehatan device ke GUI. Backend=None = kamera virtual gagal."""
        if not self.on_health:
            return
        try:
            tts.cable_output_device()
            mic_ok = True
        except tts.TtsUnavailable:
            mic_ok = False
        self.on_health(
            {
                "virtual_cam": backend,
                "virtual_mic": mic_ok,
                "model": self._model is not None,
            }
        )

    def run(self) -> None:
        """Sampai stop() atau error. Melempar RuntimeError bila device gagal."""
        try:
            self._model = recognizer.build_model()
            smoother = recognizer.Smoother()
        except FileNotFoundError as exc:
            raise RuntimeError(f"model belum diunduh: {exc}") from exc

        # speak_async: antre + thread daemon. Blocking sd.play di loop video
        # akan menghentikan cap.read() selama durasi audio tiap kata.
        speak = tts.speak_async if self.enable_tts else None
        if self.enable_tts:
            try:
                tts.cable_output_device()
            except tts.TtsUnavailable as exc:
                self.tts_error = str(exc)  # video tetap jalan tanpa suara
                speak = None
            else:
                # Muat voice SEBELUM loop/pipeline: biaya sekali jalan (~1,7 s)
                # di luar loop, jadi tidak muncul sebagai freeze saat kata
                # pertama terbentuk.
                tts.prewarm()
        self._pipeline = text_pipeline.TextPipeline(speak=speak)

        try:
            cap, cam_w, cam_h = open_webcam()
        except RuntimeError as exc:
            raise RuntimeError(f"webcam gagal dibuka: {exc}") from exc

        try:
            # vcam SETELAH webcam: resolusi stream harus cocok dengan frame
            # yang dikirim, kalau tidak pyvirtualcam mis-render.
            cam, backend = open_vcam(cam_w, cam_h)
        except RuntimeError as exc:
            cap.release()
            raise RuntimeError(f"kamera virtual gagal: {exc}") from exc
        self.running = True
        self.backend = backend
        self._health(backend=backend)
        self._status(f"berjalan ({backend})")

        model, labels = self._model
        detector = HandDetector()
        missed = 0
        try:
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    missed += 1
                    if missed > 60:
                        raise RuntimeError("webcam berhenti memberi frame")
                    continue
                missed = 0
                # Frame dikirim apa adanya (tanpa flip): output kamera virtual
                # harus sama dengan gambar asli webcam.
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                marks = detector.detect(rgb)
                if marks is None:
                    self._pipeline.clear_hand()   # tangan hilang -> timer absen
                    smoother.reset()
                    # tick() WAJIB di cabang ini: tanpa ini timer absen tak
                    # pernah jalan (tangan hilang tak menghasilkan huruf)
                    self._flush()
                    self._preview(frame, None, self._pipeline.text)
                    draw_text(frame, self._pipeline.text, self.font_size)
                    self.sent_frames += 1
                    cam.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    cam.sleep_until_next_frame()
                    continue

                self.hand_frames += 1
                # tangan ADA di frame -> jangan jalankan timer absen walau model
                # belum memberi huruf baru (tanpa ini kata tak pernah selesai)
                self._pipeline.mark_present()
                cropped = crop_hand(frame, marks)
                if cropped is None:   # tangan terlalu jauh/kecil: skip prediksi
                    self._flush()
                    self._preview(frame, marks, self._pipeline.text, None)
                    draw_text(frame, self._pipeline.text, self.font_size)
                    self.sent_frames += 1
                    cam.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    cam.sleep_until_next_frame()
                    continue

                with torch.no_grad():
                    # Crop DIBALIK sebelum preprocess (keputusan user: tampilan
                    # tanpa mirror, pemrosesan memakai citra mirror).
                    probs = torch.softmax(
                        model(recognizer.preprocess(cv2.flip(cropped, 1))), 1)[0]
                self.inferred_frames += 1
                letter = labels[int(probs.argmax())]
                self.last_letter = letter
                stable = smoother.update(letter)
                # Ambang 0,3: rata-rata conf pada crop benar 0,48, jadi 0,5
                # membuang ~setengah prediksi benar. Smoother 4-dari-5 yang
                # menyaring jitter — bukan confidence.
                if stable and probs.max() > 0.3:
                    self._pipeline.add_letter(stable)
                self._flush()
                if self.on_candidate:
                    self.on_candidate(letter)
                self._preview(
                    frame, marks, self._pipeline.text, cropped,
                    letter, float(probs.max()),
                )
                draw_text(frame, self._pipeline.text, self.font_size)
                cam.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cam.sleep_until_next_frame()
                self._status(self._pipeline.text)
        finally:
            cap.release()
            cam.close()
            if self.debug:
                cv2.destroyAllWindows()
            if self.enable_tts and self.tts_error is None:
                # drain menunggu daemon selesai; TTS_DRAIN_MAX membatasinya
                # supaya BERHENTI tak menggantung, tapi tidak memotong kata.
                tts.drain(TTS_DRAIN_MAX)
            self.running = False


def main() -> None:
    worker = Worker(on_status=lambda t: print(t) if t else None)
    try:
        worker.run()
    except KeyboardInterrupt:
        worker.stop()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
