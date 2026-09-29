"""Worker: webcam -> pengenalan -> teks -> kamera virtual + TTS.

Satu thread kerja (dipanggil dari QThread di main.py, atau langsung lewat
main() untuk smoke test). Setiap frame: flip -> model -> smoothing 4-dari-5 ->
pipeline teks -> overlay -> pyvirtualcam.
"""

from __future__ import annotations

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
BACKEND_ORDER = ("obs", "unitycapture")  # OBS dulu: terbukti di M1


def open_vcam(width: int, height: int, preferred: Optional[str] = None) -> tuple:
    """Buka kamera virtual di resolusi frame webcam. (camera, backend_name)."""
    order = (preferred,) if preferred else BACKEND_ORDER
    errors = []
    for name in order:
        try:
            cam = pyvirtualcam.Camera(
                width=width, height=height, fps=FPS, backend=name
            )
            return cam, name
        except Exception as exc:  # backend tidak terpasang / device lemah
            errors.append(f"{name}: {exc}")
    raise RuntimeError(
        " | ".join(errors)
        + " (nyalakan OBS Virtual Camera: Controls > Start Virtual Camera)"
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


def draw_text(frame: np.ndarray, text: str, font_size: int = 22) -> np.ndarray:
    """Subtitle gaya film: teks putih + halo hitam tipis, terpusat.

    Sesuai referensi: tanpa band gelap, teks langsung di atas gambar.
    OpenCV 5 membatasi ketebalan stroke putText (th 10 == th 3), jadi halo
    dibuat lewat mask + dilate — bukan stroke lebih tebal."""
    if not text:
        return frame
    h, w = frame.shape[:2]
    scale = font_size / 20.0
    thick = max(1, int(round(font_size / 7)))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x = max(10, (w - tw) // 2)
    y = max(th + 8, h - int(h * 0.08))

    # mask teks -> dilate = halo, tempel hitam di bawah glyph putih
    mask = np.zeros((h, w), np.uint8)
    cv2.putText(mask, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                255, thick, cv2.LINE_AA)
    halo = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    halo_region = halo > 0
    frame[halo_region] = (0, 0, 0)
    frame[mask > 0] = (255, 255, 255)
    return frame


class Worker:
    """Loop utama. on_status(teks) dipanggil tiap frame untuk GUI."""

    def __init__(
        self,
        on_status: Optional[Callable[[str], None]] = None,
        enable_tts: bool = True,
        font_size: int = 22,
        debug: bool = False,
    ) -> None:
        self.on_status = on_status
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

    def stop(self) -> None:
        self.running = False

    def _preview(self, frame, marks, text: str, crop=None) -> None:
        """Mode debug saja: jendela debug dengan bbox + keterangan deteksi."""
        if not self.debug:
            return
        show = frame.copy()
        for x, y in marks if marks else []:
            cv2.circle(show, (int(x * show.shape[1]), int(y * show.shape[0])), 3,
                       (0, 255, 255), -1)
        if marks:
            xs = [x * show.shape[1] for x, _ in marks]
            ys = [y * show.shape[0] for _, y in marks]
            cv2.rectangle(show, (int(min(xs)), int(min(ys))),
                          (int(max(xs)), int(max(ys))), (255, 0, 255), 2)
        crop_note = "-" if crop is None else f"{crop.shape[1]}x{crop.shape[0]}px"
        cv2.putText(show, f"tangan: {'ya' if marks else 'tidak'} | "
                          f"crop: {crop_note}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # overlay huruf: posisi & gaya sama persis dengan yang dikirim ke vcam
        draw_text(show, text, self.font_size)
        cv2.imshow("IsyaratKu debug (tekan q untuk tutup jendela)", show)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.running = False

    def _status(self, text: str) -> None:
        if self.on_status:
            self.on_status(text)

    def run(self) -> None:
        """Sampai stop() atau error. Melempar RuntimeError bila device gagal."""
        try:
            self._model = recognizer.build_model()
            smoother = recognizer.Smoother()
        except FileNotFoundError as exc:
            raise RuntimeError(f"model belum diunduh: {exc}") from exc

        speak = tts.speak if self.enable_tts else None
        if self.enable_tts:
            try:
                tts.cable_output_device()
            except tts.TtsUnavailable as exc:
                self.tts_error = str(exc)  # video tetap jalan tanpa suara
                speak = None
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
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                marks = detector.detect(rgb)
                if marks is None:
                    self._pipeline.clear_hand()   # tangan hilang -> timer absen
                    smoother.reset()
                    self._preview(frame, None, self._pipeline.text)
                    draw_text(frame, self._pipeline.text, self.font_size)
                    self.sent_frames += 1
                    cam.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    cam.sleep_until_next_frame()
                    continue

                self.hand_frames += 1
                cropped = crop_hand(frame, marks)
                self._preview(frame, marks, self._pipeline.text, cropped)
                if cropped is None:   # tangan terlalu jauh/kecil: skip prediksi
                    draw_text(frame, self._pipeline.text, self.font_size)
                    self.sent_frames += 1
                    cam.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    cam.sleep_until_next_frame()
                    continue

                with torch.no_grad():
                    probs = torch.softmax(model(recognizer.preprocess(cropped)), 1)[0]
                self.inferred_frames += 1
                letter = labels[int(probs.argmax())]
                self.last_letter = letter
                stable = smoother.update(letter)
                # Ambang 0,3: rata-rata conf pada crop benar 0,48, jadi 0,5
                # membuang ~setengah prediksi benar. Smoother 4-dari-5 yang
                # menyaring jitter — bukan confidence.
                if stable and probs.max() > 0.3:
                    self._pipeline.add_letter(stable)
                spoken = self._pipeline.tick()
                if spoken:
                    self._status(spoken)

                self.sent_frames += 1
                draw_text(frame, self._pipeline.text, self.font_size)
                cam.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cam.sleep_until_next_frame()
                self._status(self._pipeline.text)
        finally:
            cap.release()
            cam.close()
            if self.debug:
                cv2.destroyAllWindows()
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
