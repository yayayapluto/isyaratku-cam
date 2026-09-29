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

WIDTH, HEIGHT, FPS = 1280, 720, 20
BACKEND_ORDER = ("obs", "unitycapture")  # OBS dulu: terbukti di M1


def open_vcam(preferred: Optional[str] = None) -> tuple:
    """Buka kamera virtual. Kembalikan (camera, backend_name)."""
    order = (preferred,) if preferred else BACKEND_ORDER
    errors = []
    for name in order:
        try:
            cam = pyvirtualcam.Camera(
                width=WIDTH, height=HEIGHT, fps=FPS, backend=name
            )
            return cam, name
        except Exception as exc:  # backend tidak terpasang / device lemah
            errors.append(f"{name}: {exc}")
    raise RuntimeError(
        " | ".join(errors)
        + " (nyalakan OBS Virtual Camera: Controls > Start Virtual Camera)"
    )


def open_webcam() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("webcam tidak ditemukan")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    return cap


def draw_text(frame: np.ndarray, text: str, font_size: int = 22) -> np.ndarray:
    """Posisi subtitle film: strip gelap di bawah, teks hijau di tengahnya."""
    if not text:
        return frame
    h = frame.shape[0]
    scale = font_size / 20.0
    thick = max(2, int(round(font_size / 7)))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    band = max(60, int(th * 2.2))
    y0 = h - band
    cv2.rectangle(frame, (0, y0), (frame.shape[1], h), (15, 15, 15), -1)
    x = max(10, (frame.shape[1] - tw) // 2)
    cv2.putText(frame, text, (x, h - int(band * 0.32)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), thick, cv2.LINE_AA)
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
            cam, backend = open_vcam()
        except RuntimeError as exc:
            raise RuntimeError(f"kamera virtual gagal: {exc}") from exc
        cap = open_webcam()
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
                frame = cv2.resize(frame, (WIDTH, HEIGHT))
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
                if stable and probs.max() > 0.5:
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
