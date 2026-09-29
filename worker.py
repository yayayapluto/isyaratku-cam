"""Worker: webcam -> pengenalan -> teks -> kamera virtual + TTS.

Loop utama memakai satu thread kerja. Capture MSMF (fallback DSHOW), deteksi
tangan diselang-seling dengan inferensi (inferensi setiap frame), settle 3
frame + bbox stabil sebelum huruf diterima, dan logging ke folder logs/.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from typing import Callable, Optional

import cv2
import numpy as np
import pyvirtualcam
import torch

import logs
import recognizer
import text_pipeline
import tts
from hand_detect import HandDetector, crop_hand

FPS = 20
TTS_DRAIN_MAX = 10.0   # detik; drain cukup ~4 kata (~0,98 s/kata)
BACKEND_ORDER = ("unitycapture", "obs")  # Unity Capture utama, OBS cadangan
CONF_GATE = 0.45   # prob maksimum minimum sebelum huruf diterima
SETTLE_FRAMES = 3  # crop wajib stabil segini frame berturut-turut
BOX_TOL = 0.20     # toleransi perubahan ukuran crop (fraksi dari ukuran lama)
CENTROID_TOL = 0.12  # toleransi pergeseran pusat tangan (fraksi dari frame)
DETECT_EVERY = 2   # deteksi tangan tiap N frame; inferensi tetap tiap frame
FPS_WINDOW = 30    # jumlah frame untuk rata-rata fps & latensi tahap

STAGES = ("t_cap", "t_detect", "t_crop", "t_model", "t_smooth", "t_overlay")

_log = logs.get_logger()


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
            # Kumpulkan untuk diagnosa di log, JANGAN masuk ke pesan yang
            # dilihat user: teks itu sudah ditetapkan jadi bahasa perbaikan.
            errors.append(f"{name}: {exc}")
    if errors:
        _log.error("[vcam] gagal: %s", " | ".join(errors))
    raise RuntimeError(
        "Kamera virtual tidak ditemukan. Jalankan Install.bat Unity Capture "
        "sebagai Administrator, lalu restart aplikasi. Cadangan: pasang OBS."
    )


def open_webcam() -> tuple[cv2.VideoCapture, int, int]:
    """Buka webcam di resolusi native — TIDAK diminta/diresize ke 720p:
    upscale 640x480->1280x720 menurunkan akurasi deteksi tangan (ukur:
    3/62 native vs 0/60 setelah upscale) dan buang CPU.

    Backend capture diukur di mesin ini: CAP_DSHOW hanya 6,9-7,0 read/detik
    (bottleneck, bukan model), CAP_MSMF 21,7-23,1, CAP_ANY 22,1-22,2.
    MSMF dicoba lebih dulu; webcam/driver lama yang tak punya pipeline MSMF
    jatuh kembali ke DSHOW secara transparan."""
    cap: Optional[cv2.VideoCapture] = None
    for api in (cv2.CAP_MSMF, cv2.CAP_DSHOW):
        candidate = cv2.VideoCapture(0, api)
        if candidate.isOpened():
            cap = candidate
            break
        candidate.release()
    if cap is None:
        raise RuntimeError("webcam tidak ditemukan")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (w, h) != (640, 480):
        # paksa 640x480 (resolusi yang dipakai open_vcam); kalau backend
        # menolak, resolusi asli dipakai apa adanya (vcam ikut resolusi itu).
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
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


def _panel(img: np.ndarray, lines: list[str], x: int = 10, y: int = 24) -> None:
    """Panel tepi-atas: latar putih, teks hitam. Lebar = teks terpanjang
    yang masih masuk frame (tak pernah terpotong)."""
    h, w = img.shape[:2]
    scale, thick = 0.55, 1
    widths = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0]
              for t in lines]
    panel_w = min(w - x - 6, (max(widths) + 20) if widths else 0)
    if panel_w <= 0:
        return
    line_h = 20
    cv2.rectangle(img, (x - 6, y - 18),
                  (x + panel_w, y + line_h * len(lines) - 6),
                  (255, 255, 255), -1)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick,
                    cv2.LINE_AA)


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

        # transisi gestur: crop wajib stabil SETTLE_FRAMES frame berturut
        self._prev_crop_shape: Optional[tuple[int, int]] = None
        self._prev_centroid: Optional[tuple[float, float]] = None
        self._settle = 0
        self._frame_idx = 0
        self._last_crop: Optional[np.ndarray] = None
        self._last_marks: Optional[list[tuple[float, float]]] = None

        # panel debug: fps + latensi rata-rata window terakhir
        self._frame_times: deque[float] = deque(maxlen=FPS_WINDOW)
        self._stage_ms: dict[str, deque[float]] = {
            s: deque(maxlen=FPS_WINDOW) for s in STAGES
        }
        self.fps: float = 0.0
        self.last_probs: Optional[list[tuple[str, float]]] = None
        self._smoother: Optional[recognizer.Smoother] = None

    def stop(self) -> None:
        self.running = False

    def _record_stage(self, name: str, start: float) -> None:
        """Catat latensi satu tahap: masuk log debug + rata-rata panel."""
        ms = (time.perf_counter() - start) * 1000.0
        self._stage_ms[name].append(ms)
        _log.debug("%s dur_ms=%.2f", name, ms)

    def _note_frame(self) -> None:
        self._frame_times.append(time.perf_counter())
        times = self._frame_times
        if len(times) >= 2:
            span = times[-1] - times[0]
            if span > 0:
                self.fps = (len(times) - 1) / span

    def _settle_update(
        self, frame: np.ndarray, cropped: np.ndarray, marks: list
    ) -> int:
        """Update penghitung settle. Kembalikan nilai settle sekarang.

        Crop stabil kalau ukuran & pusat tangan tak bergeser jauh dari frame
        sebelumnya. Acuan SELALU diperbarui tiap frame (bukan hanya saat
        tidak stabil) — kalau tidak, `_prev_* is None` membuat `stable`
        selalu True dan settle menaik tanpa memeriksa gerakan sama sekali.

        # ponytail: threshold ambang empirically, riwayat jitter user jadi
        # dasar tuning. Naikkan BOX_TOL/CENTROID_TOL (atau turunkan
        # SETTLE_FRAMES) kalau satu huruf butuh > 0,3 s untuk tampil; naikkan
        # SETTLE_FRAMES kalau frame transisi masih bocor."""
        h, w = frame.shape[:2]
        ch, cw = cropped.shape[:2]
        cx = sum(x for x, _ in marks) / len(marks) * w
        cy = sum(y for _, y in marks) / len(marks) * h
        stable = self._prev_crop_shape is None or (
            abs(cw - self._prev_crop_shape[0]) <= BOX_TOL * self._prev_crop_shape[0]
            and abs(ch - self._prev_crop_shape[1]) <= BOX_TOL * self._prev_crop_shape[1]
            and abs(cx - self._prev_centroid[0]) <= CENTROID_TOL * w
            and abs(cy - self._prev_centroid[1]) <= CENTROID_TOL * h
        )
        self._prev_crop_shape = (cw, ch)
        self._prev_centroid = (cx, cy)
        self._settle = self._settle + 1 if stable else 0
        return self._settle

    def _reset_hand_state(self) -> None:
        """Tangan hilang: settle, bbox acuan, dan smoother mulai dari nol."""
        self._settle = 0
        self._prev_crop_shape = None
        self._prev_centroid = None
        self._last_crop = None
        self._last_marks = None

    def _preview(
        self,
        frame: np.ndarray,
        marks,
        text: str,
        crop=None,
        letter: str = "",
        conf: float = 0.0,
    ) -> None:
        """Mode debug saja: TIGA jendela.

        1. kamera penuh — landmark, bbox, panel status, overlay teks dan
           ringkasan pipeline (fps/gate/settle/smooth/conf).
        2. crop tangan — citra PERSIS yang masuk model (TIDAK dibalik),
           label, top-3, ukuran input, gate & settle, histori Smoother.
        3. status — fps terukur, latensi per tahap, buffer kata/kalimat,
           backend vcam, dan antrean TTS.

        Tampilan jendela 1 dibalik supaya terasa seperti cermin (tangan kanan
        user muncul di kanan jendela). Landmark & bbox DIPETIK lewat (1-x):
        keduanya diukur pada frame tak dibalik, jadi tanpa pemetaan itu jatuh
        di sisi yang salah."""
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
        _panel(show, [
            f"tangan: {'ya' if marks else 'tidak'} | crop: {crop_note} |"
            f" {tts_note}",
            f"fps={self.fps:.1f} gate={CONF_GATE} settle={self._settle}"
            f"/{SETTLE_FRAMES} conf={conf:.2f}",
        ])
        # overlay huruf gaya sama dengan vcam, tapi ukurannya dibatasi:
        # jendela debug 640x480, dan draw_text menskalakan tinggi baris dari
        # font_size — 32px di situ menabrak tepi bawah.
        draw_text(show, text, min(self.font_size, 18))
        cv2.imshow("IsyaratKu debug - kamera (tekan q untuk tutup)", show)

        # Jendela 2: crop APA ADANYA yang masuk model (tidak dibalik),
        # di-upscale ke 240px supaya bentuk tangan terbaca.
        if crop is None:
            view = np.zeros((240, 240, 3), np.uint8)
            cv2.putText(view, "tidak ada crop", (16, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            view = crop
            side = max(view.shape[:2])
            if side < 240:
                view = cv2.resize(
                    view, (int(round(view.shape[1] * 240 / side)),
                           (int(round(view.shape[0] * 240 / side)))),
                    interpolation=cv2.INTER_NEAREST)
        if letter:
            cv2.putText(view, f"{letter} {conf:.2f}", (8, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                        cv2.LINE_AA)
        if self.last_probs:
            top = " / ".join(f"{n} {p:.2f}" for n, p in self.last_probs[:3])
            cv2.putText(view, f"top-3: {top}", (8, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1,
                        cv2.LINE_AA)
        iw, ih = recognizer.INPUT_SIZE
        cv2.putText(view, f"input {iw}x{ih} letterbox pad10", (8, 74),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(view,
                    f"gate {CONF_GATE} settle={self._settle}/{SETTLE_FRAMES}",
                    (8, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1,
                    cv2.LINE_AA)
        smooth_hist = self._smoother.history() if self._smoother else []
        cv2.putText(view, f"smoother={''.join(smooth_hist) or '-'}",
                    (8, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.imshow("IsyaratKu debug - crop (tekan q untuk tutup)", view)

        # Jendela 3: status satu layar — teks hitam di latar putih, muat 10
        # baris; kalau lebih, baris bawah dipotong (rare: kalimat panjang).
        word = self._pipeline.word if self._pipeline else ""
        sentence = self._pipeline.sentence if self._pipeline else ""
        queued = tts.queue_depth() if self.enable_tts else 0
        stage_rows = [
            (f"{s}={sum(v) / len(v):.1f}" if self._stage_ms[s] else f"{s}=-")
            for s in STAGES
        ]
        status = np.full((240, 480, 3), 255, np.uint8)
        _panel(status, [
            f"fps terukur: {self.fps:.1f}",
            *stage_rows,
            f"huruf kandidat: {self.last_letter or '-'}",
            f"buffer kata: {word or '-'}",
            f"buffer kalimat: {sentence.strip() or '-'}",
            f"vcam backend: {self.backend or '-'}",
            f"antrean TTS: {queued}",
        ], x=10, y=24)
        cv2.imshow("IsyaratKu debug - status (tekan q untuk tutup)", status)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.running = False

    def _status(self, text: str) -> None:
        if self.on_status:
            self.on_status(text)

    def _flush(self) -> None:
        """Jalankan timer pipeline; laporkan kalimat yang selesai."""
        flushed = self._pipeline.tick() if self._pipeline else None
        if flushed:
            if self.on_spoken:
                self.on_spoken(flushed)
            self._status(flushed)

    def _health(self, backend: Optional[str] = None) -> None:
        """Lapor kesehatan device ke GUI. Backend=None = kamera virtual gagal.

        fps opsional: widget GUI yang belum membacanya tetap aman (.get)."""
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
                "fps": round(self.fps, 1),
            }
        )

    def run(self) -> None:
        """Sampai stop() atau error. Melempar RuntimeError bila device gagal."""
        try:
            self._model = recognizer.build_model()
            smoother = recognizer.Smoother()
            self._smoother = smoother
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
        torch.set_num_threads(8)
        _log.info(
            "worker berjalan (backend=%s, input=%dx%d, gate=%.2f, settle=%d,"
            " threads=%d)", backend, recognizer.INPUT_SIZE[0],
            recognizer.INPUT_SIZE[1], CONF_GATE, SETTLE_FRAMES,
            torch.get_num_threads(),
        )
        missed = 0
        try:
            while self.running:
                self._frame_idx += 1
                start = time.perf_counter()
                ok, frame = cap.read()
                self._record_stage("t_cap", start)
                if not ok:
                    missed += 1
                    if missed > 60:
                        raise RuntimeError("webcam berhenti memberi frame")
                    continue
                missed = 0
                # Frame dikirim apa adanya (tanpa flip): output kamera virtual
                # harus sama dengan gambar asli webcam.
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._note_frame()

                # Deteksi diselang-seling dengan inferensi: deteksi tiap
                # DETECT_EVERY frame (atau saat belum pernah ada crop), frame
                # lainnya memakai crop+landmark frame deteksi terakhir.
                # Inferensi TETAP jalan setiap frame supaya huruf tak hilang.
                detect_now = (
                    self._frame_idx % DETECT_EVERY == 0
                    or self._last_crop is None
                )
                start = time.perf_counter()
                if detect_now:
                    marks = detector.detect(rgb)
                else:
                    marks = self._last_marks
                self._record_stage("t_detect", start)
                if marks is None:
                    self._reset_hand_state()
                    self._pipeline.clear_hand()   # tangan hilang -> timer absen
                    smoother.reset()
                    # tick() WAJIB di cabang ini: tanpa ini timer absen tak
                    # pernah jalan (tangan hilang tak menghasilkan huruf)
                    self._flush()
                    self._preview(frame, None, self._pipeline.text)
                    start = time.perf_counter()
                    draw_text(frame, self._pipeline.text, self.font_size)
                    self._record_stage("t_overlay", start)
                    self.sent_frames += 1
                    cam.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    cam.sleep_until_next_frame()
                    continue

                self.hand_frames += 1
                # tangan ADA di frame -> jangan jalankan timer absen walau model
                # belum memberi huruf baru (tanpa ini kata tak pernah selesai)
                self._pipeline.mark_present()
                start = time.perf_counter()
                if detect_now:
                    cropped = crop_hand(frame, marks)
                    self._last_crop = cropped
                    self._last_marks = marks
                else:
                    cropped = self._last_crop
                self._record_stage("t_crop", start)
                if cropped is None:
                    self._flush()
                    self._preview(frame, marks, self._pipeline.text, None)
                    start = time.perf_counter()
                    draw_text(frame, self._pipeline.text, self.font_size)
                    self._record_stage("t_overlay", start)
                    self.sent_frames += 1
                    cam.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    cam.sleep_until_next_frame()
                    continue

                start = time.perf_counter()
                with torch.no_grad():
                    # Crop TIDAK dibalik: terukur flip menurunkan akurasi di
                    # semua konfigurasi input (260/pad10: 68,08% -> 64,62%).
                    probs = torch.softmax(model(
                        recognizer.preprocess(cropped)), 1)[0]
                self._record_stage("t_model", start)
                self.inferred_frames += 1
                letter = labels[int(probs.argmax())]
                self.last_letter = letter
                conf = float(probs.max())
                self.last_probs = recognizer.top_k(probs, labels)
                start = time.perf_counter()
                stable = smoother.update(letter)
                settle = self._settle_update(frame, cropped, self._last_marks)
                # Ambang 0,45 (terukur pada 260/pad10: conf benar p20=0,373,
                # conf salah p80=0,344): menyisakan 71% huruf benar dan 2,7%
                # sampel prediksi salah. Settle + stabilitas bbox menolak
                # frame transisi sebelum Smoother menilai label.
                if stable and conf > CONF_GATE and settle >= SETTLE_FRAMES:
                    self._pipeline.add_letter(stable, dict(zip(
                        labels, [float(p) for p in probs])))
                self._record_stage("t_smooth", start)
                self._flush()
                if self.on_candidate:
                    self.on_candidate(letter)
                start = time.perf_counter()
                self._preview(
                    frame, self._last_marks, self._pipeline.text, cropped,
                    letter, conf,
                )
                draw_text(frame, self._pipeline.text, self.font_size)
                self._record_stage("t_overlay", start)
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
    debug = "--debug" in sys.argv
    logs.setup_logging(debug=debug)
    worker = Worker(on_status=lambda t: print(t) if t else None, debug=debug)
    try:
        worker.run()
    except KeyboardInterrupt:
        worker.stop()
    except RuntimeError as exc:
        _log.error("ERROR: %s", exc)


if __name__ == "__main__":
    main()
