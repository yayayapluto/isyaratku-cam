"""IsyaratKu Cam — GUI PySide6/qfluentwidgets.

Thread worker dipisah dari UI: WorkerThread melakukan deteksi/klasifikasi/
TTS/overlay, MainWindow hanya menerima signal Qt (tidak pernah menyentuh
widget dari thread worker). Semua komunikasi pintar-ke-UI melewati Signal.

Jalankan: python app.py [--debug]
  --debug  buka jendela pratinjau terpisah (bbox + titik landmark).
"""
from __future__ import annotations

import argparse
import os
import sys

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolTip,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CardWidget,
    FluentIcon,
    IndeterminateProgressRing,
    PrimaryPushButton,
    SegmentedWidget,
    SimpleCardWidget,
    StrongBodyLabel,
    SubtitleLabel,
    TitleLabel,
    ToolButton,
)

import hand_detect
import recognizer
import tts
import worker

STATE_STOPPED = "stopped"
STATE_PREPARING = "preparing"
STATE_RUNNING = "running"
STATE_ERROR = "error"

# Tema gelap halaman: setTheme() tak mengubah QPalette di build ini (terukur:
# palette Window tetap #efefef, TitleLabel fg tetap hitam setelah panggil),
# jadi warna latar & teks dipasang eksplisit di sini — putih terjamin terang.
PAGE_QSS = """
QWidget#PageRoot { background-color: #202020; }
"""

PLACEHOLDER = "Tunjukkan isyarat huruf ke kamera"
FONT_SIZES = {"S": 14, "M": 22, "L": 32}          # ukuran font OVERLAY
DISPLAY_SIZES = {"S": 22, "M": 28, "L": 32}       # ukuran font kartu teks
DEVICE_NAMES = {"unitycapture": "Unity Video Capture", "obs": "OBS Virtual Camera"}

MIC_FIX = (
    "Mikrofon VA-Cable tidak ditemukan. "
    "Pasang VB-CABLE, lalu pilih CABLE Output sebagai mikrofon di Zoom/Meet."
)
MODEL_FIX = (
    "Model tidak lengkap. Jalankan scripts/download_models.sh "
    "lalu letakkan bobot di models/."
)
CAM_FIX = (
    "Kamera virtual tidak ditemukan. Jalankan Install.bat Unity Capture "
    "sebagai Administrator, lalu restart aplikasi. Cadangan: pasang OBS."
)
STATE_TEXT = {
    STATE_STOPPED: ("Berhenti", "#B8BEC6"),
    STATE_PREPARING: ("Menyiapkan", "#4FA3E3"),
    STATE_RUNNING: ("Berjalan", "#3FBF7F"),
}
ERROR_TEXT = ("Terjadi masalah", "#FF6B6B")


class HealthChip(QLabel):
    """Chip kesehatan: hijau siap, kuning kurang, merah gagal + tooltip."""

    def __init__(
        self, label: str, description: str, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._label_text = label
        self._fix_text = description
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.set_state("unchecked")

    def set_state(self, level: str, detail: str = "") -> None:
        """level: ok | warn | bad | unchecked."""
        color = {
            "ok": "#1A9152",
            "warn": "#BE8214",
            "bad": "#C62C30",
            "unchecked": "#3A3F45",
        }[level]
        text = f"{self._label_text} …" if level == "unchecked" else self._label_text
        if detail and level != "unchecked":
            text = f"{self._label_text} · {detail}"
        self.setText(text)
        self.setStyleSheet(
            f"HealthChip {{ background: {color}; border-radius: 6px;"
            f" padding: 4px 10px; font-size: 13pt; color: white; }}"
        )
        # Klik (bukan hover) memperlihatkan langkah perbaikan untuk
        # kuning/merah — chip hijau tidak butuh langkah apa pun.
        self._fix_on_click = level in ("warn", "bad")
        self.setToolTip(self._fix_text if self._fix_on_click else "")

    def set_ok(self, detail: str = "") -> None:
        self.set_state("ok", detail)

    def set_warn(self, detail: str = "") -> None:
        self.set_state("warn", detail)

    def set_bad(self) -> None:
        self.set_state("bad")

    def set_unknown(self) -> None:
        self.set_state("unchecked")

    def mousePressEvent(self, event) -> None:
        if getattr(self, "_fix_on_click", False):
            QToolTip.showText(event.globalPosition().toPoint(), self._fix_text)
        super().mousePressEvent(event)

class WorkerThread(QThread):
    """Deteksi/klasifikasi/TTS/overlay di thread terpisah. Semua output ke
    GUI lewat Signal. Thread worker TIDAK boleh menyentuh widget langsung."""

    status = Signal(str)
    candidate_letter = Signal(str)
    spoken = Signal(str)
    health = Signal(dict)
    error = Signal(str)

    def __init__(
        self,
        font_size: int = 22,
        enable_tts: bool = True,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.font_size = font_size
        self.debug = debug
        self.enable_tts = enable_tts
        self._worker: worker.Worker | None = None

    def run(self) -> None:
        self._worker = worker.Worker(
            on_status=self.status.emit,
            on_candidate=self.candidate_letter.emit,
            on_spoken=self.spoken.emit,
            on_health=self.health.emit,
            enable_tts=self.enable_tts,
            font_size=self.font_size,
            debug=self.debug,
        )
        try:
            self._worker.run()
        except RuntimeError as exc:
            self.error.emit(str(exc))
        except OSError as exc:
            self.error.emit(f"perangkat gagal: {exc}")

    def stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()

    def set_font_size(self, size: int) -> None:
        self.font_size = size


CARD_BG = QColor(0x2A, 0x2A, 0x2A)   # latar card gelap (halaman gelap)


_TINT_MARK = "/*tint*/"


def tint(label: QWidget, color: str) -> None:
    """Tambahkan warna teks, JANGAN ganti stylesheet milik widget.

    Widget Fluent memasang QSS sendiri (FluentLabelBase{color:black} +
    pengaturan font). setStyleSheet("color: ...") menimpa SELURUH sheet itu —
    font ikut hilang. Append satu aturan saja; aturan terakhir yang menang,
    jadi warna harus diletakkan di akhir sheet widget itu sendiri.
    """
    cls = type(label).__name__          # selectornya kelas widget itu sendiri
    sheet = label.styleSheet()
    if _TINT_MARK in sheet:   # ganti warna tanpa numpuk aturan
        head = sheet.split(_TINT_MARK)[0]
        label.setStyleSheet(head + f"{_TINT_MARK}{cls} {{ color: {color}; }}")
    else:
        label.setStyleSheet(
            sheet + f"\n{_TINT_MARK}{cls} {{ color: {color}; }}")


def dark_card(card: QWidget) -> QWidget:
    """Paksa card jadi gelap.

    Card melukis latar di paintEvent() dari BackgroundColorObject, BUKAN dari
    QSS — jadi aturan halaman maupun setTheme tak pernah mengubahnya (terukur:
    CardWidget tetap #f0f0f0 walau isDarkTheme()==True). Menimpa warna di
    sumbernya; tanpa ini teks putih tak terbaca di card terang.
    """
    for child in card.children():
        if type(child).__name__ == "BackgroundColorObject":
            child.backgroundColor = CARD_BG
    return card


class MainWindow(QWidget):
    """Satu halaman tunggal. State mesin: Berhenti/Menyiapkan/Berjalan/Error."""

    def __init__(self, debug: bool = False) -> None:
        super().__init__()
        self.setObjectName("PageRoot")
        self.debug = debug
        self.thread: WorkerThread | None = None
        self.state = STATE_STOPPED
        self._last_spoken: list[str] = []
        self.setWindowTitle("IsyaratKu Cam")
        self.resize(420, 560)
        self.setMinimumSize(360, 480)
        self._build()
        self._apply_state(STATE_STOPPED)
        self.setStyleSheet(PAGE_QSS)   # latar gelap + label terang
        QTimer.singleShot(0, self.probe_health)

    # ---------- tampilan ----------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        head = QHBoxLayout()
        self.title_label = TitleLabel("IsyaratKu Cam", self)
        tint(self.title_label, "#FFFFFF")
        self.pin_button = ToolButton(FluentIcon.PIN, self)
        self.pin_button.setCheckable(True)
        self.pin_button.setToolTip("Selalu di atas")
        self.pin_button.toggled.connect(self.on_toggle_pin)
        head.addWidget(self.title_label)
        head.addStretch(1)
        head.addWidget(self.pin_button)
        root.addLayout(head)

        size_row = QHBoxLayout()
        size_label = BodyLabel("Ukuran", self)
        tint(size_label, "#FFFFFF")
        self.size_seg = SegmentedWidget(self)
        for key in ("S", "M", "L"):
            self.size_seg.insertItem(
                list(FONT_SIZES).index(key), key, key
            )
        # QSS milik halaman TIDAK sampai ke PivotItem: tiap item punya
        # stylesheet sendiri (15 aturan, paksa color: black di semua state),
        # dan paksa itu menang atas aturan induk. Jadi warna dipasang langsung
        # di tiap item: putih saat tak terpilih, hitam di atas pil terpilih.
        # Terukur render: glyph putih 36px di atas latar transparan.
        self.size_seg.setStyleSheet("background: transparent;")
        for item in self.size_seg.items.values():
            item.setStyleSheet(
                item.styleSheet()
                + "\nPivotItem { color: white; }"
            )
            # WAJIB setelah mengubah stylesheet item: tanpa polish ulang,
            # aturan lama (paksa hitam) tetap dipakai & huruf tak terlihat.
            item.style().unpolish(item)
            item.style().polish(item)
        self.size_seg.setCurrentItem("M")
        self.size_seg.currentItemChanged.connect(self.on_size_changed)
        size_row.addWidget(size_label)
        size_row.addStretch(1)
        size_row.addWidget(self.size_seg)
        root.addLayout(size_row)

        badge_row = QHBoxLayout()
        self.spinner = IndeterminateProgressRing(self)
        self.spinner.setFixedSize(16, 16)
        self.spinner.hide()
        self.state_label = StrongBodyLabel("Berhenti", self)
        badge_row.addWidget(self.spinner)
        badge_row.addWidget(self.state_label)
        badge_row.addStretch(1)
        root.addLayout(badge_row)

        self.text_card = dark_card(SimpleCardWidget(self))
        card_box = QVBoxLayout(self.text_card)
        card_box.setContentsMargins(18, 18, 18, 18)
        card_box.setSpacing(6)
        self.text_label = QLabel(PLACEHOLDER, self.text_card)
        self.text_label.setWordWrap(True)
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_label.setMinimumHeight(150)
        tint(self.text_label, "#9099A6")
        self.text_label.setFont(QFont("", self._display_size()))
        card_box.addWidget(self.text_label)
        self.candidate_label = CaptionLabel(
            "Tunjukkan isyarat huruf ke kamera", self.text_card
        )
        self.candidate_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tint(self.candidate_label, "#9AA3AE")
        card_box.addWidget(self.candidate_label)
        root.addWidget(self.text_card, 1)

        recent = dark_card(CardWidget(self))
        recent_box = QVBoxLayout(recent)
        recent_box.setContentsMargins(14, 12, 14, 12)
        recent_box.setSpacing(4)
        recent_head = SubtitleLabel("Diucapkan terakhir", recent)
        # SubtitleLabel juga punya QSS sendiri (color: black).
        tint(recent_head, "#FFFFFF")
        recent_box.addWidget(recent_head)
        self.recent_labels = [
            CaptionLabel("—", recent) for _ in range(3)
        ]
        for row in self.recent_labels:
            # CaptionLabel punya QSS sendiri (color: black): timpa langsung.
            tint(row, "#E8EAED")
        for row in self.recent_labels:
            row.setWordWrap(True)
            recent_box.addWidget(row)
        root.addWidget(recent)

        chips = QHBoxLayout()
        chips.setSpacing(8)
        self.chip_cam = HealthChip("Kamera virtual", CAM_FIX, self)
        self.chip_mic = HealthChip("Mikrofon", MIC_FIX, self)
        self.chip_model = HealthChip("Model", MODEL_FIX, self)
        for chip in (self.chip_cam, self.chip_mic, self.chip_model):
            chips.addWidget(chip)
        chips.addStretch(1)
        root.addLayout(chips)

        self.error_card = dark_card(CardWidget(self))
        error_box = QVBoxLayout(self.error_card)
        error_box.setContentsMargins(14, 12, 14, 12)
        error_box.setSpacing(6)
        self.error_label = StrongBodyLabel("Error", self.error_card)

        tint(self.error_label, "#FF6B6B")
        error_box.addWidget(self.error_label)
        self.error_text = CaptionLabel("", self.error_card)
        self.error_text.setWordWrap(True)
        tint(self.error_text, "#E8EAED")
        error_box.addWidget(self.error_text)
        self.retry_button = PrimaryPushButton("Coba lagi", self.error_card)
        self.retry_button.clicked.connect(self.start_worker)
        error_box.addWidget(self.retry_button, 0, Qt.AlignmentFlag.AlignLeft)
        root.addWidget(self.error_card)
        self.error_card.hide()

        self.start_button = PrimaryPushButton("MULAI", self)
        self.start_button.setMinimumHeight(48)
        self.start_button.setFont(QFont("", 14, QFont.Weight.Bold))
        self.start_button.clicked.connect(self.on_toggle)
        root.addWidget(self.start_button)

        hint = CaptionLabel(
            "Kamera virtual dipakai: ditampilkan setelah Mulai.", self
        )
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tint(hint, "#9AA3AE")
        root.addWidget(hint)
        self.hint_label = hint

    # ---------- state ----------
    def _apply_state(self, state: str) -> None:
        self.state = state
        text, color = STATE_TEXT.get(state, ERROR_TEXT)
        self.state_label.setText(text)
        tint(self.state_label, color)
        self.spinner.setVisible(state == STATE_PREPARING)
        if state == STATE_STOPPED or state == STATE_ERROR:
            self.start_button.setText("MULAI")
            self.start_button.setStyleSheet("")
        elif state == STATE_RUNNING:
            self.start_button.setText("HENTIKAN")
            self.start_button.setStyleSheet(
                "PrimaryPushButton { background-color: #C62C30;"
                " border: 1px solid #C62C30; }"
            )
        else:
            self.start_button.setText("Menyiapkan…")
            self.error_card.hide()
        if state == STATE_STOPPED:
            self.text_label.setText(PLACEHOLDER)
            tint(self.text_label, "#9099A6")
        if state != STATE_PREPARING and state != STATE_RUNNING:
            self.candidate_label.setText("Tunjukkan isyarat huruf ke kamera")

    def on_toggle(self) -> None:
        if self.state == STATE_RUNNING:
            self.stop_worker()
        else:
            self.start_worker()

    def start_worker(self) -> None:
        if self.thread is not None and self.thread.isRunning():
            return
        self.error_card.hide()
        self.text_label.setText(PLACEHOLDER)
        tint(self.text_label, "#9099A6")
        self._apply_state(STATE_PREPARING)
        self.text_label.setFont(QFont("", self._display_size()))
        self.thread = WorkerThread(
            font_size=self._font_size(),
            enable_tts=True,
            debug=self.debug,
        )
        self.thread.status.connect(self.on_status)
        self.thread.candidate_letter.connect(self.on_candidate)
        self.thread.spoken.connect(self.on_spoken)
        self.thread.health.connect(self.on_health)
        self.thread.error.connect(self.on_error)
        self.thread.finished.connect(self.on_finished)
        self.thread.start()

    def stop_worker(self) -> None:
        if self.thread is not None:
            self.thread.stop()

    def on_finished(self) -> None:
        if self.state != STATE_STOPPED and self.state != STATE_ERROR:
            self._apply_state(STATE_STOPPED)

    def _key(self) -> str:
        return self.size_seg.currentRouteKey() or "M"

    def _font_size(self) -> int:
        return FONT_SIZES[self._key()]

    def _display_size(self) -> int:
        return DISPLAY_SIZES[self._key()]

    def on_size_changed(self, _route: object) -> None:
        self.text_label.setFont(QFont("", self._display_size()))
        if self.thread is not None:
            self.thread.set_font_size(self._font_size())

    def on_status(self, text: str) -> None:
        if not text:
            return
        self.text_label.setText(text)
        tint(self.text_label, "#FFFFFF")

    def on_candidate(self, letter: str) -> None:
        if letter:
            self.candidate_label.setText(f"Kandidat: {letter}")

    def on_spoken(self, sentence: str) -> None:
        rows = [r.text() for r in self.recent_labels if r.text() != "—"]
        rows.insert(0, sentence)
        for row, value in zip(self.recent_labels, rows[:3]):
            row.setText(value)

    def on_health(self, data: dict) -> None:
        cam = data.get("virtual_cam")
        # Hanya dari worker yang baru start: probe saat aplikasi dibuka juga
        # memanggil ini saat state masih BERHENTI (jangan ubah tombol).
        # Health baru terkirim SETELAH webcam+vcam siap dan running=True.
        if cam and self.state == STATE_PREPARING:
            self._apply_state(STATE_RUNNING)
        if cam == "unitycapture":
            self.chip_cam.set_ok("Unity Video Capture")
        elif cam == "obs":
            self.chip_cam.set_warn("OBS Virtual Camera")
        else:
            self.chip_cam.set_bad()
        mic = data.get("virtual_mic")
        if mic:
            self.chip_mic.set_ok("CABLE Output")
        else:
            self.chip_mic.set_bad()
        if data.get("model"):
            self.chip_model.set_ok("siap")
        else:
            self.chip_model.set_bad()
        device = DEVICE_NAMES.get(cam, "kamera virtual cadangan")
        self.hint_label.setText(f"Pilih kamera di Zoom/Meet: {device}")

    def probe_health(self) -> None:
        """Cek kesehatan saat aplikasi dibuka — hanya file & device, tanpa
        membuka webcam/kamera virtual supaya tidak bentrok dengan worker.

        Mic dicek lewat tts.cable_output_device(); model & hand task cukup
        ada sebagai file (path sama dengan yang dipakai worker)."""
        try:
            tts.cable_output_device()
            mic = True
        except tts.TtsUnavailable:
            mic = False
        health = {
            "virtual_cam": None,
            "virtual_mic": mic,
            "model": all(
                os.path.exists(pth)
                for pth in (recognizer.MODEL_PATH, hand_detect.HAND_MODEL)
            ),
        }
        # virtual_cam=None -> on_health hanya set chip kamera, tidak
        # mengubah state. State tetap BERHENTI, teks terakhir tetap terlihat.
        self.on_health(health)

    def on_error(self, message: str) -> None:
        self._apply_state(STATE_ERROR)
        self.error_label.setText("Terjadi masalah")
        self.error_text.setText(message)
        self.error_card.show()

    def on_toggle_pin(self, pinned: bool) -> None:
        flags = self.windowFlags()
        if pinned:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        else:
            flags &= ~Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.pin_button.setIcon(FluentIcon.PIN if not pinned else FluentIcon.UNPIN)
        self.show()

    def closeEvent(self, event) -> None:
        self.stop_worker()
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait(3000)
        event.accept()


def main() -> None:
    parser = argparse.ArgumentParser(description="IsyaratKu Cam")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="buka jendela pratinjau terpisah (bbox + landmark)",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)  # Qt mengabaikan --debug argparse
    # Card melukis latar lewat paintEvent/isDarkTheme() — BUKAN QSS, jadi
    # aturan halaman tak pernah mengubahnya. setTheme(save=False) memindahkan
    # isDarkTheme()->True sehingga card jadi gelap; save=False tak menulis
    # config ke disk user.
    setTheme(Theme.DARK)
    window = MainWindow(debug=args.debug)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
