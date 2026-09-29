"""GUI minimalis (M6): tombol besar Start/Stop + status + slider font.

QThread menjalankan Worker; GUI tidak pernah memanggil model langsung.
"""

from __future__ import annotations

import sys

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QLabel, QVBoxLayout, QWidget, QSlider,
)
from PySide6.QtCore import Qt
from qfluentwidgets import CheckBox, PushButton, TitleLabel, Slider

import worker as worker_mod


class WorkerThread(QThread):
    status = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        enable_tts: bool = True,
        font_size: int = 22,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self._worker = worker_mod.Worker(
            on_status=self.status.emit,
            enable_tts=enable_tts,
            font_size=font_size,
            debug=debug,
        )

    def run(self) -> None:
        try:
            self._worker.run()
        except RuntimeError as exc:
            self.error.emit(str(exc))

    def stop(self) -> None:
        self._worker.stop()


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("IsyaratKu Cam")
        self.resize(420, 320)
        self.thread: WorkerThread | None = None

        layout = QVBoxLayout(self)
        layout.addWidget(TitleLabel("IsyaratKu Cam"))

        self.toggle = PushButton("MULAI")
        self.toggle.clicked.connect(self.on_toggle)
        font = self.toggle.font()
        font.setPointSize(18)
        self.toggle.setFont(font)
        self.toggle.setMinimumHeight(72)
        layout.addWidget(self.toggle)

        layout.addWidget(QLabel("Teks terakhir:"))
        self.text_label = QLabel("-")
        layout.addWidget(self.text_label)

        self.status = QLabel("Berhenti")
        layout.addWidget(self.status)

        layout.addWidget(QLabel("Ukuran font overlay:"))
        self.slider = Slider(Qt.Orientation.Horizontal)
        self.slider.setRange(10, 32)
        self.slider.setValue(22)
        self.slider.valueChanged.connect(self.on_font_size)
        layout.addWidget(self.slider)

        self.debug_box = CheckBox("Mode debug: pratinjau kamera + bbox tangan")
        self.debug_box.setChecked(False)
        layout.addWidget(self.debug_box)

        self.setLayout(layout)

    def on_toggle(self) -> None:
        if self.thread:
            self.thread.stop()
            self.thread.wait(3000)   # tunggu webcam + kamera virtual dilepas
            self.thread = None
            self.toggle.setText("MULAI")
            self.status.setText("Berhenti")
            return
        debug = self.debug_box.isChecked()
        self.thread = WorkerThread(
            font_size=self.slider.value(),
            debug=debug,
        )
        self.thread.status.connect(self.on_status)
        self.thread.error.connect(self.on_error)
        self.thread.start()
        self.toggle.setText("STOP")
        self.status.setText("Berjalan" + (" (debug)" if debug else ""))

    def on_status(self, text: str) -> None:
        if text:
            self.text_label.setText(text)

    def on_font_size(self, value: int) -> None:
        # live saat worker jalan; kalau berhenti, dipakai saat Start berikutnya
        if self.thread:
            self.thread._worker.font_size = value

    def on_error(self, message: str) -> None:
        self.status.setText(f"Error: {message}")
        self.toggle.setText("MULAI")
        self.thread = None

    def closeEvent(self, event) -> None:
        if self.thread:
            self.thread.stop()
            self.thread.wait(3000)
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
