"""Pipeline teks: huruf stabil -> kata -> kalimat.

Aturan (PRD FR-07/FR-08/AC-03):
- huruf stabil yang BERBEDA dari huruf terakhir ditambahkan ke kata.
- jeda 3 detik tanpa huruf baru -> kalimat selesai: diterjemahkan ke suara
  (via tts.speak) dan buffer dibersihkan.
- API: add_letter() tiap frame huruf stabil; tick() tiap frame untuk cek jeda;
  sentence_ready() mengosongkan buffer kalimat.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

IDLE_SECONDS = 3.0


class TextPipeline:
    def __init__(
        self,
        speak: Optional[Callable[[str], None]] = None,
        idle_seconds: float = IDLE_SECONDS,
    ) -> None:
        self._speak = speak
        self._idle = idle_seconds
        self.word = ""
        self.prev_letter: Optional[str] = None
        self._last_new: Optional[float] = None
        self._hand_present = False

    def _now(self) -> float:
        return time.monotonic()

    def add_letter(self, letter: str) -> None:
        """Panggil tiap kali model memberi huruf stabil."""
        if letter == self.prev_letter:
            return  # huruf sama ditahan sekali saja (satu ketukan = satu huruf)
        self.word += letter
        self.prev_letter = letter
        self._last_new = self._now()
        self._hand_present = True

    def clear_hand(self) -> None:
        """Panggil saat tangan hilang dari frame."""
        self.prev_letter = None

    def tick(self) -> Optional[str]:
        """Panggil tiap frame. Kembalikan kalimat bila jeda 3 detik tercapai."""
        if self._last_new is None:
            return None
        if self._now() - self._last_new < self._idle:
            return None
        if not self.word:
            return None
        spoken_sentence = self.word.strip()
        self.word = ""
        self.prev_letter = None
        self._last_new = None
        if self._speak:
            try:
                self._speak(spoken_sentence)
            except Exception:
                pass  # TTS gagal tidak boleh mematikan loop video
        return spoken_sentence

    @property
    def text(self) -> str:
        """Teks yang tampil di overlay: kata yang sedang dibentuk."""
        return self.word


if __name__ == "__main__":
    spoken: list[str] = []
    pipe = TextPipeline(speak=spoken.append, idle_seconds=0.2)

    pipe.add_letter("H")
    pipe.add_letter("H")  # harus diabaikan: huruf sama
    pipe.add_letter("I")
    assert pipe.text == "HI", pipe.text

    time.sleep(0.25)
    assert pipe.tick() == "HI", "kalimat harus siap setelah jeda"
    assert spoken == ["HI"]
    assert pipe.text == "", pipe.text

    pipe.clear_hand()
    pipe.add_letter("O")
    assert pipe.text == "O"
    print("text_pipeline self-check OK")
