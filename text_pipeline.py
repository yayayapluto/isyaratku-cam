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
        self._hand_absent_since: Optional[float] = None

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
        self._hand_absent_since = None

    def clear_hand(self) -> None:
        """Panggil saat tangan hilang dari frame (mulai timer absen)."""
        self.prev_letter = None
        self._hand_present = False
        if self._hand_absent_since is None and self.word:
            self._hand_absent_since = self._now()

    def tick(self) -> Optional[str]:
        """Flush hanya kalau tangan absen >= idle (TECH_SPEC §4.5 / FR-07).

        Tangan tetap di frame -> jangan ucapkan kata yang belum selesai.
        """
        if self._hand_present or self._hand_absent_since is None:
            return None
        if self._now() - self._hand_absent_since < self._idle:
            return None
        if not self.word:
            self._hand_absent_since = None
            return None
        spoken_sentence = self.word.strip()
        self.word = ""
        self.prev_letter = None
        self._last_new = None
        self._hand_absent_since = None
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

    # tangan MASIH di frame -> jangan flush walau sudah 3 detik
    time.sleep(0.3)
    assert pipe.tick() is None, "tangan ada: kata belum selesai jangan diucapkan"
    assert spoken == [], spoken

    # tangan hilang -> timer absen jalan, baru flush
    pipe.clear_hand()
    time.sleep(0.25)
    assert pipe.tick() == "HI", "kalimat harus siap setelah tangan hilang + jeda"
    assert spoken == ["HI"]
    assert pipe.text == ""

    # pipe kosong -> tick aman
    assert pipe.tick() is None

    pipe.add_letter("O")
    assert pipe.text == "O"
    print("text_pipeline self-check OK")
