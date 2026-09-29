"""Pipeline teks: huruf -> kata -> kalimat. TTS per KATA.

Aturan (diubah sesuai permintaan user):
- huruf stabil yang BERBEDA dari huruf terakhir ditambahkan ke kata.
- tangan absen >= WORD_PAUSE (1,2 dtk) -> KATA SELESAI: diucapkan lewat TTS
  dan ditambahkan ke buffer kalimat. Overlay tetap menampilkan kalimat+kata.
- tangan absen >= IDLE (3 dtk) -> KALIMAT SELESAI: buffer dibersihkan,
  tanpa TTS (sudah diucapkan per kata).
- API: add_letter() tiap frame huruf stabil; tick() tiap frame; return kalimat
  selesai hanya untuk status GUI, bukan untuk diucapkan.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

WORD_PAUSE = 1.2   # absen tangan segini -> kata selesai & diucapkan
IDLE_SECONDS = 3.0  # absen tangan segini -> kalimat selesai (buffer bersih)


class TextPipeline:
    def __init__(
        self,
        speak: Optional[Callable[[str], None]] = None,
        idle_seconds: float = IDLE_SECONDS,
        word_pause: float = WORD_PAUSE,
    ) -> None:
        self._speak = speak
        self._idle = idle_seconds
        self._word_pause = word_pause
        self.word = ""
        self.sentence = ""
        self.prev_letter: Optional[str] = None
        self._hand_present = False
        self._hand_absent_since: Optional[float] = None

    def _now(self) -> float:
        return time.monotonic()

    def add_letter(self, letter: str) -> None:
        """Panggil tiap kali model memberi huruf stabil."""
        if letter == self.prev_letter:
            return  # satu ketukan = satu huruf
        self.word += letter
        self.prev_letter = letter
        self._last_new = self._now()
        self.mark_present()

    def mark_present(self) -> None:
        """Panggil TIAP frame tangan ADA di frame, walau tak ada huruf baru —
        cegah timer absen jalan hanya karena tidak ada huruf baru."""
        self._hand_present = True
        self._hand_absent_since = None

    def clear_hand(self) -> None:
        """Panggil saat tangan hilang dari frame (mulai timer absen)."""
        self.prev_letter = None
        self._hand_present = False
        if self._hand_absent_since is None and (self.word or self.sentence):
            self._hand_absent_since = self._now()

    def tick(self) -> Optional[str]:
        """Kata: absen >= word_pause -> ucapkan + simpan ke kalimat.
        Kalimat: absen >= idle -> bersih, return untuk status GUI (tanpa TTS)."""
        if self._hand_present or self._hand_absent_since is None:
            return None
        absent = self._now() - self._hand_absent_since

        if absent >= self._word_pause and self.word:
            word = self.word.strip()
            self.word = ""
            self.prev_letter = None
            self.sentence = (self.sentence + word + " ").strip() + " "
            if self._speak:
                try:
                    self._speak(word)
                except Exception:
                    pass  # TTS gagal tidak boleh mematikan loop video
            # timer absen TETAP jalan: kalimat selesai pada idle dari absen sama

        if absent >= self._idle and self.sentence:
            done = self.sentence.strip()
            self.sentence = ""
            self.word = ""
            self.prev_letter = None
            self._hand_absent_since = None
            return done
        return None

    @property
    def text(self) -> str:
        """Overlay: kalimat terbentuk + kata yang sedang dibentuk (spasi akhir
        dipertahankan supaya kata berikutnya tidak menempel)."""
        return self.sentence + self.word


if __name__ == "__main__":
    spoken: list[str] = []
    pipe = TextPipeline(speak=spoken.append, word_pause=0.2, idle_seconds=0.4)

    pipe.add_letter("H")
    pipe.add_letter("H")  # harus diabaikan: huruf sama
    pipe.add_letter("I")
    assert pipe.text == "HI", pipe.text
    assert spoken == [], spoken

    # tangan MASIH di frame -> tidak ada yang diucapkan
    time.sleep(0.25)
    assert pipe.tick() is None
    assert spoken == [], spoken

    # tangan hilang 0.2 dtk -> KATA diucapkan, kalimat tersimpan
    pipe.clear_hand()
    time.sleep(0.25)
    assert pipe.tick() is None, "kalimat belum selesai (belum idle)"
    assert spoken == ["HI"], f"kata harus diucapkan tiap jeda: {spoken}"
    assert pipe.text == "HI ", pipe.text

    # kata kedua dalam kalimat sama
    pipe.add_letter("D")
    pipe.add_letter("U")
    pipe.add_letter("A")
    assert pipe.text == "HI DUA"
    pipe.clear_hand()
    time.sleep(0.25)
    assert pipe.tick() is None
    assert spoken == ["HI", "DUA"], spoken

    # absen >= idle -> kalimat selesai (tanpa TTS tambahan), buffer bersih
    time.sleep(0.2)
    done = pipe.tick()
    assert done == "HI DUA", done
    assert spoken == ["HI", "DUA"], "kalimat TIDAK diucapkan ulang"
    assert pipe.text == "", pipe.text

    # pipe kosong -> tick aman
    assert pipe.tick() is None
    print("text_pipeline self-check OK")
