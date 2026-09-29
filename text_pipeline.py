"""Pipeline teks: huruf -> kata -> kalimat. TTS per KATA.

Aturan (diubah sesuai permintaan user):
- huruf stabil yang BERBEDA dari huruf terakhir ditambahkan ke kata, bersama
  profil probabilitas tiap posisi (untuk koreksi KBBI).
- tangan absen >= WORD_PAUSE (1,2 dtk) -> KATA SELESAI: dikoreksi terhadap
  kamus KBBI, lalu diucapkan lewat TTS dan ditambahkan ke buffer kalimat.
  Kata non-kamus yang tidak punya kandidat tetap ditampilkan apa adanya.
- tangan absen >= IDLE (3 dtk) -> KALIMAT SELESAI: buffer dibersihkan,
  tanpa TTS (sudah diucapkan per kata).
- API: add_letter() tiap frame huruf stabil; tick() tiap frame; return kalimat
  selesai hanya untuk status GUI, bukan untuk diucapkan.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import logs
from kata import correct_word, is_valid

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
        # profil huruf per posisi kata yang sedang dibentuk; dipakai
        # correct_word() saat flush. Satu entri per huruf yang masuk kata.
        self._letters: list[dict[str, float]] = []

    def _now(self) -> float:
        return time.monotonic()

    def add_letter(self, letter: str, probs: Optional[dict[str, float]] = None) -> None:
        """Panggil tiap kali model memberi huruf stabil.

        probs: peta huruf -> probabilitas dari frame itu. Diberikan supaya
        koreksi kata bisa menimbang huruf runner-up, bukan hanya argmax.
        Tanpa probs (path uji) dipakai one-hot."""
        if letter == self.prev_letter:
            return  # satu ketukan = satu huruf
        self.word += letter
        self.prev_letter = letter
        self._letters.append(dict(probs) if probs else {letter: 1.0})
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
        """Kata: absen >= word_pause -> koreksi, ucapkan + simpan ke kalimat.
        Kalimat: absen >= idle -> bersih, return untuk status GUI (tanpa TTS)."""
        if self._hand_present or self._hand_absent_since is None:
            return None
        absent = self._now() - self._hand_absent_since

        if absent >= self._word_pause and self.word:
            raw = self.word.strip()
            self.word = ""
            self.prev_letter = None
            corrected, fixed = (
                correct_word(self._letters) if self._letters else (raw.lower(), False)
            )
            self._letters.clear()
            logs.get_logger().debug(
                "kata flush raw=%s corrected=%s fixed=%s", raw, corrected, fixed
            )
            if not fixed and not is_valid(corrected):
                # tak dikenal tetap tampil apa adanya; JANGAN disamarkan jadi
                # kata kamus lain supaya kalimat tidak berbohong.
                logs.get_logger().debug("kata tak dikenal: %s", raw)
            self.sentence = (self.sentence + corrected + " ").strip() + " "
            if self._speak:
                try:
                    self._speak(corrected)
                except Exception as exc:  # TTS rusak JANGAN membisukan diam-diam
                    logs.get_logger().error(
                        "[tts] %s: %s", type(exc).__name__, exc
                    )
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

    # "dua" & "aku" dipakai (bukan HI): keduanya kata KBBI, jadi koreksi
    # meninggalkannya apa adanya dan self-check menguji alur timer, bukan
    # koreksi.
    pipe.add_letter("D")
    pipe.add_letter("D")  # harus diabaikan: huruf sama
    pipe.add_letter("U")
    pipe.add_letter("A")
    assert pipe.text == "DUA", pipe.text
    assert spoken == [], spoken
    assert pipe._letters == [{"D": 1.0}, {"U": 1.0}, {"A": 1.0}], pipe._letters

    # tangan MASIH di frame -> tidak ada yang diucapkan
    time.sleep(0.25)
    assert pipe.tick() is None
    assert spoken == [], spoken

    # tangan hilang 0.2 dtk -> KATA diucapkan, kalimat tersimpan
    pipe.clear_hand()
    time.sleep(0.25)
    assert pipe.tick() is None, "kalimat belum selesai (belum idle)"
    assert spoken == ["dua"], f"kata harus diucapkan tiap jeda: {spoken}"
    assert pipe.text == "dua ", pipe.text
    assert pipe._letters == [], "buffer huruf harus bersih setelah flush"

    # kata kedua dalam kalimat sama
    pipe.add_letter("A")
    pipe.add_letter("K")
    pipe.add_letter("U")
    assert pipe.text == "dua AKU"
    pipe.clear_hand()
    time.sleep(0.25)
    assert pipe.tick() is None
    assert spoken == ["dua", "aku"], spoken

    # absen >= idle -> kalimat selesai (tanpa TTS tambahan), buffer bersih
    time.sleep(0.2)
    done = pipe.tick()
    assert done == "dua aku", done
    assert spoken == ["dua", "aku"], "kalimat TIDAK diucapkan ulang"
    assert pipe.text == "", pipe.text

    # probs runner-up dipakai: tebakan argmax "kuca" bukan kata KBBI, harus
    # dikoreksi ke kandidat kamus, bukan dibuang.
    pipe2 = TextPipeline(speak=spoken.append, word_pause=0.2, idle_seconds=0.4)
    pipe2.add_letter("K", {"K": 0.9, "A": 0.05})
    pipe2.add_letter("U", {"U": 0.9, "I": 0.05})
    pipe2.add_letter("C", {"C": 0.9, "K": 0.05})
    pipe2.add_letter("A", {"A": 0.9, "M": 0.05})
    pipe2.clear_hand()
    time.sleep(0.25)
    pipe2.tick()
    assert spoken[-1] != "kuca", f"tebakan non-kamus lolos mentah: {spoken[-1]}"
    assert is_valid(spoken[-1]), f"hasil koreksi bukan kata kamus: {spoken[-1]}"

    # pipe kosong -> tick aman
    assert pipe.tick() is None
    print("text_pipeline self-check OK")
