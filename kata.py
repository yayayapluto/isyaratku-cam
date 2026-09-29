"""Daftar kata bahasa Indonesia (KBBI) untuk koreksi/validasi huruf hasil isyarat.

Sumber: damzaky/kumpulan-kata-bahasa-indonesia-KBBI v1.0.0 (list_1.0.0.txt,
112.651 baris) -> disaring jadi huruf kecil a-z saja (67.662 kata).
Bukan runtime app: data/ gitignored, file ini cuma loader.

Pemakaian:
    from kata import is_valid, correct_word
    is_valid("apa")            # True
    correct_word([...])        # (kata_final, dikoreksi) dari profil huruf
"""

from __future__ import annotations

import math
import os
from typing import Optional

ROOT = os.path.dirname(os.path.abspath(__file__))
KATA_FILES = (
    os.path.join(ROOT, "data", "kata_kbbi_bersih.txt"),
    os.path.join(ROOT, "data", "kata_kbbi.txt"),
)

_EPS = 1e-6   # prob huruf yang tidak ada di distribusi posisi kandidat

_words: Optional[frozenset[str]] = None
_INDEX: Optional[dict[int, list[str]]] = None


def kata() -> frozenset[str]:
    """Muat sekali, cache modul-level. Kosong = file dataset belum diunduh."""
    global _words
    if _words is None:
        for path in KATA_FILES:
            if os.path.exists(path):
                with open(path, encoding="utf-8", errors="ignore") as fh:
                    _words = frozenset(
                        w.strip() for w in fh if w.strip().isalpha()
                        and w.strip().islower()
                    )
                break
        else:
            _words = frozenset()
    return _words


def _by_len() -> dict[int, list[str]]:
    """Bucket kata per panjang, sekali bangun (11 bucket, ~47 ms).

    Koreksi tidak mungkin menyisipkan/menghapus huruf (model memberi satu slot
    distribusi per huruf yang ditampilkan), jadi kandidat hanya butuh panjang
    yang sama — tanpa ini setiap koreksi memindai 67.662 kata (0,33-1,54 s),
    terlalu lambat untuk jeda kata 1,2 s."""
    global _INDEX
    if _INDEX is None:
        buckets: dict[int, list[str]] = {}
        for w in kata():
            buckets.setdefault(len(w), []).append(w)
        for group in buckets.values():
            group.sort()   # tie skor -> alfabetis pertama = deterministik
        _INDEX = buckets
    return _INDEX


def is_valid(word: str) -> bool:
    """Kata ada di kamus. .lower() wajib: model memberi uppercase, dataset KBBI
    lowercase — tanpa ini lookup uppercase selalu gagal."""
    return word.lower() in kata()


def correct_word(distributions: list[dict[str, float]]) -> tuple[str, bool]:
    """Koreksi kata dari profil huruf per posisi (urutan seperti dimasukkan).

    Return (kata_final_lowercase, benar_dari_hasil_koreksi).
    Kandidat hanya dari bucket panjang yang sama (model tidak menyisipkan
    atau menghapus huruf). Skor = -sum(log(prob)), prob huruf yang tak ada
    di distribusi posisi dianggap _EPS. Skor terkecil menang; tie alfabetis.
    Tanpa kandidat valid -> (tebakan, False): kata non-kamus tidak diganti
    dengan kata kamus lain."""
    guess = "".join(max(d, key=d.get) for d in distributions)
    if is_valid(guess):
        return guess.lower(), False
    best_cand: Optional[str] = None
    best_score = 0.0
    for cand in _by_len().get(len(guess), []):
        score = 0.0
        for pos, probs in enumerate(distributions):
            score -= math.log(max(probs.get(cand[pos], 0.0), _EPS))
        if best_cand is None or score < best_score:
            best_cand, best_score = cand, score
    if best_cand is not None:
        return best_cand, True
    return guess.lower(), False   # tak dikenal: parser keluarkan apa adanya


if __name__ == "__main__":
    total = kata()
    print(f"{len(total)} kata termuat")
    for w in ("apa", "KECA", "halobanget", "tangan"):
        print(f"  is_valid({w!r}) = {is_valid(w)}")

    # tebakan valid -> lowercase, bukan koreksi
    assert correct_word([{"A": .9}, {"P": .8}, {"A": .7}]) == ("apa", False)
    # tebakan invalid -> kandidat bucket panjang sama, hasil kata kamus
    kata2, fixed2 = correct_word(
        [{"K": .9, "A": .05}, {"U": .9, "I": .05},
         {"C": .9, "K": .05}, {"A": .9, "M": .05}]
    )
    assert is_valid(kata2), f"hasil koreksi bukan kata kamus: {kata2}"
    assert kata2 != "kuca", "tebakan invalid tak boleh lolos mentah"
    print(f"  correct_word(kUca-ish) = {correct_word([{'K': .9}, {'I': .9}])}")
    print("kata self-check OK")
