"""Daftar kata bahasa Indonesia (KBBI) untuk koreksi/validasi huruf hasil isyarat.

Sumber: damzaky/kumpulan-kata-bahasa-indonesia-KBBI v1.0.0 (list_1.0.0.txt,
112.651 baris) -> disaring jadi huruf kecil a-z saja (67.662 kata).
Bukan runtime app: data/ gitignored, file ini cuma loader.

Pemakaian:
    from kata import is_valid, suggest
    is_valid("apa")     # True
    suggest("apa")      # ["apa"]
    suggest("kuca")     # kandidat terdekat 1 -> edit distance
"""

from __future__ import annotations

import os
from typing import Optional

ROOT = os.path.dirname(os.path.abspath(__file__))
KATA_FILES = (
    os.path.join(ROOT, "data", "kata_kbbi_bersih.txt"),
    os.path.join(ROOT, "data", "kata_kbbi.txt"),
)

_words: Optional[frozenset[str]] = None


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


def is_valid(word: str) -> bool:
    return word in kata()


def _dist(a: str, b: str) -> int:
    """Levenshtein satu baris (ukuran kecil, dp 2 baris cukup)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def suggest(word: str, n: int = 3) -> list[str]:
    """Kandidat terdekat dengan panjang mirip. Kosong kalau dataset absen."""
    words = kata()
    if not words or not word:
        return []
    band = range(max(1, len(word) - 2), len(word) + 3)
    best: list[tuple[int, str]] = []
    for cand in words:
        if len(cand) not in band:
            continue
        d = _dist(word, cand)
        best.append((d, cand))
    best.sort()
    return [w for _, w in best[:n]]


if __name__ == "__main__":
    total = kata()
    print(f"{len(total)} kata termuat")
    for w in ("apa", "kuca", "halobanget", "tangan"):
        print(f"  is_valid({w!r}) = {is_valid(w)} | suggest = {suggest(w)}")
