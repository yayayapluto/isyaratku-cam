# IsyaratKu Cam

Aplikasi desktop Python yang berjalan sebagai **kamera virtual**: membaca webcam,
mengenali isyarat tangan BISINDO (A–Z), menggambar teks terjemahan di video, dan
mengucapkannya lewat TTS offline ke mikrofon virtual. Kamera & mikrofon dipilih di
Zoom/Meet, bukan di aplikasi ini.

Target pengguna sementara: **penyandang tunawicara (bisu)** yang bisa mendengar
tetapi tidak dapat berbicara. Subtema: **"Akses untuk Semua"** (SDG 3, 10, 16).

## Status

Fase perencanaan. Milestone dan urutan build: [`docs/BUILD_ORDER.md`](docs/BUILD_ORDER.md).

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# Prasyarat di luar pip:
# 1. Unity Capture — Install.bat sebagai Administrator (DirectShow filter)
#    https://github.com/schellingb/UnityCapture
# 2. VB-Cable — https://vb-audio.com/Cable/
# 3. espeak-ng — wajib untuk fonemisasi voice id_ID di Piper
#    https://github.com/espeak-ng/espeak-ng/releases

bash scripts/download_models.sh   # bobot model A-Z + voice Piper (~63MB)
python scripts/check_env.py       # cek device virtual + file model + espeak-ng
python main.py
```

## Prasyarat lingkungan

| Komponen | Fungsi | Cara pasang |
|---|---|---|
| Unity Capture | kamera virtual "Unity Capture Camera" (terlihat di OBS) | `Install.bat` sebagai Administrator — DirectShow filter, tanpa test mode |
| OBS Studio | tes di Zoom/Meet (fallback kamera virtual) | installer resmi; pakai "Start Virtual Camera" |
| VB-Cable | mikrofon virtual "CABLE Output" | installer resmi |
| espeak-ng | fonemisasi voice Indonesia Piper | installer `.exe` dari release GitHub |
| Model A-Z | bobot EfficientNet-B3 dari `Syizuril/bisindo-sign-language` | `scripts/download_models.sh` |
| Voice Piper | `id_ID-news_tts-medium` (ONNX) | `scripts/download_models.sh` |

Di Zoom/Meet: kamera = "Unity Capture Camera" (fallback: OBS Virtual Camera),
mikrofon = "CABLE Output".

## Peringatan lisensi [PERLU DICEK]

- Bobot model A-Z (`Syizuril/bisindo-sign-language`) repo **publik tanpa token**,
  tetapi **lisensi tidak tertulis**. Hanya diunduh saat setup, tidak di-bundle,
  tidak didistribusikan ulang sebelum lisensi terkonfirmasi.
- Voice Piper `id-ID-news_tts-medium` (dari `rhasspy/piper-voices`) — lisensi
  MODEL_CARD **belum diverifikasi**. Sama: unduh saat setup, jangan di-bundle.
- Asumsi model (urutan label A–Z, ukuran input, normalisasi) **belum
  terdokumentasi**; divalidasi saat seleksi model, lihat
  [`docs/TECH_SPEC.md` §6](docs/TECH_SPEC.md).

## Dokumentasi

- [`docs/PRD.md`](docs/PRD.md) — latar belakang, persona, user stories, FR/AC
- [`docs/TECH_SPEC.md`](docs/TECH_SPEC.md) — arsitektur, pipeline, integrasi
- [`docs/BUILD_ORDER.md`](docs/BUILD_ORDER.md) — milestone M1–M7 + checklist demo
- [`AGENTS.md`](AGENTS.md) — konvensi & aturan untuk coding agent

## Git

Format commit: `[type]: description` — type: `feat`, `fix`, `docs`, `refactor`,
`chore`, `test`, `perf`, `build`, `ci`. Micro commit dianjurkan (lihat
[`AGENTS.md`](AGENTS.md#git-commit)). Folder `models/` dan `.venv/` tidak
di-commit.

## Privasi

Semua pengenalan dan TTS berjalan **offline & lokal**. Tidak ada pengiriman data
ke server.
