# IsyaratKu Cam

Aplikasi desktop Python yang berjalan sebagai **kamera virtual**: membaca webcam,
mengenali isyarat tangan BISINDO (A–Z), menggambar teks terjemahan di video, dan
mengucapkannya lewat TTS offline ke mikrofon virtual. Kamera & mikrofon dipilih di
Zoom/Meet, bukan di aplikasi ini.

Target pengguna sementara: **penyandang tunawicara (bisu)** yang bisa mendengar
tetapi tidak dapat berbicara. Subtema: **"Akses untuk Semua"** (SDG 3, 10, 16).

## Status

M1–M6 selesai: webcam → deteksi tangan MediaPipe → crop bbox → model A–Z →
smoothing 4-dari-5 → pipeline huruf→kata→kalimat → overlay → OBS Virtual
Camera + suara Piper ke "CABLE Output". GUI (`python main.py`) sudah teruji.
Milestone & urutan build: [`docs/BUILD_ORDER.md`](docs/BUILD_ORDER.md).

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# Prasyarat di luar pip:
# 1. OBS Studio — untuk "OBS Virtual Camera" (kamera virtual output utama).
#    Start Virtual Camera agar device aktif di Zoom/Meet.
# 2. VB-Cable — https://vb-audio.com/Cable/  → mikrofon virtual "CABLE Output".
# 3. espeak-ng — wajib untuk fonemisasi voice id_ID di Piper
#    https://github.com/espeak-ng/espeak-ng/releases
#    (OPSIONAL, tertunda: Unity Capture https://github.com/schellingb/UnityCapture)

bash scripts/download_models.sh   # bobot model A-Z + voice Piper (~63MB)
python scripts/check_env.py       # cek device virtual + file model + espeak-ng
python main.py
```

## Prasyarat lingkungan

| Komponen | Fungsi | Cara pasang / status |
|---|---|---|
| OBS Studio | kamera virtual utama "OBS Virtual Camera" | installer resmi; "Start Virtual Camera" — **TERPASANG** |
| VB-Cable | mikrofon virtual "CABLE Output" | installer resmi — **TERPASANG** |
| espeak-ng | fonemisasi voice Indonesia Piper | terpasang lokal: extract MSI ke `tools/espeak-ng/` — **TERPASANG** |
| Unity Capture | kamera virtual alternatif bila OBS VC bermasalah | `Install.bat` sebagai Administrator — **DITUNDA, OPSIONAL** |
| Model A-Z | bobot EfficientNet-B3 dari `Syizuril/bisindo-sign-language` | `scripts/download_models.sh` |
| Voice Piper | `id_ID-news_tts-medium` (ONNX) | `scripts/download_models.sh` |

Di Zoom/Meet: kamera = "OBS Virtual Camera", mikrofon = "CABLE Output".
Tes lewat self-view Zoom/Meet. Playback TTS ke "CABLE Input" — pilih host API
WASAPI (48 kHz, 2 kanal); MME/DirectSound mendaftar entri ganda.

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
