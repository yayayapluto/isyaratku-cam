# IsyaratKu Cam

Aplikasi desktop Python yang berjalan sebagai **kamera virtual**: membaca webcam,
mengenali isyarat tangan BISINDO (A–Z), menggambar teks terjemahan di video, dan
mengucapkannya lewat TTS offline ke mikrofon virtual. Kamera & mikrofon dipilih di
Zoom/Meet, bukan di aplikasi ini.

Target pengguna sementara: **penyandang tunawicara (bisu)** yang bisa mendengar
tetapi tidak dapat berbicara. Subtema: **"Akses untuk Semua"** (SDG 3, 10, 16).

## Status

M1–M6 jalan, terverifikasi E2E (`scripts/verify_m36.py`):

```
webcam → MediaPipe Hands (num_hands=2) → crop bbox +20px → letterbox persegi
→ EfficientNet-B3 A–Z → smoothing 4-dari-5 → huruf→kata→kalimat
→ overlay → OBS Virtual Camera + suara Piper → "CABLE Output"
```

Kalimat diucapkan **setelah tangan hilang 3 detik** (TECH_SPEC §4.5) — menahan
satu isyarat tidak membuat kata terpotong di tengah.
Hasil pengukuran akurasi A–Z: **belum ada** (`docs/results_m2.json` belum dibuat),
lihat [Pengukuran akurasi (M2)](#pengukuran-akurasi-m2).

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# Prasyarat di luar pip:
# 1. OBS Studio — "OBS Virtual Camera" (kamera virtual output utama).
#    Start Virtual Camera agar device aktif di Zoom/Meet.
# 2. VB-Cable — https://vb-audio.com/Cable/  → mikrofon virtual "CABLE Output".
# 3. espeak-ng — fonemisasi voice id_ID di Piper; sudah disiapkan lokal di
#    tools/espeak-ng/ (hasil extract MSI, tidak masuk PATH).
#    (Opsional, ditunda: Unity Capture https://github.com/schellingb/UnityCapture)

bash scripts/download_models.sh   # bobot A-Z (~44MB) + voice Piper (~63MB) + hand landmarker (~7,5MB)
python scripts/check_env.py       # 6 cek: VB-Cable, model A-Z, model tangan, voice, espeak-ng, kamera virtual
python main.py                    # Start / Stop + slider ukuran font overlay
```

## Pakai aplikasi

1. `python main.py` → **MULAI**.
2. Isyaratkan huruf satu per satu; jeda/rupanya huruf baru cukup untuk menambah huruf.
3. Turunkan tangan dan tunggu 3 detik → kata diucapkan (Zoom/Meet memakainya
   sebagai mikrofon) dan overlay bersih untuk kata berikutnya.
4. Slider mengatur ukuran font overlay (10–32, langsung berlaku saat jalan).

Di Zoom/Meet: kamera = **"OBS Virtual Camera"**, mikrofon = **"CABLE Output"**.
Tes lewat self-view Zoom/Meet.

## Prasyarat lingkungan

| Komponen | Fungsi | Status |
|---|---|---|
| OBS Studio | kamera virtual utama "OBS Virtual Camera" | **TERPASANG** |
| VB-Cable | mikrofon virtual "CABLE Output" | **TERPASANG** |
| espeak-ng | fonemisasi voice Indonesia Piper | **TERPASANG** (lokal `tools/espeak-ng/`, hasil extract MSI) |
| Unity Capture | kamera virtual alternatif bila OBS VC bermasalah | **DITUNDA, OPSIONAL** |
| Model A-Z | EfficientNet-B3, `Syizuril/bisindo-sign-language` | diunduh via script |
| Hand landmarker | deteksi tangan MediaPipe Tasks | diunduh via script |
| Voice Piper | `id_ID-news_tts-medium` (ONNX) | diunduh via script |

Catatan device: playback TTS mencari perangkat dengan "CABLE" dan
`max_output_channels > 0`, memilih WASAPI stereo 48 kHz. MME/DirectSound
mendaftar entri ganda (16 kanal) untuk perangkat yang sama.

## Pengukuran akurasi (M2)

```bash
python scripts/test_letters.py              # kondisi aplikasi: crop + letterbox
python scripts/test_letters.py --full-frame  # baseline: frame penuh
```

Jalankan **keduanya di kamera & cahaya yang sama** supaya bisa dibandingkan:
tanpa baseline full-frame, akurasi buruk tidak bisa diatribusi ke crop.
Tulis hasilnya ke `docs/MODEL_SELECTION.md` sebelum menyebut angka akurasi.

Peringatan yang masih terbuka (jangan lulus sebelum diuji):

- Ukuran input 224×224 dan normalisasi ImageNet adalah **[ASUMSI]** — model card
  tidak mendokumentasikannya.
- Letterbox persegi di crop adalah **[ASUMSI]** tentang cara data latih disiapkan.
- Variasi regional isyarat, pencahayaan, dan jarak tangan belum diukur.

## Peringatan lisensi [PERLU DICEK]

- Bobot model A–Z (`Syizuril/bisindo-sign-language`) repo **publik tanpa token**,
  tetapi **lisensi tidak tertulis**. Hanya diunduh saat setup, tidak di-bundle,
  tidak didistribusikan ulang sebelum lisensi terkonfirmasi.
- Voice Piper `id-ID-news_tts-medium` (dari `rhasspy/piper-voices`) — lisensi
  MODEL_CARD **belum diverifikasi**. Sama: unduh saat setup, jangan di-bundle.
- MediaPipe hand landmarker: Apache-2.0 (aman dipakai).

## Dokumentasi

- [`docs/PRD.md`](docs/PRD.md) — latar belakang, persona, user stories, FR/AC
- [`docs/TECH_SPEC.md`](docs/TECH_SPEC.md) — arsitektur, pipeline, integrasi
- [`docs/BUILD_ORDER.md`](docs/BUILD_ORDER.md) — milestone M1–M7 + checklist demo
- [`AGENTS.md`](AGENTS.md) — konvensi & aturan untuk coding agent

## Git

Format commit: `[type]: description` — type: `feat`, `fix`, `docs`, `refactor`,
`chore`, `test`, `perf`, `build`, `ci`. Micro commit dianjurkan (lihat
[`AGENTS.md`](AGENTS.md#git-commit)). Folder `models/`, `tools/`, dan `.venv/`
tidak di-commit.

## Privasi

Semua pengenalan dan TTS berjalan **offline & lokal**. Tidak ada pengiriman data
ke server.
