# AGENTS.md — IsyaratKu Cam

Ringkasan untuk coding agent (Claude Code / Cursor). Ikuti aturan di bawah tanpa
pengecualian.

## Proyek

Aplikasi desktop Python "kamera virtual": webcam → kenali isyarat tangan
BISINDO (A–Z) → teks overlay di frame keluar → ucapan otomatis lewat Piper TTS
ke mikrofon virtual. Kamera & mikrofon dipilih di Zoom/Meet, bukan di aplikasi.
Target pengguna sementara: penyandang tunawicara. Subtema: "Akses untuk Semua".

## Perintah

# Setup
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
bash scripts/download_models.sh      # bobot model A-Z + voice Piper (~63MB), tidak di-commit
python scripts/check_env.py          # cek device virtual + file model + espeak-ng

# Run
python main.py

# Test model seleksi (M2)
python scripts/test_webcam.py
```

## Struktur folder

```
docs/        PRD.md TECH_SPEC.md BUILD_ORDER.md MODEL_SELECTION.md
main.py      GUI PySide6: tombol start/stop, status, slider font
worker.py    thread kerja: capture, deteksi, recognizer, overlay, vcam send
recognizer.py  load model PyTorch + prediksi + smoothing 4-of-5
text_pipeline.py huruf -> kata -> kalimat + timer jeda 3 detik
tts.py       Piper ONNX -> device VB-Cable (cari by name)
virtual_cam.py pyvirtualcam send + gambar overlay teks
scripts/     download_models.sh, test_webcam.py, check_env.py
models/      bobot model + voice (tidak di-commit)
```

Baca `docs/PRD.md` minimal bagian FR/AC saat menyentuh requirement,
`docs/TECH_SPEC.md` saat menyentuh pipeline/integrasi, `docs/BUILD_ORDER.md`
saat mengerjakan milestone. ID `FR-xx`/`AC-xx` konsisten antar dokumen.

## Konvensi kode

- Python 3.10+, type hints pada fungsi publik, nama snake_case.
- Modul kecil, satu tanggung jawab; tanggung jawab lihat `TECH_SPEC.md §3`.
- Tanpa dependency baru bila stdlib/terpasang sudah cukup.
- Komentar hanya untuk "kenapa" non-obvious. Tandai penyederhanaan sadar:
  `# ponytail: <batas>, upgrade path <...>`.
- Model & voice tidak masuk git; `models/.gitkeep` saja. Unduh via
  `scripts/download_models.sh`.

## Aturan anti-overengineering (WAJIB)

- Satu proses, satu thread kerja untuk kamera+AI; tanpa asyncio, database,
  plugin system, config UI, logging framework, atau abstraksi berlapis.
- Total kode inti < ~600 baris, maksimal ~8 file.
- Pakai kode/contoh dari repo dan model yang sudah ada sebelum menulis baru.
- Fitur di luar bagian MUST di PRD TIDAK boleh dikerjakan.
- Pengujian cukup smoke test manual + checklist demo; bukan test suite.
- Luar scope: preview di aplikasi, GUI rekam/latih, kalimat isyarat kontinu,
  dua tangan penuh, hotkey, akun/cloud, pilihan perangkat.

## Prasyarat lingkungan (di luar pip)

- Unity Capture: `Install.bat` sebagai Administrator (DirectShow filter, tanpa test
  mode). Nanti muncul sebagai kamera "Unity Capture Camera".
- VB-Cable: installer resmi. Nanti muncul sebagai "CABLE Output" (input) /
  "CABLE Input" (output).
- espeak-ng: wajib untuk fonemisasi voice `id_ID` di Piper.
- Lisensi BELUM diverifikasi: voice Piper `id-ID-news_tts-medium` (MODEL_CARD) dan
  repo model `Syizuril/bisindo-sign-language` (tanpa lisensi tertulis). Tidak
  boleh di-bundle/distribusi ulang sebelum dicek — tulis di README.
- Model A-Z: urutan label A-Z, ukuran input, dan normalisasi BELUM
  terdokumentasi → `[ASUMSI]` sampai divalidasi webcam di M2.

## Git commit

- **Commit setiap perubahan yang masuk akal**, micro commit diperbolehkan dan
  dianjurkan — jangan menumpuk banyak fitur dalam satu commit.
- Format: `[type]: description` — deskripsi singkat, bahasa Inggris, imperative,
  tanpa tanda titik akhir.
- Type: `feat`, `fix`, `docs`, `refactor`, `chore`, `test`, `perf`, `build`, `ci`.
- Contoh:
  ```bash
  git add -A && git commit -m "feat: add webcam capture worker thread"
  git add -A && git commit -m "fix: guard missing vcam device on start"
  git add -A && git commit -m "docs: add model selection notes"
  ```
- Jaga commit tetap fokus satu perubahan logis (walking skeleton → recognizer →
  pipeline → tts → gui).
- Jangan commit folder `models/`, `.venv/`, atau artefak demo.

## Verifikasi minimal sebelum commit

- Jalankan yang terpengaruh: `python main.py` atau `python scripts/check_env.py`.
- Smoke: klik Start → status `Berjalan`; stream tampil di OBS.
- Bila perubahan menyentuh overlay/TTS, ulangi checklist di
  `docs/BUILD_ORDER.md`.
