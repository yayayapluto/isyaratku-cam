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
python app.py

# Test & ukur
python scripts/test_webcam.py
python scripts/eval_offline.py       # akurasi dataset VOC-AC, log.* stdout
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
- Logging: stdlib `logging` saja, dipusatkan di `logs.py` (`setup_logging()`,
  `get_logger()`); **tanpa framework logging pihak ketiga**. Handler dipasang
  pada logger bernama `isyaratku`, BUKAN root logger (root INFO membuang semua
  DEBUG). Produksi jangan `print()` kecuali `__main__` self-check. Keluaran
  masuk `logs/isyaratku.log` (INFO) dan `logs/isyaratku-debug.log` (DEBUG,
  beserta `dur_ms=` per tahap); dua file itu di `.gitignore`.

## Prasyarat lingkungan (di luar pip)

- Unity Capture: `Install.bat` sebagai Administrator (DirectShow filter, tanpa test
  mode). Nanti muncul sebagai kamera "Unity Video Capture"
  (nama device sebenarnya — bukan "Unity Capture Camera").
- VB-Cable: installer resmi. Nanti muncul sebagai "CABLE Output" (input) /
  "CABLE Input" (output).
- espeak-ng: wajib untuk fonemisasi voice `id_ID` di Piper.
- Lisensi BELUM diverifikasi: voice Piper `id-ID-news_tts-medium` (MODEL_CARD) dan
  repo model `Syizuril/bisindo-sign-language` (tanpa lisensi tertulis). Tidak
  boleh di-bundle/distribusi ulang sebelum dicek — tulis di README.
- Model A-Z: label A-Z sudah diverifikasi dari checkpoint. Ukuran input
  (260×260) & padding (10 px) sudah terukur (`docs/MODEL_SELECTION.md §3`,
  `scripts/eval_offline.py`) → ganti `[ASUMSI]` di PRD bila masih tertulis.

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