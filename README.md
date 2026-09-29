# IsyaratKu Cam

Aplikasi desktop Python yang berjalan sebagai **kamera virtual**: membaca webcam,
mengenali isyarat tangan BISINDO (A–Z), menggambar teks terjemahan di video, dan
mengucapkannya lewat TTS offline ke mikrofon virtual. Kamera & mikrofon dipilih di
Zoom/Meet, bukan di aplikasi ini.

Target pengguna sementara: **penyandang tunawicara (bisu)** yang bisa mendengar
tetapi tidak dapat berbicara. Subtema: **"Akses untuk Semua"** (SDG 3, 10, 16).

---

## Arsitektur

```
webcam (native res, backend MSMF) → MediaPipe Hands (num_hands=2, tiap 2 frame)
  → crop bbox +10px, letterbox persegi (hand_detect.py)
  → resize 260×260, ImageNet norm, tanpa flip (recognizer.py)
  → EfficientNet-B3 A–Z → confidence gate 0,45
  → settle 3 frame + bbox stabil → Smoother 4-dari-5 (recognizer.py)
  → huruf → kata (koreksi KBBI saat flush) → kalimat (text_pipeline.py)
  → overlay subtitle putih, wrap baris (worker.py) → Unity Video Capture
                        (Unity Capture utama, OBS Virtual Camera cadangan)
                        └→ tangan absen 1,2 dtk → Piper TTS per KATA → VB-Cable
```

Semua berjalan di **satu proses**, satu thread video + satu thread audio (TTS).
Tidak ada server, database, atau koneksi jaringan — 100% lokal.

**Aturan flush:**

| Kondisi | Aksi |
|---|---|
| Tangan ada di frame | kata belum selesai, tidak ada ucapan |
| Tangan hilang ≥ 1,2 dtk | **kata** diucapkan + masuk buffer kalimat |
| Tangan hilang ≥ 3 dtk | kalimat selesai, buffer bersih, overlay kosong |

Batas yang perlu diketahui: `WORD_PAUSE=1,2 dtk` juga memisahkan huruf dalam
satu kata — jeda antar-huruf lebih dari itu akan memecah kata.

---

## Struktur modul

| File | Tanggung jawab |
|---|---|
| `app.py` | GUI PySide6/qfluentwidgets: tombol MULAI/HENTIKAN, badge status, chip kesehatan, S/M/L. Thread worker dipisah dari UI. |
| `worker.py` | Loop utama: buka webcam → deteksi → klasifikasi → smoothing → overlay → kirim ke kamera virtual. `draw_text()` adalah satu-satunya penerjemah teks (dipakai debug & produksi). |
| `hand_detect.py` | `HandDetector` (MediaPipe Tasks, `num_hands=2`, singleton modul) dan `crop_hand()` (bbox + padding 10px, letterbox ke persegi). |
| `recognizer.py` | `build_model()` memuat bobot EfficientNet-B3 + urutan label; `preprocess()` normalisasi 260×260; `Smoother()` kebijakan 4-dari-5. |
| `kata.py` | Kamus KBBI (67.662 kata) + `correct_word()`: skor −Σlog(p) per posisi memilih kata kamus paling mungkin saat buffer huruf di-flush. |
| `logs.py` | stdlib `logging` (satu-satunya jalur): `logs/isyaratku.log` (INFO) + `logs/isyaratku-debug.log` (DEBUG, `dur_ms=` per tahap pipeline). |
| `text_pipeline.py` | Buffer huruf → kata → kalimat; `tick()` memicu TTS per kata (tangan absen ≥ 1,2 dtk) dan membersihkan buffer saat kalimat selesai (≥ 3 dtk). `mark_present()` dipanggil tiap frame tangan ada agar timer absen tidak salah jalan. |
| `tts.py` | Piper ONNX → VB-Cable "CABLE Input" (WASAPI 48 kHz), resample 22050→48000. `speak_async()` antre tanpa memotong backlog + thread daemon (COM diinisialisasi supaya WASAPI mau memutar audio) supaya audio tidak menghentikan loop video; `drain()` tunggu kata terakhir saat BERHENTI. |
| `scripts/` | verifikasi & tooling, lihat bawah. |

Rincian pipeline: [`docs/TECH_SPEC.md`](docs/TECH_SPEC.md). Pilihan model + semua
asumsi terukur: [`docs/MODEL_SELECTION.md`](docs/MODEL_SELECTION.md).

---

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# Prasyarat di luar pip:
# 1. Unity Capture — "Unity Video Capture" (kamera virtual output UTAMA).
#    Pasang lewat Install.bat di folder repo sebagai Administrator.
# 2. OBS Studio — "OBS Virtual Camera" (kamera virtual CADANGAN).
# 3. VB-Cable — https://vb-audio.com/Cable/  → mikrofon virtual "CABLE Output".
# 4. espeak-ng — fonemisasi voice id_ID di Piper; sudah lokal di
#    tools/espeak-ng/ (hasil extract MSI, tidak masuk PATH).

bash scripts/download_models.sh   # bobot A-Z (~44MB) + voice Piper (~63MB) + hand landmarker (~7,5MB)
python scripts/check_env.py       # 6 cek: VB-Cable, model A-Z, model tangan, voice, espeak-ng, kamera virtual
python app.py                    # MULAI / HENTIKAN + ukuran S/M/L + debug: app.py --debug
```

Worker mencari kamera virtual urut: **Unity Video Capture** dulu, kalau tidak
ada **OBS Virtual Camera**. Kalau dua-duanya belum aktif, MULAI batal dengan
pesan perbaikan. Kalau yang dipakai cadangan, nyalakan OBS → *Controls → Start
Virtual Camera*.

---

## Pakai aplikasi

1. `python app.py` → **MULAI**.
2. Isyaratkan huruf satu per satu; pause pendek cukup untuk menambah huruf.
3. Turunkan tangan ±1,2 detik → **kata** diucapkan dan masuk kalimat; overlay tetap menampilkan kalimat + kata berikutnya. Tangan turun ±3 detik → kalimat selesai, overlay bersih.
4. Segmented **S/M/L** mengatur ukuran font overlay (14/22/32, langsung berlaku
   saat jalan).
5. **Debug** (opsional): `python app.py --debug` → dua jendela pratinjau
   terpisah: **kamera** (landmark kuning, bbox magenta, panel status, overlay
   teks tampil ter-mirror) dan **crop** (citra persis yang masuk model —
   sudah mirror dan di-upscale 240px — plus huruf teratas + confidence).
   `q` menutup jendela dan menghentikan worker.

Di Zoom/Meet: kamera = **"Unity Video Capture"** (atau "OBS Virtual Camera"
kalau dipakai cadangan), mikrofon = **"CABLE Output"**.

---

## Verifikasi

Script ada di folder `scripts/`, semuanya bisa dijalankan tanpa Zoom:

| Script | Yang dibuktikan |
|---|---|
| `verify_m1.py` | stream worker → OBS Virtual Camera: overlay putih terbaca ulang sebagai input |
| `verify_m36.py` | worker mulai, frame terkirim, MediaPipe menemukan tangan, model mengklasifikasi crop |
| `verify_m5.py` | TTS offline: Piper → VB-Cable, RMS & peak diukur |
| `test_letters.py` | uji A–Z per huruf, `--full-frame` untuk baseline frame penuh |
| `check_env.py` | 6 prasyarat lingkungan (VB-Cable, model, voice, espeak-ng, kamera virtual) |

---

## Status akurasi

**Angka akurasi webcam A–Z belum diukur** (`docs/results_m2.json` belum ada).
Jangan menyebut angka akurasi sebelum:

```bash
python scripts/test_letters.py               # kondisi aplikasi: crop + letterbox
python scripts/test_letters.py --full-frame  # baseline: frame penuh
```

Keduanya harus dijalankan di kamera & cahaya yang sama, supaya akurasi buruk bisa
diatribusi ke crop atau ke model.

**Sudah terukur** (detail di `docs/MODEL_SELECTION.md`):

- Crop tangan > full frame: **64% vs 23%** → model dilatih pada crop tangan,
  jalur produksi benar.
- Deteksi MediaPipe di citra dataset: **76/78 (97%)** → detektor sehat.
- Preprocessing: 260×260 + ImageNet normalisasi konfigurasi terbaik pada
  dataset VOC 520 citra (68,65% vs 224×224 58,27%); padding 10px > 20px
  (68,65% vs 68,08%); flip crop malah menurunkan (68,65% → 64,62%).
  Ukur ulang: `python scripts/eval_offline.py`.

**Isu terbuka:** model dilatih dengan 9.169 citra; dataset publik BISINDO hanya
520 citra. Retrain dari dataset publik berisiko regresi akurasi. Dataset tetap
dipakai sebagai baseline regresi. Model weights juga tidak punya lisensi tertulis
— hanya diunduh saat setup, tidak di-bundle.

---

## Privasi

Semua pengenalan dan TTS berjalan **offline & lokal**. Tidak ada pengiriman data
ke server.
