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
webcam (native res) → flip → MediaPipe Hands (num_hands=2)
  → crop bbox +20px, letterbox persegi (hand_detect.py)
  → resize 224×224, ImageNet norm (recognizer.py)
  → EfficientNet-B3 A–Z → confidence
  → Smoother 4-dari-5 (recognizer.py)
  → huruf → kata → kalimat (text_pipeline.py)
  → overlay subtitle putih, wrap baris (worker.py) → OBS Virtual Camera
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
| `main.py` | GUI PySide6/qfluentwidgets: tombol MULAI/BERHENTI, slider ukuran font, checkbox Mode debug. Thread worker dipisah dari UI. |
| `worker.py` | Loop utama: buka webcam → deteksi → klasifikasi → smoothing → overlay → kirim ke kamera virtual. `draw_text()` adalah satu-satunya penerjemah teks (dipakai debug & produksi). |
| `hand_detect.py` | `HandDetector` (MediaPipe Tasks, `num_hands=2`, singleton modul) dan `crop_hand()` (bbox + padding 20px, letterbox ke persegi). |
| `recognizer.py` | `build_model()` memuat bobot EfficientNet-B3 + urutan label; `preprocess()` normalisasi 224×224; `Smoother()` kebijakan 4-dari-5. |
| `text_pipeline.py` | Buffer huruf → kata → kalimat; `tick()` memicu TTS per kata (tangan absen ≥ 1,2 dtk) dan membersihkan buffer saat kalimat selesai (≥ 3 dtk). `mark_present()` dipanggil tiap frame tangan ada agar timer absen tidak salah jalan. |
| `tts.py` | Piper ONNX → VB-Cable "CABLE Input" (WASAPI 48 kHz), resample 22050→48000. `speak_async()` antre + thread daemon (backlog maks 2) supaya audio tidak menghentikan loop video; `drain()` tunggu kata terakhir saat BERHENTI. |
| `scripts/` | verifikasi & tooling, lihat bawah. |

Rincian pipeline: [`docs/TECH_SPEC.md`](docs/TECH_SPEC.md). Pilihan model + semua
asumsi terukur: [`docs/MODEL_SELECTION.md`](docs/MODEL_SELECTION.md).

---

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# Prasyarat di luar pip:
# 1. OBS Studio — "OBS Virtual Camera" (kamera virtual output utama).
# 2. VB-Cable — https://vb-audio.com/Cable/  → mikrofon virtual "CABLE Output".
# 3. espeak-ng — fonemisasi voice id_ID di Piper; sudah lokal di
#    tools/espeak-ng/ (hasil extract MSI, tidak masuk PATH).

bash scripts/download_models.sh   # bobot A-Z (~44MB) + voice Piper (~63MB) + hand landmarker (~7,5MB)
python scripts/check_env.py       # 6 cek: VB-Cable, model A-Z, model tangan, voice, espeak-ng, kamera virtual
python main.py                    # Start / Stop + slider ukuran font overlay
```

**Nyalakan OBS Virtual Camera sebelum MULAI** (OBS → *Controls → Start Virtual
Camera*). Worker membatalkan sendiri dengan pesan jelas kalau device tidak aktif.

---

## Pakai aplikasi

1. `python main.py` → **MULAI**.
2. Isyaratkan huruf satu per satu; pause pendek cukup untuk menambah huruf.
3. Turunkan tangan ±1,2 detik → **kata** diucapkan dan masuk kalimat; overlay tetap menampilkan kalimat + kata berikutnya. Tangan turun ±3 detik → kalimat selesai, overlay bersih.
4. Slider mengatur ukuran font overlay (10–32, langsung berlaku saat jalan).
5. **Mode debug** (opsional): centang sebelum MULAI → jendela CCTV dengan bbox
   magenta, landmark kuning, ukuran crop, dan teks overlay. `q` menutup jendela
   dan menghentikan worker.

Di Zoom/Meet: kamera = **"OBS Virtual Camera"**, mikrofon = **"CABLE Output"**.

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
- Preprocessing: 224×224 + ImageNet normalisasi konfigurasi terbaik
  (65% vs 56% half, 38% raw).

**Isu terbuka:** model dilatih dengan 9.169 citra; dataset publik BISINDO hanya
520 citra. Retrain dari dataset publik berisiko regresi akurasi. Dataset tetap
dipakai sebagai baseline regresi. Model weights juga tidak punya lisensi tertulis
— hanya diunduh saat setup, tidak di-bundle.

---

## Privasi

Semua pengenalan dan TTS berjalan **offline & lokal**. Tidak ada pengiriman data
ke server.
