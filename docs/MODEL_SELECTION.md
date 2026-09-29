# MODEL_SELECTION — pemilihan & asumsi model A–Z BISINDO

Catatan jujur satu file untuk semua asumsi pipeline pengenalan. Angka yang
sudah **terukur** dipisahkan dari yang masih **[ASUMSI]**.

## 1. Model terpilih

| Item | Nilai |
|---|---|
| Nama | `Syizuril/bisindo-sign-language` (Hugging Face) |
| Arsitektur | EfficientNet-B3 ImageNet-pretrained, classifier head custom |
| Kelas | 26 (A–Z BISINDO) |
| Jumlah data latih | 9.169 citra (dari model card) |
| Akurasi validasi | 0,9860 @ epoch 25 (dari model card) |
| Lisensi | **Tidak tertulis** — bobot hanya diunduh, tidak didistribusikan ulang |

Head classifier diverifikasi cocok dengan checkpoint: `Dropout(0,4) → Linear(1536,512) → SiLU → Dropout(0,3) → Linear(512,26)`.
Urutan label diverifikasi dari `class_to_idx` di checkpoint: A→0 … Z→25.

**Asal dataset latih: belum teridentifikasi.** Model card tidak menyebut
dataset. Kandidat publik yang mungkin (belum dikonfirmasi):

- [rhiosutoyo/Indonesian-Sign-Language-BISINDO-Hand-Sign-Detection-Dataset](https://github.com/rhiosutoyo/Indonesian-Sign-Language-BISINDO-Hand-Sign-Detection-Dataset) — 20 foto × A–Z (520 citra)
- [agungmrf/indonesian-sign-language-bisindo](https://www.kaggle.com/datasets/agungmrf/indonesian-sign-language-bisindo) — alfabet BISINDO
- [achmadnoer/alfabet-bisindo](https://www.kaggle.com/datasets/achmadnoer/alfabet-bisindo) — A–Z skala 1:1, 3 latar

Akurasi 0,9860 itu **akurasi validasi latih**, bukan akurasi di webcam kami.

## 2. Kandidat lain (kenapa tidak dipakai)

| Kandidat | Alasan |
|---|---|
| `Louisljz/bisindo-sign-lang-recog` | Berbasis kata, bukan huruf; isi `labels.json` belum diperiksa |
| WL-BISINDO / AceKinnn (Kaggle) | Word-level; butuh frame video berurutan |
| `agungmrf` langsung sebagai model | Cuma dataset, tanpa bobot terlatih |
| Roboflow bisindo detection (v4/v9) | Object detection, bukan classifier A–Z |

## 3. Asumsi pipeline — status per item

| # | Asumsi | Status | Dasar |
|---|---|---|---|
| A1 | Ukuran input 224×224 | **[ASUMSI]** | Model card diam; default EfficientNet-B3 |
| A2 | Normalisasi ImageNet (0,485/0,456/0,406 + 0,229/0,224/0,225) | **[ASUMSI]** | Model card diam; sesuai riset EfficientNet-B3 |
| A3 | Frame masuk BGR (cv2 default) | Divalidasi | `recognizer.preprocess:51` `COLOR_BGR2RGB` |
| A4 | Crop bbox tangan + padding 20px | Divalidasi | TECH_SPEC §4.2 |
| A5 | **Letterbox persegi** sebelum resize 224×224 | **[ASUMSI]** | Bbox tangan TINGGI; squash merusak bentuk huruf |
| A6 | `num_hands=2` (landmark kedua tangan digabung) | **[ASUMSI]** | Sebagian huruf BISINDO memakai dua tangan |
| A7 | Gate confidence > 0,5 sebelum huruf diterima | **[ASUMSI]** | Angka hibrida; tuning dokumentasi |
| A8 | Smoothing 4-dari-5 | Divalidasi | FR/AC + TEST |
| A9 | Flush kalimat saat tangan absen ≥ 3 detik | Divalidasi | FR-07/AC-03/TECH_SPEC §4.5 |

## 4. Sudah terukur (bukan asumsi)

| Ukuran | Hasil | Script |
|---|---|---|
| Crop tangan nyata (20 frame) | min 158px, median 169px, max 182px | pengukuran langsung |
| `MIN_CROP = 32` | Guard hanya untuk tangan jauh; lapangan 158–182px | pengukuran langsung |
| Akurasi crop bbox pada dataset rhiosutoyo (130 citra berlabel) | **82/129 = 64%**, conf rata 0,48 | uji 3 jalur, §7 |
| worker E2E di resolusi native (640×480) | `sent=448`, gate tangan/klasifikasi menunggu tangan di frame | `scripts/verify_m36.py` |
| Overlay subtitle putih | 2593 pixel putih di area bawah, read-back 640×480 (ambang 150) | `scripts/verify_m1.py` |
| Hand-absence flush | self-check hijau: `tick()` dua kali → "HI" | `text_pipeline.py` |
| TTS ke VB-Cable | RMS=183,9, peak=32768 | `scripts/verify_m5.py` |

## 5. Belum terukur — prasyyat angka akurasi di iklan

1. `python scripts/test_letters.py` (**crop + letterbox**) → `docs/results_m2.json`
2. `python scripts/test_letters.py --full-frame` (baseline, kamera & cahaya sama)
   → kedua file tersimpan terpisah; A/B crop-vs-full
3. Faktor penyebab: variasi regional isyarat, pencahayaan, jarak tangan, latar

**Jangan sebut angka akurasi** di README, PR, lomba tanpa data ini.

## 6. Mode debug (opsional)

Checkbox "Mode debug" di `main.py` membuka jendela pratinjau: bbox magenta +
landmark kuning + ukuran crop + huruf + overlay identik dengan output.
Berguna untuk membedakan "model salah" dari "crop jelek / tangan jauh".

## 7. Uji 3 jalur input pada dataset rhiosutoyo (terukur)

Diukur pada 104–130 citra berlabel A–Z dari `data/bisindo_rhio/collectedimages/`:

| Jalur input | Benar | Conf rata |
|---|---|---|
| full frame langsung | 30/130 (23%) | 0.27 |
| crop bbox, resize squash | **82/129 (64%)** | 0.48 |
| crop bbox + letterbox | **82/129 (64%)** | 0.48 |

Simpulan:

- Model dilatih pada **crop tangan**, bukan frame penuh → jalur produksi
  `crop_hand()` benar.
- Letterbox == squash pada toleransi ini (82/129 identik). Perhatikan: crop uji
  memakai anotasi bbox VOC, sedangkan `worker.py` memakai landmark MediaPipe
  + PADDING=20 — geometri crop berbeda. Kesetaraan di sini berarti letterbox
  **tidak teruji terpisah**, bukan terbukti gratis/benar.
- Deteksi MediaPipe pada citra dataset: **76/78 (97%)** → detektor sehat.
  Hasil 0/40 di webcam berarti tidak ada tangan di frame, bukan bug.
- Sweep ukuran/normalisasi: 224×224 + ImageNet **65%**, half 56%, raw 38%,
  300×300 67% → asumsi A1/A2 benar.

**Sisa pertanyaan:** 64% jauh di bawah 98.6% di model card. Bobot `Syizuril`
dilatih dengan 9.169 citra (bukan 520 dari dataset ini), jadi gap wajar.
Dataset rhiosutoyo berguna sebagai **baseline regresi**, bukan jalur
fine-tune. Fine-tune sebaiknya memakai citra webcam milik pengguna.
