# TRAINING_PLAN — rencana fine-tuning A–Z BISINDO

Rencana menutup gap akurasi dataset VOC 520 citra (**68,65% terukur**,
`scripts/eval_offline.py`) terhadap angka model card 0,9860 (akurasi validasi
latih penulis model, bukan akurasi di webcam kami — `docs/MODEL_SELECTION.md`
§1). Semua angka di bawah berasal dari run verifikasi di repo ini; yang belum
terukur ditandai **[ASUMSI]** / **[UNVERIFIED]**.

## 1. Ringkasan dataset (terverifikasi)

| Dataset | Sumber | Lisensi | Citra masuk | Crop 260×260 jadi | Status |
|---|---|---|---|---|---|
| `data/bisindo_rhio` (VOC) | github.com/rhiosutoyo/Indonesian-Sign-Language-BISINDO-Hand-Sign-Detection-Dataset | MIT (`data/bisindo_rhio/LICENSE`) | 520 | 520 | dipakai, baseline regresi |
| `alfredo_bisindo_letter` | [UNVERIFIED URL] kandidat: kaggle.com/datasets/alfredolorentiars/bisindo-letter-dataset | license tidak diketahui — periksa sebelum distribusi | 936 | 570 | dipakai (89 tanpa tangan + 277 duplikat crop) |
| `meisyavira_abjad_bisindo` | kaggle.com/datasets/meisyavira/abjad-bahasa-isyarat-indonesia-bisindo | license tidak diketahui — periksa sebelum distribusi | 650 | 619 | dipakai (31 dilewati) |
| `achmadnoer_alfabet_bisindo` | kaggle.com/datasets/achmadnoer/alfabet-bisindo | license tidak diketahui — periksa sebelum distribusi | 312 | 0 | seluruh crop duplikat dari meisyavira (312 dilewati) — sumbernya sudah tercakup meisyavira |
| `agungmrf_bisindo` | kaggle.com/datasets/agungmrf/indonesian-sign-language-bisindo | license tidak diketahui — periksa sebelum distribusi | — | — | **di-skip: unduh gagal/lambat (user instruction)** — arsip rusak (tanpa EOCD), redownload 2× gagal, partial dihapus |

Catatan penting: 312 dari 650 citra meisyavira **byte-identik** dengan
achmadnoer (hash MD5, `data/datasets/meisyavira_abjad_bisindo/DATASET.md`) —
kartu dataset meisyavira sendiri menyebutnya "combined dataset between Achmad
Noer's dataset and my original data". `scripts/build_dataset.py` sekarang
meng-**dedupe crop lintas sumber lewat MD5 piksel crop**: kemunculan pertama
menang, duplikat berikutnya dilewati dan dihitung di baris `dilewati` tiap
sumber (ditandai di log: `crop duplikat lintas sumber, dilewati`).
Akibatnya achmadnoer tidak menambah satu pun crop baru (0 dari 312) dan
alfredo turun dari 847 ke 570 (277 crop alfredo byte-identik dengan crop
meisyavira — latar & pose yang sama). Data efektif: 520 + 570 + 619 =
**1.709 crop unik**. Karena dedupe lintas-sumber, pasangan identik tidak lagi
bisa terbagi antara train dan val.

## 2. Skenario fine-tuning

| # | Skenario | Estimasi waktu (GPU sehari) | Kapan dipakai |
|---|---|---|---|
| a | Baseline: bobot lama, tanpa latih, ukur ulang di holdout webcam | ~5 menit | wajib — patokan harus dari mesin & cahaya kita |
| b | Fine-tune **head saja** (`classifier` beku, backbone beku) | ~25–40 menit | rekomendasi pertama |
| c | Fine-tune penuh (backbone + head), lr kecil | ~2–3 jam | bila b stagnan di bawah 75% |

**Rekomendasi: (b) dulu, lalu (c) bila perlu.** Data tambahan kita hanya
~1709 crop unik (setelah dedupe lintas sumber), jauh di bawah 9.169 citra
milik penulis model — discriminative lr 1e-4 pada head cukup menyesuaikan
kalibrasi kelas tanpa merusak fitur ImageNet+isyarat yang sudah terbukti
(crop bbox + letterbox memberi lompatan 68,65% vs 58,27% pada 224×224,
`docs/MODEL_SELECTION.md` §4). Fine-tune penuh pada data sebesar ini berisiko
overfit ke satu persona.

## 3. Hyperparameter

| Item | Head (b) | Full (c) |
|---|---|---|
| Optimizer | AdamW, weight_decay 0,01 | AdamW |
| lr | 1e-4 | 1e-5 backbone, 1e-4 head |
| Batch | 16 | 32 (batch 16 bila VRAM < 4 GB) |
| Epoch | 30 + early stop | 15 + early stop |
| Early stop | patience 5 pada `val_loss` | patience 3 |
| Scheduler | Cosine ke 1e-6 | Cosine per parameter-group |
| Label smoothing | 0,05 | 0,05 |
| Input | 260×260, normalisasi ImageNet | sama |
| Loss | CrossEntropy | sama |

Augmentasi ringkas: `ColorJitter(brightness 0,2, contrast 0,2, saturation 0,2)`,
rotasi kecil ±15°, random resized crop berskala 0,9–1,0.

**Flip horizontal: TIDAK dipakai.** Handedness (tangan kiri/kanan) tidak
dapat ditentukan dari nama berkas maupun metadata citra (EXIF hanya memuat
`CreatorTool`/tanggal). Beberapa huruf BISINDO bersifat spekular — membalik
membalik label — dan pengukuran di repo ini sudah menunjukkan flip menurunkan
akurasi di semua konfigurasi (260/pad10: 68,65% → 64,62%,
`docs/MODEL_SELECTION.md` §4). **[ASUMSI: tanpa flip]** sampai handedness
tiap dataset diverifikasi manual.

## 4. Strategi split

- Per-kelas stratified, val 20%, seed 42 — persis konfigurasi
  `scripts/build_dataset.py` yang sudah dijalankan.
- Dedupe crop lintas-sumber oleh MD5 piksel hasil crop (bukan MD5 berkas
  mentah): kemunculan pertama menang, duplikat dilewati dan dihitung di baris
  `dilewati` sumber tersebut. Diverifikasi deterministik: build dua kali →
  MD5 pohon output identik.
- Val tidak boleh dipakai untuk model selection berulang lebih dari sekali
  per eksperimen; uji akhir selalu di webcam (§5).

## 5. Evaluasi & gate

1. Confusion matrix 26×26 + akurasi per huruf pada `data/finetune/val`
2. **Gate akhir: `python scripts/test_letters.py` di webcam** (kamera &
   cahaya sama dengan skenario a). Fine-tune hanya mengganti
   `models/bisindo_alphabet/` bila **mengalahkan 68,65%** pada holdout
   webcam itu.
3. Backup bobot lama sebelum swap: `models/bisindo_alphabet/` →
   `models/bisindo_alphabet_backup_<tanggal>/` (folder `models/` tidak masuk
   git, jadi backup murni lokal).
4. Gate confidence > 0,45 dan smoothing 4-dari-5 tetap aktif; perubahan di
   luar head tidak menyentuh `worker.py`/`recognizer.py`.

## 6. Risiko

| Risiko | Mitigasi |
|---|---|
| Overfit ke 520 citra rhio (identitas, latar, pencahayaan tunggal) | campur beberapa sumber; early stop; ukur di webcam |
| Variasi regional isyarat & persona | rekam data sendiri (`scripts/recorder_cli.py`) sebagai split kelima |
| Latar/pencahayaan dataset ≠ webcam | ColorJitter; uji webcam sebagai gate |
| Lisensi tidak diketahui (alfredo, meisyavira, achmadnoer) | **tidak boleh redistribusi sebelum dicek**; model hasil latih yang di-share wajib cantumkan sebagai derivative |
| Duplikat lintas-sumber (meisyavira↔achmadnoer, meisyavira↔alfredo) | dedupe MD5 crop di `build_dataset.py`; duplikat tercatat di baris `dilewati` |
| Citra kecil alfredo (100–128 px) → crop up-scaled dari sedikit detail | pertahankan (variasi resolusi = augmentasi gratis), tapi jangan beri bobot khusus |
| agungmrf hilang (±11,4 rb citra tambahan) | ulang unduhan di jaringan stabil; jangan pakai `curl -C -` pada unduhan yang pernah terputus (muncul hole 414 MB) |

## 7. Referensi skrip

| Skrip | Peran |
|---|---|
| `scripts/build_dataset.py` | rakit crop 260×260 dari sumber `voc`/`folders`, dedupe MD5 lintas sumber, split stratified seed 42 |
| `scripts/recorder_cli.py` | rekam data webcam pengguna sendiri per huruf |
| `scripts/eval_offline.py` | akurasi & conf pada dataset VOC (68,65% terukur) |
| `scripts/test_letters.py` | uji webcam — dasar gate §5 |
| `data/datasets/*/DATASET.md` | inventaris per dataset (count, resolusi, MediaPipe hit-rate) |

Perintah build yang menghasilkan angka §1 (semua path repo-relative):

```bash
python scripts/build_dataset.py \
  --sources "data/bisindo_rhio:voc" \
            "data/datasets/alfredo_bisindo_letter/BISINDO:folders" \
            "data/datasets/meisyavira_abjad_bisindo/dataset_merged:folders" \
            "data/datasets/achmadnoer_alfabet_bisindo/Citra BISINDO:folders" \
  --out data/finetune --val-split 0.2 --seed 42
```

Catatan sumber VOC: `read_voc_source` menerima KEDUA layout —
`<root>/collectedimages/{train,test}/` (VOC rhio asli) dan
`<root>/{train,test}/` (layout repo saat ini); direktori yang sama hanya
dibaca sekali bila keduanya ada.
