# TRAINING_PLAN — rencana fine-tuning A–Z BISINDO

Rencana menutup gap akurasi dataset VOC BISINDO (**68,65% terukur**,
`scripts/eval_offline.py`) terhadap angka model card 0,9860 (akurasi validasi
latih penulis model, bukan akurasi di webcam kami — `docs/MODEL_SELECTION.md`
§1). Semua angka di bawah berasal dari run verifikasi di repo ini; yang belum
terukur ditandai **[ASUMSI]** / **[UNVERIFIED]**.

## 1. Ringkasan dataset (terverifikasi)

Angka "citra masuk" di bawah dihitung dari isi folder (pasangan `*.jpg` +
`*.xml`, atau berkas citra per huruf), bukan dari readme sumber.

| Dataset | Sumber | Lisensi | Citra masuk | Crop 260×260 jadi | Status |
|---|---|---|---|---|---|
| `data/bisindo_rhio` (VOC) | github.com/rhiosutoyo/Indonesian-Sign-Language-BISINDO-Hand-Sign-Detection-Dataset | MIT (`data/bisindo_rhio/LICENSE`) | **514** pasangan unik | **514** | dipakai; baseline regresi |
| `alfredo_bisindo_letter` | [UNVERIFIED URL] kandidat: kaggle.com/datasets/alfredolorentiars/bisindo-letter-dataset | license tidak diketahui — periksa sebelum distribusi | 936 (650 unik) | **570** | dipakai (89 tanpa tangan + 277 crop duplikat internal; detail di `DATASET.md`) |
| `meisyavira_abjad_bisindo` | kaggle.com/datasets/meisyavira/abjad-bahasa-isyarat-indonesia-bisindo | license tidak diketahui — periksa sebelum distribusi | 650 | **619** | dipakai (31 tanpa tangan) |
| `achmadnoer_alfabet_bisindo` | kaggle.com/datasets/achmadnoer/alfabet-bisindo | license tidak diketahui — periksa sebelum distribusi | 312 | **0** | seluruh crop duplikat file-level dari meisyavira (312 dilewati) — sumbernya sudah tercakup meisyavira |
| `agungmrf_bisindo` | kaggle.com/datasets/agungmrf/indonesian-sign-language-bisindo | license tidak diketahui — periksa sebelum distribusi | — | — | **di-skip: unduh gagal/lambat (user instruction)** — arsip rusak (tanpa EOCD), redownload 2× gagal, partial dihapus |

Catatan rhio (diverifikasi per berkas, bukan dari readme upstream):

- Readme upstream menyebut 20 citra/huruf = 520. Isi folder VOC sebenarnya
  **514 pasangan** `*.jpg` + `*.xml`: A 19, F 19, H 19, Q 18, Y 19, sisanya 20.
- Ada **12 pasangan `" - Copy"`** (10 train + 2 test) yang byte-identik dengan
  berkas aslinya (MD5 cocok). Jadi di disk ada 526 berkas `.xml` = 514 anotasi
  asli + 12 salinan. `build_dataset.py` sekarang **melewati** nama yang memuat
  `" - Copy"` (log: `salinan ' - Copy' dilewati`): 526 `.xml` → **514
  ter-stage**, 12 pasangan salinan dilewati.
- `data/bisindo_rhio/collectedimages/` hanya berisi **520 berkas `.jpg`
  tanpa `.xml`**, tersebar di `<LETTER>/` (20 per huruf) — bukan layout VOC.
  Karena itu `_voc_candidate_dirs` aman: base `collectedimages` tidak
  memenuhi syarat (tanpa `.xml`) dan stage ganda rhio tidak terjadi. Yang
  terbaca tetap `<root>/{train,test}`.

Catatan dedupe lintas sumber (MD5 *piksel crop*, kemunculan pertama menang):

- **alfredo**: 936 berkas = **650 unik** (286 berkas duplikat *internal*,
  terverifikasi MD5 — aritmatika di
  `data/datasets/alfredo_bisindo_letter/DATASET.md`). Hasil cage per citra
  (dihitung sekali jalan, bukan estimasi): 89 tanpa tangan MediaPipe (9 di
  antaranya duplikat internal), 277 crop terduplikat (dihitung ulang dari
  piksel crop), sisa **570 crop unik**. Jadi 936 → 570 = 89 no-hand
  + 277 crop-duplikat + 570 crop bertahan.
- **meisyavira**: 650 − 31 tanpa tangan = **619** crop.
- **achmadnoer**: 312 citra semuanya **byte-identik** dengan citra meisyavira
  (terverifikasi MD5 file-level: 312 beririsan), jadi 0 crop baru.
- Data efektif: 514 + 570 + 619 = **1.703 crop unik** — persis jumlah di
  `data/finetune` (1.362 train + 341 val), karena seluruh crop sudah unik
  lintas-sumber (tidak ada pasangan identik yang terbagi antara train dan val).

## 2. Skenario fine-tuning

| # | Skenario | Estimasi waktu (GPU sehari) | Kapan dipakai |
|---|---|---|---|
| a | Baseline: bobot lama, tanpa latih, ukur ulang di holdout webcam | ~5 menit | wajib — patokan harus dari mesin & cahaya kita |
| b | Fine-tune **head saja** (`classifier` beku, backbone beku) | ~25–40 menit | rekomendasi pertama |
| c | Fine-tune penuh (backbone + head), lr kecil | ~2–3 jam | bila b stagnan di bawah 75% |

**Rekomendasi: (b) dulu, lalu (c) bila perlu.** Data tambahan kita hanya
**1.703 crop unik** (setelah dedupe lintas sumber, lihat §1), jauh di bawah
9.169 citra milik penulis model — discriminative lr 1e-4 pada head cukup
menyesuaikan kalibrasi kelas tanpa merusak fitur ImageNet+isyarat yang sudah
terbukti (crop bbox + letterbox memberi lompatan 68,65% vs 58,27% pada
224×224, `docs/MODEL_SELECTION.md` §4). Fine-tune penuh pada data sebesar ini
berisiko overfit ke satu persona.

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
- Staging rhio deterministik & bebas salinan: berkas VOC bernama `" - Copy"`
  dilewati (`scripts/build_dataset.py`), dan hanya SATU base VOC per root
  yang dipakai (base pertama yang punya `.xml` menang).
- Val tidak boleh dipakai untuk model selection berulang lebih dari sekali
  per eksperimen; uji akhir selalu di webcam (§5).

Hasil build terakhir (`--val-split 0.2 --seed 42`): **1.703 crop** di
`data/finetune` = **1.362 train + 341 val**, sama dengan total crop unik per
sumber di §1 (514 + 570 + 619) — tidak ada crop lain yang hilang di luar
baris `dilewati`:

```bash
python scripts/build_dataset.py \
  --sources "data/bisindo_rhio:voc" \
            "data/datasets/alfredo_bisindo_letter/BISINDO:folders" \
            "data/datasets/meisyavira_abjad_bisindo/dataset_merged:folders" \
            "data/datasets/achmadnoer_alfabet_bisindo/Citra BISINDO:folders" \
  --out data/finetune --val-split 0.2 --seed 42
```

## 5. Evaluasi & gate

1. Validasi `data/finetune/val`: metrik otomatis saat ini hanya `val_acc`
   dari `scripts/train_finetune.py` (per epoch, tercetak ke stdout) —
   confusion matrix 26×26 per huruf **belum ada skripnya** (gap, lihat §10).
   Catatan: `val_acc` pada `data/finetune/val` penuh **tidak setara**
   dengan baseline 68,65% — 1.189 dari 1.703 crop (alfredo 570 +
   meisyavira 619) berasal dari deteksi MediaPipe nyata
   (`build_dataset.py`, `read_folders_source` → landmark None), sedangkan
   baseline memakai landmark sintetis dari sudut bbox VOC. Namun subset
   rhio-saja dari val (514 crop → 103 crop val ter-split) memakai jalur
   crop sintetis bbox yang IDENTIK dengan `eval_offline.py`, jadi subset itu
   **setara langsung** dengan 68,65% dan boleh dipakai sebagai gate regresi
   G1; untuk bisa dipakai, subset itu harus diekstrak dulu (belum ada
   skripnya).
2. **Gate akhir: `python scripts/test_letters.py` di webcam** (kamera &
   cahaya sama dengan skenario a). Fine-tune hanya mengganti
   `models/bisindo_alphabet/` bila **mengalahkan 68,65%** pada holdout
   webcam itu.
3. Backup bobot lama sebelum swap: `models/bisindo_alphabet/` →
   `models/bisindo_alphabet_backup_<tanggal>/` (folder `models/` tidak masuk
   git, jadi backup murni lokal).
4. Gate confidence > 0,45 dan smoothing 4-dari-5 tetap aktif; perubahan di
   luar head tidak menyentuh `worker.py`/`recognizer.py`.

### Bar akurasi (NFR) — target produk

**Target produk: akurasi alfabet A–Z ≥ 85%** pada cahaya ruangan normal &
posisi tangan wajar — itu Acceptance Criteria NFR `Akurasi` di
`docs/PRD.md` §5 (`≥ 85%`, ditandai **[ASUMSI: target internal]** di PRD). Ini
target produk, **bukan** hasil terukur.

**Baseline terukur sekarang: 68,65%** (`scripts/eval_offline.py`, dataset VOC
520 citra, 260×260 + pad10 tanpa flip) — dan **itulah satu-satunya sumber
angka terukur di repo ini: `docs/MODEL_SELECTION.md` §3/§4. Jangan pakai
angka lain; jangan hitung ulang di dokumen ini.** Angka ini datang dari
jalur crop bbox VOC sintetis (`eval_offline.py` menyusun landmark dari sudut
bbox), jadi **tidak langsung setara** dengan metrik fine-tune apa pun di
`data/finetune/val` (crop dari deteksi MediaPipe nyata + PADDING=10 — asal
crop beda, lihat §10).

Gate fine-tuning (keduanya wajib, urut):

| # | Gate | Target | Status |
|---|---|---|---|
| G1 | `scripts/eval_offline.py` pada VOC rhio (`DATA_DIR` hardcoded ke `data/bisindo_rhio`, hanya baca `{train,test}`) | tetap **68,65%** sebagai cek regresi saja (turun berarti lingkungan rusak) — baseline ini diukur lewat jalur crop bbox VOC sintetis, jadi **tidak setara** dengan metrik fine-tune di `data/finetune/val` | baseline saja; belum dijalankan ulang pasca-rebuild |
| G2 | `scripts/test_letters.py` di webcam → `docs/results_m2.json` | **> 68,65%** pada holdout webcam yang sama dengan skenario a, menuju target 85%; A/B baseline vs bobot fine-tune wajib lewat jalur crop yang sama — tanpa skrip eval folder-crop untuk `data/finetune/val`, A/B yang valid hari ini hanya uji webcam ini (`val_loss`/`val_acc` `train_finetune.py` tetap internal latih, belum terverifikasi pada data penuh) | **belum ada berkasnya — wajib dibuat dulu; jumlahnya tidak boleh dikarang** |

## 6. Risiko

| Risiko | Mitigasi |
|---|---|
| Overfit ke 514 citra rhio (identitas, latar, pencahayaan tunggal) | campur beberapa sumber; early stop; ukur di webcam |
| Variasi regional isyarat & persona | rekam data sendiri (`scripts/recorder_cli.py`) sebagai split kelima |
| Latar/pencahayaan dataset ≠ webcam | ColorJitter; uji webcam sebagai gate |
| Lisensi tidak diketahui (alfredo, meisyavira, achmadnoer) | **tidak boleh redistribusi sebelum dicek**; model hasil latih yang di-share wajib cantumkan sebagai derivative |
| Duplikat lintas-sumber: meisyavira↔achmadnoer (312 citra byte-identik, terverifikasi MD5 file-level) | dedupe MD5 crop di `build_dataset.py`; duplikat tercatat di baris `dilewati` |
| Duplikat internal alfredo: 286 berkas byte-identik di dalam folder huruf yang sama (936 berkas → 650 unik, `data/datasets/alfredo_bisindo_letter/DATASET.md`) | dedupe MD5 piksel crop di `build_dataset.py` (277 crop duplikat alfredo dilewati, tercatat di baris `dilewati`); dedupe file-level saja tidak cukup — citra berbeda bisa jadi crop identik setelah resize INTER_AREA. Tidak terkait meisyavira/achmadnoer: MD5 file-level alfredo vs keduanya = 0 beririsan, dan resolusi alfredo (100–128 px) memang berbeda grid dari meisyavira (720–879 px) |
| Salinan `" - Copy"` di VOC rhio (12 pasangan byte-identik) | `build_dataset.py` melewati nama yang memuat `" - Copy"` (log `salinan ' - Copy' dilewati`) |
| Citra kecil alfredo (100–128 px) → crop up-scaled dari sedikit detail | pertahankan (variasi resolusi = augmentasi gratis), tapi jangan beri beban khusus |
| agungmrf hilang (±11,4 rb citra tambahan) | ulang unduhan di jaringan stabil; jangan pakai `curl -C -` pada unduhan yang pernah terputus (muncul hole 414 MB) |

## 7. Referensi skrip

| Skrip | Peran |
|---|---|
| `scripts/build_dataset.py` | rakit crop 260×260 dari sumber `voc`/`folders`, dedupe MD5 crop lintas sumber, lewati salinan `" - Copy"`, pakai satu base VOC per root, split stratified seed 42 |
| `scripts/train_finetune.py` | fine-tune EfficientNet-B3 (`--mode head|full`, seed 42, early stop) — augmentasi ada DI SINI, bukan di `build_dataset.py` |
| `scripts/recorder_cli.py` | rekam data webcam pengguna sendiri per huruf |
| `scripts/eval_offline.py` | akurasi & conf pada dataset VOC (68,65% terukur) |
| `scripts/test_letters.py` | uji webcam — dasar gate §5 |
| `data/datasets/*/DATASET.md` | inventaris per dataset (count, resolusi, MediaPipe hit-rate, hitungan duplikat) |

Flag `train_finetune.py` (default di kurung):

| Flag | Arti |
|---|---|
| `--data` | folder dataset (`data/finetune`) |
| `--out` | folder keluaran (`models/bisindo_alphabet_finetuned`) |
| `--mode` | `head` (backbone beku) / `full` (semua dilatih, backbone lr÷10) |
| `--epochs` | 30 |
| `--batch` | 16 |
| `--lr` | 1e-4 |
| `--seed` | 42 |
| `--device` | `auto` / `cpu` / `cuda` |
| `--resume` | path `.pth` bobot awal (default: bobot `recognizer.MODEL_PATH`) |

Perintah latih nyata:

```bash
# latih head saja
python scripts/train_finetune.py --data data/finetune --mode head

# fine-tune penuh
python scripts/train_finetune.py --data data/finetune --mode full
```

Keluaran: satu baris per epoch (`epoch`, `train_loss`, `val_loss`,
`val_acc`, lr, durasi), lalu `model.pth` + `labels.json` + `config.json` di
folder `--out`, best weights dipulihkan dari `val_loss` terendah.

## 8. Reproduksi dari clone bersih

`data/` ada di `.gitignore` (baris 32) — clone baru **tidak punya dataset
sama sekali**. Semua angka §1 butuh langkah di bawah.

Urutan langkah (semua path repo-relative; jalankan dari akar repo):

1. **rhio** (MIT, wajib — baseline regresi kita):

   ```bash
   git clone https://github.com/rhiosutoyo/Indonesian-Sign-Language-BISINDO-Hand-Sign-Detection-Dataset data/bisindo_rhio
   ```

   Layout: `train/` + `test/` berisi pasangan `<LETTER>.<uuid>.jpg` +
   `.xml` (VOC). `collectedimages/<LETTER>/*.jpg` juga ada tapi **tanpa
   `.xml`**, jadi tidak dipakai skrip VOC.

2. **Kaggle** (`pip install kaggle`, lalu letakkan `kaggle.json` di
   `~/.kaggle/`):

   ```bash
   kaggle datasets download -d achmadnoer/alfabet-bisindo -p data/datasets
   kaggle datasets download -d meisyavira/abjad-bahasa-isyarat-indonesia-bisindo -p data/datasets
   kaggle datasets download -d [UNVERIFIED slug] -p data/datasets   # alfredo
   ```

3. Unpack, lalu **path akar sumber** yang dipakai `--sources`:

   | Slug | Berkas hasil unduh | Folder setelah di-unpack | Root `--sources` |
   |---|---|---|---|
   | `achmadnoer/alfabet-bisindo` | `alfabet-bisindo.zip` | `data/datasets/achmadnoer_alfabet_bisindo/` | `.../Citra BISINDO:folders` |
   | `meisyavira/abjad-bahasa-isyarat-indonesia-bisindo` | `abjad-bahasa-isyarat-indonesia-bisindo.zip` | `data/datasets/meisyavira_abjad_bisindo/` | `.../dataset_merged:folders` |
   | alfredo — **[UNVERIFIED slug]** kandidat `alfredolorentiars/bisindo-letter-dataset` | zip (nama persis belum dicatat) | `data/datasets/alfredo_bisindo_letter/` | `.../BISINDO:folders` |

   Nama folder setelah unzip bisa menambahkan subfolder versi; sesuaikan
   sampai `<root>/` langsung berisi folder huruf `A`…`Z`. Slug alfredo belum
   diverifikasi — periksa nama folder hasil unzip sebelum menjalankan build.

4. `agungmrf/indonesian-sign-language-bisindo` **di-skip** (unduhan gagal,
   arsip rusak tanpa EOCD, `curl -C -` meninggalkan hole 414 MB).

5. Build dataset dengan perintah di §4.

**Lisensi alfredo, meisyavira, achmadnoer tidak diketahui.** Sampai itu
jelas: dataset **tidak boleh diredistribusi**; berbagi hasil latih wajib
dinyatakan sebagai turunan (derivative) dan menyebutkan sumbernya (§6).

## 9. Arsitektur landmark alternatif (opsi terpisah)

Penting supaya tidak tertukar: **pendekatan repo ini bukan model landmark.**
MediaPipe hanya dipakai untuk mengambil bbox tangan (`crop_hand`,
`PADDING=10`) → crop 260×260 → **EfficientNet-B3 klasifikasi citra**
(`recognizer.py`). Koordinat landmark tidak pernah jadi fitur model.

Jalur landmark nyata (mis. `suryaadji/bisindo-alphabet-mediapipe-hand-landmarks`,
[UNVERIFIED URL], 21 koordinat × x/y/z per tangan, CC BY 4.0) butuh hal lain:

| Item | Isi |
|---|---|
| Model | Dense/LSTM head kecil di atas deret 21×3 koordinat — **bukan** fine-tune bobot Syizuril |
| Bobot awal | tidak ada: dari nol, bukan `recognizer.MODEL_PATH` |
| Data | butuh sampling tangan nyata; dataset citra di §1 tidak berlaku |
| Status di rencana ini | **OUT OF SCOPE** — opsi terpisah, jangan diklaim bisa "fine-tune langsung" |

Bila nanti dikerjakan, jadikan dokumen rencana sendiri (dataset, split,
metrik, model head) di luar berkas ini.

## 10. Keterbatasan skrip (gap, jangan dianggap ada)

- **Tidak ada augmentasi saat build.** Augmentasi (ColorJitter 0,2, rotasi
  ±15°, RandomResizedCrop 0,9–1,0) hanya ada di `scripts/train_finetune.py`,
  jalan saat melatih.
- **Tidak ada self-test.** Verifikasi hasil build lewat hitungan berkas di
  §1/§4 dan determinisme MD5 pohon, bukan skrip otomatis.
- **Tidak ada skrip evaluasi/confusion matrix untuk `data/finetune/val`.**
  `scripts/eval_offline.py` tidak bisa dipakai di sana (`DATA_DIR` hardcoded ke
  `data/bisindo_rhio`, hanya baca format VOC `{train,test}`). Satu-satunya
  metrik val otomatis sekarang: baris `val_loss`/`val_acc` yang dicetak
  `scripts/train_finetune.py` setiap epoch. Confusion matrix 26×26 dan
  akurasi per huruf masih belum ada skripnya.
- **Belum ada skrip evaluasi folder-crop untuk `data/finetune/val`.**
  Bila nanti dibuat, bobot baseline dan hasil fine-tune WAJIB dijalankan
  lewat jalur crop yang identik dulu sebelum angkanya dibandingkan: crop
  `data/finetune` memakai deteksi MediaPipe nyata + PADDING=10
  (`scripts/build_dataset.py`), sedangkan `eval_offline.py` memakai landmark
  sintetis dari sudut bbox VOC — dua asal bbox berbeda, angkanya tidak
  dapat dibandingkan langsung.
- **Hasil build bergantung isi folder.** Klaim "20 citra/huruf = 520" dari
  readme rhio tidak cocok dengan isi folder (§1); pakai angka terverifikasi,
  bukan readme.

## 11. Ringkasan perubahan (2026-09-30)

- `scripts/build_dataset.py`: staging rhio kini bebas salinan `" - Copy"`
  (514 pasangan, bukan 526) dan satu base VOC per root
  (`collectedimages` rhio adalah jpg-only, jadi tidak menimbulkan stage ganda).
- `scripts/train_finetune.py`: skrip fine-tune baru (smoke-tested, mode
  `head` + `full`, CPU-only, 2 epoch).
- `docs/training_plan.md`: angka §1 direkonsiliasi, gate + NFR akurasi,
  langkah reproduksi clone, opsi landmark terpisah, gap terkait.
- `data/datasets/alfredo_bisindo_letter/DATASET.md`: aritmatika duplikat
  dibetulkan (829 → **650 unik**), atribusi duplikat sekarang menyebut
  internal-alfredo, bukan lintas-sumber ke meisyavira.
