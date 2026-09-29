# BUILD_ORDER — IsyaratKu Cam

Urutan milestone M1..M7. Bukan jadwal harian — selesai milestone = satu bukti
bisa dites. ID `FR-xx`/`AC-xx` merujuk `docs/PRD.md`.

| M | Nama | Output |
|---|---|---|
| M1 | Walking skeleton ✅ | webcam → overlay dummy → Unity Video Capture (OBS cadangan) |
| M2 | Seleksi model | `docs/MODEL_SELECTION.md` + model A-Z terpilih |
| M3 | Pengenalan A–Z | huruf terbaca di overlay |
| M4 | Logika huruf→kata→kalimat | kalimat terbentuk otomatis |
| M5 | TTS ke mikrofon virtual | kalimat terdengar di VB-Cable |
| M6 | GUI minimalis | tombol besar + status + slider font |
| M7 | Kata (opsional) | pengenalan kata BISINDO |

## M1 — Walking skeleton ✅ SELESAI

**Keputusan (update 2026-09-29):** output kamera utama = **Unity Video
Capture** (Unity Capture). OBS Virtual Camera hanya **cadangan** bila Unity
Capture tidak terpasang. Kalau dua-duanya absen, worker berhenti dengan pesan
perbaikan (`worker.py` -> `open_vcam()`). Tes hasil video lewat
**self-view Zoom/Meet**.

**Riwayat keputusan (supaya tabel di bawah tidak terbaca bertentangan):**
saat M1 dulu (2026-09-29) `unitycapture` gagal `"No camera registered"` karena
Unity Capture belum terpasang, jadi OBS dipakai sementara. Unity Capture sudah
dipasang lewat `Install.bat` (Administrator), jadi urutan dibalik:
`BACKEND_ORDER = ("unitycapture", "obs")`.

**Selesai (definisi of done):**
(a) `pyvirtualcam` terbuka tanpa exception, device "Unity Video Capture" terbentuk
(backend `unitycapture`, dengan fallback `obs`);
(b) frame + teks dummy terbaca ulang dari device virtual (`scripts/verify_m1.py`);
(c) VB-Cable terpasang → `sounddevice.query_devices()` mendeteksi "CABLE Output"
dan `sd.play()` ke "CABLE Input" berhasil.

**Bukti (2026-09-29):** `pyvirtualcam` memilih backend `unitycapture`, device
`Unity Video Capture`; fallback dicoba lewat nama backend palsu → error persis
"Kamera virtual tidak ditemukan. Jalankan Install.bat Unity Capture sebagai
Administrator, lalu restart aplikasi. Cadangan: pasang OBS.";
`scripts/verify_m1.py` exit 0, 1714 pixel teks terdeteksi di header stream
(ukuran lama, masih valid — berbasis frame, bukan nama device);
`scripts/verify_m36.py` exit 0 dengan `backend = unitycapture`
(`sent=29 hand=1 inferred=1 letter_terakhir=G`);
VB-Cable — 10 device "CABLE" terdaftar, nada 440 Hz 0.3 s terputar ke idx 21
(WASAPI, 48 kHz).
Satu instance only: satu nama device virtual berbasis pyvirtualcam.

**Tes jangka panjang (diulang setiap demo):** self-view Zoom/Meet memilih
"Unity Video Capture" + "CABLE Output" (atau "OBS Virtual Camera" bila cadangan).
**Praktik:** commit kecil tiap langkah — `feat: ...`, `fix: ...`.
**Belum ada MediaPipe/AI** di sini; M1 hanya membuktikan capture → kamera virtual
dan perangkat audio virtual hidup di mesin ini.

## M2 — Seleksi model [FR-06, FR-12]

**Sel:** Semua kandidat di `TECH_SPEC.md §6` diuji di webcam nyata; catat hasil
di `docs/MODEL_SELECTION.md` (tabel: kandidat, benar/salah per huruf, catatan
cahaya, keputusan akhir). Pilih satu model A–Z, unduh ke `models/`.
**Tes:** `python scripts/test_webcam.py` menampilkan label+confidence live;
dokumen catatan terisi.
**Catatan jujur wajib:** variasi regional, pencahayaan, jarak tangan.
**Cadangan:** bila kandidat utama gagal → Kaggle `agungmrf`; bila keraguan
tinggi → tetap lanjut dengan kandidat terbaik + tandai sebagai risiko demo.
**Tidak dilakukan:** mengambil keluhan "kurang akurat" sebagai alasan menambah
data sendiri — itu di luar MVP (FR-13).

## M3 — Pengenalan A–Z [FR-05, FR-06]

**Sel:** MediaPipe Hands + model A-Z + smoothing (4 dari 5 frame).
**Tes:** Isyaratkan A–Z satu-satu; minimal 17 dari 26 benar muncul di overlay.
**Commit:** `feat: add mediapipe hand detection`, `feat: add alphabet recognizer`,
`feat: add 5-frame label smoothing`.

## M4 — Logika huruf→kata→kalimat [FR-07]

**Sel:** `text_pipeline.py` — huruf stabil → tambah ke kata; tangan hilang
1,2 detik → **kata** diucapkan (TTS) + masuk buffer kalimat; tangan hilang
3 detik → kalimat selesai (buffer bersih, overlay kosong).
**Tes:** isyarat kata "HALO" → muncul `HALO`, tangan turun ±1,2 detik → terdengar
"HALO"; turun 3 detik → overlay kosong.

## M5 — TTS ke mikrofon virtual [FR-10]

**Sel:** `tts.py` — Piper voice `id-ID-news_tts-medium`, cari device VB-Cable by name.
**Tes:** Ulangi M4 → dengar ucapan di playback device lain yang memantau
"CABLE Output". Bila VB-Cable tidak ada → status Error jelas, tidak crash.
**Commit:** `feat: add piper tts to virtual output`.

## M6 — GUI minimalis [FR-01, FR-02, FR-03, FR-11]

**Sel:** `app.py` — tombol besar MULAI/HENTIKAN (48px), badge status berwarna
Berhenti/Menyiapkan/Berjalan/Error, kartu teks besar, segmented ukuran S/M/L,
chip kesehatan (kamera virtual / mikrofon / model), kartu error + "Coba lagi",
tombol pin "Selalu di atas". Semua pembaruan dari worker lewat **Signal Qt**,
bukan widget yang disentuh dari thread. Pipeline di thread terpisah; UI tetap
lancar.
**Tes:** Klik Start/Stop 5× tanpa hang; drag jendela saat jalan; overlay ikut
mengikuti nilai S/M/L; `python app.py --debug` membuka jendela pratinjau
terpisah (tanpa checkbox di GUI).

## M7 — Kata (opsional) [FR-12]

**Sel:** hanya jika kandidat kata lolos di M2.
**Tes:** 10 kata diperagakan; minimal 8 benar diucapkan.

## Checklist tes Zoom/Meet (self-view)

- [ ] Zoom/Meet: daftar kamera memuat "Unity Video Capture" → pilih → self-view menampilkan
      frame + overlay teks IsyaratKu Cam.
- [ ] Zoom/Meet: mikrofon = "CABLE Output" (VB-Cable) → suara TTS terdengar di self-view.
- [ ] Frame mengalir tanpa freeze selama ≥ 60 detik.
- [ ] Stop di aplikasi → stream berhenti bersih (tidak freeze hitam).
- [ ] Kalau Unity Capture tidak muncul: pakai "OBS Virtual Camera" (cadangan).

## Skrip demo 3 langkah

1. Buka aplikasi → klik **START** (status `Berjalan`).
2. Buka Zoom/Meet → pilih **Unity Video Capture** (atau **OBS Virtual Camera**
   bila cadangan) + **CABLE Output** (self-view).
3. Peragakan isyarat `HALO SAYA RINA` → peserta membaca overlay yang bertambah,
   mendengar setiap kata ±1,2 detik setelah tangan turun.

## Rencana cadangan

1. OBS Virtual Camera tidak muncul / tidak streaming → hidupkan "Start Virtual Camera"
   di OBS lalu ulangi. Unity Capture opsional, pasang hanya bila OBS VC bermasalah.
2. Model A–Z terpilih akurasi rendah → naikkan ambang smoothing (tampilkan contoh huruf
   lebih lama), demo dengan kata pendek.
3. Piper bermasalah → suara tetap didengar dengan volume perangkat naik;
   bila tetap gagal, tampilkan transkrip overlay + jelaskan pada demo.
4. Deadline mepet → potong M7 (opsional), jaga M1..M6.

## Fitur dipotong duluan (urut)

1. M7 kata BISINDO.
2. Slider ukuran font (tetap, nilai default tetap).
3. Recorder CLI rekam data sendiri.
