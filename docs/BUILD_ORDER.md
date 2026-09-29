# BUILD_ORDER — IsyaratKu Cam

Urutan milestone M1..M7. Bukan jadwal harian — selesai milestone = satu bukti
bisa dites. ID `FR-xx`/`AC-xx` merujuk `docs/PRD.md`.

| M | Nama | Output |
|---|---|---|
| M1 | Walking skeleton ✅ | webcam → overlay dummy → OBS Virtual Camera |
| M2 | Seleksi model | `docs/MODEL_SELECTION.md` + model A-Z terpilih |
| M3 | Pengenalan A–Z | huruf terbaca di overlay |
| M4 | Logika huruf→kata→kalimat | kalimat terbentuk otomatis |
| M5 | TTS ke mikrofon virtual | kalimat terdengar di VB-Cable |
| M6 | GUI minimalis | tombol besar + status + slider font |
| M7 | Kata (opsional) | pengenalan kata BISINDO |

## M1 — Walking skeleton ✅ SELESAI

**Keputusan (update):** output kamera utama = **OBS Virtual Camera** (terbukti
di M1). Unity Capture **ditunda, opsional saja**. Tes hasil video lewat
**self-view Zoom/Meet**, bukan OBS.

**Selesai (definisi of done):**
(a) `pyvirtualcam` terbuka tanpa exception, device "OBS Virtual Camera" terbentuk
(backend `obs`; backend `unitycapture` dicoba lebih dulu dan aman gagal);
(b) frame + teks dummy terbaca ulang dari device virtual (`scripts/verify_m1.py`);
(c) VB-Cable terpasang → `sounddevice.query_devices()` mendeteksi "CABLE Output"
dan `sd.play()` ke "CABLE Input" berhasil.

**Bukti (2026-09-29):** `pyvirtualcam` memilih backend `obs`
(`unitycapture` gagal: "No camera registered"); `scripts/verify_m1.py` exit 0,
1714 pixel teks terdeteksi di header stream; VB-Cable — 10 device "CABLE"
terdaftar, nada 440 Hz 0.3 s terputar ke idx 21 (WASAPI, 48 kHz).
Satu instance only: OBS Virtual Camera tidak bisa ditangkap ulang OBS.

**Tes jangka panjang (diulang setiap demo):** self-view Zoom/Meet memilih
"OBS Virtual Camera" + "CABLE Output".
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

**Sel:** `text_pipeline.py` — huruf stabil → tambah ke kata; jeda tanpa tangan
3 detik → kalimat siap diucapkan (sementara `print`), overlay dikosongkan.
**Tes:** isyarat kata "HALO" → muncul `HALO`, berhenti 3 detik → overlay kosong.

## M5 — TTS ke mikrofon virtual [FR-10]

**Sel:** `tts.py` — Piper voice `id-ID-news_tts-medium`, cari device VB-Cable by name.
**Tes:** Ulangi M4 → dengar ucapan di playback device lain yang memantau
"CABLE Output". Bila VB-Cable tidak ada → status Error jelas, tidak crash.
**Commit:** `feat: add piper tts to virtual output`.

## M6 — GUI minimalis [FR-01, FR-02, FR-03, FR-11]

**Sel:** `main.py` — tombol besar Start/Stop, status Berhenti/Berjalan/Error,
slider ukuran font overlay. Pipeline di thread terpisah; UI tetap lancar.
**Tes:** Klik Start/Stop 5× tanpa hang; drag jendela saat jalan; overlay ikut
mengikuti nilai slider.

## M7 — Kata (opsional) [FR-12]

**Sel:** hanya jika kandidat kata lolos di M2.
**Tes:** 10 kata diperagakan; minimal 8 benar diucapkan.

## Checklist tes Zoom/Meet (self-view)

- [ ] Zoom/Meet: daftar kamera memuat "OBS Virtual Camera" → pilih → self-view menampilkan
      frame + overlay teks IsyaratKu Cam.
- [ ] Zoom/Meet: mikrofon = "CABLE Output" (VB-Cable) → suara TTS terdengar di self-view.
- [ ] Frame mengalir tanpa freeze selama ≥ 60 detik.
- [ ] Stop di aplikasi → stream berhenti bersih (tidak freeze hitam).
- [ ] Opsional (bila Unity Capture dipasang): OBS source "Unity Capture Camera" tampil.

## Skrip demo 3 langkah

1. Buka aplikasi → klik **START** (status `Berjalan`).
2. Buka Zoom/Meet → pilih **OBS Virtual Camera** + **CABLE Output** (self-view).
3. Peragakan isyarat `HALO SAYA RINA` → peserta membaca overlay,
   mendengar kalimat setelah jeda 3 detik.

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
