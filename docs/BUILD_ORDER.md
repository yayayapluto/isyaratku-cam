# BUILD_ORDER — IsyaratKu Cam

Urutan milestone M1..M7. Bukan jadwal harian — selesai milestone = satu bukti
bisa dites. ID `FR-xx`/`AC-xx` merujuk `docs/PRD.md`.

| M | Nama | Output |
|---|---|---|
| M1 | Walking skeleton | webcam → overlay dummy → terlihat di OBS |
| M2 | Seleksi model | `docs/MODEL_SELECTION.md` + model A-Z terpilih |
| M3 | Pengenalan A–Z | huruf terbaca di overlay |
| M4 | Logika huruf→kata→kalimat | kalimat terbentuk otomatis |
| M5 | TTS ke mikrofon virtual | kalimat terdengar di VB-Cable |
| M6 | GUI minimalis | tombol besar + status + slider font |
| M7 | Kata (opsional) | pengenalan kata BISINDO |

## M1 — Walking skeleton

**Selesai (definisi of done):**
(a) python → `pyvirtualcam` terbuka tanpa exception, device "Unity Capture Camera"
terbentuk; (b) OBS: source "Unity Capture Camera" menampilkan frame + teks dummy;
(c) VB-Cable terpasang → `scripts/check_env.py` mendeteksi "CABLE Output".

**Tes M1 (dijalankan pada mesin demo):**
1. `python main.py` → Start → OBS tampil overlay → **AC-04**.
2. Zoom/Meet: daftar kamera memuat "Unity Capture Camera" → pilih → video keluar.
3. `python scripts/check_env.py` → device "CABLE Output" terdeteksi → **AC-06**.
4. Kalau langkah 2 gagal (device tidak muncul setelah `Install.bat` Admin) →
   fallback OBS Virtual Camera, jalankan lagi langkah 2.

**Praktik:** commit kecil tiap langkah — `feat: ...`, `fix: ...`.

**Belum ada MediaPipe/AI** di sini; M1 hanya membuktikan `pyvirtualcam` +
DirectShow filter Unity Capture + VB-Cable hidup di mesin ini.

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

## Checklist tes OBS & Zoom/Meet

- [ ] OBS: source "Unity Capture Camera" tampil, frame mengalir, overlay teks muncul.
- [ ] OBS: output rekaman 30 detik tanpa putus.
- [ ] Zoom: daftar kamera memuat "Unity Capture Camera" (fallback: OBS Virtual Camera)
      → peserta lain melihat overlay.
- [ ] Meet: sama seperti Zoom.
- [ ] Zoom/Meet: mikrofon = "CABLE Output" (VB-Cable) → peserta lain mendengar TTS.
- [ ] Stop di aplikasi → stream berhenti bersih (tidak freeze hitam).

## Skrip demo 3 langkah

1. Buka aplikasi → klik **START** (status `Berjalan`).
2. Buka Zoom/Meet → pilih **Unity Capture Camera** (fallback: OBS Virtual Camera)
   + **CABLE Output**.
3. Peragakan isyarat `HALO SAYA RINA` → peserta membaca overlay,
   mendengar kalimat setelah jeda 3 detik.

## Rencana cadangan

1. Unity Capture tidak terdeteksi di Zoom/Meet setelah `Install.bat` (Admin) →
   pakai OBS Virtual Camera langsung ke Zoom/Meet. Bila keduanya gagal, hanya OBS
   yang dipakai untuk demo.
2. Model A–Z terpilih akurasi rendah → tampilkan plus contoh huruf lebih lama
   (naikkan ambang smoothing), demo dengan kata pendek.
3. Piper bermasalah → suara tetap didengar dengan volume perangkat naik;
   bila tetap gagal, tampilkan transkrip overlay + jelaskan pada demo.
4. Deadline mepet → potong M7 (opsional), jaga M1..M6.

## Fitur dipotong duluan (urut)

1. M7 kata BISINDO.
2. Slider ukuran font (tetap, nilai default tetap).
3. Recorder CLI rekam data sendiri.
