# PRD — IsyaratKu Cam

**Status:** Draft v1 · **Tanggal:** 2026-09-29 · **Pemilik:** Tim FIK FAIr
**Dokumen terkait:** `TECH_SPEC.md` (arsitektur & integrasi), `BUILD_ORDER.md` (urutan eksekusi)

## 1. Latar Belakang & Masalah

Penyandang tunawicara (bisu) yang tetap memiliki pendengaran normal tidak bisa
berbicara, sementara orang di sekitarnya umumnya tidak memahami BISINDO.
Akibatnya, di rapat diawali, kuliah daring, layanan publik, atau percakapan
sehari-hari, penyandang tunawicara harus bergantung pada juru tulis, tulisan, atau
tidak berkomunikasi sama sekali — hambatan yang menutup akses pendidikan,
pekerjaan, dan partisipasi sosialnya. IsyaratKu Cam menjawab hambatan itu
dengan aplikasi desktop Python yang membaca webcam, mengenali isyarat tangan
BISINDO, menuliskan terjemahannya di atas video, lalu mengucapkannya lewat
text-to-speech ke mikrofon virtual, sehingga lawan bicara mendengar suara
langsung di Zoom/Meet tanpa tahu apa pun tentang isyarat. Solusi ini selaras
dengan **SDG 3** (kesehatan & kesejahteraan melalui alat bantu komunikasi),
**SDG 10** (mengurangi kesenjangan untuk penyandang disabilitas), dan
**SDG 16** (institusi & ruang diskusi yang inklusif) pada subtema
**"Akses untuk Semua"**.

## 2. Persona

| # | Persona | Konteks | Kebutuhan utama | Hambatan |
|---|---|---|---|---|
| P1 | **Rina**, mahasiswa tunawicara, 20 tahun | Kuliah online & seminar fakultas | Ikut diskusi kelas tanpa juru tulis; dosen/teman mendengar jawabannya | Dosen/teman tidak paham isyarat; menulis chat terlalu lambat saat presentasi |
| P2 | **Pak Budi**, karyawan tunawicara, 34 tahun | Rapat kerja & layanan tatap muka | Menyampaikan pendapat/permintaan secara mandiri dan cepat | Lawan bicara buta isyarat; papan tulis lambat dan merepotkan |

## 3. User Stories (MoSCoW)

**Must**

- US-01 — Sebagai pengguna, saya ingin menekan **Start** dan sistem langsung membaca
  webcam saya, tanpa perlu memilih kamera, agar tidak ada langkah menyiapkan.
- US-02 — Sebagai pengguna, saya ingin melihat status jelas (Berhenti / Berjalan /
  Error beserta pesan singkat), agar saya tahu aplikasi bekerja atau tidak.
- US-03 — Sebagai pengguna, saya ingin huruf yang saya isyaratkan muncul sebagai teks
  di atas video, agar lawan bicara bisa membaca.
- US-04 — Sebagai pengguna, saya ingin kalimat yang terbentuk diucapkan otomatis
  setelah saya berhenti beberapa detik, agar saya "bisa bicara" tanpa menekan tombol.
- US-05 — Sebagai pengguna, saya ingin aplikasi terlihat sebagai **kamera** dan
  terdengar sebagai **mikrofon** di Zoom/Meet, agar saya cukup memilih device sekali.
- US-06 — Sebagai pengguna, saya ingin semua pengenalan berjalan **offline & lokal**,
  agar rapat saya tidak tersimpan ke cloud.

**Should**

- US-07 — Sebagai pengguna, saya ingin mengubah ukuran font overlay, agar teks
  terbaca di layar yang berbeda.
- US-08 — Sebagai pengguna, saya ingin pengenalan **kata** BISINDO (bukan hanya
  huruf), agar kalimat tersusun lebih cepat.

**Could**

- US-09 — Sebagai pengguna, saya ingin merekam data isyarat sendiri lewat skrip CLI,
  agar model bisa ditingkatkan di luar MVP.

**Won't (rilis hackathon ini)**

- US-10 — Preview video di dalam aplikasi, pilihan kamera/audio, hotkey, kalimat
  isyarat kontinu (dua tangan penuh), akun/cloud.

## 4. Requirement Fungsional

| ID | Requirement | Prioritas |
|---|---|---|
| FR-01 | Tombol besar **Start/Stop** mengontrol seluruh pipeline kamera+AI | Must |
| FR-02 | Indikator status: `Berhenti` / `Berjalan` / `Error: <pesan singkat>` | Must |
| FR-03 | Pengaturan ukuran font overlay (satu kontrol) | Should |
| FR-04 | Tangkap frame dari **kamera default** (tanpa dropdown) | Must |
| FR-05 | Deteksi tangan dengan MediaPipe Hands | Must |
| FR-06 | Klasifikasi huruf A–Z dengan stabilisasi beberapa frame (FR-07) | Must |
| FR-07 | Logika otomatis: huruf stabil → tambah ke kata; tangan hilang ±1,2 detik → **kata** diucapkan via TTS; tangan hilang ±3 detik → kalimat selesai (buffer bersih, overlay kosong) | Must |
| FR-08 | Gambar teks terjemahan sebagai overlay pada frame keluar | Must |
| FR-09 | Frame keluar didorong ke kamera virtual (`pyvirtualcam`) | Must |
| FR-10 | TTS `Piper` offline, voice `id-ID-news_tts-medium`, keluar ke VB-Cable yang terdeteksi otomatis berdasarkan nama perangkat; jika tidak ditemukan → status Error dengan pesan jelas | Must |
| FR-11 | Pipeline kamera+AI berjalan di **thread terpisah** dari GUI | Must |
| FR-12 | Pengenalan kata BISINDO (opsional, hanya jika lolos tes milestone seleksi model) | Could |
| FR-13 | Skrip CLI rekam dataset sendiri (opsional, di luar MVP) | Could |

## 5. Requirement Non-Fungsional

| Aspek | Target |
|---|---|
| Latensi | Ucapan terdengar ≤ ±2 detik setelah jeda deteksi [ASUMSI: belum diukur, batas masuk akal untuk demo] |
| Throughput | ≥ 15 FPS pada webcam default [ASUMSI] |
| Akurasi | Alfabet A–Z ≥ 85% kondisi cahaya ruangan normal & posisi tangan wajar [ASUMSI: target internal; wajib diuji & dicatat di milestone seleksi model] |
| Privasi | 100% offline: tidak ada unggah data, tidak ada telemetri |
| Platform | Windows 10/11 x64, Python 3.10+ |
| Portabilitas | Kamera & mikrofon dipilih di Zoom/Meet; aplikasi tidak punya dropdown perangkat |
| Ukuran kode | Total kode inti < ~600 baris, maksimal ~8 file (lihat `AGENTS.md`) |

## 6. Acceptance Criteria Terukur

- **AC-01** Tombol Start → status menjadi `Berjalan` dalam ≤ 2 detik, tanpa dialog tambahan.
- **AC-02** 20 isyarat huruf berbeda diperagakan satu per satu; minimal 17 huruf muncul benar
  pada overlay (sesuai hasil tes milestone seleksi model, dicatat).
- **AC-03** Setiap kata terdengar di output VB-Cable ±1,2 detik setelah tangan
  hilang; setelah 3 detik tanpa tangan, kalimat terakhir terlihat di overlay lalu
  buffer kosong dan overlay bersih.
- **AC-04** Kamera virtual menampilkan stream dengan overlay teks terbaca.
  Nama device: **"Unity Video Capture"** (nama asli Unity Capture — bukan
  "Unity Capture Camera"), atau "OBS Virtual Camera" bila pakai cadangan.
- **AC-05** Zoom/Meet: dipilih "Unity Video Capture" (atau "OBS Virtual Camera"
  sebagai fallback) → peserta lain melihat overlay; dipilih "CABLE Output"
  (VB-Cable) → peserta lain mendengar ucapan.
- **AC-06** VB-Cable tidak terpasang → status `Error: VB-Cable tidak ditemukan` dan
  aplikasi tidak crash.
- **AC-07** Model file hilang/di-skip → status `Error: model tidak ditemukan` dengan pesan,
  tidak crash.
- **AC-08** UI tetap responsif (tombol bisa diklik, jendela bisa di-drag) selama pipeline berjalan.

## 7. Wireframe ASCII

```
┌──────────────────────────────────────────┐
│  IsyaratKu Cam                       Pin │
├──────────────────────────────────────────┤
│  ● Berjalan                              │  ← badge berwarna + spinner
│                                          │
│  ┌────────────────────────────────────┐  │
│  │        Teks sekarang (besar)       │  │  ← 28-32pt, tengah, wrap
│  │        Kandidat: A                 │  │  ← huruf kandidat terkini
│  └────────────────────────────────────┘  │
│                                          │
│  Diucapkan terakhir                      │  ← maks 3 kalimat
│   - APA KABAR                            │
│                                          │
│  Ukuran teks:   ( S )( M )( L )          │  ← segmented, ganti slider
│  [Kamera virtual] [Mikrofon] [Model]     │  ← chip + tooltip perbaikan
│                                          │
│  ┌────────────────────────────────────┐  │
│  │              MULAI                  │  │  ← permanen, merah saat jalan
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
```

Tidak ada preview video, tidak ada dropdown, tidak ada halaman lain.
Output (frame + overlay) hanya terlihat di kamera virtual (Zoom/Meet).

## 8. Out of Scope

Preview video di aplikasi · GUI rekam/latih data · pilihan kamera/audio ·
hotkey · kalimat isyarat kontinu & dua tangan penuh · akun/cloud/tanpa
internet-sync · test suite otomatis (cukup smoke test + checklist demo).