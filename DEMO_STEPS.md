# DEMO_STEPS — ASR + TTS Bahasa Indonesia (PTU 2026)

Dokumen ini berisi urutan demo yang aman, ringkas, dan siap digunakan saat presentasi UAS/sidang.

## A. Pre-Demo Checklist (1–2 menit)

1. Pastikan environment siap

- Aktifkan venv:
  - PowerShell: `.\.venv\Scripts\Activate.ps1`
  - CMD: `.\.venv\Scripts\activate.bat`
- Jalankan GUI: `python app.py`

2. Pastikan artefak model tersedia

- `models/asr_mlp.joblib`
- `models/asr_meta.json`

3. Pastikan perangkat audio

- Windows Sound Settings → Input device: pilih mikrofon yang benar
- Pastikan tidak sedang dipakai aplikasi lain (Zoom/Meet/Discord)

4. Pastikan artefak evaluasi siap (untuk ditunjukkan bila diperlukan)

- `outputs/confusion_matrix.png`
- `outputs/classification_report.txt`
- `outputs/train_report.json`

Jika folder `outputs/` masih kosong, jalankan training sekali:

- `python train/train_mlp.py`

## B. Urutan Demo Utama (6–10 menit)

### 1) Opening singkat (30–45 detik)

- Jelaskan tujuan: mengenali 10 keyword negara ASEAN (Bahasa Indonesia)
- Tekankan: ASR dibuat sendiri (MFCC + MLP), bukan API / Whisper

### 2) Demo GUI: overview (30 detik)

- Tunjukkan 2 panel: ASR (kiri) dan TTS (kanan)
- Tunjukkan indikator: prediksi, confidence, top-3, dan plot waveform+MFCC

### 3) Skenario Demo Realtime ASR (2–3 menit)

Tujuan: menunjukkan inferensi streaming + smoothing + confidence.

Langkah:

1. Klik **Start Realtime**
2. Ucapkan 3 keyword berturut-turut (misalnya):
   - “Indonesia” → tunggu prediksi stabil
   - “Malaysia” → tunggu prediksi stabil
   - “Singapura” → tunggu prediksi stabil
3. Jelaskan 3 hal saat layar menampilkan hasil:
   - Prediksi label + confidence score
   - Top-3 prediction (untuk transparansi ketidakpastian)
   - Smoothing: prediksi lebih stabil dibanding raw per-window

Catatan demo:

- Bila noise tinggi, minta audiens diam 5–10 detik
- Ucapkan keyword jelas, durasi 1 kata sesuai dataset

### 4) Skenario Demo Single-shot (Record Once) (2 menit)

Tujuan: menunjukkan mode “rekam sekali” yang stabil.

Langkah:

1. Klik **Record Once**
2. Ucapkan 1 keyword (misal: “Thailand”) selama durasi rekam
3. Tunjukkan hasil prediksi + plot waveform+MFCC
4. Jelaskan perbedaan dengan realtime:
   - Single-shot lebih stabil untuk demo bila streaming mic bermasalah

### 5) Skenario Demo Auto-TTS (1–2 menit)

Tujuan: menunjukkan integrasi ASR → TTS otomatis berbasis confidence.

Langkah:

1. Pastikan toggle **Auto TTS** aktif
2. Pastikan threshold default 85% (atau atur di slider)
3. Ucapkan keyword yang mudah (misal: “Indonesia”) sampai confidence melewati threshold
4. Jelaskan:
   - Auto-TTS hanya berjalan jika confidence cukup tinggi
   - Ada cooldown agar tidak "spam" suara saat prediksi berulang

### 6) Demo TTS manual + export audio (1 menit)

Tujuan: menunjukkan TTS Bahasa Indonesia + ekspor.

Langkah:

1. Di panel TTS, ketik kalimat pendek (contoh): “Selamat siang, kami siap presentasi.”
2. Klik **Play**
3. Klik **Save Audio** lalu simpan sebagai WAV/MP3 (sesuai opsi)

Opsional:

- Klik **Use ASR** untuk menyalin teks prediksi ASR ke textbox TTS

## C. Backup Plan Jika Mic Error (Wajib Disiapkan)

Berikut skenario cadangan yang tetap terlihat profesional.

### Level 1 — Streaming gagal start

Gejala:

- Status menampilkan error streaming / mic
- Tombol realtime tidak bisa digunakan

Solusi demo:

- Jelaskan bahwa aplikasi memiliki **Safe Mode Fallback**
- Langsung demo **Record Once** (lebih stabil)

### Level 2 — Mic tidak terbaca / device conflict

Langkah cepat:

1. Tutup aplikasi yang mengakses mic (Zoom/Meet/Discord)
2. Windows Sound Settings → set input device
3. Restart GUI

Jika tetap gagal:

- Lanjutkan demo bagian TTS dan evaluasi model (confusion matrix + classification report)

### Level 3 — Ruangan terlalu bising / confidence rendah

Strategi:

- Naikkan threshold Auto-TTS sementara (atau matikan Auto-TTS)
- Gunakan single-shot Record Once
- Dekatkan mic (jarak 10–20 cm) dan ucapkan keyword lebih jelas

## D. Script Cadangan (Tanpa Demo Live ASR)

Jika demo ASR tidak memungkinkan, tetap dapat menunjukkan aspek akademis:

1. Buka `outputs/confusion_matrix.png` dan jelaskan interpretasinya
2. Buka `outputs/classification_report.txt` dan jelaskan precision/recall/F1
3. Jelaskan arsitektur sistem menggunakan flowchart Mermaid di README
4. Demo TTS + export audio sebagai bukti integrasi sistem
