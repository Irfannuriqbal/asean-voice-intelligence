# PRESENTATION_SCRIPT — Naskah Presentasi Kelompok (ASR + TTS)

Dokumen ini adalah script formal yang bisa langsung dipakai. Silakan sesuaikan nama anggota dan pembagian peran.

Durasi target: 8–12 menit (disarankan).

---

## 0) Pembagian Peran (Opsional)

- **Presenter 1 (Pembukaan & problem statement)**
- **Presenter 2 (Metode: MFCC + MLP + evaluasi)**
- **Presenter 3 (Demo GUI + integrasi ASR→TTS + penutup)**

Jika anggota 2 orang, gabungkan Presenter 2 dan 3.

---

## 1) Pembukaan (Presenter 1 | 45–60 detik)

“Assalamu’alaikum warahmatullahi wabarakatuh / Selamat pagi/siang.

Kami dari kelompok [NAMA KELOMPOK] akan mempresentasikan proyek UAS mata kuliah Pengenalan Ucapan dan Teks ke Ucapan.

Judul proyek kami adalah **ASR + TTS Bahasa Indonesia** dengan tema **pengenalan nama negara ASEAN**.

Tujuan utama proyek ini adalah membangun aplikasi yang mampu mengenali **10 keyword** dari ucapan pengguna, menampilkan confidence, dan mengintegrasikannya dengan **Text-to-Speech** di dalam GUI.”

---

## 2) Latar Belakang & Ruang Lingkup (Presenter 1 | 60–90 detik)

“Pada implementasi ASR, ada dua pendekatan besar: menggunakan model pretrained besar, atau membangun pipeline sendiri.

Dalam proyek ini, kami memilih membangun pipeline ASR sendiri untuk pembelajaran, yaitu:

- Mengumpulkan dataset audio sendiri
- Melakukan preprocessing
- Ekstraksi fitur MFCC
- Training model klasifikasi menggunakan MLPClassifier
- Menyediakan inferensi realtime dan single-shot di aplikasi GUI

Ruang lingkup kami adalah keyword recognition untuk 10 kata agar sistem dapat stabil dan evaluasinya terukur.”

---

## 3) Penjelasan Sistem (Presenter 2 | 2–3 menit)

### 3.1 Pipeline ASR

“Alur ASR kami adalah:

1. Audio dari mikrofon
2. Preprocessing: resample, trim silence, normalisasi, dan penyesuaian durasi
3. Ekstraksi fitur MFCC
4. Vektor fitur MFCC (mean+std)
5. Klasifikasi menggunakan MLPClassifier
6. Output: label prediksi + confidence score

Pada realtime mode, kami mengambil window audio secara periodik, lalu melakukan smoothing pada probabilitas beberapa window terakhir supaya prediksi lebih stabil.”

### 3.2 Kenapa MFCC dan MLP

“MFCC dipilih karena menjadi fitur standar yang ringkas dan relevan dengan karakteristik spektral ucapan.

MLPClassifier dipilih sebagai baseline neural network sederhana yang cukup kuat untuk klasifikasi non-linear pada fitur MFCC, dan dapat menghasilkan probabilitas untuk confidence score.”

---

## 4) Evaluasi Model (Presenter 2 | 1–2 menit)

“Kami mengevaluasi model dengan pembagian train/test terstratifikasi.

Metrik yang digunakan:

- Accuracy
- Precision, Recall, dan F1-score (macro dan weighted)
- Classification report per kelas
- Confusion matrix

Confusion matrix membantu melihat pasangan keyword yang sering tertukar, misalnya karena kemiripan pelafalan atau noise.

Artefak evaluasi tersimpan pada folder `outputs/`.”

---

## 5) Demo Aplikasi (Presenter 3 | 3–5 menit)

“Berikut kami demo aplikasi GUI.

Pertama, ini adalah panel ASR (kiri) dan panel TTS (kanan).”

### 5.1 Demo Realtime ASR

“Untuk realtime, kami tekan **Start Realtime** lalu mengucapkan keyword.

Aplikasi menampilkan:

- Prediksi label
- Confidence score
- Top-3 prediction
- Visualisasi waveform dan MFCC

Untuk membuat hasil lebih stabil, sistem menerapkan smoothing pada probabilitas realtime.”

### 5.2 Demo Single-shot (Record Once)

“Selain realtime, kami menyediakan mode **Record Once** untuk kondisi yang lebih stabil, terutama jika streaming mic bermasalah.

Mode ini merekam sekali sesuai durasi target, lalu memprediksi.”

### 5.3 Demo Auto-TTS & Integrasi

“Kami juga menambahkan integrasi ASR ke TTS.

- Jika **Auto-TTS** aktif dan confidence melewati threshold, sistem akan membacakan hasil prediksi.
- Atau kami bisa menekan tombol **Use ASR** untuk memindahkan teks prediksi ke panel TTS, lalu menekan **Play**.

Kami juga menyediakan fitur **Save Audio** untuk menyimpan output TTS.”

---

## 6) Penutup (Presenter 3 | 30–60 detik)

“Kesimpulannya, proyek ini berhasil membangun aplikasi ASR keyword recognition berbasis MFCC + MLP dengan inferensi realtime dan single-shot, serta terintegrasi dengan TTS Bahasa Indonesia dalam GUI.

Ke depan, pengembangan yang dapat dilakukan adalah augmentasi data untuk meningkatkan robust terhadap noise dan perbedaan mikrofon, serta eksperimen model pembanding seperti SVM atau CNN kecil.

Terima kasih. Kami siap menerima pertanyaan.”

---

## 7) Jika Ada Pertanyaan

Gunakan dokumen `DOSEN_QNA.md` sebagai panduan jawaban teknis.
