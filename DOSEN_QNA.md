# DOSEN_QNA — Prediksi Pertanyaan & Jawaban Teknis

Dokumen ini berisi pertanyaan yang umum muncul pada sidang/UAS, beserta jawaban teknis yang ringkas, akademis, dan dapat dipertanggungjawabkan.

## 1) Kenapa MFCC?

**Jawaban singkat:** MFCC adalah fitur standar untuk pengenalan ucapan karena memodelkan karakteristik spektral yang relevan dengan persepsi manusia dan relatif robust dibanding sinyal mentah.

**Jawaban teknis:**

- MFCC merangkum informasi frekuensi dari sinyal suara setelah STFT, lalu dipetakan ke skala Mel yang mendekati persepsi pendengaran.
- Setelah DCT, koefisien MFCC menjadi representasi kompak yang cocok untuk model klasifikasi.
- Untuk keyword recognition (1 kata), MFCC efektif karena perbedaan pola formant/energi antar kata cukup jelas.

## 2) Kenapa MLPClassifier?

**Jawaban singkat:** MLP adalah baseline neural network yang sederhana namun cukup kuat untuk klasifikasi fitur MFCC, serta mudah dilatih dan dijelaskan.

**Jawaban teknis:**

- Input proyek ini berupa vektor fitur (ringkasan MFCC), sehingga MLP cocok sebagai classifier non-linear.
- Training cepat, inference ringan untuk realtime.
- Model menghasilkan probabilitas kelas (`predict_proba`) yang dapat dipakai sebagai confidence score dan smoothing.

## 3) Kenapa keyword recognition (ASR terbatas), bukan ASR kalimat bebas?

**Jawaban singkat:** Scope proyek akademik dibatasi pada 10 keyword agar pipeline ASR buatan sendiri dapat diuji end-to-end dengan dataset sendiri dan bisa stabil untuk realtime.

**Jawaban teknis:**

- ASR kalimat bebas memerlukan model akustik + bahasa (language model) atau end-to-end model besar dan dataset jauh lebih besar.
- Dengan keyword recognition, evaluasi lebih terukur (kelas terbatas) dan dapat dibangun dari nol.

## 4) Kenapa dataset dibuat sendiri?

**Jawaban singkat:** Untuk memenuhi tujuan pembelajaran dan kontrol kualitas: dataset sesuai domain (aksen, mic, kondisi ruangan) serta dapat didokumentasikan prosesnya.

**Jawaban teknis:**

- Dataset publik sering berbeda domain (bahasa, aksen, noise) sehingga terjadi domain shift.
- Dengan dataset sendiri, pipeline rekam → preprocessing → fitur → model dapat dipertanggungjawabkan.

## 5) Tantangan Bahasa Indonesia pada proyek ini?

**Jawaban singkat:** Variasi pelafalan, aksen, kecepatan bicara, noise lingkungan, dan kualitas mikrofon.

**Jawaban teknis:**

- Kata-kata tertentu memiliki fonem yang mirip sehingga perlu data cukup untuk memisahkan kelas.
- Bahasa Indonesia cenderung memiliki vokal yang jelas, tetapi perbedaan intonasi/tempo antar penutur dapat menggeser pola spektral.

## 6) Kenapa tidak pakai Whisper?

**Jawaban singkat:** Fokus proyek adalah membangun ASR dari dasar dengan MFCC + model klasifikasi, bukan memanfaatkan model besar siap pakai.

**Jawaban teknis:**

- Whisper membutuhkan resource lebih besar dan mengubah fokus proyek menjadi integrasi model pretrained.
- Secara akademik, pipeline buatan sendiri lebih cocok untuk menjelaskan fitur, training, evaluasi, dan interpretasi.

## 7) Bagaimana evaluasi model dilakukan?

**Jawaban singkat:** Menggunakan split terstratifikasi train/test dan metrik klasifikasi multi-kelas.

**Jawaban teknis:**

- Dataset dibagi dengan `StratifiedShuffleSplit` agar proporsi kelas seimbang di train/test.
- Metrik yang ditampilkan:
  - Accuracy
  - Precision, Recall, F1-score (macro dan weighted)
  - Classification report per kelas
  - Confusion matrix

Artefak evaluasi tersimpan di:

- `outputs/confusion_matrix.png`
- `outputs/classification_report.txt`
- `outputs/train_report.json`

## 8) Apa fungsi smoothing pada realtime?

**Jawaban singkat:** Untuk menstabilkan prediksi yang fluktuatif akibat noise dan perbedaan window audio.

**Jawaban teknis:**

- Realtime ASR mengambil window audio per interval tertentu.
- Prediksi per-window bisa berubah cepat karena noise atau potongan fonem.
- Smoothing menghitung rata-rata probabilitas beberapa window terakhir sehingga label lebih stabil.

## 9) Apa fungsi confidence score?

**Jawaban singkat:** Sebagai ukuran keyakinan model terhadap kelas yang dipilih, dan sebagai dasar keputusan otomatis (Auto-TTS).

**Jawaban teknis:**

- Confidence diambil dari probabilitas kelas tertinggi (`max(predict_proba)`).
- Dipakai untuk:
  - Menampilkan kualitas prediksi ke pengguna
  - Auto-TTS hanya aktif bila confidence melewati threshold
  - Menghindari output salah saat audio noisy/hening

## 10) Kenapa top-3 prediction ditampilkan?

**Jawaban singkat:** Untuk transparansi: pengguna dapat melihat alternatif kelas teratas ketika confidence tidak sangat tinggi.

**Jawaban teknis:**

- Jika top-3 berdekatan, itu indikasi kata sulit dibedakan pada input tertentu.
- Informasi ini membantu analisis kesalahan dan perbaikan dataset.

## 11) Apa yang terjadi jika audio hening/terlalu pelan?

**Jawaban singkat:** Sistem menolak prediksi dan menampilkan error agar tidak menghasilkan output yang menyesatkan.

**Jawaban teknis:**

- Ada pengecekan energi (RMS) pada audio setelah preprocessing.
- Jika terlalu kecil, proses prediksi dibatalkan dan user diminta mengulang lebih dekat ke mic.

## 12) Apa rencana pengembangan berikutnya?

**Jawaban singkat:** Robustness & generalisasi.

**Jawaban teknis (contoh):**

- Augmentasi data (noise, time-shift, gain)
- K-fold validation
- Model pembanding (SVM/CNN kecil)
- Fitur “load WAV for inference” agar demo bisa tanpa mic
