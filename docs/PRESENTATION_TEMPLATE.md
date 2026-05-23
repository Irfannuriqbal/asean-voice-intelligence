# Template Presentasi — Projek Akhir PTU 2026

> Tema: Pengenalan Nama Negara ASEAN (Bahasa Indonesia)

## Slide 1 — Judul

- Nama project
- Nama/NIM/Kelas
- Mata kuliah: Pengenalan Ucapan & Teks ke Ucapan (PTU)

## Slide 2 — Latar Belakang

- Kenapa ASR terbatas (keyword spotting) cocok untuk demo
- Target: 10 kata (nama negara ASEAN)

## Slide 3 — Tujuan

- ASR: suara → teks (label negara)
- TTS: teks → suara (Bahasa Indonesia)
- Real-time prediction + confidence

## Slide 4 — Dataset

- Daftar label
- Format folder `dataset/<label>/*.wav`
- Jumlah data per label
- Durasi & sample rate

## Slide 5 — Pipeline ASR (Dibuat Sendiri)

- Rekam mic → preprocessing
- Ekstraksi MFCC
- Fitur (mean+std) → MLPClassifier

## Slide 6 — Metode MFCC (Intuisi)

- Dari sinyal waktu ke spektrum
- Skala Mel (mendekati persepsi manusia)
- DCT → koefisien MFCC

## Slide 7 — Model Klasifikasi (MLP)

- Input: vektor fitur MFCC
- Hidden layers
- Output: probabilitas 10 kelas

## Slide 8 — Evaluasi

- Akurasi
- Confusion matrix
- Contoh error: label yang sering tertukar

## Slide 9 — GUI Demo

- Panel kiri: ASR (status, prediksi, confidence, MFCC)
- Panel kanan: TTS (input, speed, play, save)

## Slide 10 — Integrasi ASR → TTS

- Prediksi ASR dikirim ke textbox TTS
- Play untuk membacakan hasil

## Slide 11 — Kesimpulan & Saran

- Hasil utama
- Keterbatasan: dataset kecil/noise/mic
- Saran: tambah data, augmentasi, tuning model

## Slide 12 — Q&A
