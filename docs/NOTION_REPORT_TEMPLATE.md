# Template Laporan (Notion) — PTU 2026

## 1. Ringkasan

- Judul project:
- Tema:
- Fitur utama:

## 2. Ruang Lingkup

- ASR terbatas (10 kata), bukan transkripsi bebas
- Bahasa: Indonesia

## 3. Dataset

- Daftar label:
- Sumber data: rekaman sendiri
- Format dataset:
- Total sample & pembagian train/test:

## 4. Metodologi

### 4.1 Preprocessing

- Resample ke 16 kHz
- Trim silence
- Normalisasi amplitude
- Pad/Truncate durasi

### 4.2 Ekstraksi Fitur MFCC

- Parameter: n_mfcc, n_fft, hop, win
- Output: MFCC (n_mfcc x n_frames)
- Feature vector: mean + std

### 4.3 Model Klasifikasi (MLP)

- Arsitektur:
- Optimizer/solver:
- Early stopping:

## 5. Implementasi

- Struktur folder
- Modul penting:
  - `utils/asr_infer.py`
  - `utils/features.py`
  - `gui/app_gui.py`

## 6. Evaluasi

- Akurasi:
- Confusion matrix:
- Analisis kesalahan:

## 7. Fitur Tambahan

- Confidence score
- Visualisasi MFCC
- Integrasi ASR→TTS

## 8. Cara Menjalankan

- Setup venv
- Recording dataset
- Training
- Running GUI

## 9. Kesimpulan

- Hasil dan pembelajaran

## 10. Lampiran

- Screenshot GUI
- Confusion matrix
- Link repo
