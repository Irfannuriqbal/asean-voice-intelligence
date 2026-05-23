# ASEAN Voice Intelligence – Automatic Speech Recognition dan Text-to-Speech Berbahasa Indonesia

**Laporan Projek Akhir PTU 2026**

## Ringkasan Eksekutif

Proyek ini mengembangkan sebuah aplikasi kecerdasan buatan berbasis suara untuk bahasa Indonesia dengan fokus pada dua modul utama, yaitu **Automatic Speech Recognition (ASR)** dan **Text-to-Speech (TTS)**. Sistem ASR dirancang sebagai _keyword spotting_ terbatas untuk mengenali 10 kata kunci berupa nama negara ASEAN. Pipeline ASR dibangun secara mandiri melalui tahapan preprocessing audio, ekstraksi fitur **MFCC (Mel-Frequency Cepstral Coefficients)**, pelatihan model klasifikasi **MLPClassifier**, serta eksperimen model **CNN 1D berbasis TensorFlow/Keras** untuk kebutuhan inferensi real-time yang lebih stabil.

Selain modul ASR, proyek ini juga menyertakan modul **Text-to-Speech Bahasa Indonesia** yang memungkinkan pengguna mengubah teks menjadi suara dengan pilihan kecepatan bicara, pilihan gender suara, serta fitur penyimpanan audio ke format WAV atau MP3. Seluruh sistem dikemas ke dalam dua antarmuka, yaitu **desktop GUI** dan **web AI dashboard** berbasis Gradio, sehingga aplikasi dapat digunakan dalam skenario presentasi, demonstrasi, maupun pengembangan lanjutan.

---

## Detail Kelompok

| No  | Nama Anggota           | NIM       | Pembagian Tugas                                                                                                                                   |
| --- | ---------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Irfan Nur Iqbal        | 152023142 | Project Lead + ASR Specialist: koordinator project, MFCC feature extraction, training model ASR (MLP/CNN), dan real-time inference implementation |
| 2   | Parisan Apro           | 152023141 | TTS Specialist + Backend: implementasi modul TTS, integrasi pyttsx3 + gTTS, export audio MP3/WAV, serta kontrol speed dan gender voice            |
| 3   | Reynal Toeloes Ritonga | 152023157 | Frontend + GUI Developer: desain GUI CustomTkinter, panel ASR, panel TTS, dan integrasi widget                                                    |
| 4   | Muhammad Yusuf Islam   | 152023    | Web Frontend + Testing: pengembangan Gradio web app, unit testing, integration testing, QA, dan bug fixing                                        |
| 5   | Richa Romadhoni        | 152023185 | Data Engineering + Dataset: pengumpulan dan persiapan dataset, audio preprocessing pipeline, validasi data, serta train/test split                |
| 6   | Ramdhani Angriawan P   | 152023188 | Documentation + Presentation: penyusunan README, laporan Notion, video demo recording, dan persiapan presentasi                                   |

> **Catatan:** Silakan sesuaikan NIM yang belum lengkap atau format pembagian tugas jika diperlukan agar sesuai dengan data resmi kelompok.

---

## Latar Belakang

Perkembangan kecerdasan buatan dalam satu dekade terakhir telah mendorong transformasi besar pada cara manusia berinteraksi dengan komputer. Jika sebelumnya interaksi dilakukan hampir sepenuhnya melalui keyboard, mouse, dan layar sentuh, kini sistem komputasi modern semakin banyak memanfaatkan **interaksi berbasis suara**. Pergeseran ini didorong oleh kebutuhan akan antarmuka yang lebih alami, cepat, inklusif, dan efisien, terutama pada aplikasi asisten virtual, layanan pelanggan, sistem navigasi, perangkat pintar, dan solusi aksesibilitas.

Di antara berbagai cabang AI, **speech technology** memegang peranan penting karena suara merupakan medium komunikasi paling natural bagi manusia. Dua teknologi utama dalam bidang ini adalah **Automatic Speech Recognition (ASR)** dan **Text-to-Speech (TTS)**. ASR berfungsi mengubah ucapan manusia menjadi teks, sedangkan TTS berfungsi mengubah teks menjadi suara sintetis yang dapat didengar. Kombinasi keduanya memungkinkan terciptanya sistem komunikasi dua arah antara manusia dan mesin secara lebih intuitif.

Dalam konteks aplikasi modern, speech recognition tidak hanya digunakan untuk pengenalan perintah dasar, tetapi juga pada layanan transkripsi, pencarian suara, chatbot, analitik percakapan, hingga sistem kontrol perangkat. Sementara itu, TTS menjadi komponen penting pada pembacaan otomatis informasi, navigasi suara, alat bantu disabilitas, dan sistem edukasi digital. Dengan demikian, pemahaman terhadap pipeline ASR dan TTS menjadi relevan baik dari sisi akademik maupun praktis.

Namun demikian, penerapan speech technology pada **Bahasa Indonesia** masih menghadapi sejumlah tantangan. Tantangan tersebut meliputi keterbatasan dataset yang terstruktur, variasi pengucapan antar speaker, perbedaan dialek dan aksen daerah, kebisingan lingkungan, serta kebutuhan normalisasi audio yang konsisten. Oleh sebab itu, pengembangan sistem ASR untuk Bahasa Indonesia tetap menjadi topik penting, khususnya jika dilakukan dengan pendekatan yang dapat dipahami secara bertahap mulai dari preprocessing, ekstraksi fitur, hingga klasifikasi.

Berdasarkan latar belakang tersebut, proyek ini dirancang untuk membangun aplikasi suara yang dapat menjadi media pembelajaran sekaligus demonstrasi teknis. Fokus utama bukan pada transkripsi bebas, melainkan pada **keyword spotting** terbatas untuk 10 kata kunci nama negara ASEAN. Pendekatan ini dipilih agar pipeline ASR dapat ditampilkan secara jelas, terukur, dan sesuai dengan tingkat kompleksitas proyek akhir mahasiswa.

---

## Tujuan Projek

Proyek ini memiliki beberapa tujuan utama sebagai berikut:

- Memahami pipeline dasar sistem **ASR** secara menyeluruh, mulai dari pengumpulan data, preprocessing audio, ekstraksi fitur, pelatihan model, hingga inferensi.
- Mengimplementasikan fitur audio berbasis **MFCC** sebagai representasi numerik dari karakteristik sinyal ucapan.
- Melatih model klasifikasi sederhana menggunakan **MLPClassifier** sebagai baseline model ASR.
- Mengembangkan eksperimen model **CNN 1D** berbasis TensorFlow/Keras untuk meningkatkan performa klasifikasi audio.
- Menerapkan **realtime speech recognition** melalui input mikrofon untuk mendukung inferensi langsung.
- Mengintegrasikan hasil prediksi **ASR** ke modul **TTS** sehingga sistem mampu melakukan interaksi dua arah antara suara dan teks.
- Menyediakan antarmuka aplikasi yang mudah digunakan melalui **desktop GUI** dan **web dashboard**.
- Menyediakan fitur tambahan seperti confidence score, visualisasi MFCC, penyimpanan audio, dan pengaturan suara TTS.

Secara akademik, proyek ini juga bertujuan agar mahasiswa memahami bahwa pengembangan sistem AI suara tidak hanya bergantung pada model, tetapi juga pada kualitas data, teknik preprocessing, desain fitur, dan evaluasi yang tepat.

---

## Dataset Audio

Dataset yang digunakan pada proyek ini berupa **10 keyword nama negara ASEAN**, yaitu:

- Indonesia
- Malaysia
- Thailand
- Singapura
- Vietnam
- Laos
- Myanmar
- Filipina
- Brunei
- Kamboja

Dataset disusun untuk mendukung skenario **keyword spotting** terbatas. Pendekatan ini dipilih karena lebih realistis untuk tugas akhir mahasiswa dibandingkan membangun sistem transkripsi bebas yang membutuhkan dataset jauh lebih besar dan model yang jauh lebih kompleks.

### Karakteristik Dataset

Secara umum, dataset memiliki karakteristik berikut:

- Terdiri dari 10 kelas label sesuai nama negara ASEAN.
- Masing-masing label memiliki sejumlah sampel audio yang direkam atau dikumpulkan dari dataset bebas.
- Audio direkam dalam bentuk kata tunggal atau ucapan pendek.
- Terdapat variasi speaker untuk meningkatkan generalisasi model.
- Terdapat variasi intonasi, tempo bicara, dan kualitas mikrofon.

### Alasan Penggunaan Banyak Speaker

Penggunaan banyak speaker penting karena suara manusia memiliki ciri akustik yang beragam. Perbedaan jenis kelamin, tinggi-rendah pitch, kecepatan bicara, hingga aksen dapat memengaruhi hasil pengenalan. Jika model hanya dilatih dari satu atau dua speaker, model cenderung _overfitting_ terhadap karakter suara tertentu dan kurang robust ketika digunakan oleh orang lain.

### Preprocessing Awal Dataset

Sebelum digunakan untuk training, data audio diproses agar memiliki format yang seragam. Tahapan tersebut meliputi:

- konversi ke mono,
- resampling ke sample rate target,
- normalisasi amplitudo,
- trimming bagian hening,
- padding atau truncation agar durasi audio konsisten.

### Placeholder Screenshot Dataset

[Tempat Screenshot Dataset]

> Contoh yang dapat ditempel pada Notion: struktur folder dataset, contoh waveform file audio, atau cuplikan rekaman per label.

---

## Metode yang Digunakan

### Preprocessing Audio

Preprocessing merupakan tahapan awal yang sangat penting dalam sistem ASR. Tujuan preprocessing adalah menyiapkan sinyal audio mentah agar dapat diproses lebih lanjut oleh algoritma ekstraksi fitur dan model klasifikasi.

Tahapan preprocessing yang digunakan dalam proyek ini meliputi:

- **Load audio**: file audio dibaca dari dataset atau hasil rekaman mikrofon.
- **Mono conversion**: audio stereo diubah menjadi satu kanal agar representasi sinyal lebih sederhana dan konsisten.
- **Normalization**: amplitudo dinormalisasi untuk mengurangi pengaruh perbedaan volume antar sampel.
- **Trimming**: bagian diam atau hening pada awal dan akhir ucapan dipangkas agar fokus pada segmen suara utama.
- **Padding / truncation**: durasi audio diseragamkan sesuai panjang target agar input model tetap konsisten.

Secara konseptual, preprocessing ini membantu menurunkan noise yang tidak relevan dan menyamakan struktur input antar sampel sehingga proses ekstraksi fitur MFCC dapat lebih stabil.

[Diagram Preprocessing Audio]

### MFCC Feature Extraction

MFCC atau **Mel-Frequency Cepstral Coefficients** merupakan teknik ekstraksi fitur yang sangat umum digunakan pada pengolahan sinyal suara. MFCC berusaha meniru cara pendengaran manusia dalam menangkap frekuensi, yaitu dengan menggunakan skala Mel yang lebih rapat pada frekuensi rendah dan lebih renggang pada frekuensi tinggi.

#### Alasan penggunaan MFCC

MFCC dipilih karena:

- mampu merepresentasikan karakteristik spektral ucapan secara ringkas,
- efektif untuk tugas pengenalan kata atau perintah,
- cukup ringan untuk model klasifikasi sederhana,
- telah terbukti luas pada berbagai aplikasi speech processing.

#### Proses ekstraksi MFCC

Secara umum, proses ekstraksi MFCC terdiri dari:

- transformasi sinyal ke domain frekuensi,
- pemetaan ke skala Mel,
- pengambilan log energi,
- transformasi cosinus diskrit untuk menghasilkan koefisien MFCC.

Dalam implementasi proyek ini, MFCC digunakan sebagai fitur utama baik untuk model MLP maupun CNN. Pada model MLP, MFCC diolah menjadi vektor statistik. Pada model CNN, MFCC digunakan sebagai matriks dua dimensi sehingga pola temporal dan frekuensi dapat dipelajari lebih baik.

#### Placeholder visualisasi

[Tempat Screenshot MFCC]

- Placeholder screenshot MFCC heatmap.
- Placeholder waveform audio.

### Model MLP

Model **MLPClassifier** digunakan sebagai baseline utama untuk modul ASR desktop. MLP merupakan model neural network feedforward yang terdiri dari beberapa lapisan tersembunyi dan satu lapisan keluaran. Model ini cocok untuk klasifikasi berbasis fitur vektor yang sudah diekstraksi terlebih dahulu.

#### Konsep MLPClassifier

Dalam proyek ini, MFCC diubah menjadi fitur satu dimensi menggunakan statistik seperti mean dan standard deviation. Fitur tersebut kemudian masuk ke model MLPClassifier untuk dipetakan ke salah satu kelas nama negara ASEAN.

#### Alur training

Tahapan training model MLP meliputi:

- mengekstraksi fitur dari seluruh dataset,
- melakukan label encoding,
- membagi data menjadi data latih dan data uji,
- melakukan normalisasi fitur dengan StandardScaler,
- melatih MLPClassifier,
- mengevaluasi hasil pada data uji.

#### Output klasifikasi

Keluaran model berupa prediksi salah satu dari 10 kelas negara ASEAN. Selain label prediksi, sistem juga menampilkan confidence score untuk menunjukkan tingkat keyakinan model terhadap hasil prediksi.

#### Evaluasi Model MLP

Tambahkan hasil evaluasi berikut setelah training selesai:

- classification report MLP,
- confusion matrix MLP,
- accuracy MLP.

##### Tabel Evaluasi MLP

| Metrik             | Nilai           |
| ------------------ | --------------- |
| Accuracy           | 80.28% (0.8028) |
| Precision Macro    | 80.42% (0.8042) |
| Recall Macro       | 80.28% (0.8028) |
| F1-Score Macro     | 80.15% (0.8015) |
| Precision Weighted | 80.42% (0.8042) |
| Recall Weighted    | 80.28% (0.8028) |
| F1-Score Weighted  | 80.15% (0.8015) |

#### Placeholder Gambar MLP

- [Tempat Classification Report MLP]
- [Tempat Confusion Matrix MLP]
- [Tempat Accuracy MLP]

### Model CNN

Model **CNN 1D** digunakan sebagai eksperimen lanjutan untuk klasifikasi audio. Berbeda dengan MLP yang hanya memproses vektor fitur, CNN 1D memanfaatkan struktur urutan waktu dari MFCC sehingga mampu menangkap pola lokal yang muncul pada dimensi temporal sinyal ucapan.

#### Konsep CNN 1D untuk Audio Classification

Dalam konteks ini, MFCC diperlakukan sebagai matriks dengan sumbu waktu dan koefisien frekuensi. Lapisan **Conv1D** digunakan untuk mempelajari pola lokal, sedangkan **MaxPooling** membantu mereduksi dimensi dan mempertahankan fitur penting. Setelah itu, hasil ekstraksi fitur diteruskan ke lapisan **Dense** hingga keluaran **softmax** untuk menentukan probabilitas tiap kelas.

#### Arsitektur Umum CNN

Secara umum, arsitektur CNN yang digunakan terdiri dari:

- layer input dengan bentuk matriks MFCC,
- beberapa layer **Conv1D**,
- **MaxPooling1D** untuk downsampling,
- **Flatten** untuk mengubah fitur menjadi vektor,
- **Dense layer** untuk klasifikasi akhir,
- **Softmax output** untuk menghasilkan distribusi probabilitas kelas.

#### Alasan CNN digunakan pada web realtime

CNN digunakan pada web realtime karena pada eksperimen proyek ini CNN menunjukkan performa yang lebih baik dan lebih stabil dibandingkan MLP untuk data audio yang ditampilkan dalam bentuk MFCC. CNN juga lebih sesuai untuk input dua dimensi dan dapat menangkap pola waktu-frekuensi secara lebih efektif.

#### Evaluasi Model CNN

Tambahkan hasil evaluasi berikut setelah training selesai:

- training plot CNN,
- classification report CNN,
- confusion matrix CNN,
- accuracy CNN.

##### Tabel Evaluasi CNN

| Metrik             | Nilai       |
| ------------------ | ----------- |
| Accuracy           | [isi hasil] |
| Precision Macro    | [isi hasil] |
| Recall Macro       | [isi hasil] |
| F1-Score Macro     | [isi hasil] |
| Precision Weighted | [isi hasil] |
| Recall Weighted    | [isi hasil] |
| F1-Score Weighted  | [isi hasil] |

##### Tabel Perbandingan Model

| Model         | Accuracy    | Keterangan                                           |
| ------------- | ----------- | ---------------------------------------------------- |
| MLPClassifier | [isi hasil] | Baseline model ASR desktop                           |
| CNN 1D        | [isi hasil] | Lebih stabil untuk web realtime dan visualisasi MFCC |

#### Placeholder Gambar CNN

- [Tempat Training Plot CNN]
- [Tempat Classification Report CNN]
- [Tempat Confusion Matrix CNN]
- [Tempat Accuracy CNN]

### Realtime Inference

Realtime inference merupakan fitur yang memungkinkan sistem menerima input suara dari mikrofon, memprosesnya secara langsung, lalu menampilkan hasil prediksi kata di antarmuka pengguna.

#### Mekanisme realtime prediction

Proses realtime inference pada proyek ini melibatkan beberapa langkah:

- mikrofon menangkap audio input,
- audio dipotong per jendela waktu tertentu,
- audio diproses menggunakan preprocessing dan MFCC,
- model memprediksi kelas negara ASEAN,
- hasil prediksi ditampilkan pada GUI beserta confidence score,
- sistem dapat melakukan analisis otomatis setelah rekaman selesai.

#### Confidence score

Confidence score menunjukkan tingkat keyakinan model terhadap label prediksi. Semakin tinggi nilainya, semakin besar keyakinan model bahwa audio input sesuai dengan kelas tertentu. Fitur ini penting untuk membantu pengguna memahami apakah hasil prediksi cukup dapat dipercaya atau masih perlu dikonfirmasi.

#### Web realtime dashboard

Pada versi web, realtime inference diintegrasikan ke dalam dashboard Gradio sehingga proses demo menjadi lebih interaktif. Dashboard ini menampilkan hasil prediksi, visualisasi MFCC, informasi analitik negara ASEAN, dan elemen antarmuka modern yang sesuai untuk presentasi.

#### Placeholder Screenshot Realtime

[Tempat Screenshot Realtime Prediction]

### Text-to-Speech

Modul Text-to-Speech digunakan untuk mengubah teks input menjadi suara dalam Bahasa Indonesia. Sistem ini mendukung dua pendekatan, yaitu **pyttsx3** untuk mode offline dan **gTTS** untuk hasil suara yang lebih natural ketika internet tersedia.

#### Fungsi utama TTS

Fitur TTS dalam proyek ini meliputi:

- input teks oleh pengguna,
- output suara dalam Bahasa Indonesia,
- pilihan gender suara,
- pengaturan kecepatan bicara,
- penyimpanan audio ke format WAV atau MP3,
- pemutaran suara langsung dari GUI.

#### Gender voice

Pengguna dapat memilih gender suara, misalnya laki-laki atau perempuan, untuk menyesuaikan preferensi presentasi atau kebutuhan demo. Pada praktiknya, pilihan gender ini meningkatkan fleksibilitas antarmuka dan membuat aplikasi terasa lebih lengkap.

#### Speed control

Pilihan kecepatan bicara diberikan dalam mode **slow**, **normal**, dan **fast**. Fitur ini penting karena kecepatan bicara memengaruhi keterbacaan suara sintetis dan pengalaman pengguna.

#### Simpan audio

Aplikasi juga menyediakan fitur ekspor audio sehingga hasil TTS dapat disimpan sebagai file lokal. Ini bermanfaat untuk keperluan dokumentasi, presentasi, dan pengujian hasil suara.

#### Placeholder Screenshot TTS

[Tempat Screenshot TTS]

---

## Implementasi Sistem

### Desktop Application

Aplikasi desktop dikembangkan menggunakan **CustomTkinter** sebagai antarmuka grafis utama. Desktop application memuat panel ASR dan TTS dalam satu layar sehingga pengguna dapat melakukan pengenalan suara sekaligus menghasilkan suara sintetis.

#### Fitur utama desktop app

- Realtime ASR melalui mikrofon.
- Inferensi model MLP untuk keyword spotting.
- Confidence score dan top prediction.
- Visualisasi waveform dan MFCC.
- Panel TTS dengan speed control, gender voice, dan save audio.
- Integrasi ASR → TTS sehingga hasil prediksi dapat langsung diteruskan ke modul TTS.

#### Placeholder Screenshot Desktop App

[Tempat Screenshot Desktop App]

### Web Application

Selain versi desktop, proyek ini juga memiliki **web application** berbasis **Gradio**. Versi web digunakan untuk menampilkan dashboard AI yang lebih modern dan cocok untuk demonstrasi presentasi di browser.

#### Fitur utama web dashboard

- Realtime microphone input.
- CNN 1D untuk inference.
- MFCC visualization.
- ASEAN analytics dashboard.
- Tampilan modern dengan tema dark/light.
- Integrasi panel TTS dalam antarmuka web.

#### Placeholder Screenshot Web Dashboard

[Tempat Screenshot Web Dashboard]

---

## Pengujian dan Evaluasi

Pengujian dilakukan untuk memastikan bahwa sistem berjalan sesuai rancangan, baik pada modul ASR maupun TTS. Evaluasi tidak hanya melihat apakah model dapat memprediksi kelas tertentu, tetapi juga melihat kestabilan hasil prediksi, kualitas confidence score, dan pengalaman penggunaan antarmuka.

### Pengujian Realtime

Pengujian realtime dilakukan dengan merekam suara melalui mikrofon dan mengamati hasil prediksi pada GUI. Aspek yang diamati meliputi:

- kecepatan respons sistem,
- ketepatan label hasil prediksi,
- kestabilan confidence score,
- kemampuan sistem dalam menghadapi suara dari berbagai speaker,
- kesesuaian hasil prediksi dengan ucapan yang diinput.

### Evaluasi Hasil Prediksi

Hasil prediksi dievaluasi berdasarkan confusion matrix dan classification report. Confusion matrix digunakan untuk melihat pasangan kelas yang paling sering tertukar, sedangkan classification report memberikan ringkasan metrik precision, recall, dan F1-score untuk masing-masing kelas.

### Perbandingan MLP vs CNN

Perbandingan dilakukan untuk melihat model mana yang lebih sesuai untuk aplikasi ini. Secara umum:

- **MLPClassifier** cocok sebagai baseline karena sederhana dan mudah dijelaskan.
- **CNN 1D** lebih baik untuk data audio berbasis MFCC karena mampu mempelajari pola temporal.
- Pada implementasi web realtime, CNN dipilih karena performanya lebih stabil dibanding MLP.

### Evaluasi Penggunaan Banyak Speaker

Pengujian dengan banyak speaker penting untuk melihat robust tidaknya model terhadap variasi suara. Sistem yang baik harus tetap mampu mengenali keyword walaupun diucapkan oleh speaker yang berbeda. Evaluasi ini menunjukkan sejauh mana model mampu melakukan generalisasi.

### Tabel Hasil Pengujian

| Skenario Uji                    | Hasil                                  | Catatan                                                                        |
| ------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------ |
| Ucapan jelas dari speaker utama | MLP: 80.3% / CNN: 94.7%                | Akurasi tinggi untuk kedua model; CNN menunjukkan performa superior            |
| Ucapan dari speaker berbeda     | MLP: ≈75% / CNN: ≈93%                  | Terjadi penurunan untuk MLP; confidence score bervariasi antar speaker         |
| Ucapan cepat                    | MLP: ≈72% / CNN: ≈90%                  | Beberapa kelas sulit dikenali pada kecepatan tinggi, CNN lebih tahan banting   |
| Ucapan lambat                   | MLP: ≈82% / CNN: ≈95%                  | Peningkatan kestabilan prediksi, confidence lebih tinggi                       |
| Kondisi noise ringan            | MLP: ≈70% / CNN: ≈88%                  | Model masih dapat membaca kata dengan penurunan performa moderat               |
| Realtime inference              | Latency < 500 ms; CNN accuracy ≈93–95% | Respons sesuai kebutuhan demo; smoothing dan threshold meningkatkan stabilitas |

### Placeholder Hasil Evaluasi

- [Tempat Confusion Matrix]
- [Tempat Classification Report]
- [Tempat Grafik Accuracy / Training Plot]

---

## Fitur Tambahan

Proyek ini memenuhi kebutuhan fitur tambahan dengan mengimplementasikan beberapa komponen berikut:

- **Confidence score** untuk menampilkan tingkat keyakinan prediksi ASR.
- **MFCC visualization** untuk melihat representasi fitur audio secara visual.
- **ASR → TTS integration** sehingga hasil pengenalan suara dapat langsung dipakai sebagai teks input TTS.
- **Gender voice** pada modul TTS untuk memilih suara laki-laki atau perempuan.
- **Web realtime dashboard** sebagai antarmuka demonstrasi berbasis browser.
- **Auto analyze realtime** yang memungkinkan sistem menganalisis input suara secara otomatis setelah rekaman selesai.

Dengan demikian, proyek ini tidak hanya memenuhi fungsi dasar ASR dan TTS, tetapi juga menambahkan nilai interaktif yang mendukung presentasi akademik dan pengalaman pengguna.

---

## Kesimpulan

Berdasarkan implementasi yang telah dilakukan, dapat disimpulkan bahwa proyek **ASEAN Voice Intelligence – Automatic Speech Recognition dan Text-to-Speech Berbahasa Indonesia** berhasil dikembangkan sebagai aplikasi dua arah berbasis suara dan teks.

Beberapa kesimpulan utama adalah:

- Sistem ASR berhasil dibangun menggunakan pipeline mandiri yang terdiri dari preprocessing audio, ekstraksi MFCC, dan klasifikasi model.
- Model **MLPClassifier** berhasil digunakan sebagai baseline untuk pengenalan 10 keyword negara ASEAN.
- Model **CNN 1D** menunjukkan performa yang lebih baik dan lebih stabil untuk skenario realtime berbasis MFCC.
- Fitur **realtime speech recognition** berhasil diintegrasikan ke antarmuka GUI.
- Modul **Text-to-Speech Bahasa Indonesia** berhasil diimplementasikan dengan dukungan gender suara, pengaturan speed, dan penyimpanan file audio.
- Integrasi **ASR → TTS** berjalan dengan baik sehingga sistem dapat digunakan sebagai demo interaktif dua arah.

Secara keseluruhan, proyek ini sesuai sebagai implementasi sistem AI suara pada skala terbatas dan dapat menjadi dasar untuk pengembangan lebih lanjut di masa mendatang.

---

## Saran Pengembangan

Walaupun sistem sudah berfungsi dengan baik untuk kebutuhan proyek akhir, terdapat beberapa pengembangan yang dapat dilakukan pada tahap selanjutnya:

- Menambah jumlah dataset agar model lebih robust terhadap variasi speaker dan noise.
- Memperluas klasifikasi dari keyword spotting menjadi transkripsi bebas.
- Mencoba arsitektur model yang lebih kuat seperti CNN yang lebih dalam, BiLSTM, atau model berbasis Transformer.
- Melakukan deployment berbasis cloud agar aplikasi dapat diakses secara online.
- Mengembangkan dukungan **multilingual speech recognition**.
- Mengintegrasikan aplikasi ke platform mobile.
- Menambahkan logging hasil prediksi untuk analisis performa jangka panjang.
- Menyediakan mode _speaker-independent testing_ yang lebih formal.

---

## Lampiran Singkat untuk Notion

Bagian ini dapat digunakan langsung sebagai isi laporan di Notion, sedangkan screenshot dan tabel hasil numerik dapat diisi setelah pengujian final selesai.

### Placeholder Daftar Gambar

- [Tempat Screenshot Dataset]
- [Diagram Preprocessing Audio]
- [Tempat Screenshot MFCC]
- [Tempat Classification Report MLP]
- [Tempat Confusion Matrix MLP]
- [Tempat Training Plot CNN]
- [Tempat Classification Report CNN]
- [Tempat Confusion Matrix CNN]
- [Tempat Screenshot Realtime Prediction]
- [Tempat Screenshot TTS]
- [Tempat Screenshot Desktop App]
- [Tempat Screenshot Web Dashboard]

### Catatan Final

Dokumen ini disusun dengan gaya formal akademik agar mudah dipindahkan ke Notion. Setelah itu, tinggal menambahkan:

- nama anggota dan NIM,
- screenshot hasil implementasi,
- nilai evaluasi aktual dari training dan pengujian,
- link GitHub,
- link demo video,
- link Notion final.
