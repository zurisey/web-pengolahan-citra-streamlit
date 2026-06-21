# SnapSense - Aplikasi Web Pengolahan Citra Digital

SnapSense adalah aplikasi web interaktif yang dikembangkan menggunakan arsitektur Streamlit. Aplikasi ini berfokus pada dua fungsi utama pengolahan citra digital: optimasi ukuran gambar (kompresi) dengan mempertahankan kualitas visual, serta analisis biometrik untuk pencocokan wajah.

Repositori Github: https://github.com/zurisey/web-pengolahan-citra-streamlit.git

## Fitur Utama

1. Optimasi Gambar (Kompresi Berbasis PCA)
Fitur ini memungkinkan pengguna untuk memperkecil ukuran memori gambar menggunakan metode reduksi dimensi Principal Component Analysis (PCA). Sistem bekerja secara matematis untuk mengekstrak komponen utama gambar.
- Mendukung pemrosesan citra Grayscale (1-Channel) dan RGB (3-Channel).
- Menyediakan kendali tingkat kompresi dinamis (jumlah komponen).
- Menampilkan metrik evaluasi kompresi secara langsung, meliputi:
  - Mean Squared Error (MSE)
  - Peak Signal-to-Noise Ratio (PSNR)
  - Structural Similarity Index Measure (SSIM)
  - Rasio Penghematan dan Explained Variance.

2. Pencocokan Wajah (Analisis Biometrik)
Fitur ini mengomparasi dua foto wajah yang berbeda untuk menentukan tingkat kemiripan identitas.
- Metode Deep Learning: Memanfaatkan arsitektur Convolutional Neural Network (CNN) dan ekstraksi model DeepFace dengan basis TensorFlow untuk akurasi tinggi menggunakan pengukuran Cosine Distance.
- Metode Klasik: Menggunakan kombinasi Eigenface dan Local Binary Pattern (LBP) untuk komputasi spasial yang lebih ringan menggunakan Cosine Similarity.
- Menghasilkan persentase kemiripan lengkap dengan status konklusi kecocokan identitas secara instan.

## Prasyarat Sistem

Pastikan sistem Anda sudah terpasang Python (versi 3.8 atau lebih baru disarankan). Berkas pemrosesan gambar dan model deep learning pada aplikasi ini membutuhkan beberapa pustaka eksternal.

## Panduan Instalasi dan Penggunaan

Ikuti langkah-langkah di bawah ini untuk menjalankan aplikasi di komputer lokal Anda:

1. Kloning repositori ini ke direktori lokal Anda
```bash
git clone [https://github.com/zurisey/web-pengolahan-citra-streamlit.git](https://github.com/zurisey/web-pengolahan-citra-streamlit.git)