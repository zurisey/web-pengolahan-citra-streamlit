import os
import time
import numpy as np
import streamlit as st
from PIL import Image

# Import modul pencocokan wajah bawaan (Pastikan folder 'model' ada di direktori yang sama)
import model.eigenface as ef

# ==========================================
# KONFIGURASI DIREKTORI SEMENTARA
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_UPLOAD_FOLDER = os.path.join(BASE_DIR, "temp_uploads")
os.makedirs(TEMP_UPLOAD_FOLDER, exist_ok=True)

# Inisialisasi Model Eigenface
if not ef.model.is_trained:
    ef.run_training_manually()

# ==========================================
# FUNGSI KOMPUTASI PCA
# ==========================================
def hitung_pca_core(matrix, n_comp):
    h, w = matrix.shape
    rata_rata = np.mean(matrix, axis=0)
    centered = matrix - rata_rata
    
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    n = min(n_comp, len(S), w, h)
    S_diag = np.diag(S[:n])
    
    reconstructed = np.dot(U[:, :n], np.dot(S_diag, Vt[:n, :])) + rata_rata
    return np.clip(reconstructed, 0, 255).astype(np.uint8)

def proses_pca_grayscale(img_arr, n_components):
    return hitung_pca_core(img_arr, n_components)

def proses_pca_rgb(img_arr, n_components):
    r_chan = hitung_pca_core(img_arr[:, :, 0], n_components)
    g_chan = hitung_pca_core(img_arr[:, :, 1], n_components)
    b_chan = hitung_pca_core(img_arr[:, :, 2], n_components)
    return np.stack([r_chan, g_chan, b_chan], axis=2)

# ==========================================
# ANTARMUKA STREAMLIT
# ==========================================
st.set_page_config(page_title="Sistem Analisis Wajah", layout="wide")

st.sidebar.title("Navigasi Menu")
menu = st.sidebar.radio("Pilih Fitur:", ["Kompresi Gambar (PCA)", "Analisis Kemiripan Wajah"])

# --- MENU 1: KOMPRESI PCA ---
if menu == "Kompresi Gambar (PCA)":
    st.title("Kompresi Gambar menggunakan PCA")
    st.write("Unggah gambar untuk melihat hasil kompresi reduksi dimensi.")
    
    uploaded_file = st.file_uploader("Pilih Gambar", type=["jpg", "jpeg", "png"])
    n_components = st.slider("Pilih Jumlah Komponen Utama (Semakin kecil semakin buram & ringan)", min_value=1, max_value=300, value=50)

    if uploaded_file is not None:
        # Tampilkan Gambar Asli
        img_original = Image.open(uploaded_file)
        lebar, tinggi = img_original.size
        ukuran_asli_kb = uploaded_file.size / 1024

        st.subheader("Gambar Asli")
        st.image(img_original, caption=f"Dimensi: {lebar}x{tinggi} | Ukuran: {ukuran_asli_kb:.2f} KB", use_column_width=True)

        if st.button("Proses Kompresi"):
            with st.spinner('Sedang memproses kompresi...'):
                MAX_RESOLUTION = 450
                if lebar > MAX_RESOLUTION or tinggi > MAX_RESOLUTION:
                    img_original.thumbnail((MAX_RESOLUTION, MAX_RESOLUTION))
                    lebar, tinggi = img_original.size

                # Proses Grayscale
                img_gray = img_original.convert('L')
                arr_gray = np.array(img_gray)
                res_gray = proses_pca_grayscale(arr_gray, n_components)
                img_gray_res = Image.fromarray(res_gray)
                
                # Proses RGB
                img_rgb = img_original.convert('RGB')
                arr_rgb = np.array(img_rgb)
                res_rgb = proses_pca_rgb(arr_rgb, n_components)
                img_rgb_res = Image.fromarray(res_rgb)

                # Tampilan Hasil
                st.markdown("---")
                st.subheader("Hasil Kompresi")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(img_gray_res, caption=f"Grayscale ({n_components} Komponen)", use_column_width=True)
                
                with col2:
                    st.image(img_rgb_res, caption=f"RGB ({n_components} Komponen per Channel)", use_column_width=True)

# --- MENU 2: KEMIRIPAN WAJAH ---
elif menu == "Analisis Kemiripan Wajah":
    st.title("Analisis Kemiripan Wajah (Bayi vs Dewasa)")
    st.write("Sistem mendeteksi fitur biometrik wajah untuk mencari nilai kecocokan identitas.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Foto 1 (Masa Kecil/Bayi)")
        file1 = st.file_uploader("Unggah Foto Pertama", type=["jpg", "jpeg", "png"], key="file1")
        if file1:
            st.image(Image.open(file1), use_column_width=True)

    with col2:
        st.subheader("Foto 2 (Dewasa)")
        file2 = st.file_uploader("Unggah Foto Kedua", type=["jpg", "jpeg", "png"], key="file2")
        if file2:
            st.image(Image.open(file2), use_column_width=True)

    if file1 and file2:
        if st.button("Mulai Analisis Kemiripan"):
            with st.spinner("Ekstraksi LBP dan Eigenface sedang berjalan..."):
                # Simpan file sementara untuk dibaca oleh cv2.imread di eigenface.py
                path1 = os.path.join(TEMP_UPLOAD_FOLDER, "temp_face1.jpg")
                path2 = os.path.join(TEMP_UPLOAD_FOLDER, "temp_face2.jpg")
                
                with open(path1, "wb") as f:
                    f.write(file1.getbuffer())
                with open(path2, "wb") as f:
                    f.write(file2.getbuffer())

                try:
                    sim_percentage, cosine_score, kesimpulan = ef.compare_faces(path1, path2)
                    
                    st.success("Analisis Selesai!")
                    st.markdown(f"""
                    * **Persentase Kemiripan:** {sim_percentage}%
                    * **Jarak Cosine (Cosine Score):** {cosine_score}
                    * **Kesimpulan Sistem:** **{kesimpulan}**
                    """)
                except Exception as e:
                    st.error(f"Gagal memproses gambar: {str(e)}")
                    st.info("Pastikan gambar memuat wajah yang jelas dan menghadap ke depan.")
