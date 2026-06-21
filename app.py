import os
import gc
import numpy as np
import streamlit as st
from PIL import Image
from scipy.sparse.linalg import svds
from skimage.metrics import mean_squared_error as mse_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

# ==========================================
# KONFIGURASI AWAL & DIREKTORI
# ==========================================
st.set_page_config(
    page_title="SnapSense", 
    page_icon="images/icon.jpeg", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_UPLOAD_FOLDER = os.path.join(BASE_DIR, "temp_uploads")
os.makedirs(TEMP_UPLOAD_FOLDER, exist_ok=True)

@st.cache_resource
def inisialisasi_deep_learning_model():
    import model.deepface_analyser as dfa
    return dfa.muat_model_terlebih_dahulu()

# ==========================================
# INJEKSI CSS CUSTOM (RE-DESIGN TOTAL)
# ==========================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] { 
        font-family: 'Plus Jakarta Sans', sans-serif !important; 
    }
    h1, h2, h3, p, span, label { 
        color: #0F172A !important; 
    }
    
    [data-testid="stAppViewContainer"] { 
        background-color: #F8FAFC !important; 
    }
    
    [data-testid="stHeader"] { 
        background-color: transparent !important;
    }

    /* Modifikasi Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1E5E60 !important; 
        border-right: none;
        padding-top: 15px;
    }
    [data-testid="stSidebar"] * { 
        color: #FFFFFF !important; 
    }
    
    .sidebar-logo { 
        font-size: 1.6rem; 
        font-weight: 700; 
        margin-bottom: 25px; 
        letter-spacing: -0.5px;
        color: #FFFFFF !important;
    }
    .sidebar-hero { 
        font-size: 2rem; 
        font-weight: 800; 
        line-height: 1.25; 
        margin-bottom: 12px; 
        color: #FFFFFF !important;
    }
    .sidebar-desc { 
        color: #E2E8F0 !important; 
        font-size: 0.9rem; 
        line-height: 1.6; 
        margin-bottom: 35px; 
    }

    /* Styling Tombol Navigasi Radio */
    div[role="radiogroup"] > label {
        background-color: #164E50 !important; 
        padding: 14px 16px !important;
        border-radius: 10px !important;
        border: 2px solid transparent !important;
        margin-bottom: 12px;
        transition: all 0.25s ease-in-out;
        display: flex;
        align-items: center;
        cursor: pointer;
    }
    div[role="radiogroup"] > label:hover { 
        background-color: #0D3335 !important; 
        transform: translateY(-1px); 
    }
    div[role="radiogroup"] > label[data-checked="true"] {
        background-color: #0D3335 !important;
        border-color: #38B2AC !important; 
    }
    div[role="radiogroup"] div[data-testid="stMarkdownContainer"] p {
        font-size: 0.95rem !important; 
        font-weight: 600 !important; 
        margin: 0;
        color: #FFFFFF !important;
    }
    div[role="radiogroup"] span[data-baseweb="radio"] { 
        display: none !important; 
    }

    /* Styling Button Utama */
    .stButton > button {
        background-color: #38B2AC !important; 
        color: #FFFFFF !important;
        border: none !important;
        width: 100%;
        padding: 14px 24px !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.3px;
        box-shadow: 0 4px 6px -1px rgba(56, 178, 172, 0.2);
        transition: all 0.2s ease-in-out !important;
    }
    .stButton > button:hover { 
        background-color: #2C8E8A !important; 
        box-shadow: 0 10px 15px -3px rgba(56, 178, 172, 0.3);
        transform: translateY(-1px);
    }

    /* File Uploader Dropzone */
    [data-testid="stFileUploadDropzone"] {
        background-color: #FFFFFF !important;
        border: 2px dashed #CBD5E1 !important;
        border-radius: 14px !important;
        padding: 45px 20px !important;
    }

    /* Tipografi Utama */
    .section-title { 
        font-size: 2.2rem; 
        text-align: center; 
        margin-top: 10px;
        margin-bottom: 10px; 
        font-weight: 800; 
        color: #0F172A !important;
    }
    .section-desc { 
        text-align: center; 
        color: #64748B !important; 
        font-size: 1.05rem; 
        margin-bottom: 40px; 
    }
    .micro-instruction {
        font-size: 0.85rem !important;
        color: #64748B !important;
        margin-top: -8px;
        margin-bottom: 15px;
        display: block;
    }

    /* Styling St.Metric sebagai Kartu Cantik */
    [data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        padding: 20px 15px !important;
        border-radius: 12px !important;
        border: 1px solid #E2E8F0 !important;
        text-align: center !important;
        box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.05) !important;
    }
    [data-testid="stMetricValue"] {
        justify-content: center !important;
        font-size: 1.8rem !important;
        color: #1E5E60 !important;
        font-weight: 800 !important;
        margin-top: 5px !important;
    }
    [data-testid="stMetricLabel"] {
        justify-content: center !important;
        font-size: 0.95rem !important;
        color: #64748B !important;
        font-weight: 600 !important;
    }

    /* Tabs Styling */
    button[data-baseweb="tab"] {
        font-weight: 600 !important;
        font-size: 1rem !important;
    }

    .main-footer {
        padding: 25px 0; 
        margin-top: 60px; 
        border-top: 1px solid #E2E8F0; 
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# FUNGSI KOMPRESI (PCA) DENGAN METRIK STATISTIK
# ==========================================
def hitung_pca_core_ekonomis(matrix, n_comp):
    h, w = matrix.shape
    rata_rata = np.mean(matrix, axis=0)
    centered = (matrix - rata_rata).astype(np.float32)
    n = min(n_comp, w-1, h-1)
    if n < 1: n = 1
    
    U, S, Vt = svds(centered, k=n)
    
    total_variance = np.sum(centered ** 2)
    explained_variance = np.sum(S ** 2) / total_variance if total_variance > 0 else 1.0
    
    U = U[:, ::-1]
    S = S[::-1]
    Vt = Vt[::-1, :]
    
    reconstructed = np.dot(U, np.dot(np.diag(S), Vt)) + rata_rata
    del centered, U, Vt
    gc.collect()
    return np.clip(reconstructed, 0, 255).astype(np.uint8), n, explained_variance

def proses_pca_grayscale(img_arr, n_components):
    h, w = img_arr.shape
    reconstructed, n_aktif, exp_var = hitung_pca_core_ekonomis(img_arr, n_components)
    
    mse_val = float(mse_metric(img_arr, reconstructed))
    psnr_val = float(psnr_metric(img_arr, reconstructed, data_range=255))
    ssim_val = float(ssim_metric(img_arr, reconstructed, data_range=255))
    
    ukuran_asli = h * w
    ukuran_kompresi = n_aktif * (h + w + 1)
    
    stats = {
        "ukuran_asli": ukuran_asli,
        "ukuran_kompresi": ukuran_kompresi,
        "mse": mse_val,
        "psnr": psnr_val,
        "ssim": ssim_val,
        "explained_var": exp_var
    }
    return reconstructed, stats

def proses_pca_rgb(img_arr, n_components):
    h, w, c = img_arr.shape
    r_chan, n_aktif, ev_r = hitung_pca_core_ekonomis(img_arr[:, :, 0], n_components)
    g_chan, _, ev_g = hitung_pca_core_ekonomis(img_arr[:, :, 1], n_components)
    b_chan, _, ev_b = hitung_pca_core_ekonomis(img_arr[:, :, 2], n_components)
    
    reconstructed = np.stack([r_chan, g_chan, b_chan], axis=2)
    
    mse_val = float(mse_metric(img_arr, reconstructed))
    psnr_val = float(psnr_metric(img_arr, reconstructed, data_range=255))
    ssim_val = float(ssim_metric(img_arr, reconstructed, data_range=255, channel_axis=2))
    
    ukuran_asli = h * w * 3
    ukuran_kompresi = n_aktif * (h + w + 1) * 3
    
    stats = {
        "ukuran_asli": ukuran_asli,
        "ukuran_kompresi": ukuran_kompresi,
        "mse": mse_val,
        "psnr": psnr_val,
        "ssim": ssim_val,
        "explained_var": (ev_r + ev_g + ev_b) / 3.0
    }
    return reconstructed, stats

# ==========================================
# SIDEBAR UI
# ==========================================
with st.sidebar:
    st.markdown('<div class="sidebar-logo">SnapSense</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-hero">Maksimalkan Potensi<br>Setiap Gambar</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-desc">Kompres gambar lebih efisien, analisis wajah lebih akurat.</div>', unsafe_allow_html=True)
    
    menu = st.radio(
        label="Navigasi Menu", 
        options=["Optimasi Gambar", "Pencocokan Wajah", "Segera Hadir"],
        label_visibility="collapsed"
    )
    
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style="display: flex; flex-direction: column; gap: 12px; font-size: 0.85rem; opacity: 0.9;">
            <a href="https://www.instagram.com/zurykami_" style="color: #FFFFFF; text-decoration: none;" target="_blank">Muhammad Ibrahim Zuhri (ketua)</a>
            <a href="https://www.instagram.com/krimmy2050cheese/" style="color: #FFFFFF; text-decoration: none;" target="_blank">Haikal Raka Pratama</a>
            <a href="https://www.instagram.com/hasnanafissa/" style="color: #FFFFFF; text-decoration: none;" target="_blank">Afifah Hasna</a>
        </div>
    """, unsafe_allow_html=True)

# ==========================================
# KONTEN UTAMA (MAIN AREA)
# ==========================================
if menu == "Optimasi Gambar":
    st.markdown('<div class="section-title">Perkecil Ukuran, Pertahankan Kualitas</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Optimalkan ukuran berkas gambar memanfaatkan rekonstruksi matematis matriks PCA.</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Tarik dan lepas gambar di sini atau klik untuk memilih berkas.", type=["jpg", "jpeg", "png"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    n_components = st.slider("Tingkat Kompresi (Jumlah Komponen Utama)", min_value=5, max_value=150, value=50)
    st.markdown('<span class="micro-instruction">Tips: Nilai komponen lebih tinggi mempertahankan akurasi detail visual.</span>', unsafe_allow_html=True)

    if uploaded_file is not None:
        img_original = Image.open(uploaded_file)
        lebar, tinggi = img_original.size
        ukuran_asli_kb = uploaded_file.size / 1024

        st.markdown(f'<p style="text-align:center; font-size:0.95rem; font-weight:600; color:#475569;">Dimensi Asli: {lebar}x{tinggi} px | Ukuran Awal: {ukuran_asli_kb:.2f} KB</p>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Mulai Optimasi Gambar"):
            with st.spinner('Memproses ekstraksi matriks komponen gambar...'):
                MAX_RESOLUTION = 400
                if lebar > MAX_RESOLUTION or tinggi > MAX_RESOLUTION:
                    img_original.thumbnail((MAX_RESOLUTION, MAX_RESOLUTION))
                
                img_gray = img_original.convert('L')
                arr_gray = np.array(img_gray)
                res_gray, stats_gray = proses_pca_grayscale(arr_gray, n_components)
                img_gray_res = Image.fromarray(res_gray)
                
                img_rgb = img_original.convert('RGB')
                arr_rgb = np.array(img_rgb)
                res_rgb, stats_rgb = proses_pca_rgb(arr_rgb, n_components)
                img_rgb_res = Image.fromarray(res_rgb)

            st.markdown('<hr style="border:1px solid #E2E8F0; margin-top: 40px; margin-bottom: 40px;">', unsafe_allow_html=True)
            st.markdown('<h3 style="text-align:center; margin-top:0; margin-bottom: 30px; font-weight:800; color:#0F172A;">Hasil Rekonstruksi Citra</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_gray_res, caption="Format Hasil Grayscale (1-Channel)", use_container_width=True)
            with col2:
                st.image(img_rgb_res, caption="Format Hasil RGB (3-Channel)", use_container_width=True)
            
            rasio_gray = stats_gray["ukuran_asli"] / stats_gray["ukuran_kompresi"]
            save_gray = (1 - (stats_gray["ukuran_kompresi"] / stats_gray["ukuran_asli"])) * 100
            rasio_rgb = stats_rgb["ukuran_asli"] / stats_rgb["ukuran_kompresi"]
            save_rgb = (1 - (stats_rgb["ukuran_kompresi"] / stats_rgb["ukuran_asli"])) * 100

            st.markdown('<h4 style="text-align:center; font-weight:800; color:#0F172A; margin-top:50px; margin-bottom:20px;">Laporan Analisis Struktur Kompresi</h4>', unsafe_allow_html=True)
            
            tab1, tab2 = st.tabs(["Analisis Grayscale", "Analisis RGB"])
            
            with tab1:
                st.markdown("<br>", unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Ukuran Asli", f"{stats_gray['ukuran_asli']} px")
                c2.metric("Ukuran Kompresi", f"{stats_gray['ukuran_kompresi']} px")
                c3.metric("Rasio Kompresi", f"{rasio_gray:.2f}:1")
                c4.metric("Penghematan", f"{save_gray:.2f}%")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                c5, c6, c7, c8 = st.columns(4)
                c5.metric("Nilai MSE", f"{stats_gray['mse']:.4f}")
                c6.metric("Nilai PSNR", f"{stats_gray['psnr']:.2f} dB")
                c7.metric("Nilai SSIM", f"{stats_gray['ssim']:.4f}")
                c8.metric("Explained Var.", f"{stats_gray['explained_var']*100:.2f}%")
                
            with tab2:
                st.markdown("<br>", unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Ukuran Asli", f"{stats_rgb['ukuran_asli']} px")
                c2.metric("Ukuran Kompresi", f"{stats_rgb['ukuran_kompresi']} px")
                c3.metric("Rasio Kompresi", f"{rasio_rgb:.2f}:1")
                c4.metric("Penghematan", f"{save_rgb:.2f}%")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                c5, c6, c7, c8 = st.columns(4)
                c5.metric("Nilai MSE", f"{stats_rgb['mse']:.4f}")
                c6.metric("Nilai PSNR", f"{stats_rgb['psnr']:.2f} dB")
                c7.metric("Nilai SSIM", f"{stats_rgb['ssim']:.4f}")
                c8.metric("Explained Var.", f"{stats_rgb['explained_var']*100:.2f}%")

            st.markdown("""
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top:40px;">
                    <div style="background-color: #F8FAFC; padding: 18px; border-radius: 10px; font-size: 0.85rem; text-align: center; border: 1px solid #E2E8F0; color: #475569;">Keamanan Terjamin: Berkas diproses sepenuhnya di sisi lokal browser Anda.</div>
                    <div style="background-color: #F8FAFC; padding: 18px; border-radius: 10px; font-size: 0.85rem; text-align: center; border: 1px solid #E2E8F0; color: #475569;">Metode Algoritma: Reduksi dimensi berbasis Principal Component Analysis (PCA) Core System.</div>
                </div>
            """, unsafe_allow_html=True)
            
            del arr_gray, res_gray, arr_rgb, res_rgb
            gc.collect()

elif menu == "Pencocokan Wajah":
    st.markdown('<div class="section-title">Analisis Kemiripan Biometrik Wajah</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Deteksi dan analisis tingkat kecocokan struktur dua wajah menggunakan komparasi komputasi citra digital.</div>', unsafe_allow_html=True)

    metode_dipilih = st.selectbox(
        "Pilih Mesin Algoritma Kecerdasan Buatan:",
        ["Deep Learning (CNN + DeepFace + TensorFlow) - Akurasi Tinggi", "Metode Klasik (Eigenface + LBP) - Performa Ringan"]
    )
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p style="text-align:center; font-weight:700; margin-bottom:10px;">Foto Pertama (Referensi)</p>', unsafe_allow_html=True)
        file1 = st.file_uploader("Unggah foto pertama", type=["jpg", "jpeg", "png"], key="ufile1", label_visibility="collapsed")
        if file1:
            st.image(Image.open(file1), use_container_width=True)

    with col2:
        st.markdown('<p style="text-align:center; font-weight:700; margin-bottom:10px;">Foto Kedua (Target Verifikasi)</p>', unsafe_allow_html=True)
        file2 = st.file_uploader("Unggah foto kedua", type=["jpg", "jpeg", "png"], key="ufile2", label_visibility="collapsed")
        if file2:
            st.image(Image.open(file2), use_container_width=True)

    if file1 and file2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("Jalankan Analisis Perbandingan"):
            path1 = os.path.join(TEMP_UPLOAD_FOLDER, "face_src.jpg")
            path2 = os.path.join(TEMP_UPLOAD_FOLDER, "face_tgt.jpg")
            
            with open(path1, "wb") as f:
                f.write(file1.getbuffer())
            with open(path2, "wb") as f:
                f.write(file2.getbuffer())

            try:
                if "Deep Learning" in metode_dipilih:
                    with st.spinner("Mengintegrasikan jaringan konvolusional (CNN) & ekstraksi model DeepFace..."):
                        import model.deepface_analyser as dfa
                        model_siap = inisialisasi_deep_learning_model()
                        sim_percentage, metric_score, kesimpulan = dfa.bandingkan_wajah_deep(path1, path2)
                        nama_metrik = "Cosine Distance"
                else:
                    with st.spinner("Memetakan tekstur permukaan citra lokal (Local Binary Pattern) & Ruang Eigen..."):
                        import model.eigenface as ef
                        sim_percentage, metric_score, kesimpulan = ef.compare_faces(path1, path2)
                        nama_metrik = "Cosine Similarity"

                # FIX: Mengganti HTML wrapper dengan pembatas bawaan Streamlit
                st.markdown('<hr style="border:1px solid #E2E8F0; margin-top: 40px; margin-bottom: 40px;">', unsafe_allow_html=True)
                st.markdown('<h3 style="color:#0F172A; margin-top:0; font-weight:800; text-align:center; margin-bottom:30px;">Hasil Komparasi Biometrik</h3>', unsafe_allow_html=True)
                
                c1, c2 = st.columns(2)
                c1.metric(f"Skor Jarak ({nama_metrik})", f"{metric_score:.4f}")
                c2.metric("Persentase Kemiripan", f"{sim_percentage:.2f}%")

                nilai_progress = min(int(sim_percentage), 100)
                st.markdown(f'<p style="margin-top:35px; font-weight:700; color:#475569; font-size:1rem; text-align:center;">Match Score: {sim_percentage:.2f}%</p>', unsafe_allow_html=True)
                st.progress(nilai_progress / 100.0)
                
                warna_status = "#059669" if "cocok" in kesimpulan.lower() or "sama" in kesimpulan.lower() else "#E11D48"
                bg_status = "#D1FAE5" if "cocok" in kesimpulan.lower() or "sama" in kesimpulan.lower() else "#FFE4E6"
                
                st.markdown(f"""
                    <div style="background-color: {bg_status}; border-left: 6px solid {warna_status}; padding: 22px 25px; border-radius: 8px; margin-top: 40px; text-align: center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
                        <span style="font-size: 1.05rem; color: #334155; font-weight: 500;">Status Konklusi:</span><br>
                        <span style="font-size: 1.35rem; font-weight: 800; color: {warna_status};">{kesimpulan}</span>
                    </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Sistem gagal mengeksekusi analisis wajah: {str(e)}")
            finally:
                if os.path.exists(path1): os.remove(path1)
                if os.path.exists(path2): os.remove(path2)

elif menu == "Segera Hadir":
    st.markdown('<div class="section-title">Fitur Sedang Dikembangkan</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Laboratorium kami sedang merancang sistem optimasi cerdas tingkat lanjut.</div>', unsafe_allow_html=True)
    st.markdown("""
        <div style="background-color: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 14px; padding: 70px 20px; text-align: center; margin-top: 30px;">
            <p style="color: #64748B; font-size: 1.1rem; margin: 0; font-weight: 500;">Sistem dalam perkembangan, pantau pembaruan setiap waktu ya!</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("""
    <footer class="main-footer">
        <p style="font-weight: 800; color: #1E5E60; font-size: 1.2rem; margin-bottom: 6px;">SnapSense</p>
        <p style="font-weight: 500;">&copy; 2026 SnapSense Inc. Hak Cipta Dilindungi Undang-Undang.</p>
        <p style="font-size: 0.85rem; color: #94A3B8; margin-top: 6px; font-weight: 500;">Universitas Negeri Semarang | UNNES | Teknik Informatika</p>
    </footer>
""", unsafe_allow_html=True)