import os
import warnings
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

warnings.filterwarnings("ignore")

from deepface import DeepFace

MODEL_NAME = "Facenet" 
DETECTOR_BACKEND = "opencv"  

def muat_model_terlebih_dahulu():
    try:
        dummy_img = np.zeros((112, 112, 3), dtype=np.uint8)
        DeepFace.verify(dummy_img, dummy_img, model_name=MODEL_NAME, detector_backend=DETECTOR_BACKEND, enforce_detection=False)
        return True
    except Exception as e:
        print(f"Gagal inisiasi model: {str(e)}")
        return False

def bandingkan_wajah_deep(image1_path, image2_path):
    try:
        # Ekstraksi representasi vektor embeddings wajah untuk komparasi metrik kustom
        res1 = DeepFace.represent(img_path=image1_path, model_name=MODEL_NAME, detector_backend=DETECTOR_BACKEND, enforce_detection=True)
        res2 = DeepFace.represent(img_path=image2_path, model_name=MODEL_NAME, detector_backend=DETECTOR_BACKEND, enforce_detection=True)
        
        emb1 = np.array(res1[0]["embedding"])
        emb2 = np.array(res2[0]["embedding"])
        
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        cosine_sim = float(dot_product / (norm1 * norm2))
        cosine_distance = 1.0 - cosine_sim
        euclidean_dist = float(np.linalg.norm(emb1 - emb2))
        
        # Ambang batas default model Facenet untuk metrik Cosine Distance adalah 0.40
        # Yang sepadan dengan nilai ambang batas Cosine Similarity sebesar 0.60
        threshold_model = 0.60 
        
        if cosine_sim >= threshold_model:
            similarity_percentage = round(100 - (cosine_distance / (1.0 - threshold_model)) * 30, 2)
            kesimpulan = "Kemungkinan Besar Orang Yang Sama (Identitas COCOK)"
        else:
            similarity_percentage = round(max(0, cosine_sim * 100), 2)
            if similarity_percentage > 70: 
                similarity_percentage = 68.5 
            kesimpulan = "Kemungkinan Besar Orang Berbeda (Identitas TIDAK COCOK)"
            
        return similarity_percentage, round(cosine_sim, 4), round(euclidean_dist, 4), threshold_model, kesimpulan

    except ValueError as ve:
        raise ValueError("Sistem gagal mendeteksi struktur wajah pada salah satu atau kedua foto. Pastikan posisi wajah tegak dan pencahayaan cukup.")
    except Exception as e:
        raise RuntimeError(f"Terjadi kesalahan pada internal DeepFace Engine: {str(e)}")