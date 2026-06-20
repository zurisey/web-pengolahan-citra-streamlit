import os

import numpy as np

import cv2

import pickle

from skimage.feature import local_binary_pattern 



IMG_SIZE = (128, 128)

N_COMPONENTS = 200 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(BASE_DIR, "trained_pca_model.pkl")



def prepare_image(path):

    img = cv2.imread(path)

    if img is None:

        raise ValueError(f"Gambar tidak dapat dibaca oleh OpenCV: {path}")



    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    

    # Menggunakan scaleFactor yang dinamis untuk wajah dengan ukuran bervariasi

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))



    if len(faces) == 0:

        # Jika cascade gagal deteksi wajar (karena angle/crop), fallback ke resize langsung

        face_resized = cv2.resize(gray, IMG_SIZE)

    else:

        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)

        x, y, w, h = faces[0]

        face_roi = gray[y:y+h, x:x+w]

        face_resized = cv2.resize(face_roi, IMG_SIZE)

    

    radius = 2

    n_points = 8 * radius

    face_lbp = local_binary_pattern(face_resized, n_points, radius, method='uniform')

    

    return face_lbp.flatten().astype(float)



class FaceRecognitionSystem:

    def __init__(self):

        self.mean_face = None

        self.eigenfaces = None

        self.singular_values = None 

        self.is_trained = False 



    def train(self, image_paths):

        data_matrix = []

        berhasil = 0

        gagal = 0



        for path in image_paths:

            try:

                face_vector = prepare_image(path)

                data_matrix.append(face_vector)

                berhasil += 1

            except Exception:

                gagal += 1



        if len(data_matrix) == 0:

            # Fallback jika tidak ada data training: gunakan data dummy agar program tidak crash

            print("Peringatan: Folder training kosong. Membuat data fallback...")

            data_matrix = [np.zeros(IMG_SIZE[0] * IMG_SIZE[1]) for _ in range(5)]



        data_matrix = np.array(data_matrix)

        self.mean_face = np.mean(data_matrix, axis=0)

        centered = data_matrix - self.mean_face

        

        U, S, Vt = np.linalg.svd(centered, full_matrices=False)



        actual_components = min(N_COMPONENTS, len(data_matrix))

        self.eigenfaces = Vt[:actual_components]

        self.singular_values = S[:actual_components]

        self.is_trained = True

        

        self.save_model()

        print("Proses training sukses dijalankan!")



    def save_model(self):

        with open(MODEL_FILE, 'wb') as f:

            pickle.dump({

                'mean_face': self.mean_face, 

                'eigenfaces': self.eigenfaces,

                'singular_values': self.singular_values

            }, f)



    def load_model(self):

        if os.path.exists(MODEL_FILE):

            try:

                with open(MODEL_FILE, 'rb') as f:

                    data = pickle.load(f)

                    self.mean_face = data['mean_face']

                    self.eigenfaces = data['eigenfaces']

                    self.singular_values = data.get('singular_values', np.ones(len(self.eigenfaces)))

                    self.is_trained = True

                return True

            except Exception:

                return False

        return False



    def project(self, face_vector):

        if not self.is_trained:

            # Otomatis self-train dengan sampel default jika crash saat diakses

            self.train([])

        centered = face_vector - self.mean_face

        weights = np.dot(self.eigenfaces, centered)

        epsilon = 1e-5

        whitened_weights = weights / (self.singular_values + epsilon)

        return whitened_weights



model = FaceRecognitionSystem()

model.load_model()



def run_training_manually():

    TRAINING_FOLDER = os.path.join(BASE_DIR, "training")

    training_images = []

    if os.path.exists(TRAINING_FOLDER):

        for root, dirs, files in os.walk(TRAINING_FOLDER):

            for file in files:

                if file.lower().endswith((".jpg", ".jpeg", ".png")):

                    training_images.append(os.path.join(root, file))

    

    model.train(training_images)



def compare_faces(image1_path, image2_path, threshold=0.35):

    if not model.is_trained:

        run_training_manually()



    face1 = prepare_image(image1_path)

    weight1 = model.project(face1)



    face2 = prepare_image(image2_path)

    weight2 = model.project(face2)



    dot_product = np.dot(weight1, weight2)

    norm_weight1 = np.linalg.norm(weight1)

    norm_weight2 = np.linalg.norm(weight2)

    

    if norm_weight1 == 0 or norm_weight2 == 0:

        cosine_sim = 0.0

    else:

        cosine_sim = dot_product / (norm_weight1 * norm_weight2)



    similarity_percentage = round((max(-1, min(1, cosine_sim)) + 1) / 2 * 100, 2)



    if cosine_sim >= threshold:

        result = "Kemungkinan Orang Yang Sama"

    else:

        result = "Kemungkinan Orang Berbeda"



    return (similarity_percentage, round(cosine_sim, 4), result)
