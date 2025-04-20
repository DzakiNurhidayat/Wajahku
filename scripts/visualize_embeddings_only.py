import os
import sys

# Tambahkan root proyek (satu level di atas scripts/) ke sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.face_similarity import FaceSimilarity

# Inisialisasi model dan muat model yang sudah dilatih
face_sim = FaceSimilarity(model_path='models/face_similarity_model.pth')

# Visualisasi embedding
face_sim.visualize_embeddings('data/augmented/augmented_dataset.csv')