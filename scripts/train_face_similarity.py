import os
import sys

# Tambahkan root proyek (satu level di atas scripts/) ke sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.face_similarity import FaceSimilarity
# Inisialisasi dan latih model
face_sim = FaceSimilarity()
face_sim.train('data/augmented/augmented_dataset.csv', epochs=10, batch_size=32)

# Visualisasi embedding
face_sim.visualize_embeddings('data/augmented/augmented_dataset.csv')