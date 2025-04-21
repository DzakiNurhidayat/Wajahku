# Wajahku - Face Similarity System

Wajahku adalah sebuah sistem berbasis deep learning untuk mendeteksi wajah, menghitung tingkat kemiripan antar wajah menggunakan jaringan Siamese Network, dan mendeteksi suku.

## Fitur
- Deteksi wajah menggunakan MTCNN dan Haar Cascade sebagai fallback.
- Ekstraksi embedding wajah menggunakan Siamese Network berbasis ResNet50.
- Perhitungan kemiripan wajah menggunakan Triplet Loss.
- Visualisasi embedding wajah menggunakan t-SNE.

---

## Cara Setup

### 1. Clone Repository
Clone repository ini ke komputer Anda:
```bash
git clone https://github.com/DzakiNurhidayat/Wajahku.git
cd wajahku
```

### 2. Buat Virtual Environtment
Buat dan aktifkan virtual environtment untuk mengelola dependensi:
```bash
python -m venv venv
# Aktifkan Virtual Environtment
.\venv\Scripts\activate
```

### 3. Install Dependensi
```bash
pip install -r requirement.txt
```

## Cara Menjalankan Sistem

### 1. Memasukkan folder dataset yang telah diberikan melalui zip
- Extract folder yang telah dikirim
- Hasil extract tersebut dimasukkan ke dalam folder dataset

### 2. Menjalankan untuk membuat split dan augmentasi
```bash
python scripts/prepare_data.py --step analyze
python scripts/prepare_data.py --step split
python scripts/prepare_data.py --step augment
python scripts/prepare_data.py --step analyze_augmented
```

### 3. Menjalankan untuk melatih dan menilai data
```bash
python scripts/train_face_similarity.py
python scripts/evaluate_face_similarity.py
python -m app.utils.ethnicity_classifier #train dan evaluasi
```
