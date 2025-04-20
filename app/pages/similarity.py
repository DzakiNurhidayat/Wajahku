import streamlit as st
import cv2
import os
import tempfile
from torchvision import transforms
import numpy as np
from PIL import Image
from app.utils.image_processor import detect_crop_face, resize_with_padding
from models.face_similarity import FaceSimilarity
import torch

def process_images(image, target_size=(224, 224)):
    if isinstance(image, Image.Image):
        image = np.array(image.convert('RGB'))
    elif isinstance(image, np.ndarray) and image.shape[2] == 4:  
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
        tmp_path = tmp_file.name
        cv2.imwrite(tmp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    try:
        processed_image = detect_crop_face(tmp_path)
        if processed_image is None:
            st.error("Deteksi wajah gagal untuk gambar ini.")
            return None

        processed_image = resize_with_padding(processed_image, target_size)

        return processed_image
    except Exception as e:
        st.error(f"Error saat memproses gambar: {str(e)}")
        return None
    finally:
        os.unlink(tmp_path)

def preprocess_images(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image) 
    return image  

def get_embeddings(image):
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    
    try:
        if not isinstance(image, np.ndarray):
            raise ValueError("Input harus berupa array NumPy")
        if image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("Gambar harus berupa array NumPy RGB dengan dtype uint8")

        face = Image.fromarray(image)

        face = preprocess_images(face)  
        face = face.unsqueeze(0) 
        face = face.to(device) 

        model_path = "models/face_similarity2_model.pth"
        face_sim = FaceSimilarity(model_path=model_path, device=device)
        face_sim.model.eval()

        with torch.no_grad():
            embedding = face_sim.model.forward_one(face)
        return embedding.cpu().numpy()[0]

    except Exception as e:
        print(f"Error dalam get_embeddings: {e}")
        raise

def compute_similarity_score(distance, scale=1.0):
    similarity_score = np.exp(-distance / scale)
    return similarity_score

def main():
    st.title("Deteksi Kemiripan Wajah")

    threshold = 0.7
    
    input_option = st.radio("Pilih metode input:", ("Unggah dua foto", "Gunakan webcam dan unggah satu foto"))

    if input_option == "Unggah dua foto":
        st.subheader("Unggah Dua Foto")
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_file1 = st.file_uploader("Unggah foto pertama", type=["jpg", "jpeg", "png"], key="file1")
            if uploaded_file1:
                image1 = Image.open(uploaded_file1)
                st.image(image1, caption="Foto Pertama", use_column_width=True)
                processed_image1 = process_images(image1)
                if processed_image1 is not None:
                    st.image(processed_image1, caption="Foto Pertama yang Diproses (224x224)", use_column_width=True)
        
        with col2:
            uploaded_file2 = st.file_uploader("Unggah foto kedua", type=["jpg", "jpeg", "png"], key="file2")
            if uploaded_file2:
                image2 = Image.open(uploaded_file2)
                st.image(image2, caption="Foto Kedua", use_column_width=True)
                processed_image2 = process_images(image2)
                if processed_image2 is not None:
                    st.image(processed_image2, caption="Foto Kedua yang Diproses (224x224)", use_column_width=True)

        if uploaded_file1 and uploaded_file2 and processed_image1 is not None and processed_image2 is not None:
            if st.button("Bandingkan Wajah"):
                try:
                    embedding1 = get_embeddings(processed_image1)
                    embedding2 = get_embeddings(processed_image2)
                    
                    distance = np.sqrt(np.sum((embedding1 - embedding2) ** 2))
                    
                    similarity_score = compute_similarity_score(distance, scale=1.0)
                    
                    is_similar = distance <= threshold
                    
                    st.subheader("Hasil Kemiripan")
                    st.write(f"Skor kemiripan (0-1): {similarity_score:.4f}")
                    st.write(f"Jarak Euclidean: {distance:.4f}")
                    st.write(f"Ambang Batas Kemiripan (Euclidean Distance): {threshold:.4f}")
                    if is_similar:
                        st.success("Wajah mirip!")
                    else:
                        st.error("Wajah tidak mirip.")
                except Exception as e:
                    st.error(f"Error saat menghitung kemiripan: {str(e)}")

    else:
        st.subheader("Gunakan Webcam dan Unggah Satu Foto")
        col1, col2 = st.columns(2)
        
        with col1:
            webcam_image = st.camera_input("Ambil foto dari webcam")
            if webcam_image:
                image1 = Image.open(webcam_image)
                st.image(image1, caption="Foto Webcam", use_column_width=True)
                processed_image1 = process_images(image1)
                if processed_image1 is not None:
                    st.image(processed_image1, caption="Foto Webcam yang Diproses (224x224)", use_column_width=True)
        
        with col2:
            uploaded_file = st.file_uploader("Unggah foto kedua", type=["jpg", "jpeg", "png"], key="file3")
            if uploaded_file:
                image2 = Image.open(uploaded_file)
                st.image(image2, caption="Foto yang Diunggah", use_column_width=True)
                processed_image2 = process_images(image2)
                if processed_image2 is not None:
                    st.image(processed_image2, caption="Foto yang Diunggah Diproses (224x224)", use_column_width=True)

        if webcam_image and uploaded_file and processed_image1 is not None and processed_image2 is not None:
            if st.button("Bandingkan Wajah"):
                try:
                    embedding1 = get_embeddings(processed_image1)
                    embedding2 = get_embeddings(processed_image2)
                    
                    distance = np.sqrt(np.sum((embedding1 - embedding2) ** 2))
                    
                    similarity_score = compute_similarity_score(distance, scale=1.0)
                    
                    is_similar = distance <= threshold
                    
                    st.subheader("Hasil Kemiripan")
                    st.write(f"Skor kemiripan (0-1): {similarity_score:.4f}")
                    st.write(f"Jarak Euclidean: {distance:.4f}")
                    st.write(f"Ambang Batas Kemiripan (Euclidean Distance): {threshold:.4f}")
                    if is_similar:
                        st.success("Wajah mirip!")
                    else:
                        st.error("Wajah tidak mirip.")
                except Exception as e:
                    st.error(f"Error saat menghitung kemiripan: {str(e)}")

if __name__ == "__main__":
    main()