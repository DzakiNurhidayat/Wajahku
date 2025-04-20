import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
from app.utils.image_processor import detect_and_crop_face
from app.utils.ethnicity_classifier import preprocess_image, CLASSES

MODEL_PATH = Path("models/ethnicity_model.keras")

def main():
    st.title("Prediksi Suku/Etnis")
    
    # Load model
    if not MODEL_PATH.exists():
        st.error("Model belum tersedia. Harap latih model terlebih dahulu!")
        return
    
    model = load_model(MODEL_PATH)
    
    # Upload gambar
    uploaded_file = st.file_uploader("Upload foto wajah", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Baca gambar
        img = Image.open(uploaded_file).convert('RGB')
        
        # Proses dan prediksi
        with st.spinner("Memproses gambar..."):
            # Crop wajah
            cropped_img = detect_and_crop_face(img)
            if cropped_img is None:
                st.error("Gagal mendeteksi wajah. Pastikan gambar berisi wajah yang jelas.")
                return
            
            # Debugging: Simpan cropped_img untuk pemeriksaan
            cropped_img.save("debug_cropped.png")
            
            # Preprocess: alignment, histogram equalization, dan normalisasi model
            model_img = preprocess_image(cropped_img)
            if model_img is None:
                st.error("Gagal memproses gambar.")
                return

            # Gunakan cropped_img untuk visualisasi (tanpa histogram equalization tambahan)
            visual_img = cropped_img
            
            # Konversi ke PIL Image
            visual_img_pil = visual_img
            
            # Simpan sementara untuk debugging
            visual_img_pil.save("debug_normalized.png")
            
            # Tampilkan citra wajah
            st.image(visual_img_pil, caption="Wajah yang Diproses", use_column_width=True)
            
            # Prediksi
            model_img = np.expand_dims(model_img, axis=0)
            prediction = model.predict(model_img)[0]
            
            predicted_class = CLASSES[np.argmax(prediction)]
            confidence = prediction[np.argmax(prediction)]
            
            # Tampilkan hasil
            st.subheader("Hasil Prediksi")
            st.write(f"**Suku:** {predicted_class}")
            st.write(f"**Confidence:** {confidence:.2f}")
            
            # Plot confidence scores
            st.subheader("Distribusi Confidence")
            fig, ax = plt.subplots()
            ax.bar(CLASSES, prediction)
            ax.set_title("Confidence Scores")
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

if __name__ == "__main__":
    main()