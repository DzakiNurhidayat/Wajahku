import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
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
        # Tampilkan gambar asli
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        st.image(img, caption="Gambar Asli", use_column_width=True)
        
        # Proses dan prediksi
        with st.spinner("Memproses gambar..."):
            # Crop wajah dari gambar yang diunggah
            cropped_img = detect_and_crop_face(img)
            if cropped_img is None:
                st.error("Gagal mendeteksi wajah. Pastikan gambar berisi wajah yang jelas.")
                return
            
            # Preprocess untuk ResNet50
            processed_img = preprocess_image(cropped_img)
            if processed_img is None:
                st.error("Gagal memproses gambar. Pastikan gambar valid.")
                return
            
            processed_img = np.expand_dims(processed_img, axis=0)
            prediction = model.predict(processed_img)[0]
            
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

# import streamlit as st
# import cv2
# import numpy as np
# from pathlib import Path
# from tensorflow.keras.models import load_model
# from app.utils.image_processor import detect_and_crop_face
# from app.ethnicity_classifier import preprocess_image, CLASSES

# MODEL_PATH = Path("models/ethnicity_model.h5")

# def main():
#     st.title("Prediksi Suku/Etnis")
    
#     # Load model
#     if not MODEL_PATH.exists():
#         st.error("Model belum tersedia. Harap latih model terlebih dahulu!")
#         return
    
#     model = load_model(MODEL_PATH)
    
#     # Upload gambar
#     uploaded_file = st.file_uploader("Upload foto wajah", type=["jpg", "jpeg", "png"])
    
#     if uploaded_file:
#         # Tampilkan gambar asli
#         img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
#         st.image(img, caption="Gambar Asli", use_column_width=True)
        
#         # Proses dan prediksi
#         with st.spinner("Memproses gambar..."):
#             processed_img = preprocess_image(img)
#             processed_img = np.expand_dims(processed_img, axis=0)
#             prediction = model.predict(processed_img)[0]
            
#             predicted_class = CLASSES[np.argmax(prediction)]
#             confidence = prediction[np.argmax(prediction)]
            
#             # Tampilkan hasil
#             st.subheader("Hasil Prediksi")
#             st.write(f"**Suku:** {predicted_class}")
#             st.write(f"**Confidence:** {confidence:.2f}")
            
#             # Plot confidence scores
#             st.subheader("Distribusi Confidence")
#             fig, ax = plt.subplots()
#             ax.bar(CLASSES, prediction)
#             ax.set_title("Confidence Scores")
#             ax.tick_params(axis='x', rotation=45)
#             st.pyplot(fig)

# if __name__ == "__main__":
#     main()