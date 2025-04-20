import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings
import streamlit as st
from app.pages.capture import main as capture_page
from app.pages.similarity import main as face_similarity_page
# from app.pages.predict import main as predict_page

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select a page", ["Capture", "Face Similarity"])

    if page == "Capture":
        capture_page()
    elif page == "Face Similarity":
        face_similarity_page()
    # elif page == "Prediksi Suku":
    #     predict_page()

if __name__ == "__main__":
    main()
