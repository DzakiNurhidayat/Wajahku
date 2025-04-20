import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings
import streamlit as st
from app.pages.capture import main as capture_page
from app.pages.predict import main as predict_page

def main():
    st.sidebar.title("Navigasi")
    page = st.sidebar.radio("Pilih Halaman", ["Pengumpulan Data", "Prediksi Suku"])
    
    if page == "Pengumpulan Data":
        capture_page()
    elif page == "Prediksi Suku":
        predict_page()

if __name__ == "__main__":
    main()
