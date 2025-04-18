import streamlit as st
from PIL import Image

class PhotoUploadComponent:
    def __init__(self):
        self.ekspresi_options = ["Netral", "Senyum", "Sedih", "Marah", "Terkejut", 
                                "Tertawa", "Serius", "Cemberut", "Sinis"]
        self.sudut_options = ["Frontal", "Profil", "Miring", "Atas", "Bawah"]
        self.pencahayaan_options = ["Terang", "Normal", "Gelap"]
        self.suku_options = ["Sunda", "Jawa", "Batak", "Padang", "Palembang", "Cina"]
    
    def render_user_input(self):
        """Render name and ethnicity input fields"""
        col1, col2 = st.columns(2)
        with col1:
            nama = st.text_input("Nama")
        with col2:
            suku = st.selectbox("Suku", self.suku_options)
        return nama, suku
    
    def render_photo_upload(self):
        """Render photo upload section"""
        uploaded_files = st.file_uploader(
            "Upload 4 foto dengan ekspresi berbeda",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
        return uploaded_files
    
    def render_photo_preview(self, uploaded_files):
        """Render photo preview and metadata input"""
        if not uploaded_files:
            return None
        
        st.subheader("Photo Preview")
        cols = st.columns(4)
        photo_data = []
        
        for idx, file in enumerate(uploaded_files[:4]):
            with cols[idx]:
                img = Image.open(file)
                st.image(img, caption=f"Photo {idx+1}", use_column_width=True)
                
                photo_data.append({
                    'image': img,
                    'expression': st.selectbox(
                        f"Ekspresi {idx+1}",
                        self.ekspresi_options,
                        key=f"expr_{idx}"
                    ),
                    'angle': st.selectbox(
                        f"Sudut {idx+1}",
                        self.sudut_options,
                        key=f"angle_{idx}"
                    ),
                    'lighting': st.selectbox(
                        f"Pencahayaan {idx+1}",
                        self.pencahayaan_options,
                        key=f"light_{idx}"
                    )
                })
        
        return photo_data
