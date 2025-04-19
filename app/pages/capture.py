import streamlit as st
from pathlib import Path
from ..components.photo_upload import PhotoUploadComponent
from ..utils.dataset_manager import DatasetManager
from ..utils.image_processor import save_image, detect_and_crop_face

def process_photos(photo_data, user_path):
    """Process and save photos, return entries for dataset"""
    entries = []
    
    for idx, data in enumerate(photo_data, 1):
        # Save original image
        orig_path = save_image(data['image'], user_path / f"img{idx}")
        
        # Process and save cropped image
        processed_img = detect_and_crop_face(data['image'])
        crop_path = save_image(processed_img, user_path / f"img{idx}", is_processed=True)
        
        entries.append({
            'path_gambar': str(crop_path),
            'nama_orang': user_path.parent.name,
            'suku': user_path.parent.parent.name,
            'ekspresi': data['expression'].lower(),
            'sudut': data['angle'].lower(),
            'pencahayaan': data['lighting'].lower()
        })
    
    return entries

def main():
    st.title("Self Photobooth - Data Collection")
    
    # Initialize components
    upload_component = PhotoUploadComponent()
    dataset_manager = DatasetManager()
    
    # Render user input
    nama, suku = upload_component.render_user_input()
    
    # Render photo upload
    uploaded_files = upload_component.render_photo_upload()
    
    if uploaded_files:
        # Render photo preview and get metadata
        photo_data = upload_component.render_photo_preview(uploaded_files)
        
        if photo_data and st.button("Simpan Data"):
            if len(photo_data) < 4:
                st.error("Mohon upload minimal 4 foto!")
            elif not nama or not suku:
                st.error("Mohon lengkapi nama dan suku!")
            else:
                with st.spinner("Menyimpan data..."):
                    user_path = dataset_manager.create_user_folder(suku, nama)
                    entries = process_photos(photo_data, user_path)
                    dataset_manager.add_entries(entries)

                    st.success("Data berhasil disimpan!")
                    st.info(f"Data tersimpan di folder: dataset/{suku.lower()}/{nama.lower()}/")

                    st.subheader("Preview Hasil Crop:")
                    cols = st.columns(4)
                    for idx, entry in enumerate(entries[:4]):
                        with cols[idx]:
                            st.image(entry['path_gambar'], caption=f"Wajah {idx+1}", use_column_width=True)

if __name__ == "__main__":
    main()
