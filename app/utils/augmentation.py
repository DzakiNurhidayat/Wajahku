import albumentations as A
import cv2
import os
import pandas as pd
from pathlib import Path

def augment_image(image_path, output_dir, num_augmentations=3, original_labels=None):
    # Baca gambar
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Definisikan pipeline augmentasi
    transform = A.Compose([
        A.Rotate(limit=15, p=0.5),  # Rotasi ±15°
        A.HorizontalFlip(p=0.5),     # Flip horizontal
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # Brightness & contrast ±20%
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),  # Gaussian noise ringan
    ])
    
    # Buat folder output jika belum ada
    os.makedirs(output_dir, exist_ok=True)
    
    # Simpan gambar asli
    base_name = os.path.basename(image_path).split('.')[0]
    orig_path = os.path.join(output_dir, f"{base_name}_orig.jpg")
    cv2.imwrite(orig_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    # Generate gambar augmentasi
    augmented_data = [{'path_gambar': orig_path, **original_labels}]
    for i in range(num_augmentations):
        augmented = transform(image=image)['image']
        aug_path = os.path.join(output_dir, f"{base_name}_aug_{i}.jpg")
        cv2.imwrite(aug_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
        
        # Perbarui label untuk gambar augmentasi
        aug_labels = original_labels.copy()
        if aug_labels['sudut'] != 'frontal':
            aug_labels['sudut'] = 'samping' 
        aug_labels['pencahayaan'] = 'variabel' 
        augmented_data.append({'path_gambar': aug_path, **aug_labels})
    
    return augmented_data

def augment_dataset(csv_path, output_dir, num_augmentations=3):
    df = pd.read_csv(csv_path)
    augmented_data = []
    
    for _, row in df.iterrows():
        image_path = row['path_gambar']
        labels = {
            'nama_orang': row['nama_orang'],
            'suku': row['suku'],
            'ekspresi': row['ekspresi'],
            'sudut': row['sudut'],
            'pencahayaan': row['pencahayaan']
        }
        suku = row['suku']
        person = row['nama_orang']
        
        # Augmentasi gambar
        aug_data = augment_image(image_path, os.path.join(output_dir, suku, person), num_augmentations, labels)
        augmented_data.extend(aug_data)
    
    # Simpan ke CSV di output_dir
    augmented_df = pd.DataFrame(augmented_data)
    augmented_df.to_csv(os.path.join(output_dir, 'augmented_dataset.csv'), index=False)
    return augmented_df