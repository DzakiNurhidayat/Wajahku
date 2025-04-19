import argparse
import os
import sys
from pathlib import Path

# Sesuaikan path untuk import relatif
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.utils.dataset_split import split_dataset, analyze_dataset_distribution
from app.utils.augmentation import augment_dataset

def step_analyze_dataset(csv_path):
    """Langkah 1: Analisis distribusi dataset awal."""
    print(f"\n=== Analisis Dataset: {csv_path} ===")
    df = analyze_dataset_distribution(csv_path)
    return df

def step_split_dataset(csv_path):
    """Langkah 2: Split dataset menjadi training, validation, dan testing."""
    print("\n=== Splitting Dataset ===")
    train_df, val_df, test_df = split_dataset(csv_path)
    print(f"Training: {len(train_df)} images")
    print(f"Validation: {len(val_df)} images")
    print(f"Testing: {len(test_df)} images")
    return train_df, val_df, test_df

def step_augment_dataset(train_csv_path, output_dir, num_augmentations):
    """Langkah 3: Augmentasi training set."""
    print(f"\n=== Augmenting Training Dataset: {train_csv_path} ===")
    augmented_df = augment_dataset(train_csv_path, output_dir, num_augmentations=num_augmentations)
    print(f"Augmented training images: {len(augmented_df)}")
    return augmented_df

def step_analyze_augmented_dataset(augmented_csv_path):
    """Langkah 4: Analisis distribusi dataset augmentasi."""
    print(f"\n=== Analisis Augmented Dataset: {augmented_csv_path} ===")
    augmented_df = analyze_dataset_distribution(augmented_csv_path)
    return augmented_df

def main():
    # Parser untuk argumen baris perintah
    parser = argparse.ArgumentParser(description="Pipeline untuk pengolahan dataset pengenalan wajah dan deteksi suku.")
    parser.add_argument('--step', type=str, choices=['all', 'analyze', 'split', 'augment', 'analyze_augmented'],
                        default='all', help="Langkah yang akan dijalankan")
    parser.add_argument('--csv_path', type=str, default='dataset/dataset.csv', help="Path ke dataset CSV")
    parser.add_argument('--augmented_dir', type=str, default='data/augmented', help="Direktori output augmentasi")
    parser.add_argument('--num_augmentations', type=int, default=3, help="Jumlah augmentasi per gambar")
    args = parser.parse_args()

    # Konversi path relatif ke path absolut berdasarkan root proyek
    project_root = Path(__file__).parent.parent  # Naik 1 level dari scripts/ ke root
    csv_path = project_root / args.csv_path
    augmented_dir = project_root / args.augmented_dir
    train_csv_path = project_root / 'data/train.csv'
    augmented_csv_path = augmented_dir / 'augmented_dataset.csv'

    # Pastikan path valid
    if not csv_path.exists():
        print(f"Error: File {csv_path} tidak ditemukan")
        return

    # Jalankan langkah berdasarkan argumen
    try:
        if args.step == 'all' or args.step == 'analyze':
            step_analyze_dataset(str(csv_path))
        
        if args.step == 'all' or args.step == 'split':
            step_split_dataset(str(csv_path))
        
        if args.step == 'all' or args.step == 'augment':
            if not train_csv_path.exists():
                print(f"Error: File {train_csv_path} tidak ditemukan. Jalankan 'split' terlebih dahulu.")
                return
            step_augment_dataset(str(train_csv_path), str(augmented_dir), args.num_augmentations)
        
        if args.step == 'all' or args.step == 'analyze_augmented':
            if not augmented_csv_path.exists():
                print(f"Error: File {augmented_csv_path} tidak ditemukan. Jalankan 'augment' terlebih dahulu.")
                return
            step_analyze_augmented_dataset(str(augmented_csv_path))
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()