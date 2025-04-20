import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def analyze_dataset_distribution(csv_path):
    df = pd.read_csv(csv_path)
    print("Distribusi Parameter di Dataset Awal:")
    print("\nSuku:")
    print(df['suku'].value_counts(normalize=True))
    print("\nEkspresi:")
    print(df['ekspresi'].value_counts(normalize=True))
    print("\nSudut:")
    print(df['sudut'].value_counts(normalize=True))
    print("\nPencahayaan:")
    print(df['pencahayaan'].value_counts(normalize=True))
    return df

def split_dataset(csv_path, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=42):
    # Baca dataset
    df = pd.read_csv(csv_path)
    
    # Pastikan proporsi benar
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Proporsi split harus total 1.0"
    
    # Ambil daftar orang unik beserta sukunya
    person_suku = df[['nama_orang', 'suku']].drop_duplicates()

    # Stratify berdasarkan suku orang (bukan baris)
    train_persons, temp_persons = train_test_split(
        person_suku,
        train_size=train_ratio,
        stratify=person_suku['suku'],
        random_state=random_state
    )
    val_persons, test_persons = train_test_split(
        temp_persons,
        train_size=val_ratio / (val_ratio + test_ratio),
        stratify=temp_persons['suku'],
        random_state=random_state
    )
    
    # Bikin dataframe berdasarkan nama_orang
    train_df = df[df['nama_orang'].isin(train_persons['nama_orang'])]
    val_df = df[df['nama_orang'].isin(val_persons['nama_orang'])]
    test_df = df[df['nama_orang'].isin(test_persons['nama_orang'])]
    
    # Verifikasi bahwa semua kelas ada di setiap split
    classes = df['suku'].unique()
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        missing_classes = set(classes) - set(split_df['suku'].unique())
        if missing_classes:
            print(f"Peringatan: Kelas {missing_classes} tidak ada di {split_name}.")
    
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # Simpan ke CSV di data/
    train_df.to_csv(data_dir / 'train.csv', index=False)
    val_df.to_csv(data_dir / 'val.csv', index=False)
    test_df.to_csv(data_dir / 'test.csv', index=False)
    
    # Cetak distribusi kelas
    print("\nDistribusi kelas setelah split:")
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\n{split_name}.csv:")
        print(split_df['suku'].value_counts())
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    split_dataset("dataset/dataset.csv")