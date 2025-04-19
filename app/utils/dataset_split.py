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

def split_dataset(csv_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    # Baca dataset
    df = pd.read_csv(csv_path)
    
    # Pastikan proporsi benar
    assert train_ratio + val_ratio + test_ratio == 1.0, "Proporsi split harus total 1.0"
    
    # Kolom 'person' diubah menjadi 'nama_orang' untuk konsistensi dengan struktur CSV
    unique_persons = df['nama_orang'].unique()
    train_persons, temp_persons = train_test_split(unique_persons, train_size=train_ratio, random_state=random_state)
    val_persons, test_persons = train_test_split(temp_persons, train_size=val_ratio/(val_ratio + test_ratio), random_state=random_state)
    
    # Buat dataframe untuk masing-masing split
    train_df = df[df['nama_orang'].isin(train_persons)]
    val_df = df[df['nama_orang'].isin(val_persons)]
    test_df = df[df['nama_orang'].isin(test_persons)]
    
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # Simpan ke CSV di data/
    train_df.to_csv(data_dir / 'train.csv', index=False)
    val_df.to_csv(data_dir / 'val.csv', index=False)
    test_df.to_csv(data_dir / 'test.csv', index=False)
    
    return train_df, val_df, test_df