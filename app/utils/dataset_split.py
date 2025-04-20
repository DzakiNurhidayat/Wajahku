import pandas as pd
import random
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

def split_dataset(csv_path, random_state=42):
    random.seed(random_state)
    
    # Baca dataset
    df = pd.read_csv(csv_path)
    
    # Dataframe hasil split
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()
    test_df = pd.DataFrame()

    # Proses split per suku
    for suku, suku_group in df.groupby("suku"):
        persons = suku_group["nama_orang"].unique().tolist()
        
        if len(persons) < 5:
            raise ValueError(f"Suku '{suku}' hanya memiliki {len(persons)} orang. Minimal 5 orang diperlukan.")
        
        # Acak urutan orang
        random.shuffle(persons)
        
        train_persons = persons[:3]
        val_person = persons[3]
        test_person = persons[4]
        
        # Tambahkan ke masing-masing dataframe
        train_df = pd.concat([train_df, suku_group[suku_group["nama_orang"].isin(train_persons)]])
        val_df = pd.concat([val_df, suku_group[suku_group["nama_orang"] == val_person]])
        test_df = pd.concat([test_df, suku_group[suku_group["nama_orang"] == test_person]])

    # Simpan ke CSV
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / 'data'
    data_dir.mkdir(exist_ok=True)

    train_df.to_csv(data_dir / 'train.csv', index=False)
    val_df.to_csv(data_dir / 'val.csv', index=False)
    test_df.to_csv(data_dir / 'test.csv', index=False)

    return train_df, val_df, test_df