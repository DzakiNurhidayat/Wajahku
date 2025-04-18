import pandas as pd
from pathlib import Path

class DatasetManager:
    def __init__(self, base_path="dataset"):
        self.base_path = Path(base_path)
        self.csv_path = self.base_path / "dataset.csv"
        self.columns = ['path_gambar', 'nama_orang', 'suku', 'ekspresi', 'sudut', 'pencahayaan']
    
    def create_user_folder(self, suku, nama):
        """Create folder structure for user data"""
        user_path = self.base_path / suku.lower() / nama.lower()
        user_path.mkdir(parents=True, exist_ok=True)
        return user_path
    
    def load_dataset(self):
        """Load existing dataset or create new one"""
        if self.csv_path.exists():
            return pd.read_csv(self.csv_path)
        return pd.DataFrame(columns=self.columns)
    
    def save_dataset(self, df):
        """Save dataset to CSV"""
        df.to_csv(self.csv_path, index=False)
    
    def add_entries(self, entries):
        """Add new entries to dataset"""
        df = self.load_dataset()
        new_df = pd.concat([df, pd.DataFrame(entries)], ignore_index=True)
        self.save_dataset(new_df)
