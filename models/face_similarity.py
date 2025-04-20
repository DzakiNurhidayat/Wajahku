import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torchvision import transforms
from app.utils.image_processor import detect_crop_face, preprocess_image
import cv2
from PIL import Image

class TripletFaceDataset(Dataset):
    def __init__(self, df, augment=True):
        self.df = df
        self.names = df['nama_orang'].unique()
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
        ]) if augment else transforms.ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx, max_attempts=5):
        attempt = 0
        while attempt < max_attempts:
            try:
                anchor_name = self.df.iloc[idx]['nama_orang']
                anchor_path = self.df.iloc[idx]['path_gambar']

                positive_candidates = self.df[self.df['nama_orang'] == anchor_name]['path_gambar'].values
                if len(positive_candidates) < 2:
                    raise ValueError(f"Not enough positive samples for {anchor_name}")

                positive_path = np.random.choice([p for p in positive_candidates if p != anchor_path])

                negative_name = np.random.choice([n for n in self.names if n != anchor_name])
                negative_path = np.random.choice(self.df[self.df['nama_orang'] == negative_name]['path_gambar'].values)

                anchor = detect_crop_face(anchor_path)  
                positive = detect_crop_face(positive_path)
                negative = detect_crop_face(negative_path)

                if anchor is None or positive is None or negative is None:
                    raise ValueError("Face detection failed for one or more images")

                for img in [anchor, positive, negative]:
                    if img.shape[0] < 50 or img.shape[1] < 50:
                        raise ValueError("Image too small or distorted")

                anchor = Image.fromarray(anchor)
                positive = Image.fromarray(positive)
                negative = Image.fromarray(negative)

                if self.transform:
                    anchor = self.transform(anchor)
                    positive = self.transform(positive)
                    negative = self.transform(negative)

                return anchor, positive, negative

            except Exception as e:
                print(f"Skipping triplet at index {idx}: {e}")
                idx = (idx + 1) % len(self.df)
                attempt += 1

        raise ValueError(f"Failed to load valid triplet after {max_attempts} attempts. Check your dataset for issues.")

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = torch.sqrt(torch.sum((anchor - positive) ** 2, dim=1) + 1e-10)
        distance_negative = torch.sqrt(torch.sum((anchor - negative) ** 2, dim=1) + 1e-10)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return torch.mean(losses)

class SiameseNetwork(nn.Module):
    def __init__(self, backbone='resnet50'):
        super(SiameseNetwork, self).__init__()
        if backbone == 'resnet50':
            from torchvision.models import resnet50
            self.base_network = resnet50(weights='IMAGENET1K_V1')
            self.base_network.fc = nn.Identity()
            in_features = 2048

        self.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
        )

    def forward_one(self, x):
        x = self.base_network(x)
        x = self.fc(x)
        return x

    def forward(self, anchor, positive, negative):
        anchor_out = self.forward_one(anchor)
        positive_out = self.forward_one(positive)
        negative_out = self.forward_one(negative)
        return anchor_out, positive_out, negative_out

class FaceSimilarity:
    def __init__(self, model_path='models/face_similarity2_model.pth', device='cuda:1' if torch.cuda.is_available() else 'cpu', backbone='resnet50'):
        self.device = device
        self.model = SiameseNetwork(backbone=backbone).to(device)
        self.criterion = TripletLoss(margin=1.0)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)
        
        if os.path.exists(model_path):
            print(f"Memuat model dari {model_path}")
            self.model.load_state_dict(torch.load(model_path))
            self.model.to(device)
        else:
            print("Model tidak ditemukan, menggunakan model baru yang belum dilatih.")
        
        self.model.eval()

    def train(self, csv_path, epochs=50, batch_size=16):
        df = pd.read_csv(csv_path)
        dataset = TripletFaceDataset(df, augment=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        best_loss = float('inf')
        patience = 5
        counter = 0
        
        for epoch in range(epochs):
            total_loss = 0
            for anchor, positive, negative in dataloader:
                anchor, positive, negative = anchor.to(self.device), positive.to(self.device), negative.to(self.device)
                
                self.optimizer.zero_grad()
                anchor_out, positive_out, negative_out = self.model(anchor, positive, negative)
                loss = self.criterion(anchor_out, positive_out, negative_out)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
            
            self.scheduler.step(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                counter = 0
                os.makedirs('models', exist_ok=True)
                torch.save(self.model.state_dict(), 'models/face_similarity2_model.pth')
                torch.save(self.model.base_network.state_dict(), 'models/face_embedding2_model.pth')
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping: Loss tidak membaik setelah {} epoch".format(patience))
                    break
        
        print("Pelatihan selesai. Model disimpan di 'models/face_similarity2_model.pth'")

    def get_embedding(self, image_path):
        self.model.eval()
        
        try:
            face = detect_crop_face(image_path)  
            if face is None:
                raise ValueError("Face detection failed")

            face_pil = Image.fromarray(face)
            face = preprocess_image(face_pil)
            face = torch.tensor(face, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.model.forward_one(face)
            return embedding.cpu().numpy()[0]

        except Exception as e:
            print(f"Error in get_embedding for {image_path}: {e}")
            raise

    def visualize_embeddings(self, csv_path, output_path='results/embeddings2_tsne.png'):
        os.makedirs('results', exist_ok=True)
        df = pd.read_csv(csv_path)
        embeddings = []
        labels = []
        
        for _, row in df.iterrows():
            try:
                embedding = self.get_embedding(row['path_gambar'])
                embeddings.append(embedding)
                labels.append(row['nama_orang'])
            except ValueError as e:
                print(f"Skipping image {row['path_gambar']}: {e}")
                continue
        
        if not embeddings:
            raise ValueError("No valid embeddings generated. All images failed face detection.")
        
        embeddings = np.array(embeddings)
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=labels, palette='tab20')
        plt.title("t-SNE Visualization of Face Embeddings")
        plt.savefig(output_path)
        plt.close()