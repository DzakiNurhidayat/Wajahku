import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.face_similarity import FaceSimilarity

def evaluate_face_similarity(csv_path, model_path='models/face_similarity2_model.pth'):
    face_sim = FaceSimilarity(model_path=model_path)
    face_sim.model.eval()
    
    df = pd.read_csv(csv_path)
    pairs = []
    labels = []
    
    for name, group in df.groupby('nama_orang'):
        image_paths = group['path_gambar'].values
        for i in range(len(image_paths)):
            for j in range(i + 1, len(image_paths)):
                pairs.append((image_paths[i], image_paths[j]))
                labels.append(1)
    
    unique_names = df['nama_orang'].unique()
    for i in range(len(unique_names)):
        for j in range(i + 1, len(unique_names)):
            name1 = unique_names[i]
            name2 = unique_names[j]
            images1 = df[df['nama_orang'] == name1]['path_gambar'].values
            images2 = df[df['nama_orang'] == name2]['path_gambar'].values
            pairs.append((images1[0], images2[0]))
            labels.append(0)
    
    scores = []
    valid_labels = []
    skipped_pairs = []
    
    for idx, (img1_path, img2_path) in enumerate(pairs):
        try:
            embedding1 = face_sim.get_embedding(img1_path)
            embedding2 = face_sim.get_embedding(img2_path)
            
            embedding1 = torch.tensor(embedding1).to(face_sim.device)
            embedding2 = torch.tensor(embedding2).to(face_sim.device)
            
            distance = torch.sqrt(torch.sum((embedding1 - embedding2) ** 2, dim=0) + 1e-10)
            scores.append(distance.cpu().numpy())
            valid_labels.append(labels[idx])
        except ValueError as e:
            print(f"Skipping pair ({img1_path}, {img2_path}): {e}")
            skipped_pairs.append((img1_path, img2_path))
            continue
    
    if not scores:
        raise ValueError("No valid pairs processed. All pairs failed face detection.")
    
    with open('results/skipped_pairs2.txt', 'w') as f:
        for img1_path, img2_path in skipped_pairs:
            f.write(f"{img1_path}, {img2_path}\n")
    
    scores = np.array(scores)
    valid_labels = np.array(valid_labels)
    
    print("Distance scores range (Euclidean Distance):")
    print(f"Min: {scores.min():.4f}, Max: {scores.max():.4f}, Mean: {scores.mean():.4f}")
    
    pos_scores = scores[valid_labels == 1]
    neg_scores = scores[valid_labels == 0]
    plt.figure()
    plt.hist(pos_scores, bins=30, alpha=0.5, label='Positive Pairs', color='blue')
    plt.hist(neg_scores, bins=30, alpha=0.5, label='Negative Pairs', color='red')
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Frequency')
    plt.title('Euclidean Distance Distribution for Positive and Negative Pairs')
    plt.legend()
    plt.savefig('results/euclidean_distance_distribution2.png')
    plt.close()
    
    fpr, tpr, thresholds = roc_curve(valid_labels, -scores)
    roc_auc = auc(fpr, tpr)
    
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_idx]
    
    selected_threshold = 0.7
    predictions = (scores <= selected_threshold).astype(int)
    
    true_positives = np.sum((predictions == 1) & (valid_labels == 1))
    false_positives = np.sum((predictions == 1) & (valid_labels == 0))
    true_negatives = np.sum((predictions == 0) & (valid_labels == 0))
    false_negatives = np.sum((predictions == 0) & (valid_labels == 1))
    
    tar = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    far = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
    frr = false_negatives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    precision, recall, f1, _ = precision_recall_fscore_support(valid_labels, predictions, average='binary')
    
    os.makedirs('results', exist_ok=True)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('results/roc_curve2.png')
    plt.close()
    
    with open('results/evaluation_metrics2.txt', 'w') as f:
        f.write("Evaluasi Face Similarity:\n")
        f.write(f"True Acceptance Rate (TAR): {tar:.4f}\n")
        f.write(f"False Acceptance Rate (FAR): {far:.4f}\n")
        f.write(f"False Rejection Rate (FRR): {frr:.4f}\n")
        f.write(f"Equal Error Rate (EER): {eer:.4f}\n")
        f.write(f"Area Under Curve (AUC): {roc_auc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"Selected Threshold: {selected_threshold:.4f}\n")
    
    # Print hasil evaluasi
    print("\nEvaluasi Face Similarity (using threshold 0.7):")
    print(f"True Acceptance Rate (TAR): {tar:.4f}")
    print(f"False Acceptance Rate (FAR): {far:.4f}")
    print(f"False Rejection Rate (FRR): {frr:.4f}")
    print(f"Equal Error Rate (EER): {eer:.4f}")
    print(f"Area Under Curve (AUC): {roc_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Selected Threshold: {selected_threshold:.4f}")

if __name__ == "__main__":
    evaluate_face_similarity('data/test.csv')