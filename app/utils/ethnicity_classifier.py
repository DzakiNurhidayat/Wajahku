import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image

# Path konfigurasi
MODEL_PATH = Path("models/ethnicity_model.keras")
OUTPUT_PATH = Path("outputs")
OUTPUT_PATH.mkdir(exist_ok=True)

# Kelas suku
CLASSES = ["Sunda", "Jawa", "Batak", "Padang", "Palembang", "Cina"]

def preprocess_image(image):
    """
    Preprocess image for ResNet50 input.
    """
    try:
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        elif isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Pastikan gambar valid dan berukuran 224x224
        if image.size == 0 or image.shape[0] != 224 or image.shape[1] != 224:
            print(f"Error: Gambar tidak valid atau ukuran salah ({image.shape})")
            return None
        
        # Pastikan tipe data adalah uint8
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        # Normalisasi untuk ResNet50
        image = image / 255.0
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return image
    
    except Exception as e:
        print(f"Error in preprocess_image: {str(e)}")
        return None

def create_model(num_classes=len(CLASSES)):
    """
    Create ResNet50-based model for ethnicity classification.
    """
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze base model untuk pelatihan awal
    for layer in base_model.layers:
        layer.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_data(csv_path, target_size=(224, 224)):
    """
    Load dataset from CSV and create data generators.
    """
    df = pd.read_csv(csv_path)
    
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_image,
        rotation_range=15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2]
    )
    
    generator = datagen.flow_from_dataframe(
        dataframe=df,
        x_col='path_gambar',
        y_col='suku',
        target_size=target_size,
        batch_size=32,
        class_mode='categorical',
        shuffle=True,
        validate_filenames=True
    )
    return generator

def train_model():
    """
    Train ethnicity classification model with initial training and fine-tuning.
    """
    # Load data generators
    train_generator = load_data('data/train.csv')
    val_generator = load_data('data/val.csv')
    
    # Pastikan semua kelas ada
    if len(train_generator.class_indices) != len(CLASSES) or len(val_generator.class_indices) != len(CLASSES):
        raise ValueError("Tidak semua kelas suku ada di dataset pelatihan atau validasi. Periksa distribusi kelas.")
    
    # Buat model
    model = create_model()
    
    # Callback untuk menyimpan model terbaik
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(MODEL_PATH), save_best_only=True)
    ]
    
    # Pelatihan awal
    print("Melatih model dengan lapisan beku...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=20,
        callbacks=callbacks
    )
    
    # Fine-tuning: Unfreeze 10 lapisan terakhir
    print("Fine-tuning model...")
    for layer in model.layers[-10:]:
        layer.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    history_fine = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10,
        callbacks=callbacks
    )
    
    # Simpan plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'] + history_fine.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'] + history_fine.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'] + history_fine.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'] + history_fine.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.savefig(OUTPUT_PATH / 'training_history.png')
    plt.close()
    
    return model

def save_misclassified_images(generator, y_true, y_pred, n=5):
    """
    Save top-N misclassified images for analysis.
    """
    errors = np.where(y_true != y_pred)[0]
    if len(errors) == 0:
        print("Tidak ada kesalahan klasifikasi.")
        return
    print(f"Top-{min(n, len(errors))} gambar yang salah diklasifikasi:")
    for i, idx in enumerate(errors[:n]):
        img_path = generator.filenames[idx]
        img = cv2.imread(img_path)
        plt.figure(figsize=(5, 5))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f'True: {CLASSES[y_true[idx]]}, Pred: {CLASSES[y_pred[idx]]}')
        plt.axis('off')
        plt.savefig(OUTPUT_PATH / f'misclassified_{i}.png')
        plt.close()
        print(f"Gambar: {img_path}, True: {CLASSES[y_true[idx]]}, Pred: {CLASSES[y_pred[idx]]}")

def visualize_tsne(model, generator):
    """
    Visualize t-SNE of face embeddings.
    """
    generator.reset()
    # Buat model untuk ekstraksi embedding (sebelum softmax)
    embedding_model = Model(inputs=model.input, outputs=model.layers[-3].output)  # Ambil lapisan Dense(1024)
    embeddings = embedding_model.predict(generator)
    # Gunakan perplexity kecil karena dataset test kecil
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10, 8))
    for i, cls in enumerate(CLASSES):
        idx = np.where(generator.classes == i)[0]
        plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], label=cls, alpha=0.5)
    plt.legend()
    plt.title('t-SNE Visualization of Face Embeddings')
    plt.savefig(OUTPUT_PATH / 'tsne_embeddings.png')
    plt.close()

def evaluate_model(model):
    """
    Evaluate model on test set and visualize results.
    """
    test_generator = load_data('data/test.csv')
    test_generator.reset()
    
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # Classification report
    print(classification_report(y_true, y_pred, target_names=CLASSES))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(OUTPUT_PATH / 'confusion_matrix.png')
    plt.close()
    
    # ROC-AUC (One-vs-Rest)
    y_true_bin = label_binarize(y_true, classes=range(len(CLASSES)))
    plt.figure(figsize=(10, 8))
    for i in range(len(CLASSES)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], predictions[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{CLASSES[i]} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (One-vs-Rest)')
    plt.legend(loc='lower right')
    plt.savefig(OUTPUT_PATH / 'roc_curve.png')
    plt.close()
    
    # Simpan gambar yang salah diklasifikasi
    save_misclassified_images(test_generator, y_true, y_pred)
    
    # Visualisasi t-SNE
    visualize_tsne(model, test_generator)

def predict_image(image_path, model):
    """
    Predict ethnicity for a single image (for testing dataset images).
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Gagal membaca gambar {image_path}")
        return None, None
    
    processed_img = preprocess_image(img)
    if processed_img is None:
        print(f"Error: Gagal memproses gambar {image_path}")
        return None, None
    
    processed_img = np.expand_dims(processed_img, axis=0)
    
    prediction = model.predict(processed_img)[0]
    predicted_class = CLASSES[np.argmax(prediction)]
    confidence = prediction[np.argmax(prediction)]
    
    # Visualisasi hasil
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB))
    plt.title(f'Predicted: {predicted_class} ({confidence:.2f})')
    plt.axis('off')
    
    # Plot confidence scores
    plt.figure(figsize=(8, 4))
    plt.bar(CLASSES, prediction)
    plt.title('Confidence Scores')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / f'prediction_{Path(image_path).stem}.png')
    plt.close()
    
    return predicted_class, confidence

def main():
    """
    Main function to train and evaluate model.
    """
    # Latih model
    model = train_model()
    
    # Evaluasi model
    evaluate_model(model)
    
if __name__ == "__main__":
    main()

# import os
# import cv2
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path
# from PIL import Image
# from app.utils.image_processor import detect_and_crop_face

# # Path konfigurasi
# BASE_PATH = Path("dataset")
# MODEL_PATH = Path("models/ethnicity_model.keras")
# OUTPUT_PATH = Path("outputs")
# OUTPUT_PATH.mkdir(exist_ok=True)

# # Kelas suku
# CLASSES = ["Sunda", "Jawa", "Batak", "Padang", "Palembang", "Cina"]

# def preprocess_image(image):
#     """
#     Preprocess image for ResNet50 input.
#     """
#     if isinstance(image, Image.Image):
#         image = np.array(image.convert('RGB'))
#     elif isinstance(image, np.ndarray):
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     # Deteksi dan crop wajah
#     face = detect_and_crop_face(image)
#     face = np.array(face)
    
#     # Normalisasi untuk ResNet50
#     face = face / 255.0
#     face = (face - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
#     return face

# def create_model(num_classes=len(CLASSES)):
#     """
#     Create ResNet50-based model for ethnicity classification.
#     """
#     base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
#     # Freeze base model
#     for layer in base_model.layers:
#         layer.trainable = False
    
#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)
#     x = Dense(1024, activation='relu')(x)
#     x = Dropout(0.5)(x)
#     predictions = Dense(num_classes, activation='softmax')(x)
    
#     model = Model(inputs=base_model.input, outputs=predictions)
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# def load_data(csv_path, target_size=(224, 224)):
#     """
#     Load dataset from CSV and create data generators.
#     """
#     df = pd.read_csv(csv_path)
    
#     datagen = ImageDataGenerator(
#         preprocessing_function=preprocess_image,
#         rotation_range=15,
#         horizontal_flip=True,
#         brightness_range=[0.8, 1.2]
#     )
    
#     generator = datagen.flow_from_dataframe(
#         dataframe=df,
#         x_col='path_gambar',
#         y_col='suku',
#         target_size=target_size,
#         batch_size=32,
#         class_mode='categorical',
#         shuffle=True
#     )
#     return generator

# def train_model():
#     """
#     Train ethnicity classification model.
#     """
#     # Load data generators
#     train_generator = load_data('data/train.csv')
#     val_generator = load_data('data/val.csv')
    
#     # Buat model
#     model = create_model()
    
#     # Callback untuk menyimpan model terbaik
#     callbacks = [
#         tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
#         tf.keras.callbacks.ModelCheckpoint(str(MODEL_PATH), save_best_only=True)
#     ]
    
#     # Latih model
#     history = model.fit(
#         train_generator,
#         validation_data=val_generator,
#         epochs=20,
#         callbacks=callbacks
#     )
    
#     # Simpan plot training history
#     plt.figure(figsize=(12, 4))
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['accuracy'], label='Train Accuracy')
#     plt.plot(history.history['val_accuracy'], label='Val Accuracy')
#     plt.title('Model Accuracy')
#     plt.legend()
#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['loss'], label='Train Loss')
#     plt.plot(history.history['val_loss'], label='Val Loss')
#     plt.title('Model Loss')
#     plt.legend()
#     plt.savefig(OUTPUT_PATH / 'training_history.png')
#     plt.close()
    
#     return model

# def evaluate_model(model):
#     """
#     Evaluate model on test set and visualize results.
#     """
#     test_generator = load_data('data/test.csv')
#     test_generator.reset()
    
#     predictions = model.predict(test_generator)
#     y_pred = np.argmax(predictions, axis=1)
#     y_true = test_generator.classes
    
#     # Classification report
#     print(classification_report(y_true, y_pred, target_names=CLASSES))
    
#     # Confusion matrix
#     cm = confusion_matrix(y_true, y_pred)
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
#     plt.title('Confusion Matrix')
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.savefig(OUTPUT_PATH / 'confusion_matrix.png')
#     plt.close()

# def predict_image(image_path, model):
#     """
#     Predict ethnicity for a single image.
#     """
#     img = cv2.imread(str(image_path))
#     if img is None:
#         print(f"Error: Gagal membaca gambar {image_path}")
#         return None, None

#     processed_img = preprocess_image(img)
#     if processed_img is None:
#         print(f"Error: Gagal memproses gambar {image_path}")
#         return None, None

#     processed_img = np.expand_dims(processed_img, axis=0)
    
#     prediction = model.predict(processed_img)[0]
#     predicted_class = CLASSES[np.argmax(prediction)]
#     confidence = prediction[np.argmax(prediction)]
    
#     # Visualisasi hasil
#     plt.figure(figsize=(8, 6))
#     plt.imshow(cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB))
#     plt.title(f'Predicted: {predicted_class} ({confidence:.2f})')
#     plt.axis('off')
    
#     # Plot confidence scores
#     plt.figure(figsize=(8, 4))
#     plt.bar(CLASSES, prediction)
#     plt.title('Confidence Scores')
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.savefig(OUTPUT_PATH / f'prediction_{Path(image_path).stem}.png')
#     plt.close()
    
#     return predicted_class, confidence

# def main():
#     """
#     Main function to train and evaluate model.
#     """
#     # Latih model
#     model = train_model()
    
#     # Evaluasi model
#     evaluate_model(model)
    
#     # Contoh prediksi pada satu gambar
#     sample_image = BASE_PATH / "sunda" / "nama1" / "img1_crop.jpg"
#     if sample_image.exists():
#         predicted_class, confidence = predict_image(sample_image, model)
#         print(f"Sample prediction: {predicted_class} with confidence {confidence:.2f}")

# if __name__ == "__main__":
#     main()