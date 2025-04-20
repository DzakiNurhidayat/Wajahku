import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from facenet_pytorch import MTCNN
# import insightface
# import mediapipe as mp

# Load Haar Cascade sekali saja
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# Load Models Sekali Saja
mtcnn_detector = MTCNN(
    keep_all=True,
    min_face_size=15,  # Reduced to detect smaller faces
    thresholds=[0.5, 0.6, 0.7],  # Lowered thresholds to be less strict
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# RetinaFace
# retinaface_detector = insightface.app.FaceAnalysis(name='antelopev2')
# retinaface_detector.prepare(ctx_id=0 if torch.cuda.is_available() else -1)

# Mediapipe
# mp_face_detection = mp.solutions.face_detection
# mp_drawing = mp.solutions.drawing_utils

# Load SSD Mobilenet
# ssd_model_path = cv2.data.haarcascades.replace("haarcascade_frontalface_default.xml", "deploy.prototxt")
# ssd_weights_path = cv2.data.haarcascades.replace("haarcascade_frontalface_default.xml", "res10_300x300_ssd_iter_140000.caffemodel")
# ssd_net = cv2.dnn.readNetFromCaffe(ssd_model_path, ssd_weights_path)

# Fungsi padding dan resize

def resize_with_padding(image, target_size=(224, 224)):
    h, w = image.shape[:2]
    target_h, target_w = target_size
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2
    pad_right = target_w - new_w - pad_left
    pad_bottom = target_h - new_h - pad_top
    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                cv2.BORDER_REPLICATE)
    return padded

# 1. MTCNN

def detect_and_crop_face_mtcnn(image):
    """
    Detect and crop face from a NumPy array using MTCNN.
    Args:
        image (np.ndarray): Input image in RGB format.
    Returns:
        np.ndarray: Cropped and resized face as a NumPy array.
    Raises:
        ValueError: If no face is detected or the face region is invalid.
    """
    if image.shape[-1] != 3:
        raise ValueError("Input image must be in RGB format")

    boxes, _ = mtcnn_detector.detect(image)
    if boxes is None or len(boxes) == 0:
        raise ValueError("No face detected in image")

    # Take the first detected face
    x1, y1, x2, y2 = map(int, boxes[0])

    # Validate and adjust bounding box
    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1

    # Ensure coordinates are within image boundaries
    h, w = image.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    # Re-check after adjustment
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid face region detected after adjustment")

    face = image[y1:y2, x1:x2]
    face = resize_with_padding(face, target_size=(224, 224))
    return face

def detect_and_crop_face_haar(image):
    """
    Detect and crop face from a NumPy array using Haar Cascade.
    Args:
        image (np.ndarray): Input image in RGB format.
    Returns:
        np.ndarray: Cropped and resized face as a NumPy array, or None if no face is detected.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(15, 15)
    )

    if len(faces) == 0:
        return None

    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    x, y, w, h = faces[0]
    face = image[y:y+h, x:x+w]

    if face.shape[0] == 0 or face.shape[1] == 0:
        return None

    face = resize_with_padding(face, target_size=(224, 224))
    return face

def detect_crop_face(image_path):
    """
    Detect and crop face from image path using MTCNN, with fallback to Haar Cascade.
    Args:
        image_path (str): Path to the image file.
    Returns:
        np.ndarray: Processed face image as a NumPy array resized with padding.
    Raises:
        ValueError: If the image cannot be loaded or no face is detected.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")

    min_dim = 200
    h, w = image.shape[:2]
    if min(h, w) < min_dim:
        scale = min_dim / min(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    image = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
    image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Try MTCNN first
    try:
        face = detect_and_crop_face_mtcnn(image_np)
        return face
    except ValueError as e:
        print(f"MTCNN failed: {e}")

    # Fallback to Haar Cascade
    face = detect_and_crop_face_haar(image_np)
    if face is not None:
        return face

    raise ValueError(f"No face detected in image: {image_path}")

def detect_and_crop_face(image):
    """
    Detect and crop face from image.
    Returns processed face image resized with padding.
    """
    try:
        # Konversi input ke numpy array
        if isinstance(image, Image.Image):
            image_np = np.array(image.convert('RGB'))
        else:
            image_np = np.array(image)

        # Pastikan gambar tidak kosong
        if image_np.size == 0 or image_np is None:
            print("Error: Gambar kosong atau tidak valid")
            return Image.fromarray(image_np) if image_np.size > 0 else None

        # Konversi ke grayscale dengan tipe data CV_8U
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)

        # Deteksi wajah
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        if len(faces) == 0:
            print("Warning: Tidak ada wajah terdeteksi")
            return Image.fromarray(image_np)

        # Ambil wajah terbesar
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h = faces[0]

        face_crop = image_np[y:y+h, x:x+w]
        face_crop = resize_with_padding(face_crop, target_size=(224, 224))

        return Image.fromarray(face_crop)

    except Exception as e:
        print(f"Error in detect_and_crop_face: {str(e)}")
        return Image.fromarray(image_np) if image_np.size > 0 else None
    
def preprocess_image(image):
    """
    Preprocess image for Siamese Network.
    - Normalize pixel values to [0, 1].
    - Ensure image is a NumPy array with shape (224, 224, 3).
    """
    # Ensure image is a NumPy array
    if isinstance(image, Image.Image):
        image = np.array(image.convert('RGB'))
    elif not isinstance(image, np.ndarray):
        raise ValueError("Image must be a NumPy array or PIL Image")

    # Ensure image is in RGB format
    if image.shape[-1] != 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize if necessary (should already be 224x224 from detect_and_crop_face)
    if image.shape[:2] != (224, 224):
        image = resize_with_padding(image, target_size=(224, 224))

    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    return image

def save_image(image, path, is_processed=False):
    """
    Save image to specified path
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        
    if not str(path).lower().endswith(".jpg"):
        path = str(path) + ( "_crop.jpg" if is_processed else ".jpg")
    else:
        path = str(path).replace(".jpg", "_crop.jpg") if is_processed else str(path)
    
    save_path = Path(path)
    image.save(save_path)
    return save_path