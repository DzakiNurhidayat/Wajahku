import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from facenet_pytorch import MTCNN
# import insightface
import mediapipe as mp

# Load Haar Cascade sekali saja
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# Load Models Sekali Saja
mtcnn_detector = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

# RetinaFace
# retinaface_detector = insightface.app.FaceAnalysis(name='antelopev2')
# retinaface_detector.prepare(ctx_id=0 if torch.cuda.is_available() else -1)

# Mediapipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Load SSD Mobilenet
ssd_model_path = cv2.data.haarcascades.replace("haarcascade_frontalface_default.xml", "deploy.prototxt")
ssd_weights_path = cv2.data.haarcascades.replace("haarcascade_frontalface_default.xml", "res10_300x300_ssd_iter_140000.caffemodel")
ssd_net = cv2.dnn.readNetFromCaffe(ssd_model_path, ssd_weights_path)

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
    boxes, _ = mtcnn_detector.detect(image)
    faces = []
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = image[y1:y2, x1:x2]
            face = resize_with_padding(face)
            faces.append(face)
    return faces

# 2. RetinaFace

# def detect_and_crop_face_retinaface(image):
#     faces = []
#     detections = retinaface_detector.get(image)
#     for detection in detections:
#         x1, y1, x2, y2 = map(int, detection.bbox)
#         face = image[y1:y2, x1:x2]
#         face = resize_with_padding(face)
#         faces.append(face)
#     return faces

# 3. SSD MobileNet

def detect_and_crop_face_ssdmobilenet(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),
                                 1.0, (300, 300), (104.0, 177.0, 123.0))
    ssd_net.setInput(blob)
    detections = ssd_net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            face = image[y1:y2, x1:x2]
            face = resize_with_padding(face)
            faces.append(face)
    return faces

# 4. Mediapipe Face Detection

def detect_and_crop_face_mediapipe(image):
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(img_rgb)
        faces = []
        if results.detections:
            ih, iw, _ = image.shape
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x1 = int(bboxC.xmin * iw)
                y1 = int(bboxC.ymin * ih)
                w_box = int(bboxC.width * iw)
                h_box = int(bboxC.height * ih)
                x2 = x1 + w_box
                y2 = y1 + h_box
                face = image[max(0, y1):min(ih, y2), max(0, x1):min(iw, x2)]
                face = resize_with_padding(face)
                faces.append(face)
    return faces

def detect_and_crop_face(image):
    """
    Detect and crop face from image.
    Returns processed face image resized with padding.
    """
    if isinstance(image, Image.Image):
        image_np = np.array(image.convert('RGB'))
    else:
        image_np = np.array(image)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)  # Biar deteksi lebih stabil
    )

    if len(faces) == 0:
        # Kalau tidak ada wajah terdeteksi, balikin gambar asli
        return Image.fromarray(image_np)

    # Ambil wajah terbesar (kalau ada banyak deteksi)
    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    x, y, w, h = faces[0]

    face_crop = image_np[y:y+h, x:x+w]
    face_crop = resize_with_padding(face_crop, target_size=(224, 224))

    return Image.fromarray(face_crop)

def resize_with_padding(image, target_size=(224, 224)):
    """
    Resize image with padding (border replicate) to target size.
    """
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

    padded = cv2.copyMakeBorder(
        resized,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_REPLICATE
    )
    return padded

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
