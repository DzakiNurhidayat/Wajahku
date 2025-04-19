import cv2
import numpy as np
from PIL import Image
from pathlib import Path

# Load Haar Cascade sekali saja
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

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
        minSize=(50, 50)
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


# import cv2
# import numpy as np
# from PIL import Image
# from pathlib import Path

# # Load Haar Cascade once
# cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# face_cascade = cv2.CascadeClassifier(cascade_path)

# def resize_with_padding(image, target_size=(224, 224)):
#     h, w = image.shape[:2]
#     target_h, target_w = target_size

#     scale = min(target_w / w, target_h / h)
#     new_w = int(w * scale)
#     new_h = int(h * scale)

#     resized = cv2.resize(image, (new_w, new_h))

#     pad_left = (target_w - new_w) // 2
#     pad_top = (target_h - new_h) // 2
#     pad_right = target_w - new_w - pad_left
#     pad_bottom = target_h - new_h - pad_top

#     padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)
#     return padded

# def detect_and_crop_face(image_pil):
#     """Detect the first face and crop with padding"""
#     img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
#     gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    
#     if len(faces) == 0:
#         return image_pil  # fallback ke gambar asli

#     x, y, w, h = faces[0]
#     face_crop = img_cv[y:y+h, x:x+w]
#     resized_face = resize_with_padding(face_crop)

#     # Convert back to PIL
#     return Image.fromarray(cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB))