import cv2
import numpy as np
from PIL import Image
from pathlib import Path

def detect_and_crop_face(image):
    """
    Detect and crop face from image
    Returns processed image
    """
    # TODO: Implement face detection and cropping using OpenCV
    # For now just return the original image
    return image

def save_image(image, path, is_processed=False):
    """
    Save image to specified path
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    suffix = "_crop.jpg" if is_processed else ".jpg"
    save_path = Path(str(path).replace(".jpg", suffix))
    image.save(save_path)
    return save_path
