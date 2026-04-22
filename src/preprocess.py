"""
preprocess.py - Tiền xử lý ảnh trước khi trích xuất đặc trưng.

Pipeline tiền xử lý:
1. Resize về kích thước chuẩn (128x128)
2. Giảm nhiễu bằng GaussianBlur nhẹ
3. Chuẩn hóa giá trị pixel (optional)
"""

import cv2
import numpy as np
from src.utils import IMG_SIZE, setup_logger

logger = setup_logger(__name__)


def resize_image(image, size=None):
    """
    Resize ảnh về kích thước chuẩn.
    
    Parameters:
        image: numpy array (BGR)
        size: tuple (width, height), mặc định = IMG_SIZE
    
    Returns:
        resized: numpy array
    """
    if size is None:
        size = IMG_SIZE
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def denoise_image(image, ksize=3):
    """
    Giảm nhiễu nhẹ bằng GaussianBlur.
    Giúp giảm noise mà không mất quá nhiều chi tiết.
    
    Parameters:
        image: numpy array
        ksize: kích thước kernel (số lẻ)
    
    Returns:
        denoised: numpy array
    """
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def preprocess_single(image, size=None, denoise=True):
    """
    Pipeline tiền xử lý cho một ảnh.
    
    Steps:
        1. Resize
        2. Denoise (optional)
    
    Parameters:
        image: numpy array (BGR)
        size: tuple (width, height)
        denoise: bool - có giảm nhiễu không
    
    Returns:
        processed: numpy array (BGR, đã xử lý)
    """
    # Bước 1: Resize
    processed = resize_image(image, size)
    
    # Bước 2: Giảm nhiễu
    if denoise:
        processed = denoise_image(processed)
    
    return processed


def preprocess_batch(images, size=None, denoise=True):
    """
    Tiền xử lý một batch ảnh.
    
    Parameters:
        images: list of numpy arrays
        size: tuple (width, height)
        denoise: bool
    
    Returns:
        processed_images: list of numpy arrays
    """
    processed = []
    errors = 0
    
    for i, img in enumerate(images):
        try:
            p = preprocess_single(img, size, denoise)
            processed.append(p)
        except Exception as e:
            logger.warning(f"Lỗi xử lý ảnh #{i}: {e}")
            errors += 1
    
    logger.info(f"Tiền xử lý xong: {len(processed)} ảnh OK, {errors} lỗi")
    return processed


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    print("=== Test Preprocess ===")
    
    # Tạo ảnh giả để test
    dummy = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    print(f"Input shape: {dummy.shape}")
    
    result = preprocess_single(dummy)
    print(f"Output shape: {result.shape}")
    print(f"Expected: ({IMG_SIZE[1]}, {IMG_SIZE[0]}, 3)")
