"""
data_loader.py - Đọc dữ liệu ảnh từ thư mục dataset.

Cấu trúc thư mục mong đợi:
    data/raw/
    ├── fire/       (ảnh có cháy)
    └── non_fire/   (ảnh không cháy)

Label:
    fire     = 1
    non_fire = 0
"""

import os
import cv2
import numpy as np
from src.utils import (
    DATA_RAW_DIR, CLASS_NAMES, SUPPORTED_EXTENSIONS,
    is_image_file, setup_logger, imread_unicode
)

logger = setup_logger(__name__)

# Mapping label
LABEL_MAP = {"fire": 1, "non_fire": 0}


def load_image_paths(data_dir=None):
    """
    Quét thư mục dataset và trả về danh sách (đường dẫn ảnh, label).
    
    Parameters:
        data_dir: Đường dẫn gốc chứa các thư mục con fire/ và non_fire/
                  Mặc định = DATA_RAW_DIR
    
    Returns:
        image_paths: list of str - đường dẫn tới từng ảnh
        labels: list of int - nhãn tương ứng (1=fire, 0=non_fire)
    """
    if data_dir is None:
        data_dir = DATA_RAW_DIR
    
    image_paths = []
    labels = []
    
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(data_dir, class_name)
        
        if not os.path.isdir(class_dir):
            logger.warning(f"Không tìm thấy thư mục: {class_dir}")
            continue
        
        label = LABEL_MAP[class_name]
        count = 0
        
        for filename in sorted(os.listdir(class_dir)):
            if is_image_file(filename):
                filepath = os.path.join(class_dir, filename)
                image_paths.append(filepath)
                labels.append(label)
                count += 1
        
        logger.info(f"  [{class_name}] Tìm thấy {count} ảnh")
    
    logger.info(f"Tổng cộng: {len(image_paths)} ảnh")
    return image_paths, labels


def load_single_image(image_path):
    """
    Đọc một ảnh từ đường dẫn.
    
    Returns:
        image: numpy array (BGR) hoặc None nếu lỗi
    """
    if not os.path.isfile(image_path):
        logger.error(f"File không tồn tại: {image_path}")
        return None
    
    image = imread_unicode(image_path)
    if image is None:
        logger.error(f"Không đọc được ảnh: {image_path}")
        return None
    
    return image


def load_images_from_paths(image_paths):
    """
    Đọc tất cả ảnh từ danh sách đường dẫn.
    
    Returns:
        images: list of numpy arrays
        valid_indices: list of int - chỉ số các ảnh đọc được
    """
    images = []
    valid_indices = []
    
    for i, path in enumerate(image_paths):
        img = load_single_image(path)
        if img is not None:
            images.append(img)
            valid_indices.append(i)
        else:
            logger.warning(f"Bỏ qua ảnh lỗi: {path}")
    
    logger.info(f"Đọc thành công {len(images)}/{len(image_paths)} ảnh")
    return images, valid_indices


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    print("=== Test Data Loader ===")
    paths, labels = load_image_paths()
    
    if len(paths) > 0:
        print(f"\nẢnh đầu tiên: {paths[0]}")
        print(f"Label: {labels[0]}")
        
        # Thử đọc ảnh đầu tiên
        img = load_single_image(paths[0])
        if img is not None:
            print(f"Shape: {img.shape}")
    else:
        print("\nChưa có dữ liệu. Hãy đặt ảnh vào data/raw/fire/ và data/raw/non_fire/")
