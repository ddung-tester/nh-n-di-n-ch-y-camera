"""
features.py - Trích xuất đặc trưng từ ảnh cho SVM.

Phương pháp chính: HOG + Color Histogram (HSV)

Lý do chọn:
- HOG (Histogram of Oriented Gradients):
  + Bắt được cấu trúc cạnh, texture của ngọn lửa
  + Là phương pháp kinh điển, dễ giải thích trong báo cáo
  + Hoạt động tốt với SVM
  
- Color Histogram (HSV):
  + Lửa có màu đặc trưng (đỏ, cam, vàng) trong không gian HSV
  + Bổ sung thông tin màu sắc mà HOG không có
  + HSV tách riêng kênh màu và độ sáng, phù hợp hơn RGB

Kết hợp 2 loại đặc trưng cho vector đặc trưng phong phú hơn.
"""

import cv2
import numpy as np
from skimage.feature import hog
from src.utils import IMG_SIZE, setup_logger

logger = setup_logger(__name__)

# ============================================================
# CẤU HÌNH HOG
# ============================================================
HOG_PARAMS = {
    "orientations": 9,          # Số hướng gradient
    "pixels_per_cell": (16, 16), # Kích thước cell
    "cells_per_block": (2, 2),   # Số cell mỗi block
    "block_norm": "L2-Hys",      # Chuẩn hóa block
    "feature_vector": True,      # Trả về vector 1D
}

# Cấu hình Color Histogram
HIST_BINS = [16, 16, 16]  # Số bin cho H, S, V
HIST_RANGES = [0, 180, 0, 256, 0, 256]  # Range cho H, S, V


def extract_hog(image):
    """
    Trích xuất đặc trưng HOG từ ảnh.
    
    HOG hoạt động trên ảnh grayscale.
    
    Parameters:
        image: numpy array (BGR, đã resize)
    
    Returns:
        hog_features: numpy array 1D
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(gray, **HOG_PARAMS)
    return features


def extract_color_histogram(image):
    """
    Trích xuất color histogram trong không gian HSV.
    
    HSV phù hợp cho phát hiện lửa vì:
    - H (Hue): lửa nằm trong vùng 0-30 (đỏ-cam-vàng)
    - S (Saturation): lửa thường có saturation cao
    - V (Value): lửa thường sáng
    
    Parameters:
        image: numpy array (BGR, đã resize)
    
    Returns:
        hist_features: numpy array 1D (đã chuẩn hóa)
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Tính histogram 3D cho cả 3 kênh
    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],         # 3 kênh H, S, V
        None,               # Không dùng mask
        HIST_BINS,           # Số bin
        HIST_RANGES          # Range
    )
    
    # Chuẩn hóa histogram
    cv2.normalize(hist, hist)
    
    return hist.flatten()


def extract_features(image):
    """
    Trích xuất TOÀN BỘ đặc trưng cho một ảnh.
    Kết hợp HOG + Color Histogram.
    
    Parameters:
        image: numpy array (BGR, đã resize)
    
    Returns:
        features: numpy array 1D (HOG nối Color Hist)
    """
    hog_feat = extract_hog(image)
    color_feat = extract_color_histogram(image)
    
    # Nối 2 vector đặc trưng
    combined = np.concatenate([hog_feat, color_feat])
    return combined


def extract_features_batch(images):
    """
    Trích xuất đặc trưng cho batch ảnh.
    
    Parameters:
        images: list of numpy arrays (BGR, đã resize)
    
    Returns:
        X: numpy array shape (n_samples, n_features)
    """
    features_list = []
    errors = 0
    
    for i, img in enumerate(images):
        try:
            feat = extract_features(img)
            features_list.append(feat)
        except Exception as e:
            logger.warning(f"Lỗi trích xuất đặc trưng ảnh #{i}: {e}")
            errors += 1
    
    if len(features_list) == 0:
        logger.error("Không trích xuất được đặc trưng nào!")
        return np.array([])
    
    X = np.array(features_list)
    logger.info(f"Trích xuất đặc trưng xong: {X.shape[0]} mẫu, {X.shape[1]} đặc trưng")
    
    return X


def get_feature_info():
    """
    Trả về thông tin về vector đặc trưng.
    Hữu ích cho báo cáo / debug.
    """
    # Tính kích thước HOG
    # Với ảnh 128x128, cell 16x16 => 8x8 cells
    # Block 2x2 cells => 7x7 blocks (trượt)
    # Mỗi block: 2*2*9 = 36 features
    # Tổng HOG: 7*7*36 = 1764
    
    h, w = IMG_SIZE[1], IMG_SIZE[0]
    cpx, cpy = HOG_PARAMS["pixels_per_cell"]
    bx, by = HOG_PARAMS["cells_per_block"]
    orient = HOG_PARAMS["orientations"]
    
    n_cells_x = w // cpx
    n_cells_y = h // cpy
    n_blocks_x = n_cells_x - bx + 1
    n_blocks_y = n_cells_y - by + 1
    hog_size = n_blocks_x * n_blocks_y * bx * by * orient
    
    # Kích thước Color Histogram
    hist_size = 1
    for b in HIST_BINS:
        hist_size *= b
    
    total = hog_size + hist_size
    
    return {
        "hog_size": hog_size,
        "color_hist_size": hist_size,
        "total_features": total,
        "img_size": IMG_SIZE,
        "hog_params": HOG_PARAMS,
        "hist_bins": HIST_BINS,
    }


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    print("=== Test Feature Extraction ===")
    
    # Thông tin đặc trưng
    info = get_feature_info()
    print(f"\nVector đặc trưng:")
    print(f"  HOG: {info['hog_size']} chiều")
    print(f"  Color Histogram: {info['color_hist_size']} chiều")
    print(f"  Tổng: {info['total_features']} chiều")
    
    # Test với ảnh giả
    dummy = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    feat = extract_features(dummy)
    print(f"\nTest extraction:")
    print(f"  Input shape: {dummy.shape}")
    print(f"  Feature vector: {feat.shape}")
