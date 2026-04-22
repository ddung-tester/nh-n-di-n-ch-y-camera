"""
utils.py - Các hàm tiện ích dùng chung cho toàn bộ project.

Bao gồm:
- Cấu hình đường dẫn
- Hằng số dùng chung
- Hàm hỗ trợ I/O
"""

import os
import json
import logging
from datetime import datetime
import cv2
import numpy as np

# ============================================================
# ĐƯỜNG DẪN GỐC CỦA PROJECT
# ============================================================
# Lấy thư mục cha của thư mục src/ => chính là project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Các đường dẫn con
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "reports", "results")

# ============================================================
# HẰNG SỐ CẤU HÌNH
# ============================================================
IMG_SIZE = (128, 128)           # Kích thước resize ảnh (width, height)
RANDOM_SEED = 42                # Seed cho tái hiện kết quả
TEST_SIZE = 0.2                 # Tỷ lệ tập test
CLASS_NAMES = ["non_fire", "fire"]  # Index 0 = non_fire (label 0), Index 1 = fire (label 1)

# Tên file model mặc định
DEFAULT_MODEL_NAME = "svm_fire_detector.pkl"
DEFAULT_SCALER_NAME = "scaler.pkl"

# Định dạng ảnh hỗ trợ
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ============================================================
# LOGGING
# ============================================================
def setup_logger(name="fire_detection", level=logging.INFO):
    """Tạo logger với format chuẩn."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ============================================================
# HÀM HỖ TRỢ
# ============================================================
def ensure_dir(path):
    """Tạo thư mục nếu chưa tồn tại."""
    os.makedirs(path, exist_ok=True)


def is_image_file(filename):
    """Kiểm tra file có phải ảnh hỗ trợ không."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in SUPPORTED_EXTENSIONS


def imread_unicode(filepath):
    """
    Đọc ảnh hỗ trợ đường dẫn Unicode (tiếng Việt, CJK, v.v.).
    
    cv2.imread() trên Windows không đọc được path có ký tự Unicode.
    Workaround: đọc bytes bằng numpy.fromfile() rồi decode bằng cv2.imdecode().
    
    Returns:
        image: numpy array (BGR) hoặc None nếu lỗi
    """
    try:
        data = np.fromfile(filepath, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return image
    except Exception:
        return None


def save_results(results_dict, filename="results.json"):
    """Lưu kết quả đánh giá ra file JSON."""
    ensure_dir(RESULTS_DIR)
    filepath = os.path.join(RESULTS_DIR, filename)
    
    # Thêm timestamp
    results_dict["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    return filepath


def get_model_path(model_name=None):
    """Trả về đường dẫn đầy đủ đến file model."""
    if model_name is None:
        model_name = DEFAULT_MODEL_NAME
    ensure_dir(MODELS_DIR)
    return os.path.join(MODELS_DIR, model_name)


def get_scaler_path(scaler_name=None):
    """Trả về đường dẫn đầy đủ đến file scaler."""
    if scaler_name is None:
        scaler_name = DEFAULT_SCALER_NAME
    ensure_dir(MODELS_DIR)
    return os.path.join(MODELS_DIR, scaler_name)


# ============================================================
# CẤU HÌNH MỨC ĐỘ RỦI RO (Risk Assessment) - v2.0
# ============================================================
RISK_THRESHOLDS = {
    "SAFE": 0.3,       # fire_prob < 0.3 → An toàn
    "WARNING": 0.7,    # 0.3 <= fire_prob < 0.7 → Cảnh báo
    "DANGER": 1.0,     # fire_prob >= 0.7 → Nguy hiểm
}

RISK_DISPLAY = {
    "SAFE":    {"vi": "AN TOÀN",    "emoji": "🟢", "color_bgr": (0, 200, 0)},
    "WARNING": {"vi": "CẢNH BÁO",   "emoji": "🟡", "color_bgr": (0, 200, 255)},
    "DANGER":  {"vi": "NGUY HIỂM",  "emoji": "🔴", "color_bgr": (0, 0, 255)},
}


def get_risk_level(fire_probability):
    """
    Xác định mức độ rủi ro dựa trên xác suất cháy.

    Mapping:
        fire_prob < 0.3  → SAFE    (An toàn)
        fire_prob < 0.7  → WARNING (Cảnh báo)
        fire_prob >= 0.7 → DANGER  (Nguy hiểm)

    Parameters:
        fire_probability: float [0, 1] - xác suất ảnh chứa lửa

    Returns:
        risk_level: str - "SAFE", "WARNING", hoặc "DANGER"
    """
    if fire_probability < RISK_THRESHOLDS["SAFE"]:
        return "SAFE"
    elif fire_probability < RISK_THRESHOLDS["WARNING"]:
        return "WARNING"
    else:
        return "DANGER"
