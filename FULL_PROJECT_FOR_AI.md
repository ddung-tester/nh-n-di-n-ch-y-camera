# FILE TỔNG HỢP DỰ ÁN - GỬI CHO AI ĐỂ VIẾT BÁO CÁO

> **Mục đích**: File này chứa TOÀN BỘ thông tin dự án (mô tả, source code, kết quả, lý thuyết) để AI đọc hiểu và viết báo cáo kết thúc môn Trí tuệ nhân tạo.

---

## THÔNG TIN CHUNG

- **Đề tài**: Xây dựng hệ thống phát hiện và cảnh báo cháy realtime từ camera sử dụng SVM kết hợp đánh giá mức độ rủi ro
- **Môn**: Trí tuệ nhân tạo
- **MSV**: B25CHHT089
- **Phiên bản**: v3.5
- **Yêu cầu**: Dễ hiểu, chạy được, giải thích được, không dùng deep learning

---

## MÔ TẢ BÀI TOÁN

- **Đầu vào chính**: Video stream realtime từ webcam
- **Đầu vào phụ**: Ảnh tĩnh (JPG, PNG, BMP, TIFF, WEBP)
- **Đầu ra**: Nhãn fire/non_fire, xác suất cháy (0.0→1.0), mức rủi ro (AN TOÀN / CẢNH BÁO / NGUY HIỂM), overlay hiển thị trên video, cảnh báo âm thanh tự động

---

## CẤU TRÚC DỰ ÁN

```
Nhận diện cháy camera/
├── data/raw/
│   ├── fire/                     ← 872 ảnh cháy
│   └── non_fire/                 ← 450 ảnh không cháy
├── models/
│   ├── svm_fire_detector.pkl     ← Model đã huấn luyện (~21.5 MB)
│   └── scaler.pkl                ← Scaler đã fit (~138 KB)
├── reports/
│   ├── figures/                  ← Biểu đồ (confusion matrix, risk analysis)
│   └── results/                  ← Kết quả JSON
├── src/
│   ├── __init__.py
│   ├── utils.py                  ← Cấu hình, hằng số, Risk Assessment, imread_unicode
│   ├── data_loader.py            ← Đọc dữ liệu ảnh
│   ├── preprocess.py             ← Tiền xử lý (resize, denoise)
│   ├── features.py               ← Trích xuất HOG + Color Histogram
│   ├── train.py                  ← Pipeline huấn luyện SVM
│   ├── evaluate.py               ← Đánh giá + Risk analysis
│   ├── predict.py                ← Dự đoán ảnh tĩnh + Risk level
│   └── camera_realtime.py        ← Nhận diện realtime từ webcam (module chính)
├── run_train.bat
├── requirements.txt
└── README.md
```

---

## PIPELINE XỬ LÝ

### Pipeline huấn luyện (offline)
```
Ảnh gốc → Resize 128×128 → GaussianBlur → HOG (1764 chiều) + HSV Color Hist (4096 chiều)
→ StandardScaler → SVM(kernel=RBF, C=10, class_weight=balanced) → Lưu model .pkl
```

### Pipeline nhận diện realtime (camera)
```
Webcam frame → Scene Change Detector (reset nếu thay đổi cảnh >30)
→ Motion Filter (bỏ qua nếu chuyển động >12)
→ Resize 128×128 → Denoise → HOG + Color Hist → Scaler → SVM predict_proba()
→ EMA + Rolling Average Smoothing
→ State Machine + Hysteresis (SAFE ↔ WARNING ↔ DANGER, cần N frame liên tiếp)
→ Overlay hiển thị (tiếng Việt, Pillow) + Cảnh báo âm thanh (thread daemon)
```

---

## KẾT QUẢ ĐÁNH GIÁ

### Tập dữ liệu
| | Fire | Non-fire | Tổng |
|---|---|---|---|
| Sau mở rộng (v3.4) | 872 | 450 | 1,322 |
| Tỷ lệ | 65.9% | 34.1% | — |
| Train (80%) | ~697 | ~360 | ~1,057 |
| Test (20%) | 175 | 90 | 265 |

### Hiệu suất phân loại
| Chỉ số | Giá trị |
|---|---|
| Accuracy | 89.43% |
| Precision (fire) | 94.01% |
| Recall (fire) | 89.71% |
| F1-Score (fire) | 91.81% |
| Precision (non_fire) | 81.63% |
| Recall (non_fire) | 88.89% |

### Confusion Matrix
```
              Predicted
              non_fire  fire
Actual  non_fire   80     10
        fire       18    157
```

### Phân bố mức rủi ro (tập test)
| Mức rủi ro | Fire | Non-fire | Tổng |
|---|---|---|---|
| AN TOÀN | 3 | 69 | 72 |
| CẢNH BÁO | 18 | 13 | 31 |
| NGUY HIỂM | 154 | 8 | 162 |

---

## THAM SỐ KỸ THUẬT

| Tham số | Giá trị | Lý do |
|---|---|---|
| IMG_SIZE | 128×128 | Đủ chi tiết cho HOG |
| HOG orientations | 9 | Chuẩn Dalal & Triggs |
| HOG pixels_per_cell | 16×16 | Cân bằng chi tiết/tốc độ |
| Color Hist bins | [16,16,16] | 4,096 chiều |
| SVM kernel | RBF | Phi tuyến |
| SVM C | 10.0 | Regularization vừa phải |
| SVM gamma | scale | Tự động theo variance |
| SVM class_weight | balanced | Xử lý lệch class |
| test_size | 0.2 | Tỷ lệ 80/20 |
| random_state | 42 | Tái hiện kết quả |
| frame_skip | 3 | Tối ưu FPS camera |
| MOTION_THRESH | 12.0 | Ngưỡng motion filter |
| SCENE_CHANGE_THRESH | 30.0 | Ngưỡng phát hiện đổi cảnh |

---

## CƠ SỞ LÝ THUYẾT

### SVM (Support Vector Machine)
- Tìm siêu phẳng phân tách tối ưu giữa 2 lớp, tối đa hóa margin
- Kernel RBF: K(x,y) = exp(-γ||x-y||²) — ánh xạ lên không gian cao chiều cho phân loại phi tuyến
- Platt Scaling: cho phép SVM xuất xác suất (probability=True)
- class_weight=balanced: tự tính trọng số nghịch đảo tỷ lệ class (xử lý dataset lệch)

### HOG (Histogram of Oriented Gradients)
- Dalal & Triggs (2005), chia ảnh thành ô, tính histogram hướng gradient
- Nắm bắt hình dạng/texture ngọn lửa
- 9 orientations, 16×16 pixels/cell, 2×2 cells/block → 1,764 chiều

### Color Histogram (HSV)
- HSV tách Hue/Saturation/Value, lửa có hue đặc trưng 0-30 (đỏ-cam-vàng)
- 16 bins × 3 kênh → 4,096 chiều

### Risk Assessment
- Dùng predict_proba() → fire_probability → 3 mức:
  - AN TOÀN: < 0.3
  - CẢNH BÁO: 0.3 ≤ p < 0.7
  - NGUY HIỂM: ≥ 0.7

### Cơ chế chống báo nhầm
1. EMA + Rolling Average Smoothing: loại bỏ spike đơn lẻ
2. State Machine + Hysteresis: cần N frame liên tiếp mới chuyển trạng thái
3. Motion Filter: bỏ qua khi chuyển động lớn (HOG nhầm motion blur)
4. Scene Change Detector: reset khi thay đổi cảnh lớn, chờ 12 frame ổn định

---

## TOÀN BỘ SOURCE CODE

### src/utils.py (164 dòng)
```python
"""
utils.py - Các hàm tiện ích dùng chung cho toàn bộ project.
"""
import os, json, logging
from datetime import datetime
import cv2, numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "reports", "results")

IMG_SIZE = (128, 128)
RANDOM_SEED = 42
TEST_SIZE = 0.2
CLASS_NAMES = ["non_fire", "fire"]
DEFAULT_MODEL_NAME = "svm_fire_detector.pkl"
DEFAULT_SCALER_NAME = "scaler.pkl"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def setup_logger(name="fire_detection", level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in SUPPORTED_EXTENSIONS

def imread_unicode(filepath):
    """Đọc ảnh hỗ trợ đường dẫn Unicode. cv2.imread() trên Windows không đọc được path Unicode."""
    try:
        data = np.fromfile(filepath, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None

def save_results(results_dict, filename="results.json"):
    ensure_dir(RESULTS_DIR)
    filepath = os.path.join(RESULTS_DIR, filename)
    results_dict["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    return filepath

def get_model_path(model_name=None):
    if model_name is None: model_name = DEFAULT_MODEL_NAME
    ensure_dir(MODELS_DIR)
    return os.path.join(MODELS_DIR, model_name)

def get_scaler_path(scaler_name=None):
    if scaler_name is None: scaler_name = DEFAULT_SCALER_NAME
    ensure_dir(MODELS_DIR)
    return os.path.join(MODELS_DIR, scaler_name)

RISK_THRESHOLDS = {"SAFE": 0.3, "WARNING": 0.7, "DANGER": 1.0}
RISK_DISPLAY = {
    "SAFE":    {"vi": "AN TOÀN",   "emoji": "🟢", "color_bgr": (0, 200, 0)},
    "WARNING": {"vi": "CẢNH BÁO",  "emoji": "🟡", "color_bgr": (0, 200, 255)},
    "DANGER":  {"vi": "NGUY HIỂM", "emoji": "🔴", "color_bgr": (0, 0, 255)},
}

def get_risk_level(fire_probability):
    if fire_probability < RISK_THRESHOLDS["SAFE"]: return "SAFE"
    elif fire_probability < RISK_THRESHOLDS["WARNING"]: return "WARNING"
    else: return "DANGER"
```

### src/data_loader.py (128 dòng)
```python
"""data_loader.py - Đọc dữ liệu ảnh từ thư mục dataset."""
import os, cv2, numpy as np
from src.utils import DATA_RAW_DIR, CLASS_NAMES, SUPPORTED_EXTENSIONS, is_image_file, setup_logger, imread_unicode

logger = setup_logger(__name__)
LABEL_MAP = {"fire": 1, "non_fire": 0}

def load_image_paths(data_dir=None):
    if data_dir is None: data_dir = DATA_RAW_DIR
    image_paths, labels = [], []
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            logger.warning(f"Không tìm thấy thư mục: {class_dir}")
            continue
        label = LABEL_MAP[class_name]
        count = 0
        for filename in sorted(os.listdir(class_dir)):
            if is_image_file(filename):
                image_paths.append(os.path.join(class_dir, filename))
                labels.append(label)
                count += 1
        logger.info(f"  [{class_name}] Tìm thấy {count} ảnh")
    return image_paths, labels

def load_single_image(image_path):
    if not os.path.isfile(image_path): return None
    return imread_unicode(image_path)

def load_images_from_paths(image_paths):
    images, valid_indices = [], []
    for i, path in enumerate(image_paths):
        img = load_single_image(path)
        if img is not None:
            images.append(img)
            valid_indices.append(i)
    return images, valid_indices
```

### src/preprocess.py (114 dòng)
```python
"""preprocess.py - Tiền xử lý ảnh: Resize 128×128 + GaussianBlur."""
import cv2, numpy as np
from src.utils import IMG_SIZE, setup_logger

logger = setup_logger(__name__)

def resize_image(image, size=None):
    if size is None: size = IMG_SIZE
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def denoise_image(image, ksize=3):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def preprocess_single(image, size=None, denoise=True):
    processed = resize_image(image, size)
    if denoise: processed = denoise_image(processed)
    return processed

def preprocess_batch(images, size=None, denoise=True):
    processed, errors = [], 0
    for i, img in enumerate(images):
        try: processed.append(preprocess_single(img, size, denoise))
        except Exception as e: errors += 1
    return processed
```

### src/features.py (200 dòng)
```python
"""features.py - Trích xuất HOG + Color Histogram (HSV)."""
import cv2, numpy as np
from skimage.feature import hog
from src.utils import IMG_SIZE, setup_logger

logger = setup_logger(__name__)

HOG_PARAMS = {
    "orientations": 9, "pixels_per_cell": (16, 16),
    "cells_per_block": (2, 2), "block_norm": "L2-Hys", "feature_vector": True,
}
HIST_BINS = [16, 16, 16]
HIST_RANGES = [0, 180, 0, 256, 0, 256]

def extract_hog(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return hog(gray, **HOG_PARAMS)

def extract_color_histogram(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, HIST_BINS, HIST_RANGES)
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_features(image):
    return np.concatenate([extract_hog(image), extract_color_histogram(image)])

def extract_features_batch(images):
    features_list = []
    for img in images:
        try: features_list.append(extract_features(img))
        except: pass
    return np.array(features_list) if features_list else np.array([])
```

### src/train.py (199 dòng)
```python
"""train.py - Pipeline huấn luyện SVM 7 bước."""
import os, sys, time, pickle, numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.data_loader import load_image_paths, load_images_from_paths
from src.preprocess import preprocess_batch
from src.features import extract_features_batch, get_feature_info
from src.utils import RANDOM_SEED, TEST_SIZE, get_model_path, get_scaler_path, setup_logger

def train_pipeline(data_dir=None, save=True):
    # Bước 1: Load dữ liệu
    image_paths, labels = load_image_paths(data_dir)
    images, valid_indices = load_images_from_paths(image_paths)
    labels = np.array(labels)[valid_indices]
    # Bước 2: Tiền xử lý
    images = preprocess_batch(images)
    # Bước 3: Trích xuất đặc trưng
    X = extract_features_batch(images)
    y = np.array(labels)
    # Bước 4: Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)
    # Bước 5: Chuẩn hóa
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Bước 6: Huấn luyện SVM
    model = SVC(kernel="rbf", C=10.0, gamma="scale",
                class_weight="balanced", random_state=RANDOM_SEED, probability=True)
    model.fit(X_train, y_train)
    # Bước 7: Lưu model
    if save:
        with open(get_model_path(), "wb") as f: pickle.dump(model, f)
        with open(get_scaler_path(), "wb") as f: pickle.dump(scaler, f)
    return model, scaler, X_test, y_test
```

### src/evaluate.py (363 dòng)
```python
"""evaluate.py - Đánh giá mô hình: Accuracy, Precision, Recall, F1, Confusion Matrix, Risk Analysis."""
import os, json, pickle, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from src.train import train_pipeline
from src.utils import CLASS_NAMES, FIGURES_DIR, RESULTS_DIR, save_results, ensure_dir, setup_logger, get_risk_level, RISK_DISPLAY, RISK_THRESHOLDS

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, pos_label=1), 4),
        "recall": round(recall_score(y_test, y_pred, pos_label=1), 4),
        "f1_score": round(f1_score(y_test, y_pred, pos_label=1), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True),
    }, y_pred

def plot_confusion_matrix(results): # Vẽ 2 subplot: count + %
def plot_metrics_bar(results): # Biểu đồ cột 4 chỉ số
def analyze_risk_distribution(model, X_test, y_test): # Phân tích phân bố mức rủi ro + biểu đồ
```

### src/predict.py (315 dòng)
```python
"""predict.py - Dự đoán ảnh tĩnh bằng model đã train."""
import os, sys, pickle, numpy as np, cv2
from src.preprocess import preprocess_single
from src.features import extract_features
from src.utils import get_model_path, get_scaler_path, CLASS_NAMES, get_risk_level, RISK_DISPLAY, imread_unicode

def load_model(model_path=None, scaler_path=None):
    """Load model SVM và scaler từ file .pkl"""
    with open(model_path or get_model_path(), "rb") as f: model = pickle.load(f)
    with open(scaler_path or get_scaler_path(), "rb") as f: scaler = pickle.load(f)
    return model, scaler

def predict_single(image_path, model=None, scaler=None):
    """Pipeline: imread → preprocess → extract_features → scaler.transform → predict_proba → risk_level"""
    if model is None: model, scaler = load_model()
    image = imread_unicode(image_path)
    processed = preprocess_single(image)
    features = scaler.transform(extract_features(processed).reshape(1, -1))
    proba = model.predict_proba(features)[0]
    fire_prob = float(proba[1])
    risk_level = get_risk_level(fire_prob)
    return {"label": int(model.predict(features)[0]), "fire_probability": fire_prob,
            "risk_level": risk_level, "risk_label_vi": RISK_DISPLAY[risk_level]["vi"]}
```

### src/camera_realtime.py (743 dòng - Module chính)
```python
"""camera_realtime.py - Nhận diện cháy realtime từ webcam v3.5"""
import argparse, os, platform, threading, time, cv2, numpy as np
from collections import deque
from src.preprocess import preprocess_single
from src.features import extract_features
from src.predict import load_model
from src.utils import RISK_DISPLAY, CLASS_NAMES

# Các class chính:

class ProbabilitySmoother:
    """EMA + Rolling Average để làm mượt fire_prob qua nhiều frame."""
    def __init__(self, window_size, ema_alpha):
        self.window = deque(maxlen=window_size)
        self.ema = None
        self.alpha = ema_alpha
    def update(self, raw_prob):
        self.window.append(raw_prob)
        if self.ema is None: self.ema = raw_prob
        else: self.ema = self.alpha * raw_prob + (1 - self.alpha) * self.ema
        rolling = sum(self.window) / len(self.window)
        return (rolling + self.ema) / 2.0
    def reset(self): self.window.clear(); self.ema = None

class FireStateMachine:
    """3 trạng thái SAFE ↔ WARNING ↔ DANGER với hysteresis."""
    # Leo thang: cần N frame liên tiếp vượt ngưỡng
    # Hạ bậc: cần N frame liên tiếp dưới ngưỡng
    def update(self, smoothed_prob): # Cập nhật state
    def reset(self): # Reset về SAFE khi scene change

def predict_frame(frame, model, scaler):
    """Predict 1 frame theo đúng pipeline train."""
    processed = preprocess_single(frame)
    features = scaler.transform(extract_features(processed).reshape(1, -1))
    return float(model.predict_proba(features)[0][1])

def run_webcam_detection(...):
    """Vòng lặp chính:
    1. Scene Change Detection (mean_abs_diff > 30 → reset)
    2. Stabilization Period (chờ 12 frame ổn định)
    3. Motion Filter (mean_abs_diff > 12 → skip)
    4. Predict → Smooth → FSM → Overlay + Sound
    """
    # Hỗ trợ 12+ CLI arguments tùy chỉnh
    # Phím tắt: Q=thoát, M/Space=bật/tắt âm thanh
    # Cảnh báo âm thanh: WARNING=800Hz, DANGER=1200-1500Hz (winsound, thread daemon)
    # Overlay tiếng Việt bằng Pillow (PIL.ImageDraw)
```

---

## QUYẾT ĐỊNH THIẾT KẾ QUAN TRỌNG

1. **HOG + Color Histogram**: HOG nắm hình dạng lửa, Color Hist nắm màu sắc → bổ sung nhau
2. **Risk Score thay vì Multi-class**: Không cần dataset mới, giữ nguyên SVM core, thực tế hơn
3. **class_weight=balanced**: Dataset lệch (fire gấp ~2 non_fire), tự cân bằng trọng số
4. **imread_unicode()**: Đường dẫn chứa tiếng Việt, cv2.imread() không hỗ trợ → dùng np.fromfile()
5. **Ngưỡng camera riêng (0.50/0.75)**: Cao hơn evaluate (0.30/0.70) do khác biệt miền dữ liệu

---

## DEPENDENCIES (requirements.txt)
```
opencv-python>=4.8.0
scikit-learn>=1.3.0
scikit-image>=0.21.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
Pillow>=10.0.0
```

---

## ĐIỂM MẠNH
- Pipeline đơn giản, dễ hiểu, dễ tái hiện
- SVM không cần GPU, huấn luyện nhanh
- Risk Assessment 3 mức phù hợp thực tế
- Chống báo nhầm đa tầng
- Hoạt động realtime từ webcam

## HẠN CHẾ
- Lửa nhỏ ở xa khó phát hiện (hạn chế global features)
- Chưa có data augmentation
- Âm thanh chỉ hỗ trợ Windows

## HƯỚNG PHÁT TRIỂN
- Data augmentation, GridSearchCV, so sánh CNN
- Thêm lớp "smoke", deploy web/mobile
- Kết hợp IoT sensors, temporal analysis

---

## TÀI LIỆU THAM KHẢO
1. Vapnik (1995). *The Nature of Statistical Learning Theory*. Springer.
2. Dalal & Triggs (2005). Histograms of Oriented Gradients for Human Detection. *CVPR*.
3. Celik & Demirel (2009). Fire detection in video sequences using a generic color model. *Fire Safety Journal*.
4. Platt (1999). Probabilistic outputs for support vector machines.
5. Scikit-learn documentation: https://scikit-learn.org/stable/modules/svm.html
6. OpenCV documentation: https://docs.opencv.org/
