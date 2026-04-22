"""
train.py - Huấn luyện mô hình SVM phát hiện cháy.

Pipeline:
1. Load dữ liệu ảnh
2. Tiền xử lý
3. Trích xuất đặc trưng
4. Chia train/test
5. Chuẩn hóa đặc trưng (StandardScaler)
6. Huấn luyện SVM
7. Lưu model + scaler

Chạy:
    python -m src.train
"""

import os
import sys
import time
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_image_paths, load_images_from_paths
from src.preprocess import preprocess_batch
from src.features import extract_features_batch, get_feature_info
from src.utils import (
    RANDOM_SEED, TEST_SIZE, get_model_path, get_scaler_path,
    setup_logger
)

logger = setup_logger("train")


def train_pipeline(data_dir=None, save=True):
    """
    Pipeline huấn luyện đầy đủ.
    
    Parameters:
        data_dir: Đường dẫn đến thư mục chứa fire/ và non_fire/
        save: Có lưu model ra file không
    
    Returns:
        model: SVM model đã train
        scaler: StandardScaler đã fit
        X_test: dữ liệu test
        y_test: nhãn test
    """
    total_start = time.time()
    
    # ============================
    # Bước 1: Load dữ liệu
    # ============================
    print("=" * 60)
    print("BƯỚC 1: LOAD DỮ LIỆU")
    print("=" * 60)
    
    image_paths, labels = load_image_paths(data_dir)
    
    if len(image_paths) == 0:
        print("\n[LỖI] Không tìm thấy dữ liệu!")
        print("Hãy đặt ảnh vào:")
        print("  - data/raw/fire/     (ảnh có cháy)")
        print("  - data/raw/non_fire/ (ảnh không cháy)")
        sys.exit(1)
    
    images, valid_indices = load_images_from_paths(image_paths)
    labels = np.array(labels)[valid_indices]
    
    # ============================
    # Bước 2: Tiền xử lý
    # ============================
    print("\n" + "=" * 60)
    print("BƯỚC 2: TIỀN XỬ LÝ ẢNH")
    print("=" * 60)
    
    images = preprocess_batch(images)
    
    # ============================
    # Bước 3: Trích xuất đặc trưng
    # ============================
    print("\n" + "=" * 60)
    print("BƯỚC 3: TRÍCH XUẤT ĐẶC TRƯNG")
    print("=" * 60)
    
    feat_info = get_feature_info()
    print(f"  HOG: {feat_info['hog_size']} chiều")
    print(f"  Color Histogram: {feat_info['color_hist_size']} chiều")
    print(f"  Tổng: {feat_info['total_features']} chiều")
    
    X = extract_features_batch(images)
    y = np.array(labels)
    
    if X.size == 0:
        print("[LỖI] Không trích xuất được đặc trưng!")
        sys.exit(1)
    
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Phân bố: fire={np.sum(y==1)}, non_fire={np.sum(y==0)}")
    
    # ============================
    # Bước 4: Chia train/test
    # ============================
    print("\n" + "=" * 60)
    print("BƯỚC 4: CHIA TRAIN/TEST")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y  # Giữ tỷ lệ cân bằng giữa các lớp
    )
    
    print(f"  Train: {X_train.shape[0]} mẫu")
    print(f"  Test:  {X_test.shape[0]} mẫu")
    
    # ============================
    # Bước 5: Chuẩn hóa đặc trưng
    # ============================
    print("\n" + "=" * 60)
    print("BƯỚC 5: CHUẨN HÓA ĐẶC TRƯNG")
    print("=" * 60)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("  StandardScaler đã fit trên tập train")
    print(f"  Mean range: [{X_train.mean(axis=0).min():.4f}, {X_train.mean(axis=0).max():.4f}]")
    
    # ============================
    # Bước 6: Huấn luyện SVM
    # ============================
    print("\n" + "=" * 60)
    print("BƯỚC 6: HUẤN LUYỆN SVM")
    print("=" * 60)
    
    # SVM với kernel RBF - phù hợp cho dữ liệu phi tuyến
    model = SVC(
        kernel="rbf",           # Radial Basis Function kernel
        C=10.0,                 # Regularization - cân bằng giữa margin và lỗi
        gamma="scale",          # Tự động tính gamma dựa trên variance
        class_weight="balanced", # Tự cân bằng trọng số theo tỷ lệ class
        random_state=RANDOM_SEED,
        probability=True,       # Cho phép tính xác suất dự đoán
        verbose=False
    )
    
    print("  Cấu hình: kernel=RBF, C=10.0, gamma=scale, class_weight=balanced")
    print("  Đang train...")
    
    train_start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - train_start
    
    print(f"  Hoàn thành trong {train_time:.2f} giây")
    print(f"  Số support vectors: {model.n_support_}")
    
    # ============================
    # Bước 7: Lưu model
    # ============================
    if save:
        print("\n" + "=" * 60)
        print("BƯỚC 7: LƯU MODEL")
        print("=" * 60)
        
        model_path = get_model_path()
        scaler_path = get_scaler_path()
        
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"  Model: {model_path}")
        
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"  Scaler: {scaler_path}")
    
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"TỔNG THỜI GIAN: {total_time:.2f} giây")
    print(f"{'='*60}")
    
    return model, scaler, X_test, y_test


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    model, scaler, X_test, y_test = train_pipeline()
    
    # Accuracy nhanh trên test set
    accuracy = model.score(X_test, y_test)
    print(f"\n>>> Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
