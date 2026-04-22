"""
predict.py - Dự đoán ảnh mới bằng mô hình SVM đã huấn luyện.

Cách dùng:
    # Dự đoán 1 ảnh
    python -m src.predict path/to/image.jpg
    
    # Dự đoán nhiều ảnh
    python -m src.predict img1.jpg img2.png img3.jpg
    
    # Dự đoán cả thư mục
    python -m src.predict path/to/folder/
"""

import os
import sys
import pickle
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.preprocess import preprocess_single
from src.features import extract_features
from src.utils import (
    get_model_path, get_scaler_path, is_image_file,
    CLASS_NAMES, FIGURES_DIR, ensure_dir, setup_logger,
    get_risk_level, RISK_DISPLAY, imread_unicode
)

logger = setup_logger("predict")

# Nhãn hiển thị
LABEL_DISPLAY = {0: "NON-FIRE ✓", 1: "FIRE 🔥"}
LABEL_COLOR = {0: (0, 200, 0), 1: (0, 0, 255)}  # BGR cho OpenCV


def load_model(model_path=None, scaler_path=None):
    """
    Load model SVM và scaler đã lưu.
    
    Returns:
        model: SVM model
        scaler: StandardScaler
    """
    if model_path is None:
        model_path = get_model_path()
    if scaler_path is None:
        scaler_path = get_scaler_path()
    
    # Kiểm tra file tồn tại
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Không tìm thấy model: {model_path}\n"
            "Hãy chạy 'python -m src.train' trước."
        )
    if not os.path.isfile(scaler_path):
        raise FileNotFoundError(
            f"Không tìm thấy scaler: {scaler_path}\n"
            "Hãy chạy 'python -m src.train' trước."
        )
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    logger.info(f"Đã load model từ: {model_path}")
    return model, scaler


def predict_single(image_path, model=None, scaler=None):
    """
    Dự đoán một ảnh.
    
    Parameters:
        image_path: đường dẫn ảnh
        model: SVM model (nếu None sẽ tự load)
        scaler: StandardScaler (nếu None sẽ tự load)
    
    Returns:
        result: dict {
            "path": str,
            "label": int (0 hoặc 1),
            "class_name": str ("fire" hoặc "non_fire"),
            "confidence": float (xác suất),
            "display": str (nhãn hiển thị)
        }
    """
    # Load model nếu cần
    if model is None or scaler is None:
        model, scaler = load_model()
    
    # Đọc ảnh
    image = imread_unicode(image_path)
    if image is None:
        return {"path": image_path, "error": "Không đọc được ảnh"}
    
    # Tiền xử lý
    processed = preprocess_single(image)
    
    # Trích xuất đặc trưng
    features = extract_features(processed)
    features = features.reshape(1, -1)  # Reshape thành 1 mẫu
    
    # Chuẩn hóa
    features = scaler.transform(features)
    
    # Dự đoán
    label = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    confidence = proba[label]

    # v2.0: Risk Assessment
    fire_prob = float(proba[1])  # Xác suất cháy (label 1 = fire)
    risk_level = get_risk_level(fire_prob)
    risk_info = RISK_DISPLAY[risk_level]

    result = {
        "path": image_path,
        "label": int(label),
        "class_name": CLASS_NAMES[label] if label < len(CLASS_NAMES) else str(label),
        "confidence": round(float(confidence), 4),
        "display": LABEL_DISPLAY.get(label, str(label)),
        "probabilities": {
            "non_fire": round(float(proba[0]), 4),
            "fire": round(float(proba[1]), 4),
        },
        # v2.0 Risk Assessment
        "fire_probability": round(fire_prob, 4),
        "risk_level": risk_level,
        "risk_label_vi": risk_info["vi"],
        "risk_emoji": risk_info["emoji"],
    }

    return result


def predict_batch(image_paths, model=None, scaler=None):
    """
    Dự đoán nhiều ảnh.
    
    Parameters:
        image_paths: list of str
    
    Returns:
        results: list of dict
    """
    if model is None or scaler is None:
        model, scaler = load_model()
    
    results = []
    for path in image_paths:
        result = predict_single(path, model, scaler)
        results.append(result)
    
    return results


def predict_folder(folder_path, model=None, scaler=None):
    """Dự đoán tất cả ảnh trong thư mục."""
    if not os.path.isdir(folder_path):
        print(f"[LỖI] Thư mục không tồn tại: {folder_path}")
        return []
    
    image_paths = [
        os.path.join(folder_path, f)
        for f in sorted(os.listdir(folder_path))
        if is_image_file(f)
    ]
    
    if not image_paths:
        print(f"[LỖI] Không tìm thấy ảnh trong: {folder_path}")
        return []
    
    return predict_batch(image_paths, model, scaler)


def visualize_predictions(results, max_display=8, save_path=None):
    """
    Hiển thị kết quả dự đoán dạng grid ảnh.
    
    Parameters:
        results: list of predict result dicts
        max_display: số ảnh tối đa hiển thị
        save_path: đường dẫn lưu file
    """
    # Lọc kết quả hợp lệ
    valid = [r for r in results if "error" not in r][:max_display]
    
    if not valid:
        print("Không có kết quả hợp lệ để hiển thị")
        return
    
    n = len(valid)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()
    
    for i, (ax, result) in enumerate(zip(axes, valid)):
        img = imread_unicode(result["path"])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ax.imshow(img_rgb)

        # v2.0: Risk level colors
        risk_level = result.get("risk_level", "SAFE")
        risk_colors = {"SAFE": "green", "WARNING": "darkorange", "DANGER": "red"}
        title_color = risk_colors.get(risk_level, "black")
        risk_emoji = result.get("risk_emoji", "")
        risk_vi = result.get("risk_label_vi", "")

        ax.set_title(
            f"{risk_emoji} {risk_vi}\n{result['display']} ({result['confidence']*100:.1f}%)",
            fontsize=10,
            color=title_color,
            fontweight="bold"
        )
        ax.axis("off")

    # Ẩn các axes thừa
    for ax in axes[len(valid):]:
        ax.axis("off")

    plt.suptitle("SVM Fire Detection - Prediction Results (v2.0)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_path is None:
        ensure_dir(FIGURES_DIR)
        save_path = os.path.join(FIGURES_DIR, "predictions.png")
    
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nĐã lưu kết quả: {save_path}")


def print_result(result):
    """In kết quả dự đoán 1 ảnh (v2.0 - có risk level)."""
    if "error" in result:
        print(f"  ❌ {result['path']}: {result['error']}")
    else:
        filename = os.path.basename(result["path"])
        fire_prob = result.get("fire_probability", 0) * 100
        risk_emoji = result.get("risk_emoji", "")
        risk_vi = result.get("risk_label_vi", "N/A")
        risk_level = result.get("risk_level", "N/A")

        print(f"  {risk_emoji} [{risk_vi}] {result['display']}  |  {filename}")
        print(f"    Fire Prob: {fire_prob:.1f}%  |  "
              f"Confidence: {result['confidence']*100:.1f}%  |  "
              f"Risk: {risk_level}")


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cách dùng:")
        print("  python -m src.predict <ảnh_hoặc_thư_mục> [ảnh2] [ảnh3] ...")
        print("\nVí dụ:")
        print("  python -m src.predict test_image.jpg")
        print("  python -m src.predict data/raw/fire/")
        sys.exit(1)
    
    print("=" * 60)
    print("DỰ ĐOÁN ẢNH - SVM FIRE DETECTION")
    print("=" * 60)
    
    # Load model 1 lần
    model, scaler = load_model()
    
    all_results = []
    
    for arg in sys.argv[1:]:
        if os.path.isdir(arg):
            print(f"\n📁 Thư mục: {arg}")
            results = predict_folder(arg, model, scaler)
        elif os.path.isfile(arg):
            print(f"\n📄 File: {arg}")
            results = [predict_single(arg, model, scaler)]
        else:
            print(f"\n❌ Không tìm thấy: {arg}")
            continue
        
        for r in results:
            print_result(r)
        all_results.extend(results)
    
    # Tổng kết
    valid = [r for r in all_results if "error" not in r]
    fire_count = sum(1 for r in valid if r["label"] == 1)
    safe_count = sum(1 for r in valid if r.get("risk_level") == "SAFE")
    warn_count = sum(1 for r in valid if r.get("risk_level") == "WARNING")
    danger_count = sum(1 for r in valid if r.get("risk_level") == "DANGER")

    print(f"\n{'='*60}")
    print(f"TỔNG KẾT: {len(valid)} ảnh đã phân tích")
    print(f"  🔥 Fire: {fire_count}")
    print(f"  ✓  Non-fire: {len(valid) - fire_count}")
    print(f"\n  MỨC ĐỘ RỦI RO (Risk Assessment v2.0):")
    print(f"  🟢 An toàn:    {safe_count}")
    print(f"  🟡 Cảnh báo:   {warn_count}")
    print(f"  🔴 Nguy hiểm:  {danger_count}")

    # Lưu visualization
    if valid:
        visualize_predictions(valid)
