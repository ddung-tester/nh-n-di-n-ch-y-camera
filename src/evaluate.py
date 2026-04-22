"""
evaluate.py - Đánh giá mô hình SVM phát hiện cháy.

Các chỉ số đánh giá:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Xuất ra:
- In kết quả ra terminal
- Lưu confusion matrix dạng hình ảnh
- Lưu kết quả dạng JSON

Chạy:
    python -m src.evaluate
"""

import os
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Backend không cần GUI
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.train import train_pipeline
from src.utils import (
    CLASS_NAMES, FIGURES_DIR, RESULTS_DIR,
    save_results, ensure_dir, setup_logger,
    get_risk_level, RISK_DISPLAY, RISK_THRESHOLDS
)

logger = setup_logger("evaluate")


def evaluate_model(model, X_test, y_test):
    """
    Đánh giá model trên tập test.
    
    Parameters:
        model: SVM model đã train
        X_test: đặc trưng tập test (đã scale)
        y_test: nhãn thực tế
    
    Returns:
        results: dict chứa các chỉ số đánh giá
    """
    # Dự đoán
    y_pred = model.predict(X_test)
    
    # Tính các chỉ số
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="binary", pos_label=1)
    rec = recall_score(y_test, y_pred, average="binary", pos_label=1)
    f1 = f1_score(y_test, y_pred, average="binary", pos_label=1)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=CLASS_NAMES,
        output_dict=True
    )
    
    results = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "n_test_samples": len(y_test),
        "n_fire": int(np.sum(y_test == 1)),
        "n_non_fire": int(np.sum(y_test == 0)),
    }
    
    return results, y_pred


def print_results(results):
    """In kết quả đánh giá ra terminal."""
    print("\n" + "=" * 60)
    print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH")
    print("=" * 60)
    
    print(f"\n{'Chỉ số':<20} {'Giá trị':>10}")
    print("-" * 32)
    print(f"{'Accuracy':<20} {results['accuracy']:>10.4f}")
    print(f"{'Precision':<20} {results['precision']:>10.4f}")
    print(f"{'Recall':<20} {results['recall']:>10.4f}")
    print(f"{'F1-Score':<20} {results['f1_score']:>10.4f}")
    
    print(f"\nSố mẫu test: {results['n_test_samples']}")
    print(f"  - fire: {results['n_fire']}")
    print(f"  - non_fire: {results['n_non_fire']}")
    
    # Classification Report chi tiết
    print("\n" + "-" * 60)
    print("CLASSIFICATION REPORT CHI TIẾT:")
    print("-" * 60)
    report = results["classification_report"]
    print(f"\n{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 57)
    for cls in CLASS_NAMES:
        if cls in report:
            r = report[cls]
            print(f"{cls:<15} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1-score']:>10.4f} {r['support']:>10}")
    
    print(f"\n{'Macro avg':<15} {report['macro avg']['precision']:>10.4f} "
          f"{report['macro avg']['recall']:>10.4f} {report['macro avg']['f1-score']:>10.4f}")
    print(f"{'Weighted avg':<15} {report['weighted avg']['precision']:>10.4f} "
          f"{report['weighted avg']['recall']:>10.4f} {report['weighted avg']['f1-score']:>10.4f}")


def plot_confusion_matrix(results, save_path=None):
    """
    Vẽ confusion matrix và lưu ra file.
    
    Parameters:
        results: dict chứa confusion_matrix
        save_path: đường dẫn lưu file ảnh
    """
    cm = np.array(results["confusion_matrix"])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Plot 1: Confusion Matrix (số lượng) ---
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[0]
    )
    axes[0].set_title("Confusion Matrix (Count)", fontsize=13)
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    
    # --- Plot 2: Confusion Matrix (tỷ lệ %) ---
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(
        cm_pct, annot=True, fmt=".1f", cmap="Oranges",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[1]
    )
    axes[1].set_title("Confusion Matrix (%)", fontsize=13)
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")
    
    plt.suptitle("SVM Fire Detection - Evaluation Results", fontsize=15, fontweight="bold")
    plt.tight_layout()
    
    # Lưu file
    if save_path is None:
        ensure_dir(FIGURES_DIR)
        save_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nĐã lưu confusion matrix: {save_path}")
    
    return save_path


def plot_metrics_bar(results, save_path=None):
    """Vẽ biểu đồ cột các chỉ số đánh giá."""
    metrics = {
        "Accuracy": results["accuracy"],
        "Precision": results["precision"],
        "Recall": results["recall"],
        "F1-Score": results["f1_score"],
    }
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bars = ax.bar(
        metrics.keys(), metrics.values(),
        color=["#2196F3", "#4CAF50", "#FF9800", "#F44336"],
        edgecolor="white", linewidth=1.5
    )
    
    # Thêm giá trị trên mỗi cột
    for bar, val in zip(bars, metrics.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold"
        )
    
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("SVM Fire Detection - Performance Metrics", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    
    if save_path is None:
        ensure_dir(FIGURES_DIR)
        save_path = os.path.join(FIGURES_DIR, "metrics_bar.png")
    
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Đã lưu biểu đồ metrics: {save_path}")
    
    return save_path


def analyze_risk_distribution(model, X_test, y_test, save_path=None):
    """
    Phân tích phân bố mức độ rủi ro trên tập test (v2.0).

    Sử dụng predict_proba() để tính fire_probability,
    sau đó phân loại thành 3 mức: SAFE, WARNING, DANGER.

    Parameters:
        model: SVM model đã train
        X_test: đặc trưng tập test (đã scale)
        y_test: nhãn thực tế
        save_path: đường dẫn lưu biểu đồ

    Returns:
        risk_data: dict chứa phân bố risk level
    """
    probas = model.predict_proba(X_test)
    fire_probs = probas[:, 1]  # Xác suất cháy

    # Phân loại mức rủi ro cho mỗi mẫu
    risk_levels = [get_risk_level(p) for p in fire_probs]

    # Đếm theo nhãn thực tế và mức rủi ro
    risk_data = {
        "SAFE": {"fire": 0, "non_fire": 0, "total": 0},
        "WARNING": {"fire": 0, "non_fire": 0, "total": 0},
        "DANGER": {"fire": 0, "non_fire": 0, "total": 0},
    }

    for risk, true_label in zip(risk_levels, y_test):
        cls = "fire" if true_label == 1 else "non_fire"
        risk_data[risk][cls] += 1
        risk_data[risk]["total"] += 1

    # In kết quả
    print("\n" + "=" * 60)
    print("PHÂN TÍCH MỨC ĐỘ RỦI RO (v2.0)")
    print("=" * 60)
    print(f"\nNgưỡng phân loại:")
    print(f"  SAFE:    fire_prob < {RISK_THRESHOLDS['SAFE']}")
    print(f"  WARNING: {RISK_THRESHOLDS['SAFE']} <= fire_prob < {RISK_THRESHOLDS['WARNING']}")
    print(f"  DANGER:  fire_prob >= {RISK_THRESHOLDS['WARNING']}")
    print(f"\n{'Mức độ':<15} {'Tổng':>6} {'Fire':>6} {'Non-fire':>10}")
    print("-" * 40)
    for level in ["SAFE", "WARNING", "DANGER"]:
        d = risk_data[level]
        emoji = RISK_DISPLAY[level]["emoji"]
        print(f"{emoji} {level:<12} {d['total']:>6} {d['fire']:>6} {d['non_fire']:>10}")

    # Vẽ biểu đồ
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot 1: Phân bố xác suất cháy ---
    fire_mask = y_test == 1
    non_fire_mask = y_test == 0

    axes[0].hist(fire_probs[fire_mask], bins=20, alpha=0.7,
                 color="#F44336", label="Fire (thực tế)", edgecolor="white")
    axes[0].hist(fire_probs[non_fire_mask], bins=20, alpha=0.7,
                 color="#2196F3", label="Non-fire (thực tế)", edgecolor="white")

    # Vẽ đường ngưỡng risk level
    axes[0].axvline(x=RISK_THRESHOLDS["SAFE"], color="green",
                    linestyle="--", linewidth=2,
                    label=f"Safe < {RISK_THRESHOLDS['SAFE']}")
    axes[0].axvline(x=RISK_THRESHOLDS["WARNING"], color="orange",
                    linestyle="--", linewidth=2,
                    label=f"Danger \u2265 {RISK_THRESHOLDS['WARNING']}")

    axes[0].set_xlabel("Fire Probability", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title("Phân bố xác suất cháy theo nhãn thực tế", fontsize=12)
    axes[0].legend(fontsize=9)

    # --- Plot 2: Risk level bar chart ---
    levels = ["SAFE", "WARNING", "DANGER"]
    x = np.arange(len(levels))
    width = 0.35

    non_fire_counts = [risk_data[l]["non_fire"] for l in levels]
    fire_counts = [risk_data[l]["fire"] for l in levels]

    bars1 = axes[1].bar(x - width/2, non_fire_counts, width,
                        label="Non-fire", color="#2196F3", edgecolor="white")
    bars2 = axes[1].bar(x + width/2, fire_counts, width,
                        label="Fire", color="#F44336", edgecolor="white")

    # Thêm giá trị trên cột
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                axes[1].text(bar.get_x() + bar.get_width()/2, h + 0.3,
                            str(int(h)), ha="center", va="bottom", fontsize=10)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(levels)
    axes[1].set_ylabel("Count", fontsize=11)
    axes[1].set_title("Phân bố Risk Level", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.suptitle("SVM Fire Detection - Risk Analysis (v2.0)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path is None:
        ensure_dir(FIGURES_DIR)
        save_path = os.path.join(FIGURES_DIR, "risk_analysis.png")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nĐã lưu biểu đồ risk analysis: {save_path}")

    return risk_data


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ĐÁNH GIÁ MÔ HÌNH SVM PHÁT HIỆN CHÁY")
    print("=" * 60)
    
    # Train và lấy test data
    model, scaler, X_test, y_test = train_pipeline(save=True)
    
    # Đánh giá
    results, y_pred = evaluate_model(model, X_test, y_test)
    
    # In kết quả
    print_results(results)
    
    # Vẽ đồ thị
    plot_confusion_matrix(results)
    plot_metrics_bar(results)

    # v2.0: Phân tích mức độ rủi ro
    risk_data = analyze_risk_distribution(model, X_test, y_test)
    results["risk_distribution"] = risk_data

    # Lưu kết quả JSON
    filepath = save_results(results, "evaluation_results.json")
    print(f"\nĐã lưu kết quả: {filepath}")
