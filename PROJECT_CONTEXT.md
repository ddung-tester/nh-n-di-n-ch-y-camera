# Tài liệu kỹ thuật dự án

> Mô tả chi tiết kiến trúc, pipeline, quyết định thiết kế, và trạng thái của dự án.
> Dùng làm tài liệu tham khảo khi phát triển tiếp hoặc viết báo cáo.

---

## 1. Mục tiêu dự án

Xây dựng hệ thống phát hiện cháy theo thời gian thực từ webcam sử dụng mô hình SVM (Support Vector Machine).

- Bài tập kết thúc môn Trí tuệ nhân tạo
- Mã sinh viên: B25CHHT089
- Yêu cầu: dễ hiểu, chạy được, giải thích được, không dùng deep learning
- Phiên bản hiện tại: **v3.5** — Nhận diện realtime + Chống báo nhầm + Scene Change Detector

---

## 2. Lịch sử phiên bản

| Phiên bản | Ngày | Nội dung chính |
|---|---|---|
| v1.0 | 2026-04-09 | Khởi tạo: phân loại nhị phân, HOG + Color Histogram, SVM, CLI |
| v2.0 | 2026-04-10 | Thêm Risk Assessment 3 mức (SAFE / WARNING / DANGER) |
| v2.1 | 2026-04-11 | Tải dataset 962 ảnh, fix Unicode path, class_weight=balanced, Accuracy 90,16% |
| v3.0 | 2026-04-11 | Mở rộng nhận diện realtime từ camera + cảnh báo âm thanh |
| v3.4 | 2026-04-20 | Chống false positive: State Machine + Motion Filter + EMA + mở rộng dataset 1.322 ảnh |
| v3.5 | 2026-04-20 | Scene Change Detector + Stabilization Period |

---

## 3. Bài toán

```
ĐẦU VÀO (chính):  Video stream realtime từ webcam
ĐẦU VÀO (phụ):    Ảnh tĩnh (JPG, PNG, BMP, TIFF, WEBP) bất kỳ kích thước

ĐẦU RA:
  - Nhãn phân loại: fire (1) hoặc non_fire (0)
  - Xác suất cháy: fire_probability (0,0 → 1,0)
  - Mức rủi ro: AN TOÀN / CẢNH BÁO / NGUY HIỂM
  - Overlay hiển thị trên video (chế độ camera)
  - Cảnh báo âm thanh tự động (chế độ camera, Windows)
```

---

## 4. Cấu trúc thư mục

```
Nhận diện cháy camera/
├── .gitignore
├── CHANGELOG.md                  ← Lịch sử thay đổi
├── De_tai_phat_hien_chay_SVM_mo_ta_y_tuong.docx
├── PROJECT_CONTEXT.md            ← File này
├── README.md
├── REPORT_NOTES.md               ← Ghi chú nội dung viết báo cáo
├── requirements.txt
├── run_train.bat                 ← Script huấn luyện tự động (Windows)
│
├── data/raw/
│   ├── fire/                     ← 872 ảnh cháy
│   └── non_fire/                 ← 450 ảnh không cháy
│
├── models/
│   ├── svm_fire_detector.pkl     ← Model đã huấn luyện (~21,5 MB)
│   └── scaler.pkl                ← Scaler đã fit (~138 KB)
│
├── reports/
│   ├── figures/                  ← Biểu đồ (confusion matrix, risk analysis)
│   └── results/                  ← Kết quả JSON
│
└── src/
    ├── __init__.py
    ├── utils.py                  ← Cấu hình + Risk Assessment + imread_unicode
    ├── data_loader.py            ← Đọc ảnh (dùng imread_unicode)
    ├── preprocess.py             ← Resize + denoise
    ├── features.py               ← HOG + Color Histogram
    ├── train.py                  ← Pipeline huấn luyện SVM
    ├── evaluate.py               ← Đánh giá + Risk analysis
    ├── predict.py                ← Dự đoán ảnh tĩnh + Risk display
    └── camera_realtime.py        ← Nhận diện realtime từ webcam (module chính)
```

---

## 5. Pipeline xử lý

### 5.1 Pipeline huấn luyện (offline)

```
[1] Load ảnh         data_loader.py    Đọc từ data/raw/{fire,non_fire}
         ↓                              LABEL_MAP: fire=1, non_fire=0
                                        Dùng imread_unicode() cho path tiếng Việt
[2] Tiền xử lý       preprocess.py     Resize 128×128, GaussianBlur(ksize=3)
         ↓
[3] Trích xuất        features.py       HOG (1.764 chiều) + HSV Color Hist (4.096 chiều)
    đặc trưng                           Tổng: 5.860 chiều
         ↓
[4] Chia train/test   train.py          test_size=0,2, stratify=y, seed=42
         ↓
[5] Chuẩn hóa        train.py          StandardScaler (fit train, transform test)
         ↓
[6] Huấn luyện       train.py          SVM(kernel=RBF, C=10, gamma=scale,
         ↓                                  class_weight=balanced)
[7] Lưu model        train.py          pickle → models/*.pkl
         ↓
[8] Đánh giá         evaluate.py       Accuracy, Precision, Recall, F1
         ↓                              Confusion Matrix, Metrics Bar
[9] Risk Analysis     evaluate.py       Phân bố mức rủi ro
```

### 5.2 Pipeline nhận diện realtime (camera)

```
[1]  Mở webcam        camera_realtime   cv2.VideoCapture(camera_id)
          ↓
[2]  Đọc frame        camera_realtime   cap.read() → frame BGR
          ↓
[3]  Scene Change     camera_realtime   mean_abs_diff > 30 → RESET smoother + FSM
          ↓                              + chờ 12 frame ổn định
[3b] Motion Filter    camera_realtime   mean_abs_diff > 12 → bỏ qua prediction
[4]  Tiền xử lý       preprocess.py     preprocess_single() → resize 128×128 + denoise
          ↓
[5]  Trích xuất        features.py       extract_features() → HOG + Color Hist
          ↓
[6]  Chuẩn hóa        camera_realtime   scaler.transform(features)
          ↓
[7]  Dự đoán          camera_realtime   model.predict_proba() → raw fire_prob
          ↓
[8]  Smoothing         ProbabilitySmoother  EMA + Rolling Average → smoothed_prob
          ↓
[9]  State Machine     FireStateMachine     Hysteresis: SAFE ↔ WARNING ↔ DANGER
          ↓                                  (cần N frame liên tiếp mới chuyển)
[10] Overlay           camera_realtime   draw_overlay() → tiếng Việt (Pillow)
          ↓
[11] Cảnh báo          camera_realtime   trigger_alert() → âm thanh (thread daemon)
          ↓
[12] Hiển thị          camera_realtime   cv2.imshow()
```

**Điểm quan trọng:**
- Pipeline predict (bước 4–7) giữ nguyên 100% so với lúc huấn luyện
- Scene Change Detector: khi thay đổi cảnh lớn → reset smoother + state machine + chờ ổn định
- Motion Filter chặn false positive từ chuyển động (vẫy tay, người đi qua)
- Ngưỡng camera riêng (0,50/0,75) — không ảnh hưởng evaluate/report (vẫn dùng 0,30/0,70)
- Âm thanh cảnh báo chạy trên thread daemon, có cooldown 3 giây

---

## 6. Chi tiết các module

### 6.1 src/utils.py (~164 dòng)
Cấu hình trung tâm + Risk Assessment + imread_unicode
- Đường dẫn: PROJECT_ROOT, DATA_RAW_DIR, MODELS_DIR, FIGURES_DIR, RESULTS_DIR
- Hằng số: IMG_SIZE=(128,128), RANDOM_SEED=42, TEST_SIZE=0,2
- CLASS_NAMES = ["non_fire", "fire"] — index khớp với label
- Risk Assessment: RISK_THRESHOLDS, RISK_DISPLAY, get_risk_level()
- imread_unicode(): đọc ảnh qua np.fromfile() + cv2.imdecode() — fix lỗi cv2.imread() với path Unicode

### 6.2 src/data_loader.py (128 dòng)
- LABEL_MAP = {"fire": 1, "non_fire": 0}
- load_image_paths(): quét thư mục, trả (image_paths, labels)
- load_single_image(): dùng imread_unicode() thay cv2.imread()
- load_images_from_paths(): batch đọc ảnh

### 6.3 src/preprocess.py (114 dòng)
- resize_image(): cv2.resize(INTER_AREA)
- denoise_image(ksize=3): cv2.GaussianBlur()
- preprocess_single(): resize + denoise
- preprocess_batch(): xử lý batch

### 6.4 src/features.py (200 dòng)
- HOG: orientations=9, pixels_per_cell=16×16, cells_per_block=2×2 → 1.764 chiều
- Color Hist: HSV, bins=[16,16,16] → 4.096 chiều
- extract_features(): concatenate → 5.860 chiều
- get_feature_info(): metadata

### 6.5 src/train.py (199 dòng)
- train_pipeline(): pipeline huấn luyện 7 bước
- SVM: SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced", probability=True)
- Đầu ra: (model, scaler, X_test, y_test)

### 6.6 src/evaluate.py (~363 dòng)
- evaluate_model(): tính metrics
- print_results(): in kết quả ra terminal
- plot_confusion_matrix(): 2 subplot (count + %)
- plot_metrics_bar(): biểu đồ cột 4 chỉ số
- analyze_risk_distribution(): phân tích phân bố mức rủi ro

### 6.7 src/predict.py (~315 dòng)
- predict_single(): dự đoán 1 ảnh tĩnh + risk level
- predict_batch(): dự đoán nhiều ảnh
- predict_folder(): dự đoán cả thư mục
- visualize_predictions(): grid hiển thị kết quả
- load_model(): load model+scaler (dùng chung cho camera_realtime)

### 6.8 src/camera_realtime.py (~743 dòng) — Module chính
- _MotionSkip: Exception cho motion filter
- predict_frame(): predict 1 frame theo pipeline train
- ProbabilitySmoother: EMA + Rolling Average smoothing
- FireStateMachine: State machine 3 trạng thái với hysteresis
- draw_overlay(): overlay tiếng Việt (Pillow)
- play_alert_sound(): âm thanh cảnh báo (winsound, thread daemon)
- trigger_alert_if_ready(): phát cảnh báo khi WARNING/DANGER + cooldown
- run_webcam_detection(): vòng lặp chính
- CLI: hỗ trợ 12+ tham số tùy chỉnh

---

## 7. Kết quả

### 7.1 Tập dữ liệu

| | Fire | Non-fire | Tổng |
|---|---|---|---|
| v2.1 (ban đầu) | 718 | 244 | 962 |
| v3.4 (sau mở rộng) | 872 | 450 | 1.322 |
| Tỷ lệ | 65,9% | 34,1% | — |
| Train (80%) | ~697 | ~360 | ~1.057 |
| Test (20%) | 175 | 90 | 265 |

### 7.2 Hiệu suất phân loại

| Chỉ số | v2.1 | v3.4 |
|---|---|---|
| Accuracy | 90,16% | 89,43% |
| Precision (fire) | 96,99% | 94,01% |
| Recall (fire) | 89,58% | 89,71% |
| F1-Score (fire) | 93,14% | 91,81% |
| Precision (non_fire) | 75,00% | 81,63% |
| Recall (non_fire) | 91,84% | 88,89% |

Accuracy giảm nhẹ (-0,73%) do dataset đa dạng hơn (thêm lửa nhỏ + phòng thực tế). Non_fire precision tăng đáng kể (75% → 81,6%) nghĩa là ít báo nhầm hơn.

### 7.3 Confusion Matrix

```
              Predicted
              non_fire  fire
Actual  non_fire   80     10
        fire       18    157
```

### 7.4 Phân bố mức rủi ro (tập test)

| Mức rủi ro | Fire | Non-fire | Tổng |
|---|---|---|---|
| AN TOÀN | 3 | 69 | 72 |
| CẢNH BÁO | 18 | 13 | 31 |
| NGUY HIỂM | 154 | 8 | 162 |

---

## 8. Các quyết định thiết kế

### 8.1 Mở rộng sang camera realtime
- Hệ thống phát hiện cháy thực tế đều dùng camera giám sát
- Từ ảnh tĩnh sang video stream là bước mở rộng tự nhiên
- Pipeline predict giữ nguyên 100% so với lúc train, chỉ thay đầu vào
- Chạy webcam realtime trực quan hơn CLI predict ảnh

### 8.2 Risk Score thay vì Multi-class
- Không cần dataset mới (multi-class cần thêm ảnh "smoke")
- Không ảnh hưởng pipeline hiện có, chỉ thêm lớp đánh giá phía trên
- Giống cách hệ thống cảnh báo cháy thực tế hoạt động

### 8.3 Ngưỡng mức rủi ro
| Mức | Ngưỡng | Lý do |
|---|---|---|
| AN TOÀN | fire_prob < 0,3 | Thấp hơn ranh giới quyết định nhiều |
| CẢNH BÁO | 0,3 ≤ fire_prob < 0,7 | Vùng quanh ranh giới (0,5) |
| NGUY HIỂM | fire_prob ≥ 0,7 | Cao hơn ranh giới nhiều |

### 8.4 class_weight="balanced"
- Dataset lệch: fire chiếm gấp ~2 lần non_fire
- balanced tự tính trọng số nghịch đảo tỷ lệ class
- Giúp SVM không thiên vị về class đa số

### 8.5 imread_unicode()
- Đường dẫn dự án chứa tiếng Việt: "Trí tuệ nhân tạo/Nhận diện cháy camera"
- cv2.imread() trên Windows không hỗ trợ Unicode
- Giải pháp: np.fromfile() + cv2.imdecode()

### 8.6 Thiết kế camera_realtime.py
- Frame skip: predict mỗi N frame (mặc định 3) để giữ FPS cao
- Âm thanh trên thread riêng: winsound.Beep() blocking nên chạy trên daemon thread
- Cooldown: WARNING 3 giây, DANGER 1,5 giây
- Reuse code: dùng lại preprocess_single(), extract_features(), load_model() từ các module có sẵn

### 8.7 Bug đã sửa
- v2.0: CLASS_NAMES order ["fire", "non_fire"] → ["non_fire", "fire"]
- v2.1: cv2.imread() Unicode path → imread_unicode()
- v3.4: Phím M không hoạt động (IME tiếng Việt) → thêm phím Space

---

## 9. Tham số kỹ thuật

| Tham số | Giá trị | Lý do |
|---|---|---|
| IMG_SIZE | 128×128 | Đủ chi tiết cho HOG |
| HOG orientations | 9 | Chuẩn Dalal & Triggs |
| HOG pixels_per_cell | 16×16 | Cân bằng chi tiết/tốc độ |
| Color Hist bins | [16,16,16] | 4.096 chiều |
| SVM kernel | RBF | Phi tuyến |
| SVM C | 10,0 | Regularization vừa phải |
| SVM gamma | scale | Tự động theo variance |
| SVM class_weight | balanced | Xử lý lệch class |
| test_size | 0,2 | Tỷ lệ 80/20 |
| random_state | 42 | Tái hiện kết quả |
| frame_skip | 3 | Tối ưu FPS camera |
| MOTION_THRESH | 12,0 | Ngưỡng motion filter |
| SCENE_CHANGE_THRESH | 30,0 | Ngưỡng phát hiện đổi cảnh |
| SCENE_STABILIZE_FRAMES | 12 | Số frame chờ ổn định |
| WARNING beep | 800Hz, 150ms | Nhẹ nhàng |
| DANGER beep | 1200–1500Hz, 120ms×3 | Gấp gáp, thu hút chú ý |

---

## 10. Dependencies

```
opencv-python>=4.8.0
scikit-learn>=1.3.0
scikit-image>=0.21.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
Pillow>=10.0.0
```

Thư viện tích hợp sẵn (không cần cài):
- winsound — cảnh báo âm thanh (Windows only)
- threading — chạy âm thanh trên thread riêng
- argparse — xử lý CLI arguments

---

## 11. Hướng dẫn sử dụng

### Cài đặt

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Nhận diện camera (tính năng chính)

```bash
python -m src.camera_realtime
python -m src.camera_realtime --mute
python -m src.camera_realtime --camera-id 1
```

Nhấn Q để thoát, M/Space để bật/tắt âm thanh.

### Dự đoán ảnh tĩnh

```bash
python -m src.predict path/to/image.jpg
python -m src.predict path/to/folder/
```

### Huấn luyện lại (nếu thêm dữ liệu)

```bash
run_train.bat
# hoặc
python -m src.train
python -m src.evaluate
```

---

## 12. Trạng thái hiện tại (v3.5)

**Đã hoàn thành:**
1. Cấu trúc thư mục hoàn chỉnh
2. 9 file source code hoạt động
3. Risk Assessment 3 mức (AN TOÀN / CẢNH BÁO / NGUY HIỂM)
4. Dataset: 872 fire + 450 non_fire = 1.322 ảnh
5. Train + Evaluate: Accuracy 89,43%
6. Nhận diện realtime từ webcam v3.5
7. Chống báo nhầm: EMA + State Machine + Motion Filter + Scene Change Detector
8. CLI tuning: 12+ tham số tùy chỉnh
9. Cảnh báo âm thanh theo mức rủi ro
10. Overlay tiếng Việt (Pillow)

**Hạn chế đã biết:**
1. Lửa nhỏ ở xa (>1m) khó phát hiện — hạn chế của global features + SVM
2. Accuracy giảm nhẹ sau mở rộng dataset (-0,73%) — đánh đổi lấy tính đa dạng
3. Âm thanh chỉ hỗ trợ Windows (winsound)

---

## Thông tin chung

- Python: 3.10+
- OS phát triển: Windows
- Phiên bản: v3.5
- Random seed: 42
- Accuracy: 89,43%
- Dataset: 1.322 ảnh (872 fire + 450 non_fire)
