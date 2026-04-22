# Hệ thống phát hiện cháy realtime từ camera sử dụng SVM

Đề tài bài tập kết thúc môn **Trí tuệ nhân tạo** — Mã sinh viên: **B25CHHT089**

Dự án xây dựng hệ thống phát hiện cháy theo thời gian thực từ webcam, sử dụng mô hình SVM (Support Vector Machine) kết hợp đặc trưng HOG và Color Histogram. Hệ thống tích hợp đánh giá mức độ rủi ro 3 cấp và cảnh báo âm thanh tự động.

---

## Tổng quan

| Thành phần | Chi tiết |
|---|---|
| **Bài toán** | Phân loại nhị phân (fire / non_fire) + nhận diện realtime từ camera |
| **Mô hình** | SVM với kernel RBF, class_weight=balanced |
| **Đặc trưng** | HOG (1.764 chiều) + HSV Color Histogram (4.096 chiều) |
| **Đánh giá rủi ro** | 3 mức: AN TOÀN / CẢNH BÁO / NGUY HIỂM |
| **Đầu vào** | Luồng video realtime từ webcam hoặc ảnh tĩnh |
| **Cảnh báo** | Âm thanh beep tự động theo mức rủi ro (Windows) |
| **Chống báo nhầm** | State Machine + EMA Smoothing + Motion Filter + Scene Change Detector |
| **Tập dữ liệu** | 872 ảnh fire + 450 ảnh non_fire = 1.322 ảnh |
| **Accuracy** | 89,43% |

---

## Kết quả đánh giá

| Chỉ số | Giá trị |
|---|---|
| Accuracy | 89,43% |
| Precision (fire) | 94,01% |
| Recall (fire) | 89,71% |
| F1-Score (fire) | 91,81% |

### Phân bố mức rủi ro (tập test)

| Mức rủi ro | Mô tả | Số mẫu |
|---|---|---|
| AN TOÀN | fire_prob < 0,3 | 72 |
| CẢNH BÁO | 0,3 ≤ fire_prob < 0,7 | 31 |
| NGUY HIỂM | fire_prob ≥ 0,7 | 162 |

---

## Hướng dẫn cài đặt và sử dụng

### 1. Cài đặt môi trường

```bash
# Tạo virtual environment
python -m venv venv
venv\Scripts\activate          # Windows

# Cài đặt thư viện
pip install -r requirements.txt
```

**Yêu cầu:**
- Python 3.10 trở lên
- Webcam (tích hợp laptop hoặc USB)
- Windows (để có cảnh báo âm thanh bằng winsound)

### 2. Chuẩn bị dữ liệu (nếu muốn train lại)

Đặt ảnh vào các thư mục:

```
data/raw/
├── fire/           ← Ảnh có cháy (lửa lớn, bật lửa, nến, bếp gas...)
└── non_fire/       ← Ảnh không cháy (phòng, đèn vàng, ngoài trời...)
```

### 3. Huấn luyện mô hình (nếu chưa có model)

```bash
# Cách nhanh: double-click file
run_train.bat

# Hoặc chạy thủ công
python -m src.train
python -m src.evaluate
```

### 4. Chạy nhận diện cháy realtime từ camera

```bash
# Webcam mặc định
python -m src.camera_realtime

# Tắt âm thanh cảnh báo
python -m src.camera_realtime --mute

# Tùy chỉnh ngưỡng phát hiện
python -m src.camera_realtime --thresh-warning 0.40 --thresh-danger 0.60

# Tùy chỉnh smoothing
python -m src.camera_realtime --smoothing 10 --min-consec-danger 6

# Chọn camera khác
python -m src.camera_realtime --camera-id 1
```

**Khi đang chạy:**
- Cửa sổ hiển thị video realtime với thông tin dự đoán
- Viền màu thay đổi theo mức rủi ro (xanh / cam / đỏ)
- Hiển thị xác suất cháy (smoothed + raw) trên mỗi frame
- FPS hiển thị ở góc dưới trái
- Nhấn **Q** để thoát
- Nhấn **M** hoặc **Space** để bật/tắt âm thanh

**Cảnh báo âm thanh (Windows):**
- CẢNH BÁO: 1 beep nhẹ (800Hz), cooldown 3 giây
- NGUY HIỂM: 3 beep gấp (1200–1500Hz), cooldown 3 giây
- AN TOÀN: không phát âm thanh

### 5. Dự đoán trên ảnh tĩnh

```bash
python -m src.predict path/to/image.jpg
python -m src.predict path/to/folder/
```

---

## Pipeline xử lý

### Pipeline huấn luyện (offline)

```
Ảnh gốc → Resize 128×128 → Denoise → HOG + Color Hist → StandardScaler → SVM → fire/non_fire
```

### Pipeline nhận diện realtime (camera)

```
Webcam frame
    ↓
Scene Change Detector (reset hệ thống khi thay đổi khung cảnh lớn)
    ↓
Motion Filter (bỏ qua frame có chuyển động lớn)
    ↓
Resize 128×128 → Denoise → HOG + Color Hist → Scaler → SVM → predict_proba()
    ↓
EMA + Rolling Average Smoothing (làm mượt xác suất qua nhiều frame)
    ↓
Finite State Machine + Hysteresis (SAFE ↔ WARNING ↔ DANGER, cần N frame liên tiếp)
    ↓
Overlay hiển thị (tiếng Việt, Pillow) + Cảnh báo âm thanh (thread daemon)
```

---

## Cấu trúc dự án

```
Nhận diện cháy camera/
│
├── data/raw/
│   ├── fire/                     ← 872 ảnh cháy
│   └── non_fire/                 ← 450 ảnh không cháy
│
├── models/
│   ├── svm_fire_detector.pkl     ← Model đã huấn luyện
│   └── scaler.pkl                ← Scaler đã fit
│
├── reports/
│   ├── figures/                  ← Biểu đồ (confusion matrix, risk analysis)
│   └── results/                  ← Kết quả JSON
│
├── src/
│   ├── __init__.py
│   ├── utils.py                  ← Cấu hình, hằng số, imread_unicode()
│   ├── data_loader.py            ← Đọc dữ liệu ảnh
│   ├── preprocess.py             ← Tiền xử lý (resize, denoise)
│   ├── features.py               ← Trích xuất HOG + Color Histogram
│   ├── train.py                  ← Pipeline huấn luyện SVM
│   ├── evaluate.py               ← Đánh giá + Risk analysis
│   ├── predict.py                ← Dự đoán ảnh tĩnh + Risk level
│   └── camera_realtime.py        ← Nhận diện realtime từ webcam (module chính)
│
├── run_train.bat                 ← Script huấn luyện tự động (Windows)
├── requirements.txt
└── README.md
```

---

## Tham số kỹ thuật

### Mô hình SVM

| Tham số | Giá trị |
|---|---|
| Kernel | RBF |
| C | 10,0 |
| gamma | scale |
| class_weight | balanced |
| Kích thước ảnh | 128×128 |
| Số chiều HOG | 1.764 |
| Số chiều Color Histogram | 4.096 |
| Tổng số đặc trưng | 5.860 |
| Tỷ lệ tập test | 20% |
| Random seed | 42 |

### Tham số nhận diện realtime

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--camera-id` | 0 | ID webcam |
| `--frame-skip` | 3 | Predict mỗi N frame |
| `--mute` | False | Tắt cảnh báo âm thanh |
| `--thresh-warning` | 0,50 | Ngưỡng CẢNH BÁO (smoothed prob) |
| `--thresh-danger` | 0,75 | Ngưỡng NGUY HIỂM (smoothed prob) |
| `--smoothing` | 8 | Số frame lấy trung bình |
| `--ema-alpha` | 0,30 | Trọng số EMA |
| `--min-consec-danger` | 5 | Số frame liên tiếp để lên NGUY HIỂM |
| `--min-consec-warning` | 3 | Số frame liên tiếp để lên CẢNH BÁO |
| `--safe-drop` | 3 | Số frame liên tiếp dưới ngưỡng để hạ trạng thái |
| `--cooldown` | 3,0 | Giây giữa 2 lần phát âm thanh |

---

## Ghi chú kỹ thuật

- **State Machine + Hysteresis**: Tránh nhảy trạng thái khi xác suất dao động quanh ngưỡng. Cần N frame liên tiếp vượt ngưỡng mới leo thang, N frame dưới ngưỡng mới hạ bậc.
- **EMA + Rolling Average**: Kết hợp 2 phương pháp làm mượt — EMA phản ứng nhanh với thay đổi thật, rolling average ổn định tổng thể.
- **Motion Filter**: Khi frame thay đổi lớn (người đi qua, vẫy tay), bỏ qua prediction vì HOG có thể nhầm motion blur thành texture lửa.
- **Scene Change Detector**: Phát hiện thay đổi cảnh lớn (đổi góc camera, chuyển phòng) → reset toàn bộ hệ thống và chờ ổn định trước khi predict tiếp.
- **Ngưỡng camera riêng**: Camera dùng ngưỡng cao hơn (0,50/0,75) so với ngưỡng đánh giá (0,30/0,70) do webcam có sự khác biệt miền dữ liệu so với ảnh huấn luyện.
- **Pillow overlay**: Dùng PIL.ImageDraw thay cv2.putText() để hiển thị tiếng Việt có dấu trên video.
- **imread_unicode()**: Dùng np.fromfile() + cv2.imdecode() để đọc ảnh có đường dẫn tiếng Việt trên Windows.
"# nh-n-di-n-ch-y-camera" 
