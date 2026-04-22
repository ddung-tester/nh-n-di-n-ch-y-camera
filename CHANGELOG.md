# Lịch sử thay đổi

Ghi lại các thay đổi quan trọng theo từng phiên bản.

---

## v3.5 (2026-04-20) — Scene Change Detector

### Vấn đề
- Báo cháy sai khi thay đổi khung cảnh: chuyển camera từ phòng này sang phòng khác, đổi góc camera, hoặc bất kỳ thay đổi lớn nào khiến toàn bộ hình ảnh thay đổi
- Nguyên nhân: Motion Filter cũ chỉ skip 1 frame, smoother buffer chứa dữ liệu từ cảnh cũ, State Machine không reset

### Thêm mới
- **Scene Change Detector**: Phát hiện thay đổi khung cảnh lớn (mean_abs_diff > 30)
  - Khi phát hiện: reset toàn bộ smoother + state machine về AN TOÀN
  - Chờ 12 frame ổn định trước khi predict lại
  - Trong thời gian ổn định: vẫn tích lũy dữ liệu vào smoother nhưng không cập nhật state machine
- ProbabilitySmoother.reset(): xóa sạch buffer + reset EMA
- FireStateMachine.reset(): reset về AN TOÀN, xóa tất cả bộ đếm

---

## v3.4 (2026-04-20) — Chống False Positive + Motion Filter

### Vấn đề
- False positive từ đèn vàng/ánh sáng ấm: Color Histogram nhạy với màu cam/vàng giống lửa
- False positive từ chuyển động: HOG nhầm motion blur thành texture lửa
- Phím M không hoạt động do IME tiếng Việt

### Thêm mới
- **Finite State Machine + Hysteresis** (FireStateMachine):
  - 3 trạng thái: AN TOÀN ↔ CẢNH BÁO ↔ NGUY HIỂM
  - Leo thang cần N frame liên tiếp vượt ngưỡng
  - Hạ bậc cần N frame liên tiếp dưới ngưỡng
- **EMA + Rolling Average Smoothing** (ProbabilitySmoother):
  - Kết hợp EMA (phản ứng nhanh) + Rolling Average (ổn định)
  - Hiển thị cả smoothed prob và raw prob trên overlay
- **Motion Filter**: Bỏ qua prediction khi frame thay đổi lớn (mean abs diff > 12)
- **Ngưỡng camera riêng**: WARNING=0,50, DANGER=0,75 (không ảnh hưởng evaluate/report)
- **CLI arguments mới**: 12+ tham số tùy chỉnh
- Phím Space làm phím tắt thay thế cho M

### Mở rộng dataset
- Thêm ~153 ảnh fire (lửa nhỏ: bật lửa, nến, diêm, bếp gas)
- Thêm ~206 ảnh non_fire (phòng thực tế, đèn vàng, đồ vật màu ấm)
- Dataset mới: 872 fire + 450 non_fire = 1.322 ảnh

### Kết quả sau train lại
- Accuracy: 89,43% (giảm nhẹ do domain đa dạng hơn, nhưng ít báo nhầm hơn)
- Precision (fire): 94,01%
- Recall (fire): 89,71%

---

## v3.0 (2026-04-11) — Camera Realtime

### Thêm mới
- **src/camera_realtime.py**: module nhận diện cháy realtime từ webcam
  - Pipeline giữ nguyên 100% so với lúc train
  - Overlay thông tin dự đoán lên video frame (nhãn, xác suất, mức rủi ro, viền màu)
  - Cảnh báo âm thanh theo mức rủi ro (winsound, Windows)
  - Frame skip (mặc định 3) để tối ưu FPS
  - CLI arguments: --camera-id, --frame-skip, --mute
- Âm thanh chạy trên thread daemon (không block video loop)
  - CẢNH BÁO: 1 beep (800Hz, 150ms), cooldown 3 giây
  - NGUY HIỂM: 3 beep (1200–1500Hz, 120ms×3), cooldown 1,5 giây

---

## v2.1 (2026-04-11) — Dataset + Training

### Dataset
- Tải dataset: 755 ảnh fire + 244 ảnh non_fire (999 ảnh gốc)
- Xóa 37 ảnh fire trùng lặp (phát hiện bằng MD5 hash)
- Dataset cuối: 718 fire + 244 non_fire = 962 ảnh

### Sửa lỗi
- **Fix Unicode path**: cv2.imread() không đọc được đường dẫn tiếng Việt trên Windows
  - Thêm hàm imread_unicode() vào src/utils.py
  - Dùng numpy.fromfile() + cv2.imdecode() thay vì cv2.imread()
  - Cập nhật src/data_loader.py và src/predict.py

### Thay đổi
- Thêm class_weight="balanced" vào SVM (dataset lệch 2,94:1)
- Thêm run_train.bat — script chạy pipeline tự động

### Kết quả
- Accuracy: 90,16%
- Precision (fire): 96,99%, Recall (fire): 89,58%, F1 (fire): 93,14%
- Risk distribution: AN TOÀN 36, CẢNH BÁO 41, NGUY HIỂM 116

---

## v2.0 (2026-04-10) — Risk Assessment System

### Thêm mới
- Hệ thống đánh giá rủi ro 3 mức:
  - AN TOÀN: fire_prob < 0,3
  - CẢNH BÁO: 0,3 ≤ fire_prob < 0,7
  - NGUY HIỂM: fire_prob ≥ 0,7
- src/utils.py: thêm RISK_THRESHOLDS, RISK_DISPLAY, get_risk_level()
- src/evaluate.py: thêm analyze_risk_distribution() — phân tích và vẽ biểu đồ mức rủi ro
- src/predict.py: mỗi prediction hiển thị mức rủi ro

### Sửa lỗi
- Fix CLASS_NAMES order: ["fire", "non_fire"] → ["non_fire", "fire"]
  - Lỗi cũ: label 0 (non_fire) bị gán tên "fire" trong confusion matrix

### Quyết định thiết kế
- Chọn Risk Score thay vì Multi-class: không cần dataset mới, không phá pipeline, thực tế hơn

---

## v1.0 (2026-04-09) — Khởi tạo dự án

### Thêm mới
- Tạo cấu trúc thư mục dự án
- src/utils.py — cấu hình trung tâm
- src/data_loader.py — đọc ảnh từ thư mục fire/non_fire
- src/preprocess.py — tiền xử lý: resize 128×128, GaussianBlur
- src/features.py — trích xuất HOG (1.764 chiều) + HSV Color Histogram (4.096 chiều)
- src/train.py — pipeline huấn luyện SVM 7 bước
- src/evaluate.py — đánh giá Accuracy/Precision/Recall/F1 + confusion matrix
- src/predict.py — CLI predict ảnh: single, batch, folder
- README.md, requirements.txt, .gitignore

### Quyết định thiết kế
- HOG + Color Histogram làm đặc trưng
- SVM kernel RBF, C=10, gamma=scale
- Resize 128×128, StandardScaler
- Random seed = 42
