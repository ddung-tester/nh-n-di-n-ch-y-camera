# Ghi chú nội dung báo cáo

> Tổng hợp số liệu, nội dung lý thuyết, và các phần viết mẫu phục vụ viết báo cáo kết thúc môn.

---

## 1. Tiêu đề

**"Xây dựng hệ thống phát hiện và cảnh báo cháy realtime từ camera sử dụng SVM kết hợp đánh giá mức độ rủi ro"**

---

## 2. Tóm tắt (Abstract)

Báo cáo trình bày phương pháp phát hiện cháy theo thời gian thực từ webcam sử dụng mô hình Support Vector Machine (SVM). Hệ thống kết hợp đặc trưng Histogram of Oriented Gradients (HOG) để nhận diện hình dạng ngọn lửa và Color Histogram trong không gian HSV để phân tích màu sắc đặc trưng của lửa. Mô hình SVM với kernel RBF được huấn luyện trên tập dữ liệu 1.322 ảnh (872 fire, 450 non_fire).

Hệ thống hoạt động realtime từ webcam, tích hợp module đánh giá mức độ rủi ro dựa trên xác suất đầu ra của SVM, phân loại mỗi frame thành 3 mức cảnh báo: AN TOÀN, CẢNH BÁO, và NGUY HIỂM. Hệ thống còn tích hợp cơ chế chống báo nhầm gồm State Machine với Hysteresis, làm mượt xác suất bằng EMA và Rolling Average, lọc chuyển động, và phát hiện thay đổi khung cảnh. Kết quả đạt Accuracy 89,43%, F1-Score (fire) 91,81%.

---

## 3. Cơ sở lý thuyết

### 3.1 Support Vector Machine (SVM)

- **Nguyên lý**: Tìm siêu phẳng (hyperplane) phân tách tối ưu giữa 2 lớp
- **Margin maximization**: Tối đa hóa khoảng cách từ siêu phẳng đến các điểm dữ liệu gần nhất
- **Kernel trick**: Ánh xạ dữ liệu lên không gian cao chiều để phân loại phi tuyến
- **RBF kernel**: K(x,y) = exp(-γ||x-y||²) — phù hợp với dữ liệu phi tuyến
- **Tham số C**: Cân bằng giữa margin rộng và phân loại đúng trên tập huấn luyện
- **Platt Scaling**: Phương pháp cho phép SVM xuất ra xác suất (probability=True), chuyển từ decision function qua hàm sigmoid thành xác suất. Đây là nền tảng để xây dựng hệ thống đánh giá rủi ro.
- **Tham khảo**: Vapnik (1995), scikit-learn documentation

### 3.2 HOG (Histogram of Oriented Gradients)

- Dalal & Triggs (2005) — ban đầu dùng cho nhận diện người đi bộ
- Chia ảnh thành các ô (cell), tính histogram hướng gradient trong mỗi ô
- Chuẩn hóa theo khối (block normalization) để tăng độ bền vững
- Phù hợp với bài toán phát hiện cháy vì nắm bắt được hình dạng ngọn lửa
- **Tham số**: 9 orientations, 16×16 pixels/cell, 2×2 cells/block
- **Đầu ra**: 1.764 chiều

### 3.3 Color Histogram

- Biểu diễn phân bố màu sắc trong ảnh
- Không gian HSV: tách Hue (màu), Saturation (độ bão hòa), Value (độ sáng)
- Lý do chọn HSV: lửa có hue đặc trưng trong khoảng 0–30 (đỏ-cam-vàng), phân tách tốt hơn RGB
- **Tham số**: 16 bins mỗi kênh, tổng 16³ = 4.096 chiều

### 3.4 StandardScaler

- Chuẩn hóa đặc trưng về mean=0, std=1
- Quan trọng cho SVM vì SVM nhạy cảm với scale của đặc trưng
- Fit trên tập train, transform trên tập test (tránh rò rỉ dữ liệu)

### 3.5 Đánh giá mức độ rủi ro (Risk Assessment)

- Sử dụng predict_proba() của SVM (Platt Scaling) để lấy xác suất
- Thay vì chỉ trả fire/non_fire, hệ thống đánh giá mức độ rủi ro
- **3 mức cảnh báo**:
  - **AN TOÀN**: fire_probability < 0,3 — mô hình tự tin ảnh không có cháy
  - **CẢNH BÁO**: 0,3 ≤ fire_probability < 0,7 — vùng không chắc chắn, cần kiểm tra thêm (có thể là khói, ánh sáng cam, hoàng hôn...)
  - **NGUY HIỂM**: fire_probability ≥ 0,7 — mô hình tự tin phát hiện lửa, cần xử lý ngay
- **Lý do chọn ngưỡng 0,3 / 0,7**:
  - Đối xứng quanh 0,5 (ranh giới quyết định)
  - 0,3 = ngưỡng thấp để không bỏ sót (ưu tiên recall)
  - 0,7 = ngưỡng cao để giảm cảnh báo sai
  - Có thể điều chỉnh tùy yêu cầu ứng dụng thực tế
- Cách tiếp cận tương tự hệ thống cảnh báo cháy trong thực tế (NFPA: Advisory, Watch, Warning)

### 3.6 Xử lý video realtime

- **Video capture**: OpenCV cv2.VideoCapture() đọc frame từ webcam
- **Frame-by-frame processing**: Mỗi frame được xử lý như một ảnh tĩnh độc lập
- **Frame skip**: Chỉ predict mỗi N frame để giữ FPS cao, frame bị skip dùng kết quả trước đó
- **Overlay visualization**: Vẽ thông tin dự đoán trực tiếp lên video frame
- **Multi-threaded audio alert**: Cảnh báo âm thanh chạy trên thread riêng để không block vòng lặp video
- **Cooldown**: Tránh phát âm thanh quá liên tục

### 3.7 Cơ chế chống báo nhầm

- **EMA + Rolling Average Smoothing**: Làm mượt xác suất qua nhiều frame, tránh spike đơn lẻ gây báo nhầm
- **Finite State Machine + Hysteresis**: 3 trạng thái AN TOÀN ↔ CẢNH BÁO ↔ NGUY HIỂM. Leo thang cần N frame liên tiếp vượt ngưỡng. Hạ bậc cần N frame dưới ngưỡng. Tránh nhảy trạng thái khi xác suất dao động.
- **Motion Filter**: Bỏ qua prediction khi frame thay đổi lớn (người đi qua, vẫy tay) vì HOG có thể nhầm motion blur thành texture lửa.
- **Scene Change Detector**: Phát hiện thay đổi khung cảnh lớn (đổi phòng, xoay camera) → reset toàn bộ hệ thống, chờ 12 frame ổn định mới predict tiếp.

---

## 4. Phương pháp

### 4.1 Sơ đồ pipeline huấn luyện (offline)

```
Ảnh đầu vào (dataset)
    ↓
Resize 128×128 + GaussianBlur
    ↓
┌─────────────────┐
│  HOG Features   │  ← Grayscale → Gradient → Histogram
│  (1.764 chiều)  │
└────────┬────────┘
         │
    Concatenate ──→ Feature Vector (5.860 chiều)
         │
┌────────┴────────┐
│ Color Histogram │  ← BGR → HSV → Histogram 3D
│  (4.096 chiều)  │
└─────────────────┘
    ↓
StandardScaler (chuẩn hóa)
    ↓
SVM (kernel=RBF, C=10, class_weight=balanced)
    ↓
Lưu model + scaler → models/*.pkl
```

### 4.2 Sơ đồ pipeline nhận diện realtime (camera)

```
Webcam (video stream)
    ↓
Scene Change Detector (thay đổi cảnh > 30 → reset hệ thống, chờ ổn định)
    ↓
Motion Filter (chuyển động > 12 → bỏ qua frame)
    ↓
Resize 128×128 + GaussianBlur
    ↓
HOG + Color Histogram (5.860 chiều) → Scaler → SVM predict_proba()
    ↓
EMA + Rolling Average Smoothing
    ↓
State Machine + Hysteresis (SAFE ↔ WARNING ↔ DANGER)
    ↓
Overlay hiển thị (tiếng Việt) + Cảnh báo âm thanh
```

### 4.3 So sánh: ảnh tĩnh vs camera realtime

| Tiêu chí | Ảnh tĩnh | Camera Realtime |
|---|---|---|
| Đầu vào | File ảnh (JPG, PNG) | Video stream webcam |
| Xử lý | 1 ảnh/lần | Liên tục (mỗi N frame) |
| Hiển thị | Terminal text + grid ảnh | Video overlay realtime |
| Cảnh báo | Không | Âm thanh tự động |
| Ứng dụng | Phân tích offline | Giám sát thời gian thực |
| Pipeline ML | Giống nhau | Giống nhau |

### 4.4 Lý do chọn Risk Score thay vì Multi-class

| Tiêu chí | Multi-class | Risk Score (đã chọn) |
|---|---|---|
| Cần thêm dataset | Cần ảnh "smoke" | Không cần |
| Ảnh hưởng pipeline | Sửa nhiều file | Chỉ thêm lớp mới bên trên |
| Tính thực tế | Trung bình | Cao (giống hệ thống cảnh báo thật) |
| Dễ giải thích | Phức tạp | Đơn giản, trực quan |

Risk Score giữ nguyên SVM core (phân loại nhị phân), thêm tầng đánh giá rủi ro dựa trên xác suất đầu ra. Cách tiếp cận này thực tế, tương tự hệ thống cảnh báo cháy trong đời thực.

---

## 5. Kết quả thực nghiệm

### 5.1 Tập dữ liệu

| Giai đoạn | Fire | Non-fire | Tổng |
|---|---|---|---|
| Ban đầu (v2.1) | 718 | 244 | 962 |
| Sau mở rộng (v3.4) | 872 | 450 | 1.322 |

- Tỷ lệ: fire chiếm 65,9%, non_fire chiếm 34,1%
- Train/Test: 80/20, phân tầng (stratified), seed=42
- Train: khoảng 1.057 ảnh, Test: 265 ảnh
- Đã loại bỏ 37 ảnh fire trùng lặp (phát hiện bằng MD5 hash)

### 5.2 Kết quả phân loại

| Chỉ số | Giá trị |
|---|---|
| Accuracy | 89,43% |
| Precision (fire) | 94,01% |
| Recall (fire) | 89,71% |
| F1-Score (fire) | 91,81% |
| Precision (non_fire) | 81,63% |
| Recall (non_fire) | 88,89% |

### 5.3 Confusion Matrix

```
              Predicted
              non_fire  fire
Actual  non_fire   80     10
        fire       18    157
```

### 5.4 Phân bố mức rủi ro (tập test)

| Mức rủi ro | Fire | Non-fire | Tổng |
|---|---|---|---|
| AN TOÀN | 3 | 69 | 72 |
| CẢNH BÁO | 18 | 13 | 31 |
| NGUY HIỂM | 154 | 8 | 162 |

**Phân tích:**
- 154/175 ảnh fire nằm ở mức NGUY HIỂM — mô hình phát hiện cháy tốt
- 69/90 ảnh non_fire nằm ở mức AN TOÀN — ít cảnh báo sai
- Mức CẢNH BÁO chứa 18 ảnh fire + 13 non_fire — vùng mô hình chưa chắc chắn

### 5.5 Biểu đồ và file kết quả

| File | Mô tả | Trạng thái |
|---|---|---|
| reports/figures/confusion_matrix.png | Ma trận nhầm lẫn (count + %) | Có |
| reports/figures/metrics_bar.png | Biểu đồ cột 4 chỉ số | Có |
| reports/figures/risk_analysis.png | Phân bố mức rủi ro | Có |
| reports/results/evaluation_results.json | Kết quả đầy đủ dạng JSON | Có |

---

## 6. Thảo luận

### 6.1 Điểm mạnh

- Pipeline đơn giản, dễ hiểu, dễ tái hiện
- HOG nắm bắt hình dạng ngọn lửa, Color Histogram nắm bắt màu sắc — hai đặc trưng bổ sung nhau tốt
- SVM cho kết quả tốt trên dữ liệu cỡ trung bình, không cần GPU, huấn luyện nhanh
- Risk Assessment cung cấp thông tin chi tiết hơn nhãn nhị phân đơn thuần
- 3 mức cảnh báo phù hợp với ứng dụng thực tế
- Hoạt động realtime từ webcam — ứng dụng giám sát thực tế
- Cảnh báo âm thanh tự động giúp phát hiện cháy kịp thời
- Pipeline nhận diện giữ nguyên 100% so với lúc huấn luyện — đảm bảo tính nhất quán
- Hệ thống chống báo nhầm đa tầng (EMA + State Machine + Motion Filter + Scene Change)

### 6.2 Hạn chế

- HOG nhạy cảm với rotation và scale — lửa ở xa hoặc lửa nhỏ khó phát hiện
- Chất lượng phụ thuộc nhiều vào tập dữ liệu huấn luyện
- Chưa áp dụng data augmentation (lật, xoay, thay đổi độ sáng)
- Ngưỡng mức rủi ro cố định, chưa tự thích nghi theo ngữ cảnh
- Xử lý từng frame độc lập, chưa phân tích ngữ cảnh thời gian (temporal context)
- Cảnh báo âm thanh chỉ hỗ trợ Windows (winsound)
- FPS phụ thuộc vào tốc độ trích xuất đặc trưng

### 6.3 Hướng phát triển

- Thêm data augmentation (lật, xoay, thay đổi độ sáng) để tăng tính đa dạng dữ liệu
- So sánh với CNN để đánh giá trade-off giữa độ chính xác và tốc độ
- GridSearchCV tìm hyperparameter tối ưu
- Thêm lớp "smoke" (khói) cho phát hiện sớm hơn
- Deploy lên web hoặc mobile
- Ngưỡng tự thích nghi dựa trên ROC curve
- Kết hợp với IoT sensors (cảm biến nhiệt, khói) cho hệ thống lai
- Thêm temporal analysis: phân tích chuỗi frame liên tiếp
- Hỗ trợ đầu vào từ file video (ngoài webcam)
- Ghi lại video kết quả (save output)
- Hỗ trợ âm thanh đa nền tảng (thay thế winsound)

---

## 7. Đoạn viết mẫu cho báo cáo

### 7.1 Giới thiệu module Camera Realtime

> Ngoài phân tích ảnh tĩnh, hệ thống được mở rộng với module nhận diện cháy realtime từ webcam (camera_realtime.py). Module này đọc video stream từ camera, xử lý từng frame qua cùng pipeline với lúc huấn luyện (resize → denoise → HOG + Color Histogram → StandardScaler → SVM), và hiển thị kết quả dự đoán trực tiếp trên video.
>
> Hệ thống cung cấp phản hồi trực quan qua: overlay trên video (nhãn phân loại, xác suất cháy, mức rủi ro), viền màu thay đổi theo mức rủi ro (xanh/cam/đỏ), và cảnh báo âm thanh tự động.
>
> Để duy trì FPS ổn định, hệ thống sử dụng cơ chế frame skip (mặc định: 3 frame). Âm thanh cảnh báo chạy trên thread riêng (daemon) với cơ chế cooldown để tránh phát liên tục gây khó chịu.

### 7.2 Giới thiệu Risk Assessment

> Ngoài phân loại nhị phân (fire/non_fire), hệ thống tích hợp module đánh giá mức độ rủi ro dựa trên xác suất đầu ra của SVM. Thay vì chỉ thông báo "có cháy" hoặc "không cháy", hệ thống phân chia kết quả thành 3 mức cảnh báo:
>
> - **AN TOÀN**: Xác suất cháy dưới 30%. Mô hình tự tin rằng ảnh không chứa lửa. Không cần hành động.
> - **CẢNH BÁO**: Xác suất cháy từ 30% đến 70%. Vùng không chắc chắn, cần kiểm tra thêm bằng mắt thường hoặc sensor bổ sung. Hệ thống phát 1 tiếng beep nhẹ (800Hz) mỗi 3 giây.
> - **NGUY HIỂM**: Xác suất cháy trên 70%. Mô hình tự tin phát hiện lửa. Cần hành động xử lý ngay. Hệ thống phát 3 tiếng beep gấp (1200–1500Hz) mỗi 3 giây.
>
> Cách tiếp cận này tương tự với hệ thống cảnh báo cháy trong thực tế, nơi các mức cảnh báo khác nhau kích hoạt các phản ứng khác nhau.

### 7.3 Phương pháp tính Risk Level

> Mô hình SVM với tùy chọn probability=True sử dụng phương pháp Platt Scaling để chuyển đổi giá trị decision function thành xác suất. Xác suất cháy (fire_probability) được sử dụng làm đầu vào cho hệ thống đánh giá rủi ro.
>
> Ngưỡng 0,3 và 0,7 được chọn dựa trên nguyên tắc: đối xứng quanh ranh giới quyết định (0,5); ngưỡng thấp (0,3) đảm bảo không bỏ sót trường hợp nghi ngờ; ngưỡng cao (0,7) giảm thiểu cảnh báo sai.

### 7.4 Cơ chế chống báo nhầm

> Khi hoạt động realtime, hệ thống phải đối mặt với nhiều nguồn nhiễu: ánh sáng ấm từ đèn vàng, motion blur khi người đi qua camera, thay đổi đột ngột khi chuyển góc camera. Để giải quyết, hệ thống triển khai cơ chế chống báo nhầm đa tầng:
>
> 1. **Làm mượt xác suất (Probability Smoothing)**: Kết hợp EMA (Exponential Moving Average) và Rolling Average qua nhiều frame liên tiếp, loại bỏ các spike đơn lẻ.
> 2. **State Machine với Hysteresis**: Trạng thái chỉ chuyển đổi khi đủ N frame liên tiếp vượt ngưỡng (leo thang) hoặc dưới ngưỡng (hạ bậc), tránh nhảy loạn.
> 3. **Motion Filter**: Bỏ qua dự đoán khi phát hiện chuyển động lớn trong frame, vì HOG có thể nhầm motion blur thành texture lửa.
> 4. **Scene Change Detector**: Khi phát hiện thay đổi khung cảnh lớn (mean absolute difference > 30), hệ thống reset toàn bộ bộ đệm và state machine, sau đó chờ 12 frame ổn định trước khi predict tiếp.

---

## 8. Tài liệu tham khảo

1. Vapnik, V. (1995). *The Nature of Statistical Learning Theory*. Springer.
2. Dalal, N., & Triggs, B. (2005). Histograms of Oriented Gradients for Human Detection. *CVPR*.
3. Celik, T., & Demirel, H. (2009). Fire detection in video sequences using a generic color model. *Fire Safety Journal*.
4. Platt, J. (1999). Probabilistic outputs for support vector machines. *Advances in Large Margin Classifiers*.
5. Scikit-learn documentation: https://scikit-learn.org/stable/modules/svm.html
6. OpenCV documentation: https://docs.opencv.org/
7. OpenCV VideoCapture: https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html
