"""
camera_realtime.py - Nhận diện cháy realtime từ webcam + cảnh báo âm thanh.

v3.5 - Giảm false positive bằng:
  - State machine với hysteresis (tránh nhảy trạng thái)
  - EMA smoothing (trung bình trượt có trọng số)
  - Anti-flicker: cần N frame liên tiếp vượt ngưỡng mới leo thang
  - Ngưỡng camera riêng (chặt hơn ngưỡng báo cáo)
  - Scene Change Detector: phát hiện thay đổi cảnh lớn → reset toàn bộ
    smoother + state machine + chờ ổn định trước khi predict tiếp

Pipeline ML giữ nguyên 100% so với lúc train:
    frame → scene_check → motion_filter → preprocess_single
    → extract_features → scaler.transform → model.predict_proba
    → smoothing → state machine → overlay + sound

Cách dùng:
    python -m src.camera_realtime
    python -m src.camera_realtime --mute
    python -m src.camera_realtime --thresh-danger 0.90 --smoothing 20
"""

import argparse
from collections import deque
import os
import platform
import threading
import time
import cv2
import numpy as np

from src.preprocess import preprocess_single
from src.features import extract_features
from src.predict import load_model
from src.utils import RISK_DISPLAY, CLASS_NAMES, setup_logger

logger = setup_logger("camera_realtime")

# Kiểm tra winsound (chỉ có trên Windows)
_HAS_WINSOUND = False
if platform.system() == "Windows":
    try:
        import winsound
        _HAS_WINSOUND = True
    except ImportError:
        pass

# Kiểm tra Pillow cho text tiếng Việt
_HAS_PIL = False
_FONT_VI = None
try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except ImportError:
    pass


def _load_vietnamese_font(size=24):
    """Load font hỗ trợ tiếng Việt. Thử nhiều nguồn font trên Windows."""
    global _FONT_VI
    if not _HAS_PIL:
        return None

    font_candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/consola.ttf",
    ]

    for font_path in font_candidates:
        if os.path.isfile(font_path):
            try:
                _FONT_VI = ImageFont.truetype(font_path, size)
                logger.info(f"Font tiếng Việt: {os.path.basename(font_path)} (size={size})")
                return _FONT_VI
            except Exception:
                continue

    try:
        _FONT_VI = ImageFont.load_default()
        logger.warning("Dùng font mặc định (có thể không hiển thị đúng tiếng Việt)")
    except Exception:
        _FONT_VI = None
    return _FONT_VI


def _put_text_vi(img, text, position, font_size=24, color_rgb=(255, 255, 255)):
    """
    Vẽ text tiếng Việt lên ảnh OpenCV bằng Pillow.
    cv2.putText() KHÔNG hỗ trợ Unicode → dùng PIL.ImageDraw.
    """
    if not _HAS_PIL:
        cv2.putText(
            img, text, position,
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_rgb[::-1], 2, cv2.LINE_AA
        )
        return img

    global _FONT_VI
    if _FONT_VI is None or _FONT_VI.size != font_size:
        _load_vietnamese_font(font_size)

    font = _FONT_VI

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    x, y = position
    shadow_color = (0, 0, 0)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill=shadow_color)

    draw.text(position, text, font=font, fill=color_rgb)

    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    np.copyto(img, result)
    return img


# ============================================================
# CẤU HÌNH MẶC ĐỊNH
# ============================================================

# Âm thanh cảnh báo
ALERT_CONFIG = {
    "WARNING": {
        "pattern": [(800, 150)],
        "default_interval": 3.0,
    },
    "DANGER": {
        "pattern": [(1200, 120), (1500, 120), (1200, 120)],
        "default_interval": 1.5,
    },
}

OVERLAY_LABELS = {
    "fire": "CHÁY 🔥",
    "non_fire": "KHÔNG CHÁY ✓",
}

# --- Ngưỡng camera (v3.4.1) ---
CAMERA_THRESH_WARNING = 0.50
CAMERA_THRESH_DANGER  = 0.75

# --- Smoothing ---
SMOOTHING_WINDOW = 8
EMA_ALPHA        = 0.30

# --- Anti-flicker / Hysteresis ---
MIN_PREDICTIONS         = 3
MIN_CONSECUTIVE_DANGER  = 5
MIN_CONSECUTIVE_WARNING = 3
CONSECUTIVE_SAFE_TO_DROP = 3

# --- Motion filter ---
# Chặn prediction khi frame thay đổi nhiều (người đi qua, vẫy tay, vuốt tóc).
# HOG nhầm motion blur thành texture lửa → false positive.
MOTION_THRESH = 12.0  # Thấp = lọc mạnh hơn, chỉ predict khi cảnh ổn định

# --- Scene Change Detector (v3.5) ---
# Phát hiện thay đổi khung cảnh lớn (chuyển nhà→công ty, đổi camera, xoay camera).
# Khi scene change: RESET toàn bộ smoother + state machine + chờ ổn định.
# Ngưỡng scene change cao hơn motion filter nhiều vì scene change = toàn bộ
# hình ảnh thay đổi, không chỉ một phần.
SCENE_CHANGE_THRESH = 30.0     # Ngưỡng mean_abs_diff cho scene change
SCENE_STABILIZE_FRAMES = 12    # Số frame chờ ổn định sau scene change

# --- Cooldown âm thanh ---
DEFAULT_COOLDOWN = 3.0  # Giây giữa 2 lần phát âm thanh (chỉ DANGER mới phát)

# Thứ tự nghiêm trọng để so sánh
_SEVERITY = {"SAFE": 0, "WARNING": 1, "DANGER": 2}


# ============================================================
# LOAD ARTIFACTS
# ============================================================
def load_artifacts():
    """Load model SVM và scaler đã train."""
    logger.info("Đang load model và scaler...")
    model, scaler = load_model()
    logger.info("Load thành công.")
    return model, scaler


class _MotionSkip(Exception):
    """Raised khi frame bị bỏ qua do motion quá lớn."""
    pass


# ============================================================
# PREDICT 1 FRAME (pipeline giống lúc train)
# ============================================================
def predict_frame(frame, model, scaler):
    """Dự đoán 1 frame, trả về raw fire_prob."""
    processed = preprocess_single(frame)
    features = extract_features(processed)
    features = features.reshape(1, -1)
    features = scaler.transform(features)
    proba = model.predict_proba(features)[0]
    return float(proba[1])


# ============================================================
# STATE MACHINE - Quản lý trạng thái với hysteresis
# ============================================================
class FireStateMachine:
    """
    State machine 3 trạng thái: SAFE → WARNING → DANGER.

    Hysteresis:
      - Leo thang (SAFE→WARNING, WARNING→DANGER): cần smoothed_prob >= ngưỡng
        VÀ đủ consecutive frames liên tiếp vượt ngưỡng.
      - Hạ bậc (DANGER→WARNING, WARNING→SAFE): cần smoothed_prob < ngưỡng hạ
        trong consecutive_safe_to_drop frame liên tiếp.

    Điều này tránh trạng thái nhảy loạn khi xác suất dao động quanh ngưỡng.
    """

    def __init__(self, thresh_warning, thresh_danger,
                 min_consec_warning, min_consec_danger, consec_safe_to_drop):
        self.state = "SAFE"
        self.thresh_w = thresh_warning
        self.thresh_d = thresh_danger
        self.min_cw = min_consec_warning
        self.min_cd = min_consec_danger
        self.safe_drop = consec_safe_to_drop

        # Đếm frame liên tiếp cho từng hướng chuyển
        self.consec_above_warning = 0  # Liên tiếp >= thresh_warning
        self.consec_above_danger  = 0  # Liên tiếp >= thresh_danger
        self.consec_below_current = 0  # Liên tiếp dưới ngưỡng hiện tại

    def update(self, smoothed_prob):
        """Cập nhật state machine, trả về trạng thái mới."""
        # Đếm consecutive
        if smoothed_prob >= self.thresh_d:
            self.consec_above_danger += 1
            self.consec_above_warning += 1
            self.consec_below_current = 0
        elif smoothed_prob >= self.thresh_w:
            self.consec_above_danger = 0
            self.consec_above_warning += 1
            # Dưới ngưỡng DANGER → đếm hạ nếu đang DANGER
            if self.state == "DANGER":
                self.consec_below_current += 1
            else:
                self.consec_below_current = 0
        else:
            self.consec_above_danger = 0
            self.consec_above_warning = 0
            self.consec_below_current += 1

        old_state = self.state

        # --- Leo thang ---
        if self.state == "SAFE":
            if self.consec_above_warning >= self.min_cw:
                self.state = "WARNING"
                self.consec_below_current = 0
        elif self.state == "WARNING":
            if self.consec_above_danger >= self.min_cd:
                self.state = "DANGER"
                self.consec_below_current = 0

        # --- Hạ bậc ---
        if self.state == "DANGER":
            if self.consec_below_current >= self.safe_drop:
                self.state = "WARNING"
                self.consec_below_current = 0
                self.consec_above_danger = 0
        if self.state == "WARNING":
            if self.consec_below_current >= self.safe_drop:
                self.state = "SAFE"
                self.consec_below_current = 0
                self.consec_above_warning = 0

        if self.state != old_state:
            logger.info(f"STATE: {old_state} → {self.state}")

        return self.state

    def reset(self):
        """Reset về SAFE, xóa mọi counter. Gọi khi scene change."""
        old = self.state
        self.state = "SAFE"
        self.consec_above_warning = 0
        self.consec_above_danger = 0
        self.consec_below_current = 0
        if old != "SAFE":
            logger.info(f"FSM RESET: {old} → SAFE (scene change)")


# ============================================================
# SMOOTHING - EMA + Rolling Average
# ============================================================
class ProbabilitySmoother:
    """
    Kết hợp rolling average và EMA để làm mượt fire_prob.

    - Rolling average: trung bình N frame gần nhất (ổn định)
    - EMA: phản ứng nhanh hơn với thay đổi thật (cháy đột ngột)
    - Output = max(rolling_avg, ema) * 0.5 + min(...) * 0.5
      → lấy trung bình 2 giá trị, ưu tiên signal mạnh hơn một chút
    """

    def __init__(self, window_size, ema_alpha):
        self.window = deque(maxlen=window_size)
        self.ema = None
        self.alpha = ema_alpha

    def update(self, raw_prob):
        """Thêm raw_prob mới, trả về smoothed_prob."""
        self.window.append(raw_prob)

        # EMA
        if self.ema is None:
            self.ema = raw_prob
        else:
            self.ema = self.alpha * raw_prob + (1 - self.alpha) * self.ema

        # Rolling average
        rolling = sum(self.window) / len(self.window)

        # Kết hợp: trung bình rolling và EMA
        smoothed = (rolling + self.ema) / 2.0
        return smoothed

    def reset(self):
        """Reset toàn bộ history. Gọi khi scene change."""
        self.window.clear()
        self.ema = None

    @property
    def count(self):
        return len(self.window)

    @property
    def rolling_avg(self):
        if not self.window:
            return 0.0
        return sum(self.window) / len(self.window)


# ============================================================
# VẼ OVERLAY LÊN FRAME
# ============================================================
def draw_overlay(frame, prediction, mute_state=False):
    """Vẽ thông tin dự đoán lên frame (hỗ trợ tiếng Việt đầy đủ)."""
    if prediction is None:
        return frame

    output = frame.copy()
    h, w = output.shape[:2]

    risk_level = prediction["risk_level"]
    color_bgr = prediction["risk_info"]["color_bgr"]
    color_rgb = color_bgr[::-1]
    fire_prob = prediction["fire_prob"]
    class_name = prediction["class_name"]
    risk_vi = prediction["risk_info"]["vi"]
    raw_prob = prediction.get("raw_fire_prob", fire_prob)

    label_vi = OVERLAY_LABELS.get(class_name, class_name.upper())

    # Thanh nền bán trong suốt
    overlay_bar = output.copy()
    bar_height = 110
    cv2.rectangle(overlay_bar, (0, 0), (w, bar_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay_bar, 0.6, output, 0.4, 0, output)

    # Dòng 1: Kết quả + Mức rủi ro
    line1 = f"{label_vi}  |  {risk_vi}"
    _put_text_vi(output, line1, (15, 10), font_size=28, color_rgb=color_rgb)

    # Dòng 2: Smoothed prob + raw prob
    line2 = f"Smoothed: {fire_prob * 100:.1f}%  |  Raw: {raw_prob * 100:.1f}%"
    _put_text_vi(output, line2, (15, 50), font_size=20, color_rgb=color_rgb)

    # Dòng 3: Âm thanh
    if mute_state:
        sound_text = "🔇 Âm thanh: TẮT  (nhấn M/Space)"
        sound_color = (180, 180, 180)
    else:
        sound_text = "🔊 Âm thanh: BẬT  (nhấn M/Space)"
        sound_color = (0, 255, 200)
    _put_text_vi(output, sound_text, (15, 82), font_size=16, color_rgb=sound_color)

    # Viền màu theo mức risk
    border = 5
    cv2.rectangle(output, (0, 0), (w - 1, h - 1), color_bgr, border)

    return output


# ============================================================
# CẢNH BÁO ÂM THANH
# ============================================================
def play_alert_sound(risk_level):
    """Phát âm thanh cảnh báo (chạy trên thread daemon)."""
    if not _HAS_WINSOUND or risk_level not in ALERT_CONFIG:
        return
    try:
        for freq, duration in ALERT_CONFIG[risk_level]["pattern"]:
            winsound.Beep(freq, duration)
    except Exception as e:
        logger.debug(f"Lỗi phát âm thanh: {e}")


def trigger_alert_if_ready(risk_level, last_sound_time, cooldown):
    """
    Phát cảnh báo khi WARNING hoặc DANGER (với cooldown riêng).
    """
    if risk_level == "SAFE":
        return
    now = time.time()
    if now - last_sound_time.get(risk_level, 0) < cooldown:
        return
    last_sound_time[risk_level] = now
    t = threading.Thread(target=play_alert_sound, args=(risk_level,), daemon=True)
    t.start()


# ============================================================
# VÒNG LẶP CHÍNH
# ============================================================
def run_webcam_detection(
    camera_id=0,
    frame_skip=3,
    window_name="Phát hiện Cháy - Realtime",
    mute=False,
    thresh_warning=None,
    thresh_danger=None,
    smoothing_window=None,
    min_consec_danger=None,
    min_consec_warning=None,
    consec_safe_to_drop=None,
    cooldown_seconds=None,
    ema_alpha=None,
):
    """Mở webcam và chạy nhận diện cháy realtime."""

    # --- Resolve defaults ---
    tw   = thresh_warning     or CAMERA_THRESH_WARNING
    td   = thresh_danger      or CAMERA_THRESH_DANGER
    sw   = smoothing_window   or SMOOTHING_WINDOW
    mcd  = min_consec_danger  or MIN_CONSECUTIVE_DANGER
    mcw  = min_consec_warning or MIN_CONSECUTIVE_WARNING
    csd  = consec_safe_to_drop or CONSECUTIVE_SAFE_TO_DROP
    cool = cooldown_seconds   or DEFAULT_COOLDOWN
    alpha = ema_alpha         or EMA_ALPHA

    # --- Load model ---
    try:
        model, scaler = load_artifacts()
    except Exception as e:
        logger.error(f"Không load được model/scaler: {e}")
        return

    _load_vietnamese_font(28)

    # --- Khởi tạo smoother + state machine ---
    smoother = ProbabilitySmoother(window_size=sw, ema_alpha=alpha)
    fsm = FireStateMachine(
        thresh_warning=tw, thresh_danger=td,
        min_consec_warning=mcw, min_consec_danger=mcd,
        consec_safe_to_drop=csd,
    )
    last_sound_time = {}

    # --- Log config ---
    logger.info(f"Ngưỡng  → WARNING>={tw:.2f}  DANGER>={td:.2f}")
    logger.info(f"Smoothing={sw}  EMA_α={alpha:.2f}  MinPred={MIN_PREDICTIONS}")
    logger.info(f"Hysteresis → consec_warn={mcw}  consec_danger={mcd}  safe_drop={csd}")
    logger.info(f"Scene change → thresh={SCENE_CHANGE_THRESH}  stabilize={SCENE_STABILIZE_FRAMES}f")
    logger.info(f"Cooldown âm thanh: {cool}s (chỉ DANGER)")
    if mute:
        logger.info("Âm thanh: TẮT (--mute)")
    elif not _HAS_WINSOUND:
        logger.warning("winsound không khả dụng - tắt âm thanh.")
        mute = True
    else:
        logger.info("Âm thanh: BẬT")
    logger.info("Phím tắt: Q=thoát | M/Space=bật/tắt âm thanh")

    # --- Mở webcam ---
    logger.info(f"Đang mở webcam (ID={camera_id})...")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error(f"Không mở được webcam ID={camera_id}.")
        return

    cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Webcam sẵn sàng: {cap_w}x{cap_h}  frame_skip={frame_skip}")

    frame_count = 0
    last_prediction = None
    fps_time = time.time()
    prev_gray = None  # Frame trước (grayscale) cho motion detection
    scene_stabilize_remaining = 0  # Đếm frame chờ ổn định sau scene change

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame_count += 1

            # --- Predict mỗi N frame ---
            if frame_count % frame_skip == 0:
                try:
                    # ==========================================
                    # BƯỚC 1: Scene Change Detection (v3.5)
                    # ==========================================
                    cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    motion_score = 0.0
                    is_scene_change = False
                    skip_motion = False

                    if prev_gray is not None:
                        diff = cv2.absdiff(cur_gray, prev_gray)
                        motion_score = float(np.mean(diff))

                        # --- Scene change: thay đổi cảnh LỚN ---
                        if motion_score > SCENE_CHANGE_THRESH:
                            is_scene_change = True
                            scene_stabilize_remaining = SCENE_STABILIZE_FRAMES
                            smoother.reset()
                            fsm.reset()
                            logger.warning(
                                f"⚠ SCENE CHANGE detected! score={motion_score:.1f} "
                                f"> {SCENE_CHANGE_THRESH}  → RESET smoother + FSM, "
                                f"stabilizing {SCENE_STABILIZE_FRAMES} frames"
                            )
                            # Force prediction to SAFE
                            last_prediction = {
                                "label": 0,
                                "class_name": CLASS_NAMES[0],
                                "fire_prob": 0.0,
                                "raw_fire_prob": 0.0,
                                "risk_level": "SAFE",
                                "risk_info": RISK_DISPLAY["SAFE"],
                            }
                            prev_gray = cur_gray
                            raise _MotionSkip()

                        # --- Motion filter: chuyển động nhỏ hơn (người đi qua) ---
                        elif motion_score > MOTION_THRESH:
                            skip_motion = True
                            logger.debug(f"Motion skip: score={motion_score:.1f} > {MOTION_THRESH}")

                    prev_gray = cur_gray

                    # ==========================================
                    # BƯỚC 2: Stabilization sau scene change
                    # ==========================================
                    if scene_stabilize_remaining > 0:
                        scene_stabilize_remaining -= 1
                        logger.debug(
                            f"Stabilizing... {scene_stabilize_remaining} frames remaining"
                        )
                        # Trong thời gian ổn định: vẫn predict nhưng CHỈ feed vào
                        # smoother để tích lũy data mới, KHÔNG cho FSM cập nhật
                        raw_prob = predict_frame(frame, model, scaler)
                        smoother.update(raw_prob)
                        # Giữ SAFE trong suốt thời gian ổn định
                        last_prediction = {
                            "label": 0,
                            "class_name": CLASS_NAMES[0],
                            "fire_prob": smoother.rolling_avg,
                            "raw_fire_prob": raw_prob,
                            "risk_level": "SAFE",
                            "risk_info": RISK_DISPLAY["SAFE"],
                        }
                        logger.info(
                            f"#{frame_count} [STABILIZING] raw={raw_prob:.3f} "
                            f"rolling={smoother.rolling_avg:.3f} "
                            f"remain={scene_stabilize_remaining}"
                        )
                        raise _MotionSkip()

                    # ==========================================
                    # BƯỚC 3: Motion filter (chuyển động nhỏ)
                    # ==========================================
                    if skip_motion:
                        raise _MotionSkip()

                    # ==========================================
                    # BƯỚC 4: Predict bình thường
                    # ==========================================
                    raw_prob = predict_frame(frame, model, scaler)
                    smoothed = smoother.update(raw_prob)

                    # Chưa đủ data → SAFE
                    if smoother.count < MIN_PREDICTIONS:
                        state = "SAFE"
                    else:
                        state = fsm.update(smoothed)

                    label = 1 if state == "DANGER" else (1 if state == "WARNING" else 0)

                    last_prediction = {
                        "label": label,
                        "class_name": CLASS_NAMES[label],
                        "fire_prob": smoothed,
                        "raw_fire_prob": raw_prob,
                        "risk_level": state,
                        "risk_info": RISK_DISPLAY[state],
                    }

                    # Debug log
                    logger.info(
                        f"#{frame_count} raw={raw_prob:.3f} "
                        f"smooth={smoothed:.3f} ema={smoother.ema:.3f} "
                        f"state={state} motion={motion_score:.1f} "
                        f"c_warn={fsm.consec_above_warning} "
                        f"c_dang={fsm.consec_above_danger} "
                        f"c_safe={fsm.consec_below_current}"
                    )

                    # Âm thanh — chỉ DANGER, sau cooldown
                    if not mute:
                        trigger_alert_if_ready(state, last_sound_time, cool)

                except _MotionSkip:
                    pass  # Bỏ qua frame motion/scene change, giữ nguyên trạng thái cũ
                except Exception as e:
                    logger.warning(f"Lỗi predict #{frame_count}: {e}")

            # --- Overlay ---
            display = draw_overlay(frame, last_prediction, mute_state=mute)

            # FPS
            now = time.time()
            fps = 1.0 / max(now - fps_time, 1e-6)
            fps_time = now
            cv2.putText(
                display, f"FPS: {fps:.0f}", (15, display.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA
            )

            cv2.imshow(window_name, display)

            # --- Phím bấm ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                logger.info("Thoát (Q).")
                break
            if key == ord("m") or key == ord("M") or key == 32:
                mute = not mute
                logger.info(f"Âm thanh: {'TẮT' if mute else 'BẬT'}")

    except KeyboardInterrupt:
        logger.info("Ctrl+C - thoát.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Đã giải phóng webcam.")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Nhận diện cháy realtime từ webcam bằng SVM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python -m src.camera_realtime
  python -m src.camera_realtime --thresh-danger 0.90 --smoothing 20
  python -m src.camera_realtime --thresh-warning 0.60 --thresh-danger 0.90 --min-consec-danger 6 --cooldown 3
  python -m src.camera_realtime --mute --frame-skip 5
"""
    )
    parser.add_argument("--camera-id", type=int, default=0,
                        help="ID webcam (mặc định: 0)")
    parser.add_argument("--frame-skip", type=int, default=3,
                        help="Predict mỗi N frame (mặc định: 3)")
    parser.add_argument("--window-name", type=str, default="Phát hiện Cháy - Realtime",
                        help="Tên cửa sổ hiển thị")
    parser.add_argument("--mute", action="store_true",
                        help="Tắt âm thanh cảnh báo")

    # --- Ngưỡng ---
    parser.add_argument("--thresh-warning", type=float, default=None,
                        help=f"Ngưỡng WARNING (mặc định: {CAMERA_THRESH_WARNING})")
    parser.add_argument("--thresh-danger", type=float, default=None,
                        help=f"Ngưỡng DANGER (mặc định: {CAMERA_THRESH_DANGER})")

    # --- Smoothing ---
    parser.add_argument("--smoothing", type=int, default=None,
                        help=f"Rolling window size (mặc định: {SMOOTHING_WINDOW})")
    parser.add_argument("--ema-alpha", type=float, default=None,
                        help=f"EMA alpha, nhỏ=mượt hơn (mặc định: {EMA_ALPHA})")

    # --- Anti-flicker ---
    parser.add_argument("--min-consec-danger", type=int, default=None,
                        help=f"Số frame liên tiếp >= danger trước khi báo DANGER (mặc định: {MIN_CONSECUTIVE_DANGER})")
    parser.add_argument("--min-consec-warning", type=int, default=None,
                        help=f"Số frame liên tiếp >= warning trước khi báo WARNING (mặc định: {MIN_CONSECUTIVE_WARNING})")
    parser.add_argument("--safe-drop", type=int, default=None,
                        help=f"Số frame liên tiếp dưới ngưỡng để hạ trạng thái (mặc định: {CONSECUTIVE_SAFE_TO_DROP})")

    # --- Cooldown ---
    parser.add_argument("--cooldown", type=float, default=None,
                        help=f"Giây giữa 2 lần phát âm thanh DANGER (mặc định: {DEFAULT_COOLDOWN})")

    args = parser.parse_args()

    tw = args.thresh_warning or CAMERA_THRESH_WARNING
    td = args.thresh_danger  or CAMERA_THRESH_DANGER
    sw = args.smoothing      or SMOOTHING_WINDOW

    print("=" * 60)
    print("  🔥📷 PHÁT HIỆN CHÁY REALTIME - SVM (v3.5)")
    print("  Pipeline: HOG + HSV → SVM → Smoothing → State Machine")
    print(f"  WARNING>={tw:.2f}  DANGER>={td:.2f}  Smoothing={sw}f")
    print("  Phím tắt: Q=Thoát | M/Space=Bật/Tắt âm thanh")
    print("=" * 60)

    run_webcam_detection(
        camera_id=args.camera_id,
        frame_skip=args.frame_skip,
        window_name=args.window_name,
        mute=args.mute,
        thresh_warning=args.thresh_warning,
        thresh_danger=args.thresh_danger,
        smoothing_window=args.smoothing,
        min_consec_danger=args.min_consec_danger,
        min_consec_warning=args.min_consec_warning,
        consec_safe_to_drop=args.safe_drop,
        cooldown_seconds=args.cooldown,
        ema_alpha=args.ema_alpha,
    )
