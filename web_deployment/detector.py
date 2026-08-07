import cv2
import time
import os
import streamlit as st
from ultralytics import YOLO

# ==========================================
# CẤU HÌNH & CACHE MÔ HÌNH
# ==========================================
@st.cache_resource
def load_model():
    # SỬA DÒNG NÀY: Trỏ đúng vào tên file weights YOLOv11 của bạn (ví dụ: 'best.pt')
    model = YOLO('best.pt') 
    return model

# Thư mục chứa file âm thanh (Ví dụ: 'audios/Cam_Di_Nguoc_Chieu.mp3')
AUDIO_DIR = "audios"
COOLDOWN_TIME = 5.0 # Khoảng thời gian (giây) đợi trước khi phát lại cùng 1 loại cảnh báo

# ==========================================
# HÀM PHỤ TRỢ: TRÁNH ĐÈ NHÃN TRÊN ẢNH
# ==========================================
def is_overlap(box1, box2):
    """Kiểm tra xem 2 khung nhãn (x1, y1, x2, y2) có bị đè lên nhau không."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2

    if b1_x2 <= b2_x1 or b2_x2 <= b1_x1:
        return False
    if b1_y2 <= b2_y1 or b2_y2 <= b1_y1:
        return False
    return True

# ==========================================
# HÀM XỬ LÝ CỐT LÕI
# ==========================================
def process_frame(frame, show_fps=False, start_time=0, mode="image", current_video_time=None):
    model = load_model()
    results = model(frame, verbose=False)
    
    processed_frame = frame.copy()
    drawn_labels = []
    audio_triggers = [] # Mảng chứa nhiều âm thanh để đẩy ra cho utils.py phát lần lượt
    box_count = 0

    # Khởi tạo bộ đếm thời gian (cooldown) để không spam âm thanh liên tục
    if 'last_audio_time' not in st.session_state:
        st.session_state.last_audio_time = {}

    # Xác định thời gian mốc chuẩn hiện tại
    if mode == "video" and current_video_time is not None:
        current_time = current_video_time # Dùng thời gian chuẩn của video
    else:
        current_time = time.time() # Dùng thời gian thực tế của camera

    if results:
        for r in results:
            boxes = r.boxes
            for box in boxes:
                box_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls] # Lấy tên biển báo

                # 1. Vẽ Khung (Bounding Box)
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 2. Xử lý vị trí Nhãn (Tránh lỗi văng khi nhiều biển báo đè nhau)
                label_text = f"{label} {conf:.2f}"
                (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                label_x1, label_y1 = x1, y1 - 25 if y1 - 25 > 0 else y1 + 5
                label_x2, label_y2 = label_x1 + text_w, label_y1 + text_h
                
                while any(is_overlap((label_x1, label_y1, label_x2, label_y2), drawn) for drawn in drawn_labels):
                    label_y1 -= (text_h + 5)
                    label_y2 -= (text_h + 5)
                    if label_y1 < 0: # Tránh văng khỏi cạnh trên màn hình
                        label_y1 = y2 + 5
                        label_y2 = label_y1 + text_h
                        break

                drawn_labels.append((label_x1, label_y1, label_x2, label_y2))
                
                cv2.rectangle(processed_frame, (label_x1, label_y1 - text_h - 5), (label_x1 + text_w, label_y1 + 5), (0, 255, 0), -1)
                cv2.putText(processed_frame, label_text, (label_x1, label_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                # 3. XỬ LÝ ÂM THANH (Thu thập để phát lần lượt)
                # Đảm bảo tên file âm thanh khớp với tên class (VD: label là "Stop" -> "audios/Stop.mp3")
                audio_path = os.path.join(AUDIO_DIR, f"{label}.mp3")
                
                # Kiểm tra xem biển báo này có đang trong thời gian "chờ" (cooldown) không
                last_played = st.session_state.last_audio_time.get(label, -COOLDOWN_TIME)
                if (current_time - last_played) >= COOLDOWN_TIME:
                    if os.path.exists(audio_path):
                        audio_triggers.append(audio_path) # NẾU THẤY NHIỀU BIỂN CÙNG LÚC SẼ THÊM VÀO ĐÂY -> [audio1, audio2]
                        st.session_state.last_audio_time[label] = current_time
                    else:
                        print(f"Bỏ qua âm thanh: Không tìm thấy file {audio_path}")

    # Hiển thị FPS
    if show_fps and start_time > 0 and mode != "image":
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # TRẢ VỀ: Khung ảnh đã vẽ, Số lượng biển, MẢNG chứa các file âm thanh cần phát
    return processed_frame, box_count, audio_triggers
