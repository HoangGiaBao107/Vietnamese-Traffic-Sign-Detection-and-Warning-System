import cv2
import time
import os
import threading
from collections import deque
import streamlit as st
from ultralytics import YOLO
from config import MODEL_PATH, AUDIO_PATHS, ENGLISH_SUBTITLES, CONFIDENCE_THRESHOLD, COOLDOWN_SECONDS

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return YOLO(MODEL_PATH, task='detect')
    except Exception:
        return None

yolo_model = load_model()

# Tạo biến toàn cục (Thread-safe) để dùng được trong Callback WebRTC (Khắc phục lỗi Context)
_lock = threading.Lock()
_global_state = {
    'inference_cache': {'last_time': 0, 'boxes': [], 'class_confidences': {}},
    'detection_timestamps': {},
    'last_audio_time': {},
    'last_frame_time': 0,
    'smooth_fps': 0.0
}

def process_frame(frame, show_fps, start_time, mode="camera", current_video_time=0.0):
    if yolo_model is None:
        cv2.putText(frame, "Error: Model not found!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame, 0, []

    if 'last_audio_time' not in st.session_state:
        st.session_state.last_audio_time = {}

    results = yolo_model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    valid_boxes_count = 0
    drawn_labels = [] 
    detected_class_ids = set()

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            if conf < CONFIDENCE_THRESHOLD:
                continue
                
            valid_boxes_count += 1
            detected_class_ids.add(cls_id)
            
            eng_text = ENGLISH_SUBTITLES.get(cls_id, f"Class {cls_id}")
            display_text = f"{eng_text} ({conf*100:.1f}%)"
            
            (text_w, text_h), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_x1, label_y1 = x1, y1 - 25
            label_x2, label_y2 = x1 + text_w, y1

            while any(is_overlap((label_x1, label_y1, label_x2, label_y2), drawn) for drawn in drawn_labels):
                label_y1 -= (text_h + 10)
                label_y2 -= (text_h + 10)

            drawn_labels.append((label_x1, label_y1, label_x2, label_y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), (0, 255, 0), -1)
            cv2.putText(frame, display_text, (label_x1, label_y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    # Xử lý mốc thời gian tùy theo Mode (Camera thật hay Video offline)
    current_time_val = current_video_time if mode == "video" else time.time()
    audio_triggers = []

    for cls_id in detected_class_ids:
        if cls_id in AUDIO_PATHS:
            if mode == "image":
                audio_triggers.append(AUDIO_PATHS[cls_id])
            else:
                last_time = st.session_state.last_audio_time.get(cls_id, -999)
                if current_time_val - last_time > COOLDOWN_SECONDS:
                    audio_triggers.append(AUDIO_PATHS[cls_id])
                    st.session_state.last_audio_time[cls_id] = current_time_val

    if show_fps and mode != "image":
        fps = 1.0 / (time.time() - start_time + 1e-6)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    return frame, valid_boxes_count, audio_triggers
