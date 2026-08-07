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
    'last_audio_time': {}
}

def process_frame(frame, show_fps, start_time, is_image=False):
    if yolo_model is None:
        cv2.putText(frame, "Error: Model file not found!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame, 0, []

    current_time = time.time()
    valid_boxes_count = 0
    audio_triggers = []
    
    with _lock:
        # FPS Optimization: Bỏ qua frame Inference, chạy Model ở ~10 FPS, Render vẽ đồ họa ở 30 FPS
        if is_image or (current_time - _global_state['inference_cache']['last_time'] >= 0.1):
            # Tối ưu RAM: model(frame) thay vì model.predict(frame) + giảm imgsz xuống 416
            results = yolo_model(frame, imgsz=416, conf=CONFIDENCE_THRESHOLD, verbose=False)
            
            _global_state['inference_cache']['boxes'] = results[0].boxes
            _global_state['inference_cache']['last_time'] = current_time
            
            class_confidences = {}
            for box in results[0].boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                if cls_id not in class_confidences or conf > class_confidences[cls_id]:
                    class_confidences[cls_id] = conf
            _global_state['inference_cache']['class_confidences'] = class_confidences
        
        boxes = _global_state['inference_cache']['boxes']
        class_confidences = _global_state['inference_cache']['class_confidences']

        # O(1) Offset Map: Thuật toán dời label O(1) thay cho O(n^2) Check Overlap cũ
        y_offset_map = {}

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            valid_boxes_count += 1
            
            eng_text = ENGLISH_SUBTITLES.get(cls_id, f"Class {cls_id}")
            display_text = f"{eng_text} ({conf*100:.1f}%)"
            (text_w, text_h), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            x_grid = x1 // 50 
            y_offset = y_offset_map.get(x_grid, 0)
            
            label_x1 = x1
            label_y1 = max(0, y1 - 25 - y_offset)
            label_x2 = x1 + text_w
            label_y2 = label_y1 + 25

            y_offset_map[x_grid] = y_offset + 30
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), (0, 255, 0), -1)
            cv2.putText(frame, display_text, (label_x1, label_y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        if not is_image:
            for cls_id, conf in class_confidences.items():
                if cls_id not in _global_state['detection_timestamps']:
                    # Tối ưu RAM: dùng deque maxlen 10 thay cho list vô hạn
                    _global_state['detection_timestamps'][cls_id] = deque(maxlen=10)
                _global_state['detection_timestamps'][cls_id].append(current_time)
                
            for cls_id in list(_global_state['detection_timestamps'].keys()):
                valid_times = [t for t in _global_state['detection_timestamps'][cls_id] if current_time - t <= 5.0]
                _global_state['detection_timestamps'][cls_id] = deque(valid_times, maxlen=10)
                
                if len(_global_state['detection_timestamps'][cls_id]) >= 7:
                    last_time = _global_state['last_audio_time'].get(cls_id, 0)
                    if current_time - last_time > COOLDOWN_SECONDS:
                        if cls_id in AUDIO_PATHS:
                            audio_triggers.append(AUDIO_PATHS[cls_id])
                        _global_state['last_audio_time'][cls_id] = current_time
                        
                    _global_state['detection_timestamps'][cls_id].clear()
        else:
            for cls_id in class_confidences.keys():
                if cls_id in AUDIO_PATHS:
                    audio_triggers.append(AUDIO_PATHS[cls_id])

    if show_fps:
        fps = 1.0 / (current_time - start_time + 1e-6)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    return frame, valid_boxes_count, audio_triggers
