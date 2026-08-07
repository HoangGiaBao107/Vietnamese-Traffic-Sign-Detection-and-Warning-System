import cv2
import time
import os
import streamlit as st
from ultralytics import YOLO
from config import MODEL_PATH, AUDIO_PATHS, ENGLISH_SUBTITLES, CONFIDENCE_THRESHOLD, COOLDOWN_SECONDS

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"WARNING: Model file not found at {MODEL_PATH}")
        return None
    try:
        model = YOLO(MODEL_PATH, task='detect')
        return model
    except Exception as e:
        print(f"Model loading error: {e}")
        return None

yolo_model = load_model()

# Bộ nhớ đệm dùng cho Camera để lưu vị trí Bounding Box khi nhảy khung hình (Frame Skipping)
class CameraCache:
    def __init__(self):
        self.frame_count = 0
        self.last_boxes = []
        self.last_valid_count = 0
        self.detection_timestamps = {}
        self.last_audio_time = {}

cam_cache = CameraCache()

def process_frame(frame, show_fps, start_time, is_image=False, skip_frames=3):
    if yolo_model is None:
        cv2.putText(frame, "Error: Model file not found!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame, 0, []

    cam_cache.frame_count += 1
    current_time = time.time()
    audio_triggers = []

    # CHỈ CHẠY YOLO MỖI `skip_frames` KHUNG HÌNH (Hoặc khi phân tích ảnh tĩnh)
    if is_image or (cam_cache.frame_count % skip_frames == 0):
        # imgsz=320 giúp CPU xử lý nhanh gấp 3 lần so với mặc định
        results = yolo_model.predict(frame, conf=CONFIDENCE_THRESHOLD, imgsz=320, verbose=False)
        
        cached_boxes = []
        class_confidences = {}
        valid_boxes_count = 0

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                    
                valid_boxes_count += 1
                cached_boxes.append((x1, y1, x2, y2, conf, cls_id))
                
                if cls_id not in class_confidences or conf > class_confidences[cls_id]:
                    class_confidences[cls_id] = conf

        cam_cache.last_boxes = cached_boxes
        cam_cache.last_valid_count = valid_boxes_count

        # Xử lý Logic Âm thanh
        if not is_image:
            for cls_id in class_confidences.keys():
                if cls_id not in cam_cache.detection_timestamps:
                    cam_cache.detection_timestamps[cls_id] = []
                cam_cache.detection_timestamps[cls_id].append(current_time)
                
            for cls_id in list(cam_cache.detection_timestamps.keys()):
                cam_cache.detection_timestamps[cls_id] = [
                    t for t in cam_cache.detection_timestamps[cls_id]
                    if current_time - t <= 5.0
                ]
                
                # Giảm ngưỡng số lần phát hiện xuống 3 vì đã dùng Frame Skipping
                if len(cam_cache.detection_timestamps[cls_id]) >= 3:
                    last_time = cam_cache.last_audio_time.get(cls_id, 0)
                    
                    if current_time - last_time > COOLDOWN_SECONDS:
                        if cls_id in AUDIO_PATHS:
                            audio_triggers.append(AUDIO_PATHS[cls_id])
                        cam_cache.last_audio_time[cls_id] = current_time
                        
                    cam_cache.detection_timestamps[cls_id] = []
        else:
            for cls_id in class_confidences.keys():
                if cls_id in AUDIO_PATHS:
                    audio_triggers.append(AUDIO_PATHS[cls_id])

    # VẼ BOUNDING BOX (Dùng lại danh sách box cũ nếu rơi vào khung hình bị bỏ qua)
    for (x1, y1, x2, y2, conf, cls_id) in cam_cache.last_boxes:
        eng_text = ENGLISH_SUBTITLES.get(cls_id, f"Class {cls_id}")
        display_text = f"{eng_text} ({conf*100:.1f}%)"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, max(y1 - 25, 0)), (x1 + len(display_text)*10, y1), (0, 255, 0), -1)
        cv2.putText(frame, display_text, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    if show_fps:
        fps = 1.0 / (current_time - start_time + 1e-6)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    return frame, cam_cache.last_valid_count, audio_triggers
