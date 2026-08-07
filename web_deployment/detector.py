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

def is_overlap(rect1, rect2):
    x1_1, y1_1, x2_1, y2_1 = rect1
    x1_2, y1_2, x2_2, y2_2 = rect2
    if x1_1 >= x2_2 or x2_1 <= x1_2:
        return False
    if y1_1 >= y2_2 or y2_1 <= y1_2:
        return False
    return True

def process_frame(frame, show_fps, start_time, is_image=False):
    if yolo_model is None:
        cv2.putText(frame, "Error: Model file not found!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame, 0, []

    if 'detection_timestamps' not in st.session_state:
        st.session_state.detection_timestamps = {}
    if 'last_audio_time' not in st.session_state:
        st.session_state.last_audio_time = {}

    results = yolo_model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    class_confidences = {}
    valid_boxes_count = 0
    drawn_labels = [] 

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            if conf < CONFIDENCE_THRESHOLD:
                continue
                
            valid_boxes_count += 1
            if cls_id not in class_confidences or conf > class_confidences[cls_id]:
                class_confidences[cls_id] = conf
            
            eng_text = ENGLISH_SUBTITLES.get(cls_id, f"Class {cls_id}")
            display_text = f"{eng_text} ({conf*100:.1f}%)"
            
            (text_w, text_h), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            label_x1 = x1
            label_y1 = y1 - 25
            label_x2 = x1 + text_w
            label_y2 = y1

            while any(is_overlap((label_x1, label_y1, label_x2, label_y2), drawn) for drawn in drawn_labels):
                label_y1 -= (text_h + 10)
                label_y2 -= (text_h + 10)

            drawn_labels.append((label_x1, label_y1, label_x2, label_y2))
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), (0, 255, 0), -1)
            cv2.putText(frame, display_text, (label_x1, label_y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    current_time = time.time()
    audio_triggers = []

    if not is_image:
        for cls_id, conf in class_confidences.items():
            if cls_id not in st.session_state.detection_timestamps:
                st.session_state.detection_timestamps[cls_id] = []
            st.session_state.detection_timestamps[cls_id].append(current_time)
            
        for cls_id in list(st.session_state.detection_timestamps.keys()):
            st.session_state.detection_timestamps[cls_id] = [
                t for t in st.session_state.detection_timestamps[cls_id]
                if current_time - t <= 5.0
            ]
            
            if len(st.session_state.detection_timestamps[cls_id]) >= 7:
                last_time = st.session_state.last_audio_time.get(cls_id, 0)
                
                if current_time - last_time > COOLDOWN_SECONDS:
                    if cls_id in AUDIO_PATHS:
                        audio_triggers.append(AUDIO_PATHS[cls_id])
                    st.session_state.last_audio_time[cls_id] = current_time
                    
                st.session_state.detection_timestamps[cls_id] = []
    else:
        for cls_id in class_confidences.keys():
            if cls_id in AUDIO_PATHS:
                audio_triggers.append(AUDIO_PATHS[cls_id])

    if show_fps:
        fps = 1.0 / (current_time - start_time + 1e-6)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    return frame, valid_boxes_count, audio_triggers
