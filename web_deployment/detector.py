import cv2
import gc
import time
import os
from ultralytics import YOLO
from config import (
    MODEL_PATH, AUDIO_PATHS, ENGLISH_SUBTITLES,
    CONFIDENCE_THRESHOLD, COOLDOWN_SECONDS, FRAMES_TO_CONFIRM,
    MAX_INFER_SIZE
)
import streamlit as st


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


def _resize_for_inference(frame):
    """
    TỐI ƯU BỘ NHỚ:
    Video full-HD/4K sẽ khiến YOLO cấp phát rất nhiều RAM/VRAM cho mỗi frame,
    và nếu chạy hàng nghìn frame liên tục, mức tiêu thụ này rất dễ khiến
    tiến trình bị hệ điều hành/Streamlit Cloud kill do hết bộ nhớ (OOM).
    Hàm này resize frame về cạnh dài tối đa MAX_INFER_SIZE trước khi predict,
    sau đó trả về thêm hệ số scale để vẽ box đúng vị trí trên frame gốc.
    """
    h, w = frame.shape[:2]
    longest_side = max(h, w)
    if longest_side <= MAX_INFER_SIZE:
        return frame, 1.0
    scale = MAX_INFER_SIZE / float(longest_side)
    resized = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def process_frame(frame, show_fps, start_time, video_time=0.0, is_image=False):
    if yolo_model is None:
        cv2.putText(frame, "Error: Model file not found!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame, 0, []

    if 'last_audio_time' not in st.session_state:
        st.session_state.last_audio_time = {}
    if 'detection_buffer' not in st.session_state:
        st.session_state.detection_buffer = {}
    # Đếm số frame đã xử lý để biết khi nào cần dọn rác định kỳ
    if 'frames_processed_since_gc' not in st.session_state:
        st.session_state.frames_processed_since_gc = 0

    # ---- TỐI ƯU BỘ NHỚ: predict trên bản resize thay vì frame gốc ----
    infer_frame, scale = _resize_for_inference(frame)
    results = yolo_model.predict(infer_frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

    valid_boxes_count = 0
    drawn_labels = []
    detected_class_ids = set()

    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Box được model trả về theo kích thước ảnh đã resize -> quy đổi lại về kích thước gốc
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale)
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            if conf < CONFIDENCE_THRESHOLD:
                continue

            valid_boxes_count += 1
            detected_class_ids.add(cls_id)

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

    # ---- TỐI ƯU BỘ NHỚ: giải phóng kết quả predict ngay sau khi dùng xong ----
    del results
    del infer_frame

    current_sys_time = time.time()
    audio_triggers = []

    # Cập nhật bộ đệm (Buffer) lọc nhiễu cho video
    if not is_image:
        for cls_id in AUDIO_PATHS.keys():
            if cls_id in detected_class_ids:
                st.session_state.detection_buffer[cls_id] = st.session_state.detection_buffer.get(cls_id, 0) + 1
            else:
                st.session_state.detection_buffer[cls_id] = 0

    # Xử lý phát âm thanh
    for cls_id in detected_class_ids:
        if cls_id in AUDIO_PATHS:
            if is_image:
                audio_triggers.append(AUDIO_PATHS[cls_id])
            else:
                if st.session_state.detection_buffer.get(cls_id, 0) >= FRAMES_TO_CONFIRM:
                    last_time = st.session_state.last_audio_time.get(cls_id, -COOLDOWN_SECONDS - 1)
                    if video_time - last_time > COOLDOWN_SECONDS:
                        audio_triggers.append(AUDIO_PATHS[cls_id])
                        st.session_state.last_audio_time[cls_id] = video_time

    if show_fps:
        fps = 1.0 / (current_sys_time - start_time + 1e-6)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # ---- TỐI ƯU BỘ NHỚ: dọn rác định kỳ (không gọi mỗi frame vì gc.collect() khá tốn CPU) ----
    if not is_image:
        st.session_state.frames_processed_since_gc += 1
        from config import GC_COLLECT_EVERY_N_FRAMES
        if st.session_state.frames_processed_since_gc >= GC_COLLECT_EVERY_N_FRAMES:
            gc.collect()
            st.session_state.frames_processed_since_gc = 0

    return frame, valid_boxes_count, audio_triggers
