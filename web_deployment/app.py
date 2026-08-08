import streamlit as st
import cv2
import time
import numpy as np
import tempfile
import queue
import os
from utils import inject_custom_css, trigger_audio_queue
from detector import process_frame

st.set_page_config(page_title="Phát hiện Biển báo / Traffic Sign Detection", layout="wide")

inject_custom_css()

if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

def change_page(page_name):
    st.session_state.current_page = page_name
    st.rerun()

def render_home():
    st.markdown("<div class='author-badge'>Project Owner: Hoàng Gia Bảo</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='main-title' style='text-align: center;'>
            Mô hình Phát hiện và Cảnh báo Biển báo Giao thông Ứng dụng trên Thiết bị Biên<br>
            <span style='font-size: 0.55em; font-weight: normal;'>Traffic Sign Detection and Warning Model Applied on Edge Devices</span>
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div class='sub-title' style='text-align: center; margin-bottom: 2rem;'>
            Xử lý thời gian thực / Real-time processing with YOLOv11n
        </div>
        """, 
        unsafe_allow_html=True
    )
    col2, col3 = st.columns(2, gap="large")
    card_style = "height: 280px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; padding: 10px;"
            
    with col2:
        st.markdown(
            f"""
            <div class='card' style='{card_style}'>
                <div style='font-size: 4.5rem; margin-bottom: 10px; line-height: 1;'>🎥</div>
                <div style='font-size: 1.6rem; font-weight: bold; margin-bottom: 8px;'>
                    Phân tích Video<br><span style='font-size: 0.65em; font-weight: normal;'>Video Analysis</span>
                </div>
                <div style='font-size: 1.1rem; color: #555;'>Tải tệp video lên / Upload Video File</div>
            </div>
            """, unsafe_allow_html=True
        )
        if st.button("Tải Video lên / Upload Video", use_container_width=True):
            change_page('video')
            
    with col3:
        st.markdown(
            f"""
            <div class='card' style='{card_style}'>
                <div style='font-size: 4.5rem; margin-bottom: 10px; line-height: 1;'>🖼️</div>
                <div style='font-size: 1.6rem; font-weight: bold; margin-bottom: 8px;'>
                    Phân tích Ảnh<br><span style='font-size: 0.65em; font-weight: normal;'>Image Analysis</span>
                </div>
                <div style='font-size: 1.1rem; color: #555;'>Tải tệp ảnh lên / Upload Image File</div>
            </div>
            """, unsafe_allow_html=True
        )
        if st.button("Tải Ảnh lên / Upload Image", use_container_width=True):
            change_page('image')

def render_video():
    if st.button("🔙 Về trang chủ / Back to Home"):
        change_page('home')
        
    st.header("Xử lý Video / Video Processing Engine")
    uploaded_video = st.file_uploader("Chọn tệp video / Choose a video file", type=['mp4', 'avi', 'mov'])
    
    if uploaded_video:
        show_fps = st.toggle("Hiển thị FPS trên Video", key="fps_vid")
        process_btn = st.button("Bắt đầu Phân tích / Start Analysis")
        
        if process_btn:
            # 1. Lưu video gốc vào bộ nhớ tạm
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            
            cap = cv2.VideoCapture(tfile.name)
            fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 2. Cấu hình file đầu ra (Dùng định dạng WebM - vp80 để chạy trực tiếp được trên trình duyệt)
            out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
            fourcc = cv2.VideoWriter_fourcc(*'vp80')
            out = cv2.VideoWriter(out_file.name, fourcc, fps_video, (width, height))
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            frame_idx = 0
            
            with st.spinner("Hệ thống đang xử lý, vui lòng đợi..."):
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: 
                        break
                    
                    frame_idx += 1
                    start_time = time.time()
                    
                    # Truyền is_image=False, video xử lý offline nên ta không gọi audio ở đây
                    processed_frame, _, _ = process_frame(frame, show_fps, start_time, is_image=False)
                    out.write(processed_frame)
                    
                    # Cập nhật UI mỗi 5 frame để tăng tốc độ xử lý
                    if total_frames > 0 and frame_idx % 5 == 0:
                        progress_bar.progress(min(frame_idx / total_frames, 1.0))
                        status_text.text(f"Đang phân tích khung hình: {frame_idx}/{total_frames}")
                        
            cap.release()
            out.release()
            
            progress_bar.progress(1.0)
            status_text.text("Xử lý hoàn tất! Đang tải trình phát video...")
            
            # 3. Phát video bằng Streamlit
            st.success("Xử lý video thành công!")
            st.video(out_file.name)
            
            # 4. Thêm nút tải video về máy
            with open(out_file.name, "rb") as file:
                st.download_button(
                    label="⬇️ Tải Video đã phân tích về máy",
                    data=file,
                    file_name="detected_video.webm",
                    mime="video/webm"
                )

def render_image():
    if st.button("🔙 Về trang chủ / Back to Home"):
        change_page('home')
        
    st.header("Phân tích Ảnh tĩnh / Static Image Analysis")
    uploaded_image = st.file_uploader("Chọn tệp ảnh / Choose an image file", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_image:
        # Nơi chứa trình phát âm thanh ẩn
        audio_ph = st.empty()
        
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        
        with st.spinner("Đang phân tích ảnh... / Analyzing image..."):
            processed_frame, box_count, audio_triggers = process_frame(frame, show_fps=False, start_time=time.time(), is_image=True)
            
            # Gọi hàm phát audio (Hỗ trợ nhiều biển báo cùng lúc)
            trigger_audio_queue(audio_triggers, audio_ph)
            
            st.image(processed_frame, channels="BGR", width=800)
            
            if box_count == 0:
                st.warning("Không phát hiện biển báo giao thông nào trong ảnh. / No Traffic Sign Detected.")
            else:
                st.success(f"Phân tích hoàn tất! Phát hiện chính xác {box_count} biển báo. / Analysis complete!")

if __name__ == "__main__":
    if st.session_state.current_page == 'home': 
        render_home()
    elif st.session_state.current_page == 'video': 
        render_video()
    elif st.session_state.current_page == 'image': 
        render_image()
