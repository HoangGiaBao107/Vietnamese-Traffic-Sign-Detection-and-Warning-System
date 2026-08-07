import streamlit as st
import cv2
import time
import numpy as np
import tempfile
import queue
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
from utils import inject_custom_css, inject_js_audio_manager, trigger_audio_queue
from detector import process_frame

st.set_page_config(page_title="Phát hiện Biển báo / Traffic Sign Detection", layout="wide")

inject_custom_css()
inject_js_audio_manager()

# Khởi tạo Hàng đợi Audio Thread-safe cho WebRTC
if 'webrtc_audio_queue' not in st.session_state:
    st.session_state.webrtc_audio_queue = queue.Queue()

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
    col1, col2, col3 = st.columns(3, gap="large")
    card_style = "height: 280px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; padding: 10px;"

    with col1:
        st.markdown(
            f"""
            <div class='card' style='{card_style}'>
                <div style='font-size: 4.5rem; margin-bottom: 10px; line-height: 1;'>📸</div>
                <div style='font-size: 1.6rem; font-weight: bold; margin-bottom: 8px;'>
                    Camera Trực tiếp<br><span style='font-size: 0.65em; font-weight: normal;'>Real-time Camera</span>
                </div>
                <div style='font-size: 1.1rem; color: #555;'>Xử lý trên thiết bị / Edge Device Processing</div>
            </div>
            """, unsafe_allow_html=True
        )
        if st.button("Mở Camera / Open Live Camera", use_container_width=True):
            change_page('camera')
            
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

from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
def render_camera():
    if st.button("🔙 Về trang chủ / Back to Home"):
        change_page('home')
        
    st.header("Luồng Camera Trực tiếp / Real-time Camera Feed")

    col_cam, col_opt = st.columns([3, 1])
    with col_opt:
        show_fps = st.toggle("Hiển thị FPS", key="fps_cam")
        run = st.checkbox("🔴 BẬT / TẮT CAMERA", value=False)
        audio_ph = st.empty()

    with col_cam:
        stframe = st.empty()

    if run:
        cap = cv2.VideoCapture(0)
        # Tối ưu hóa việc gọi camera mượt mà
        frame_count = 0
        while run:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret: break
            
            frame_count += 1
            if frame_count % 3 != 0: # Bỏ qua 2 khung hình để giảm tải (Frame Skipping)
                continue
                
            frame = cv2.resize(frame, (640, 480))
            # Cập nhật lời gọi process_frame với mode="camera"
            processed_frame, _, audio_triggers = process_frame(frame, show_fps, start_time, mode="camera")
            
            if audio_triggers:
                trigger_audio_queue(audio_triggers, audio_ph)
                
            stframe.image(processed_frame, channels="BGR", use_container_width=True)
            time.sleep(0.01)

        cap.release()

def render_video():
    import os
    if st.button("🔙 Về trang chủ / Back to Home"):
        change_page('home')
        
    st.header("Xử lý Video / Video Processing Engine")
    uploaded_video = st.file_uploader("Chọn tệp video / Choose a video file", type=['mp4', 'avi', 'mov'])
    
    if uploaded_video:
        show_fps = st.toggle("Hiển thị FPS trên Video", key="fps_vid")
        process_btn = st.button("Bắt đầu Phân tích (Lưu & Phát lại) / Start Analysis")
        
        if process_btn:
            try:
                import moviepy.editor as mp
            except ImportError:
                st.error("Lỗi: Thư viện `moviepy` chưa được cài đặt. Hãy chạy: pip install moviepy")
                return

            tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile_in.write(uploaded_video.read())
            
            cap = cv2.VideoCapture(tfile_in.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0: fps = 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Khởi tạo VideoWriter (Lưu tạm video không tiếng)
            tfile_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(tfile_out.name, fourcc, fps, (width, height))
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            st.session_state.last_audio_time = {}  # Reset cooldown
            audio_events = [] # Lưu vết (Thời gian, Đường dẫn âm thanh)
            frame_idx = 0
            
            while cap.isOpened():
                start_time = time.time()
                ret, frame = cap.read()
                if not ret: 
                    break
                
                frame_idx += 1
                current_video_time = frame_idx / fps # Thời gian chuẩn trong video
                
                # Phân tích ngầm (không hiển thị lên giao diện)
                processed_frame, _, audio_triggers = process_frame(
                    frame, show_fps, start_time, mode="video", current_video_time=current_video_time
                )
                
                # Nếu có biển báo, lưu vết để lát nữa chèn Audio
                if audio_triggers:
                    for path in audio_triggers:
                        audio_events.append((current_video_time, path))
                        
                out.write(processed_frame)
                
                # Cập nhật thanh tiến trình (Update UI sau mỗi 5 khung hình cho đỡ lag)
                if total_frames > 0 and frame_idx % 5 == 0:
                    progress_bar.progress(min(frame_idx / total_frames, 1.0))
                    status_text.text(f"Đang xử lý ngầm: {frame_idx}/{total_frames} khung hình...")
                
            cap.release()
            out.release()
            
            # === BƯỚC GHÉP ÂM THANH ===
            status_text.text("Quá trình nhận diện xong! Đang ghép âm thanh cảnh báo vào Video...")
            try:
                video_clip = mp.VideoFileClip(tfile_out.name)
                audio_clips = []
                
                # Giữ lại âm thanh gốc của video (nếu có)
                if video_clip.audio is not None:
                    audio_clips.append(video_clip.audio)
                    
                # Chèn các cảnh báo vào đúng mốc thời gian
                for t, path in audio_events:
                    if os.path.exists(path):
                        a_clip = mp.AudioFileClip(path).set_start(t)
                        audio_clips.append(a_clip)
                        
                if audio_clips:
                    composite_audio = mp.CompositeAudioClip(audio_clips)
                    video_clip = video_clip.set_audio(composite_audio)
                    
                # Xuất ra file chuẩn MP4 H.264 để Streamlit phát được
                final_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                video_clip.write_videofile(
                    final_out.name, 
                    codec='libx264', 
                    audio_codec='aac', 
                    logger=None, # Tắt log thư viện
                    threads=4
                )
                video_clip.close()
                
                progress_bar.progress(1.0)
                status_text.success("Hoàn tất! Video đang được phát ở bên dưới.")
                
                # Hiển thị trình phát Video tích hợp sẵn âm thanh
                st.video(final_out.name)
                
            except Exception as e:
                st.error(f"Lỗi hệ thống khi ghép âm thanh: {e}")
                st.video(tfile_out.name) # Chuyển hướng phát video không tiếng nếu lỗi ghép

def render_image():
    if st.button("🔙 Về trang chủ / Back to Home"):
        change_page('home')
        
    st.header("Phân tích Ảnh tĩnh / Static Image Analysis")
    uploaded_image = st.file_uploader("Chọn tệp ảnh / Choose an image file", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_image:
        audio_ph = st.empty()
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        
        with st.spinner("Đang phân tích ảnh... / Analyzing image..."):
            # Cập nhật lời gọi process_frame với mode="image"
            processed_frame, box_count, audio_triggers = process_frame(frame, show_fps=False, start_time=time.time(), mode="image")
            
            if audio_triggers:
                trigger_audio_queue(audio_triggers, audio_ph)
            st.image(processed_frame, channels="BGR", width=800)
            
            if box_count == 0:
                st.warning("Không phát hiện biển báo giao thông nào trong ảnh. / No Traffic Sign Detected.")
            else:
                st.success(f"Phân tích hoàn tất! Phát hiện chính xác {box_count} biển báo. / Analysis complete!")


if st.session_state.current_page == 'home':
    render_home()
elif st.session_state.current_page == 'camera':
    render_camera()
elif st.session_state.current_page == 'video':
    render_video()
elif st.session_state.current_page == 'image':
    render_image()
