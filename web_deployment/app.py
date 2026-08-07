import streamlit as st
import cv2
import time
import numpy as np
import tempfile
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
from utils import inject_custom_css, trigger_audio_queue
from detector import process_frame

st.set_page_config(page_title="Phát hiện Biển báo / Traffic Sign Detection", layout="wide")

# Khởi tạo CSS ngay từ đầu
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

def render_camera():
    if st.button("🔙 Về trang chủ / Back to Home"):
        change_page('home')
        
    st.header("Luồng Camera Trực tiếp / Real-time Camera Feed")
    st.info("💡 Trên Cloud, ứng dụng dùng WebRTC để mở luồng video. Nhấn nút 'Start' bên dưới luồng camera để bắt đầu cấp quyền truy cập Camera.")

    col_cam, col_opt = st.columns([3, 1])
    
    with col_opt:
        show_fps = st.toggle("Hiển thị FPS / Show FPS", key="fps_cam")

    with col_cam:
        def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
            # Lấy luồng ảnh từ trình duyệt web
            img = frame.to_ndarray(format="bgr24")
            img = cv2.resize(img, (640, 480))
            
            # Xử lý YOLO. Lưu ý: Do webRTC chạy đa luồng, việc gọi Audio qua UI từ hàm callback bị giới hạn.
            # Do đó audio sẽ chỉ hoạt động tốt tại trang Phân tích Video/Ảnh tĩnh.
            processed_frame, _, _ = process_frame(img, show_fps, time.time(), is_image=False)
            
            return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")

        webrtc_streamer(
            key="traffic-camera",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )

def render_video():
    if st.button("🔙 Về trang chủ / Back to Home"):
        change_page('home')
        
    st.header("Xử lý Video / Video Processing Engine")
    uploaded_video = st.file_uploader("Chọn tệp video / Choose a video file", type=['mp4', 'avi', 'mov'])
    
    if uploaded_video:
        # Bắt buộc thao tác từ người dùng để Browser cấp quyền Autoplay Audio
        enable_audio = st.checkbox("🔔 Bật phát cảnh báo âm thanh", value=True)
        
        col_vid, col_opt = st.columns([3, 1])
        with col_opt:
            show_fps = st.toggle("Hiển thị FPS / Show FPS", key="fps_vid")
            process_btn = st.button("Bắt đầu Phân tích / Start Analysis")
            audio_ph = st.empty()

        with col_vid:
            stframe = st.empty()
            
        if process_btn:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            
            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            frame_idx = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: 
                    break
                
                frame_idx += 1
                start_time = time.time()
                
                processed_frame, _, audio_triggers = process_frame(frame, show_fps, start_time, is_image=False)
                
                if enable_audio:
                    trigger_audio_queue(audio_triggers, audio_ph)
                
                stframe.image(processed_frame, channels="BGR", use_container_width=True)
                
                if total_frames > 0:
                    progress_bar.progress(min(frame_idx / total_frames, 1.0))
                    status_text.text(f"Đang phân tích khung hình: {frame_idx}/{total_frames}")
                
            cap.release()
            st.success("Xử lý video hoàn tất! / Video processing completed successfully!")

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
            processed_frame, box_count, audio_triggers = process_frame(frame, show_fps=False, start_time=time.time(), is_image=True)
            
            # Kích hoạt âm thanh
            trigger_audio_queue(audio_triggers, audio_ph)
            
            st.image(processed_frame, channels="BGR", width=800)
            
            if box_count == 0:
                st.warning("Không phát hiện biển báo giao thông nào trong ảnh. / No Traffic Sign Detected.")
            else:
                st.success(f"Phân tích hoàn tất! Phát hiện chính xác {box_count} biển báo. / Analysis complete!")

if __name__ == "__main__":
    if st.session_state.current_page == 'home': 
        render_home()
    elif st.session_state.current_page == 'camera': 
        render_camera()
    elif st.session_state.current_page == 'video': 
        render_video()
    elif st.session_state.current_page == 'image': 
        render_image()
