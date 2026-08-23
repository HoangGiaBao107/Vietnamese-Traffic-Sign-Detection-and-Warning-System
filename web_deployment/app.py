import streamlit as st
import cv2
import gc
import time
import traceback
import numpy as np
import tempfile
import os
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
from utils import inject_custom_css, trigger_audio_queue
from detector import process_frame
from config import GC_COLLECT_EVERY_N_FRAMES

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
            # Reset lại thời gian và bộ đệm cảnh báo để phân tích từ đầu video
            st.session_state.last_audio_time = {}
            st.session_state.detection_buffer = {}
            st.session_state.frames_processed_since_gc = 0

            # Khai báo trước để dùng an toàn trong khối finally kể cả khi lỗi xảy ra sớm
            tfile_path = None
            temp_vid_no_audio_path = None
            final_output_path = None
            cap = None
            out = None
            video_clip = None
            audio_clips = []

            try:
                # 1. Lưu video gốc vào bộ nhớ tạm (ghi ra đĩa, không giữ toàn bộ file trong RAM)
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile_path = tfile.name
                tfile.write(uploaded_video.getbuffer())
                tfile.close()
                # Giải phóng buffer upload khỏi RAM ngay sau khi đã ghi ra đĩa
                del uploaded_video

                cap = cv2.VideoCapture(tfile_path)
                fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # 2. Tạo video output tạm thời (chỉ có hình) bằng OpenCV
                temp_vid_no_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                temp_vid_no_audio_path = temp_vid_no_audio.name
                temp_vid_no_audio.close()
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(temp_vid_no_audio_path, fourcc, fps_video, (width, height))

                progress_bar = st.progress(0)
                status_text = st.empty()

                frame_idx = 0
                audio_events = []

                with st.spinner("Bước 1/2: Đang phân tích hình ảnh AI (YOLO)..."):
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_idx += 1
                        start_time = time.time()
                        video_time_sec = frame_idx / fps_video

                        processed_frame, _, audio_triggers = process_frame(
                            frame, show_fps, start_time, video_time=video_time_sec, is_image=False
                        )
                        out.write(processed_frame)

                        if audio_triggers:
                            audio_events.append({
                                "time": video_time_sec,
                                "paths": audio_triggers
                            })

                        # Giải phóng tham chiếu frame ngay sau khi dùng xong
                        del frame, processed_frame

                        # TỐI ƯU BỘ NHỚ: dọn rác định kỳ trong vòng lặp đọc/ghi video
                        if frame_idx % GC_COLLECT_EVERY_N_FRAMES == 0:
                            gc.collect()

                        if total_frames > 0 and frame_idx % 5 == 0:
                            progress_bar.progress(min(frame_idx / total_frames, 1.0))
                            status_text.text(f"Đang phân tích khung hình: {frame_idx}/{total_frames}")

                cap.release()
                cap = None
                out.release()
                out = None
                gc.collect()
                progress_bar.progress(1.0)

                # 3. Sử dụng MoviePy để ghép âm thanh vào Video
                status_text.text("Bước 2/2: Đang tổng hợp âm thanh cảnh báo vào Video...")

                final_output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                final_output_path = final_output_file.name
                final_output_file.close()

                video_clip = VideoFileClip(temp_vid_no_audio_path)

                next_available_audio_time = 0.0
                for event in audio_events:
                    trigger_time = event["time"]
                    for path in event["paths"]:
                        if os.path.exists(path):
                            aclip = AudioFileClip(path)
                            start_time_audio = max(trigger_time, next_available_audio_time)
                            aclip = aclip.set_start(start_time_audio)
                            audio_clips.append(aclip)
                            next_available_audio_time = start_time_audio + aclip.duration

                if audio_clips:
                    final_audio = CompositeAudioClip(audio_clips)
                    video_clip = video_clip.set_audio(final_audio)

                video_clip.write_videofile(
                    final_output_path,
                    codec="libx264",
                    audio_codec="aac",
                    fps=fps_video,
                    logger=None
                )

                status_text.text("Xử lý hoàn tất!")
                st.success("Xử lý video thành công! Video đã bao gồm âm thanh cảnh báo.")

                # 4. Phát video và tải xuống
                st.video(final_output_path)

                with open(final_output_path, "rb") as file:
                    st.download_button(
                        label="⬇️ Tải Video đã phân tích (Đã ghép âm thanh) về máy",
                        data=file,
                        file_name="detected_video_with_audio.mp4",
                        mime="video/mp4"
                    )

            except Exception as e:
                # Trước đây lỗi giữa chừng (vd hết RAM, lỗi codec...) khiến Streamlit
                # âm thầm mất session và quay về trang chủ. Giờ báo lỗi rõ ràng ra màn hình.
                st.error(f"Đã xảy ra lỗi trong quá trình xử lý video: {e}")
                st.code(traceback.format_exc())

            finally:
                # TỐI ƯU BỘ NHỚ: luôn giải phóng toàn bộ tài nguyên dù thành công hay lỗi
                if cap is not None:
                    cap.release()
                if out is not None:
                    out.release()
                for aclip in audio_clips:
                    try:
                        aclip.close()
                    except Exception:
                        pass
                if video_clip is not None:
                    try:
                        video_clip.close()
                    except Exception:
                        pass

                # Xoá file tạm trung gian (không xoá final_output_path vì st.video/download_button
                # vẫn cần đọc lại file này trong lần rerun hiển thị)
                for p in [tfile_path, temp_vid_no_audio_path]:
                    if p and os.path.exists(p):
                        try:
                            os.remove(p)
                        except Exception:
                            pass

                gc.collect()


def render_image():
    if st.button("🔙 Về trang chủ / Back to Home"):
        change_page('home')

    st.header("Phân tích Ảnh tĩnh / Static Image Analysis")
    uploaded_image = st.file_uploader("Chọn tệp ảnh / Choose an image file", type=['jpg', 'jpeg', 'png'])

    if uploaded_image:
        audio_ph = st.empty()

        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        del file_bytes

        with st.spinner("Đang phân tích ảnh... / Analyzing image..."):
            processed_frame, box_count, audio_triggers = process_frame(
                frame, show_fps=False, start_time=time.time(), video_time=0, is_image=True
            )

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
