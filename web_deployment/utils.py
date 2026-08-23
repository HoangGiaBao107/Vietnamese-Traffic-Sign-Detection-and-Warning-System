import streamlit as st
import base64
import os
import streamlit.components.v1 as components
from config import BASE_DIR, BG_IMAGE_PATH


def get_base64_of_bin_file(filename):
    filepath = os.path.join(BASE_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    return ""


def trigger_audio_queue(audio_paths, placeholder=None):
    """
    Tạo một Audio Player bằng HTML5 thuần túy, phát tuần tự danh sách âm thanh.
    Khắc phục hoàn toàn lỗi xung đột iframe của Streamlit.
    """
    if not audio_paths:
        return

    b64_audios = []
    for path in audio_paths:
        if os.path.exists(path):
            with open(path, "rb") as f:
                b64_audios.append(base64.b64encode(f.read()).decode())

    if b64_audios:
        js_code = f"""
        <audio id="audio-player" autoplay></audio>
        <script>
            const audios = {b64_audios};
            let currentIndex = 0;
            const player = document.getElementById('audio-player');

            function playNext() {{
                if (currentIndex < audios.length) {{
                    player.src = "data:audio/mp3;base64," + audios[currentIndex];
                    player.play().catch(e => console.log("Trình duyệt chặn Autoplay:", e));
                    currentIndex++;
                }}
            }}

            player.onended = playNext;
            playNext(); // Bắt đầu phát
        </script>
        """
        if placeholder:
            with placeholder:
                components.html(js_code, height=0, width=0)
        else:
            components.html(js_code, height=0, width=0)


def inject_custom_css():
    bg_base64 = get_base64_of_bin_file(BG_IMAGE_PATH)
    bg_css = f"background-image: linear-gradient(rgba(224, 242, 254, 0.45), rgba(186, 230, 253, 0.55)), url('data:image/jpeg;base64,{bg_base64}');" if bg_base64 else "background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);"

    custom_css = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] {{ font-family: 'Quicksand', sans-serif !important; }}
        .stApp {{ {bg_css} background-size: cover !important; background-position: center !important; background-attachment: fixed !important; }}
        header {{ visibility: hidden; }}
        .main-title {{ text-align: center; color: #0c4a6e; font-size: 2.1rem; font-weight: 700; margin-bottom: 0px; padding-top: 35px; text-transform: uppercase; text-shadow: 0 2px 8px rgba(255, 255, 255, 0.9); }}
        .sub-title {{ text-align: center; color: #0369a1; font-size: 1rem; font-weight: 600; margin-top: 5px; margin-bottom: 30px; text-shadow: 0 1px 4px rgba(255, 255, 255, 0.9); }}
        .author-badge {{ position: absolute; top: 15px; right: 25px; background: linear-gradient(135deg, #0284c7, #0369a1); color: white; padding: 7px 18px; border-radius: 30px; font-weight: 600; font-size: 0.88rem; box-shadow: 0 4px 15px rgba(2, 132, 199, 0.4); z-index: 999; border: 1px solid rgba(255, 255, 255, 0.5); }}
        .card {{ background: rgba(255, 255, 255, 0.85); backdrop-filter: blur(14px); border-radius: 20px; padding: 15px; text-align: center; box-shadow: 0 10px 30px rgba(3, 105, 161, 0.15); display: flex; flex-direction: column; justify-content: center; align-items: center; border: 2px solid rgba(56, 189, 248, 0.6); transition: all 0.3s ease; width: 100%; }}
        .card:hover {{ transform: translateY(-3px); box-shadow: 0 15px 35px rgba(2, 132, 199, 0.3); background: rgba(255, 255, 255, 0.95); border-color: #0284c7; }}
        .stButton > button {{ background: linear-gradient(135deg, #0284c7, #0369a1); color: white; border-radius: 12px; font-weight: 600; border: none; box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3); transition: all 0.2s ease; }}
        .stButton > button:hover {{ background: linear-gradient(135deg, #0369a1, #075985); color: white; box-shadow: 0 6px 16px rgba(2, 132, 199, 0.5); }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
