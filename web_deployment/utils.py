import streamlit as st
import base64
import os
import time
from config import BASE_DIR, BG_IMAGE_PATH

def get_base64_of_bin_file(filename):
    filepath = os.path.join(BASE_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    return ""

def trigger_audio_queue(audio_paths, placeholder):
    """
    Phát âm thanh trực tiếp bằng thẻ HTML5.
    Thêm timestamp vào key/ID để ép trình duyệt Streamlit luôn tải lại âm thanh.
    """
    if not audio_paths:
        return
        
    # Lấy file âm thanh đầu tiên để phát
    path = audio_paths[0]
    
    if os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            unique_id = time.time() # Ép Streamlit render lại DOM
            
            audio_html = f"""
            <div id="audio-container-{unique_id}">
                <audio autoplay style="display:none;">
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
            </div>
            """
            placeholder.markdown(audio_html, unsafe_allow_html=True)
    else:
        print(f"Không tìm thấy tệp âm thanh: {path}")

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
        .card {{ background: rgba(255, 255, 255, 0.85); backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px); border-radius: 20px; padding: 15px; text-align: center; box-shadow: 0 10px 30px rgba(3, 105, 161, 0.15); display: flex; flex-direction: column; justify-content: center; align-items: center; border: 2px solid rgba(56, 189, 248, 0.6); transition: all 0.3s ease; width: 100%; }}
        .card:hover {{ transform: translateY(-3px); box-shadow: 0 15px 35px rgba(2, 132, 199, 0.3); background: rgba(255, 255, 255, 0.95); border-color: #0284c7; }}
        .card-rect {{ height: 385px; margin-bottom: 15px; }} 
        .card-sq {{ height: 165px; margin-bottom: 12px; }}
        .card h3, .card p {{ margin: 0 !important; padding: 0 !important; line-height: 1.2 !important; }}
        .card h3 {{ color: #0c4a6e; font-size: 1.25rem; font-weight: 700; text-align: center; width: 100%; }}
        .card p {{ color: #0369a1; font-size: 0.9rem; font-weight: 500; text-align: center; width: 100%; margin-top: 3px !important; }}
        [data-testid="stFileUploader"] {{ background: rgba(255, 255, 255, 0.85) !important; border: 2px dashed rgba(2, 132, 199, 0.4) !important; border-radius: 15px !important; padding: 20px !important; backdrop-filter: blur(10px); }}
        .stButton > button {{ background: linear-gradient(135deg, #0284c7, #0369a1); color: white; border-radius: 12px; font-weight: 600; border: none; box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3); transition: all 0.2s ease; }}
        .stButton > button:hover {{ background: linear-gradient(135deg, #0369a1, #075985); color: white; box-shadow: 0 6px 16px rgba(2, 132, 199, 0.5); }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
