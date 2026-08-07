import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "Warning_sound")

MODEL_PATH = os.path.join(BASE_DIR, "best.onnx")
BG_IMAGE_PATH = "background.jpg"

AUDIO_PATHS = {
    0: os.path.join(AUDIO_DIR, "102-cam-di-nguoc-chieu.mp3"),
    1: os.path.join(AUDIO_DIR, "103a-cam-oto.mp3"),
    2: os.path.join(AUDIO_DIR, "103b-cam-oto-re-phai.mp3"),
    3: os.path.join(AUDIO_DIR, "103c-cam-oto-re-trai.mp3"),
    4: os.path.join(AUDIO_DIR, "106-cam-oto-tai.mp3"),
    5: os.path.join(AUDIO_DIR, "107-cam-oto-khach-va-oto-tai.mp3"),
    6: os.path.join(AUDIO_DIR, "123a-cam-re-trai.mp3"),
    7: os.path.join(AUDIO_DIR, "123b-cam-re-phai.mp3"),
    8: os.path.join(AUDIO_DIR, "124a-cam-quay-dau-xe.mp3"),
    9: os.path.join(AUDIO_DIR, "124b-cam-oto-quay-dau-xe.mp3"),
    10: os.path.join(AUDIO_DIR, "124c-cam-re-trai-va-cam-quay-dau-xe.mp3"),
    11: os.path.join(AUDIO_DIR, "125-cam-vuot.mp3"),
    12: os.path.join(AUDIO_DIR, "127-toc-do-toi-da-100.mp3"),
    13: os.path.join(AUDIO_DIR, "127-toc-do-toi-da-20.mp3"),
    14: os.path.join(AUDIO_DIR, "127-toc-do-toi-da-30.mp3"),
    15: os.path.join(AUDIO_DIR, "127-toc-do-toi-da-40.mp3"),
    16: os.path.join(AUDIO_DIR, "127-toc-do-toi-da-50.mp3"),
    17: os.path.join(AUDIO_DIR, "127-toc-do-toi-da-60.mp3"),
    18: os.path.join(AUDIO_DIR, "127-toc-do-toi-da-70.mp3"),
    19: os.path.join(AUDIO_DIR, "127-toc-do-toi-da-80.mp3"),
    20: os.path.join(AUDIO_DIR, "127-toc-do-toi-da-90.mp3"),
    21: os.path.join(AUDIO_DIR, "128-cam-su-dung-coi.mp3"),
    22: os.path.join(AUDIO_DIR, "130-cam-dung-va-do-xe.mp3"),
    23: os.path.join(AUDIO_DIR, "131a-cam-do-xe.mp3"),
    24: os.path.join(AUDIO_DIR, "131b-cam-do-xe-ngay-le.mp3"),
    25: os.path.join(AUDIO_DIR, "131c-cam-do-xe-ngay-chan.mp3"),
    26: os.path.join(AUDIO_DIR, "133-het-cam-vuot.mp3"),
    27: os.path.join(AUDIO_DIR, "134-het-han-che-toc-do-toi-da.mp3"),
    28: os.path.join(AUDIO_DIR, "135-het-tat-ca-cac-lenh-cam.mp3"),
    29: os.path.join(AUDIO_DIR, "136-cam-di-thang.mp3"),
    30: os.path.join(AUDIO_DIR, "137-cam-re-trai-va-re-phai.mp3"),
    31: os.path.join(AUDIO_DIR, "201-cho-ngoat-nguy-hiem.mp3"),
    32: os.path.join(AUDIO_DIR, "203-duong-bi-hep.mp3"),
    33: os.path.join(AUDIO_DIR, "205-duong-giao-nhau.mp3"),
    34: os.path.join(AUDIO_DIR, "207-giao-nhau-voi-duong-khong-uu-tien.mp3"),
    35: os.path.join(AUDIO_DIR, "208-giao-nhau-voi-duong-uu-tien.mp3"),
    36: os.path.join(AUDIO_DIR, "209-giao-nhau-co-tin-hieu-den.mp3"),
    37: os.path.join(AUDIO_DIR, "221-duong-go-ghe.mp3"),
    38: os.path.join(AUDIO_DIR, "224-duong-nguoi-di-bo-cat-ngang.mp3"),
    39: os.path.join(AUDIO_DIR, "225-chu-y-tre-em.mp3"),
    40: os.path.join(AUDIO_DIR, "227-cong-truong.mp3"),
    41: os.path.join(AUDIO_DIR, "245-di-cham.mp3"),
    42: os.path.join(AUDIO_DIR, "301a-cac-xe-chi-duoc-di-thang.mp3"),
    43: os.path.join(AUDIO_DIR, "301b-cac-xe-chi-duoc-re-phai.mp3"),
    44: os.path.join(AUDIO_DIR, "301c-cac-xe-chi-duoc-re-trai.mp3"),
    45: os.path.join(AUDIO_DIR, "301d-cac-xe-chi-duoc-re-phai-2.mp3"),
    46: os.path.join(AUDIO_DIR, "301e-cac-xe-chi-duoc-re-trai-2.mp3"),
    47: os.path.join(AUDIO_DIR, "301h-cac-xe-chi-duoc-di-thang-va-re-trai.mp3"),
    48: os.path.join(AUDIO_DIR, "302a-huong-phai-di-vong-chuong-ngai-vat-1.mp3"),
    49: os.path.join(AUDIO_DIR, "303-noi-giao-nhau-chay-theo-vong-xuyen.mp3"),
    50: os.path.join(AUDIO_DIR, "306-toc-do-toi-thieu-60.mp3"),
    51: os.path.join(AUDIO_DIR, "407a-duong-mot-chieu.mp3"),
    52: os.path.join(AUDIO_DIR, "420-bat-dau-khu-dong-dan-cu.mp3"),
    53: os.path.join(AUDIO_DIR, "421-ket-thuc-khu-dong-dan-cu.mp3"),
    54: os.path.join(AUDIO_DIR, "423a-duong-nguoi-di-bo-sang-ngang-1.mp3"),
    55: os.path.join(AUDIO_DIR, "437-bat-dau-duong-cao-toc.mp3"),
    56: os.path.join(AUDIO_DIR, "438-ket-thuc-duong-cao-toc.mp3"),
    57: os.path.join(AUDIO_DIR, "442-cho.mp3")
}

ENGLISH_SUBTITLES = {
    0: "No Entry", 1: "No Cars", 2: "No Right Turn for Cars", 3: "No Left Turn for Cars", 
    4: "No Trucks", 5: "No Buses and Trucks", 6: "No Left Turn", 7: "No Right Turn", 
    8: "No U-Turn", 9: "No U-Turn for Cars", 10: "No Left Turn and No U-Turn", 
    11: "No Overtaking", 12: "Max Speed Limit 100", 13: "Max Speed Limit 20", 
    14: "Max Speed Limit 30", 15: "Max Speed Limit 40", 16: "Max Speed Limit 50", 
    17: "Max Speed Limit 60", 18: "Max Speed Limit 70", 19: "Max Speed Limit 80", 
    20: "Max Speed Limit 90", 21: "No Honking", 22: "No Stopping or Parking", 
    23: "No Parking", 24: "No Parking on Odd Days", 25: "No Parking on Even Days", 
    26: "End of No Overtaking", 27: "End of Speed Limit", 28: "End of All Restrictions", 
    29: "No Going Straight", 30: "No Left and Right Turn", 31: "Dangerous Curve", 
    32: "Road Narrows", 33: "Intersection Ahead", 34: "Intersection with Minor Road", 
    35: "Intersection with Priority Road", 36: "Traffic Signals Ahead", 37: "Bumpy Road", 
    38: "Pedestrian Crossing Ahead", 39: "Children Crossing", 40: "Road Work Ahead", 
    41: "Go Slow", 42: "Go Straight Only", 43: "Turn Right Only", 44: "Turn Left Only", 
    45: "Turn Right Only", 46: "Turn Left Only", 47: "Go Straight and Turn Left Only", 
    48: "Keep Right of Obstacle", 49: "Roundabout", 50: "Minimum Speed Limit 60", 
    51: "One Way Traffic", 52: "Start of Populated Area", 53: "End of Populated Area", 
    54: "Pedestrian Crossing", 55: "Start of Expressway", 56: "End of Expressway", 57: "Market"
}

CONFIDENCE_THRESHOLD = 0.50  
COOLDOWN_SECONDS = 10
