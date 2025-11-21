import os
import sys

# ORB_SLAM3 예제가 찾는 이미지 폴더 경로
IMAGE_DIR = "./ARHUD_local/tum_video_dataset/rgb"
# 생성할 rgb.txt 파일 경로
OUTPUT_FILE = "./ARHUD_local/tum_video_dataset/rgb.txt"
# 추출한 프레임 속도 (FPS)
FPS = 30 # <-- 30 FPS로 설정되어 있습니다.

# 타임스탬프 초기값 (임의의 시작 시간)
start_time = 1305031910.000000
# 프레임 간격 계산 (1 / FPS)
time_step = 1.0 / FPS

# 파일 목록 가져오기 및 정렬
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.png')])

if not image_files:
    print(f"Error: No PNG files found in {IMAGE_DIR}. Make sure step 2 ran correctly.")
    sys.exit(1)

print(f"Found {len(image_files)} images. Creating {OUTPUT_FILE}...")

with open(OUTPUT_FILE, 'w') as f:
    # TUM 데이터셋에서 요구하는 주석 (필수)
    f.write("# timestamp filename\n")
    f.write("# data/format\n")
    f.write("# note: Generated for ORB-SLAM3 mono_tum example\n")

    current_time = start_time
    for filename in image_files:
        # 타임스탬프와 파일명을 띄어쓰기로 구분하여 작성
        f.write(f"{current_time:.6f} rgb/{filename}\n")
        current_time += time_step

print(f"Successfully created {len(image_files)} timestamp entries.")
