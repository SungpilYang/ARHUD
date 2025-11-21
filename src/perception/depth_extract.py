import torch
import cv2
import numpy as np
import os
from transformers import DPTForDepthEstimation, DPTImageProcessor
from google.colab.patches import cv2_imshow
from PIL import Image

# ------------------------------------------------------------
# 1️. 상수 설정
# ------------------------------------------------------------
DEPTH_MODEL_NAME = "Intel/dpt-large"  # MiDaS Large (고해상도, 정확도 높음)
VIDEO_PATH = "./ARHUD/data/test1.MOV"
SAVE_DIR = "./ARHUD/data/depth_frames"
os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_EVERY = 5  # 5프레임마다 저장 (2100프레임 → 약 420장)

# ------------------------------------------------------------
# 2️. 모델 불러오기
# ------------------------------------------------------------
def load_depth_model():
    print(f"> Loading Depth Model: {DEPTH_MODEL_NAME} ...")
    processor = DPTImageProcessor.from_pretrained(DEPTH_MODEL_NAME)
    model = DPTForDepthEstimation.from_pretrained(DEPTH_MODEL_NAME)
    model.eval()
    print("> Depth model loaded successfully.")
    return processor, model


# ------------------------------------------------------------
# 3️. 깊이 추정 함수
# ------------------------------------------------------------
def estimate_depth(processor, model, image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    inputs = processor(images=image_pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
        predicted_depth = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth = predicted_depth.cpu().numpy()

    # 정규화 (0~255로 변환)
    depth_min, depth_max = np.percentile(depth, (5, 95))
    depth_vis = np.clip((depth - depth_min) / (depth_max - depth_min), 0, 1)
    depth_vis = (depth_vis * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

    return depth_vis, depth_colored


# ------------------------------------------------------------
# 4️. 비디오에서 주기적으로 프레임 추출
# ------------------------------------------------------------
def extract_depth_from_video():
    print(f"> Loading video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f" 총 프레임 수: {total_frames}")

    processor, model = load_depth_model()
    frame_idx = 0
    saved = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 매 SAVE_EVERY 프레임마다만 추출
        if frame_idx % SAVE_EVERY != 0:
            continue

        depth_gray, depth_colored = estimate_depth(processor, model, frame)

        depth_gray_path = os.path.join(SAVE_DIR, f"depth_{frame_idx:04d}.png")
        cv2.imwrite(depth_gray_path, depth_gray)

        saved += 1
        print(f"> Saved depth frame {frame_idx}/{total_frames} ({saved} files)")

    cap.release()
    print(f"> 모든 샘플링 프레임 저장 완료 → {SAVE_DIR}")
    print(f"총 저장된 프레임 수: {saved}")

    # 마지막 이미지 시각화 
    if os.path.exists(depth_gray_path):
        cv2_imshow(depth_colored)
        print("> 마지막 depth 시각화 완료")


# ------------------------------------------------------------
# 5️. 실행
# ------------------------------------------------------------
if __name__ == "__main__":
    extract_depth_from_video()
