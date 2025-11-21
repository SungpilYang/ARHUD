import sys, os, cv2, torch, numpy as np
from PIL import Image
from google.colab.patches import cv2_imshow

#polyfit 3차 버전
# ------------------------------------------------------------
# 1️. YOLOP 모델 로드
# ------------------------------------------------------------
YOLOP_PATH = './YOLOP'
if YOLOP_PATH not in sys.path:
    sys.path.append(YOLOP_PATH)

from lib.models.YOLOP import get_net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_net('yolop', pretrained=False)

weights_path = "./YOLOP/weights/End-to-end.pth"
checkpoint = torch.load(weights_path, map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model = model.to(device).eval()

print("> YOLOP End-to-end.pth 로드 완료")

# ------------------------------------------------------------
# 2️. 입력 동영상 설정
# ------------------------------------------------------------
video_path = "./ARHUD/data/test1.MOV"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"> 입력 영상: {width}x{height}, {fps:.2f} FPS")

# 출력 설정
save_dir = "./ARHUD"
os.makedirs(save_dir, exist_ok=True)
out_path = os.path.join(save_dir, "lane_center_output_polifit.mp4")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

# ------------------------------------------------------------
# 3️. 프레임별 중심선 검출
# ------------------------------------------------------------
frame_idx = 0
all_coords = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    print(f" > Frame {frame_idx}")

    # YOLOP 입력 전처리
    img_resized = cv2.resize(frame, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm_std = (img_norm - mean) / std
    input_tensor = torch.from_numpy(img_norm_std).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # YOLOP 추론
    with torch.no_grad():
        _, da_seg_out, _ = model(input_tensor)

    da_mask_resized = torch.nn.functional.interpolate(
        da_seg_out, size=(height, width), mode="bilinear", align_corners=False
    ).squeeze().cpu().numpy()

    # 단일 채널 보정
    if da_mask_resized.ndim == 3:
        da_mask_resized = da_mask_resized[0]
    da_mask_resized = np.nan_to_num(da_mask_resized)
    da_mask_resized = np.clip(da_mask_resized, 0, 1)
    da_mask_img = (da_mask_resized > 0.6).astype("uint8") * 255

    # Morphology 정제
    binary = cv2.morphologyEx(da_mask_img, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((50,50), np.uint8))

    # 경계 검출
    grad_x = cv2.Sobel(binary, cv2.CV_64F, 1, 0, ksize=3)
    grad_x = np.abs(grad_x)
    grad_x = (grad_x / grad_x.max() * 255).astype(np.uint8)
    edges = (grad_x > 80).astype(np.uint8)

    # 도로 높이 범위 계산
    ys_nonzero = np.where(np.any(binary > 0, axis=1))[0]
    if len(ys_nonzero) == 0:
        out.write(frame)
        continue
    y_min, y_max = ys_nonzero[0], ys_nonzero[-1]

    # 중심선 계산
    center_points = []
    for y in range(y_min, y_max, 3):
        x_edges = np.where(edges[y, :] > 0)[0]
        if len(x_edges) >= 2:
            x_center = int((np.min(x_edges) + np.max(x_edges)) / 2)
            center_points.append([x_center, y])
    center_points = np.array(center_points)

    if len(center_points) > 10:
        y_vals = center_points[:, 1]
        x_vals = center_points[:, 0]

        #  3차 다항 회귀 (도로 곡선 피팅)
        coeff = np.polyfit(y_vals, x_vals, deg=3)
        y_dense = np.linspace(y_vals.min(), y_vals.max(), num=100)
        x_smooth = np.polyval(coeff, y_dense)

        # 상단/하단 trimming
        trim_top, trim_bottom = 3, 2
        if len(y_dense) > (trim_top + trim_bottom):
            x_smooth = x_smooth[trim_bottom:-trim_top]
            y_dense = y_dense[trim_bottom:-trim_top]

        # 시각화 (빨강 점)
        for (x, y) in zip(x_smooth[::3], y_dense[::3]):
            cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)

        # 중심선 좌표 저장
        frame_coords = np.column_stack([x_smooth, y_dense])
        all_coords.append(frame_coords)

    out.write(frame)

cap.release()
out.release()
print(f"> 결과 영상 저장 완료 → {out_path}")

# ------------------------------------------------------------
# 4️. 전체 프레임 중심선 좌표 저장
# ------------------------------------------------------------
coords_path = os.path.join(save_dir, "video_centerlines_polifit.txt")
with open(coords_path, "w") as f:
    for i, frame_coords in enumerate(all_coords):
        f.write(f"# Frame {i+1}\n")
        np.savetxt(f, frame_coords, fmt="%d")
print(f"> 전체 중심선 좌표 저장 완료 → {coords_path}")
