import sys
import os
import cv2
import torch
import numpy as np
from PIL import Image
from google.colab.patches import cv2_imshow

# ------------------------------------------------------------
# 1️. YOLOP 경로 설정
# ------------------------------------------------------------
# 경로 변수는 유지합니다.
YOLOP_PATH = './YOLOP' 
if YOLOP_PATH not in sys.path:
  sys.path.append(YOLOP_PATH)
  print(f"> YOLOP 경로 추가 완료: {YOLOP_PATH}")

from lib.models.YOLOP import get_net

# ------------------------------------------------------------
# 2️. 모델 불러오기
# ------------------------------------------------------------
print("> YOLOP 모델 불러오는 중...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_net('yolop', pretrained=False)
weights_path = "./YOLOP/weights/End-to-end.pth"
checkpoint = torch.load(weights_path, map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model = model.to(device).eval()
print("> YOLOP End-to-end.pth 가중치 로드 완료!")

# ------------------------------------------------------------
# 3️. 이미지 로드
# ------------------------------------------------------------
img_path = "./ARHUD/data/road_test1.png"
if not os.path.exists(img_path):
   raise FileNotFoundError(f"warn. 이미지 파일이 존재하지 않습니다: {img_path}")

img = cv2.imread(img_path)
if img is None:
   raise ValueError(f"warn. 이미지 로딩 실패! 파일이 손상되었거나 포맷이 잘못됨: {img_path}")

H, W = img.shape[:2]
# 원본 이미지 복사본을 만들어 시각화에 사용합니다.
img_original = img.copy() 
print(f"> Image loaded: {W}x{H}")


# 4️. 전처리 (YOLOP 공식 방식: RGB 변환 + 정규화 + 표준화 적용)
# 1. BGR 이미지를 640x640으로 리사이징
img_resized = cv2.resize(img_original, (640, 640)) 
 
# 2. BGR -> RGB 변환 (PyTorch 표준)
img_resized_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

# 3. 0-1 스케일로 정규화 (float32로 변환)
img_norm = img_resized_rgb.astype(np.float32) / 255.0

# 4. ImageNet 평균 및 표준편차로 표준화 (YOLOP/YOLOv5 표준)
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 각 채널별로 표준화 적용
# NumPy의 브로드캐스팅을 활용하여 (H, W, 3) 이미지에 평균/표준편차 적용
img_norm_std = (img_norm - mean) / std

# HWC -> C H W (Tensor 형식)로 변환
input_tensor = torch.from_numpy(img_norm_std).permute(2, 0, 1).unsqueeze(0).float().to(device)

# 5️. 모델 추론

with torch.no_grad():
   det_out, da_seg_out, ll_seg_out = model(input_tensor)

# 6️. 후처리 (도로/차선 mask)
da_mask = da_seg_out
ll_mask = ll_seg_out

# 원본 크기로 복원
da_mask_resized = torch.nn.functional.interpolate(
   da_mask, size=(H, W), mode="bilinear", align_corners=False
).squeeze().cpu().numpy()

ll_mask_resized = torch.nn.functional.interpolate(
   ll_mask, size=(H, W), mode="bilinear", align_corners=False
).squeeze().cpu().numpy()

if da_mask_resized.ndim == 3:
   da_mask_resized = da_mask_resized[0]
if ll_mask_resized.ndim == 3:
   ll_mask_resized = ll_mask_resized[0]

print("도로 mask range (수정 후):", np.min(da_mask_resized), "→", np.max(da_mask_resized))
print("차선 mask range (수정 후):", np.min(ll_mask_resized), "→", np.max(ll_mask_resized))

# 임계값 재설정: 전처리 성공 시 마스크 범위가 0~1 사이에 정상적으로 나올 것이므로 0.5/0.6으로 유지해봅니다.
road_thresh = 0.6
lane_thresh = 0.42

da_mask_img = (da_mask_resized > road_thresh).astype("uint8") * 255
ll_mask_img = (ll_mask_resized > lane_thresh).astype("uint8") * 255

print("도로 mask shape:", da_mask_img.shape)
print("차선 mask shape:", ll_mask_img.shape)

# ------------------------------------------------------------
# 7️. 결과 저장
# ------------------------------------------------------------
save_base = "./ARHUD"
os.makedirs(save_base, exist_ok=True)

# 파일명에 "_final"을 추가하여 이전 파일과 구별
road_path = os.path.join(save_base, "road_mask_yolop_final3.png")
lane_path = os.path.join(save_base, "lane_mask_yolop_final3.png")
Image.fromarray(da_mask_img).save(road_path)
Image.fromarray(ll_mask_img).save(lane_path)

print(f"> Saved → road_mask_yolop_final.png / lane_mask_yolop_final.png")

# --- 8️. 시각화 (도로/차선 모두 초록색 대신 명확한 색상 사용) 👈 수정
road = cv2.imread(road_path, cv2.IMREAD_GRAYSCALE)
lane = cv2.imread(lane_path, cv2.IMREAD_GRAYSCALE)

overlay = img_original.copy()

# 도로 마스크 (초록색)
if np.any(road > 0): 
    overlay[road > 0] = (0, 255, 0)      # BGR: Green
    
# 차선 마스크 (흰색으로 변경하여 구분이 잘 되도록)
if np.any(lane > 0): 
    # 차선 마스크는 도로 마스크 위에 덮어씌워야 더 잘 보입니다. (차선이 도로보다 위에 있음)
    overlay[lane > 0] = (255, 255, 255)  # BGR: White 

# 블렌딩 비율 조정 (원본 70%, 오버레이 30%로 투명도를 높여 원본 영상을 살림)
blended = cv2.addWeighted(img_original, 0.7, overlay, 0.3, 0)

blended_path = "./ARHUD/preview_check_final.png"
#cv2.imwrite(blended_path, blended)
print(f"> {blended_path} 저장 완료 .")


print(" YOLOP 도로+차선 세그멘테이션 최종 수정 완료 ")

# ------------------------------------------------------------
# 9 도로 마스크 기반 중심선 (Opening 반복 중심 + 시각화)
# ------------------------------------------------------------
import cv2
import numpy as np
from google.colab.patches import cv2_imshow


#  도로 mask 불러오기 (도로=흰색, 배경=검정)
road_mask = da_mask_img.copy()

if np.mean(road_mask) < 127:
    print("warn 도로가 검정색이라 반전합니다.")
    road_mask = cv2.bitwise_not(road_mask)

binary = (road_mask > 127).astype(np.uint8)

#  Morphology 정제 (Opening 중심)
kernel_small = np.ones((5,5), np.uint8)
kernel_large = np.ones((50,50), np.uint8)

for _ in range(2):
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
for _ in range(5):
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_large)

cv2.imwrite("./ARHUD/road_mask_refined_opening.png", binary*255)

#  Sobel X-gradient로 도로 경계 감지
grad_x = cv2.Sobel(binary, cv2.CV_64F, 1, 0, ksize=3)
grad_x = np.abs(grad_x)
grad_x = (grad_x / grad_x.max() * 255).astype(np.uint8)
edges = (grad_x > 80).astype(np.uint8)

#  도로 영역 높이 자동 계산
ys_nonzero = np.where(np.any(binary > 0, axis=1))[0]
if len(ys_nonzero) == 0:
    raise ValueError("도로 mask가 비어 있습니다. YOLOP 출력 확인 필요.")
y_min, y_max = ys_nonzero[0], ys_nonzero[-1]
print(f"도로 세로 영역: {y_min} ~ {y_max}")

#  각 y(row)별 좌우 경계 평균 → 중심점
center_points = []
for y in range(y_min, y_max, 2):
    x_edges = np.where(edges[y, :] > 0)[0]
    if len(x_edges) < 2:
        continue

    # 좌우 경계 안정화: outlier 방지
    x_left = np.min(x_edges)
    x_right = np.max(x_edges)

    if (x_right - x_left) < 50:  # 너무 좁으면 무시
        continue

    x_center = int((x_left + x_right) / 2)
    center_points.append([x_center, y])

center_points = np.array(center_points)
print(f" 중심점 수: {len(center_points)}")

# . y 간격 보정 (간격 불균일 시 spline 보간)
if len(center_points) > 10:
    y_vals = center_points[:, 1]
    x_vals = center_points[:, 0]

    # dx/dy 급격한 변화 완화
    dx = np.abs(np.diff(x_vals))
    smooth_dx = np.convolve(dx, np.ones(5)/5, mode='same')
    threshold = np.mean(smooth_dx) + 2.5 * np.std(smooth_dx)  # 완화된 기준
    valid_idx = np.where(smooth_dx < threshold)[0]

    y_vals = y_vals[valid_idx]
    x_vals = x_vals[valid_idx]

    #  상하단부 3%만 trim (너무 끝부분 튀는 점 제거)
    trim_n = int(len(y_vals) * 0.03)
    if trim_n > 0:
        y_vals = y_vals[trim_n:-trim_n]
        x_vals = x_vals[trim_n:-trim_n]

    print(f"필터링 후 중심점 수: {len(y_vals)}")

    # y 간격이 고르지 않으면 spline으로 보간
    y_dense = np.linspace(y_vals.min(), y_vals.max(), num=200)
    smooth_x = np.interp(y_dense, y_vals, x_vals)

    # Polyfit + 이동평균으로 안정화
    coeff = np.polyfit(y_dense, smooth_x, deg=3)
    smooth_x = np.polyval(coeff, y_dense)
    window = 15
    smooth_x = np.convolve(smooth_x, np.ones(window)/window, mode='same')

     # 맨 아래 2개, 맨 위 3개 점 제거
    trim_bottom = 7
    trim_top = 7
    if len(y_dense) > (trim_bottom + trim_top):
        smooth_x = smooth_x[trim_bottom:-trim_top]
        y_dense = y_dense[trim_bottom:-trim_top]
        print(f"상단 {trim_top}개, 하단 {trim_bottom}개 점 제거 완료 (총 {len(y_dense)}점 남음)")
    else:
        print("warn 중심점 개수가 적어 trim 생략")

    # 시각화
    vis = cv2.cvtColor(binary*255, cv2.COLOR_GRAY2BGR)
    for (x, y) in zip(smooth_x[::3], y_dense[::3]):
        cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)

    save_path = "./ARHUD/road_center_spline_trimmed3.png"
    cv2.imwrite(save_path, vis)
    print(f"> 중심선 시각화 완료 (Trimmed) → {save_path}")

    np.savetxt("./ARHUD/road_center_spline_trimmed.txt",
               np.column_stack([smooth_x, y_dense]), fmt="%d")

    cv2_imshow(cv2.resize(vis, (960, 540)))
    print("> 도로 mask 기반 중심선 계산 (상단3+하단2 제외) 완료 ")