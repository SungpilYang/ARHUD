import os
import math
import cv2
import numpy as np
import pandas as pd

# ====== 0) 사용자 설정 ======
TRAJ_PATH = r".\ARHUD\data\KeyFrameTrajectory.txt"
VIDEO_IN  = r".\ARHUD\data\test1.mov"      # 네 영상 파일 이름에 맞게 수정
VIDEO_OUT = r".\ARHUD\test1_with_traj.mp4"

# 좌표 → 화면 매핑 파라미터(경로 크기/위치 맞추기)
SCALE = 120.0          # 화면 픽셀/SLAM단위 (임의로 조절)
OFFSET_X = 300         # 화면에서 경로의 기준 위치 X (px)
OFFSET_Y = 500         # 화면에서 경로의 기준 위치 Y (px)
USE_PLANE = "xz"       # 2D 투영에 쓸 평면: "xz" 또는 "xy" 또는 "yz"
SMOOTHING = 0.9        # 경로 부드럽게(지나치면 끌려감). 0.0~0.95 권장
TIME_OFFSET = 0.0      # 영상과 SLAM 시간차 보정(초). 영상이 더 늦게 시작했으면 +값

DRAW_EVERY_N = 1       # 프레임 스킵(성능용). 1이면 모두 그림
PATH_COLOR = (0, 255, 0)   # 경로선 색(BGR)
ARROW_COLOR = (0, 0, 255)  # 진행방향 화살표 색(BGR)
TEXT_COLOR = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ====== 1) 궤적 로드 ======
def load_tum_trajectory(path):
    # 주석(#) 무시하고 8열 데이터 읽기
    df = pd.read_csv(
        path, delim_whitespace=True, comment="#",
        names=["t","tx","ty","tz","qx","qy","qz","qw"]
    )
    # 일부 파일은 헤더가 없고, 일부는 첫 줄이 길이가 안맞을 수 있어 필터
    df = df.dropna()
    # 원점 정규화(선택): 첫 포즈를 (0,0,0)으로
    df["tx"] -= df["tx"].iloc[0]
    df["ty"] -= df["ty"].iloc[0]
    df["tz"] -= df["tz"].iloc[0]
    # 시간도 0 기준으로 맞추면 동기화가 편함
    df["t"]  -= df["t"].iloc[0]
    return df.reset_index(drop=True)

traj = load_tum_trajectory(TRAJ_PATH)

# ====== 2) 쿼터니언 → yaw(방위각) 계산 ======
def quat_to_yaw(qx, qy, qz, qw):
    # Z-up 또는 다양한 축 정의가 있지만, 기본적인 yaw(수평 회전) 추정
    # 참고: yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    siny_cosp = 2.0 * (qw*qz + qx*qy)
    cosy_cosp = 1.0 - 2.0 * (qy*qy + qz*qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw  # 라디안

# ====== 3) 3D → 2D 스크린 좌표 매핑 ======
def world_to_screen(x, y, z):
    if USE_PLANE == "xz":
        sx = int(x * SCALE + OFFSET_X)
        sy = int(-z * SCALE + OFFSET_Y)  # -z: 화면 y축 아래로 증가 보정
    elif USE_PLANE == "xy":
        sx = int(x * SCALE + OFFSET_X)
        sy = int(-y * SCALE + OFFSET_Y)
    else:  # "yz"
        sx = int(y * SCALE + OFFSET_X)
        sy = int(-z * SCALE + OFFSET_Y)
    return sx, sy

# ====== 4) 영상 열기 ======
cap = cv2.VideoCapture(VIDEO_IN)
assert cap.isOpened(), f"Cannot open video: {VIDEO_IN}"
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (w, h))

# 누적 경로(스크린 좌표)
path_pts = []
last_pt = None

# ====== 5) 프레임 루프 ======
frame_idx = 0
alpha = SMOOTHING
curr_sx, curr_sy = None, None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 현재 프레임의 "영상 시간"
    t_video = frame_idx / fps + TIME_OFFSET

    # SLAM 궤적에서 가장 가까운 시점 찾기 (nearest neighbor)
    # 더 고급으로는 보간(interpolation) 가능
    i = np.searchsorted(traj["t"].values, t_video)
    if i <= 0:
        i = 0
    elif i >= len(traj):
        i = len(traj) - 1

    # 해당 포즈 데이터
    row = traj.iloc[i]
    x, y, z = row["tx"], row["ty"], row["tz"]
    qx, qy, qz, qw = row["qx"], row["qy"], row["qz"], row["qw"]

    # 2D로 투영
    sx, sy = world_to_screen(x, y, z)

    # 스무딩(옵션)
    if curr_sx is None:
        curr_sx, curr_sy = sx, sy
    else:
        curr_sx = int(alpha * curr_sx + (1 - alpha) * sx)
        curr_sy = int(alpha * curr_sy + (1 - alpha) * sy)

    # 경로 누적(라인 그리기)
    path_pts.append((curr_sx, curr_sy))
    if len(path_pts) >= 2 and frame_idx % DRAW_EVERY_N == 0:
        for j in range(1, len(path_pts)):
            cv2.line(frame, path_pts[j-1], path_pts[j], PATH_COLOR, 2)

    # 진행 방향 화살표(간단 yaw 기반)
    yaw = quat_to_yaw(qx, qy, qz, qw)
    arrow_len = 50  # px
    ax = int(curr_sx + arrow_len * math.cos(yaw))
    ay = int(curr_sy - arrow_len * math.sin(yaw))  # 화면 y축 보정(-)

    cv2.arrowedLine(frame, (curr_sx, curr_sy), (ax, ay), ARROW_COLOR, 2, tipLength=0.3)

    # 정보 텍스트
    cv2.putText(frame, f"t={t_video:.2f}s  idx={i}/{len(traj)-1}", (20, 40), FONT, 0.8, TEXT_COLOR, 2)
    cv2.putText(frame, f"plane={USE_PLANE} scale={SCALE:.1f} off=({OFFSET_X},{OFFSET_Y})", (20, 70), FONT, 0.6, TEXT_COLOR, 1)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print("Done:", VIDEO_OUT)
