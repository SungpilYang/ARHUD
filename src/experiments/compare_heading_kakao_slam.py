import requests
import math
import numpy as np
import cv2

# 1️. 카카오 Directions API 호출
API_KEY = "api 키 필요"
headers = {"Authorization": f"KakaoAK {API_KEY}"}

origin = "127.046622,37.289070"
destination = "127.051037,37.285772"

url = f"https://apis-navi.kakaomobility.com/v1/directions?origin={origin}&destination={destination}&priority=TIME"
res = requests.get(url, headers=headers)
data = res.json()

route = data["routes"][0]
points = []
for section in route["sections"]:
    for road in section["roads"]:
        coords = road["vertexes"]
        for i in range(0, len(coords), 2):
            points.append((coords[i+1], coords[i]))  # (lat, lon)

# 2️. 위경도 → 미터 단위 변환
def latlon_to_m(lat, lon, lat0, lon0):
    R = 6378137
    dx = math.radians(lon - lon0) * R * math.cos(math.radians(lat0))
    dy = math.radians(lat - lat0) * R
    return np.array([dx, dy])

lat0, lon0 = points[0]
waypoints_m = np.array([latlon_to_m(lat, lon, lat0, lon0) for lat, lon in points])

# 3️. SLAM 트래젝토리 불러오기
slam_path = r".\ARHUD\data\KeyFrameTrajectory.txt"
slam_points = []
with open(slam_path, "r") as f:
    for line in f:
        if line.startswith("#"):
            continue
        data = line.strip().split()
        if len(data) >= 8:
            tx = float(data[1])
            tz = float(data[3])
            # A: 기존 테스트 조합 (SLAM X와 Z를 Y, X에 매핑)
            #slam_points.append([-tx, tz]) # [X: 동, Y: 북] = [-SLAM_X, SLAM_Z]
            #slam_points.append([tz, -tx]) # [X: 동, Y: 북] = [ SLAM_Z, -SLAM_X] 그나마 나음
            # B: SLAM의 축 순서가 반대일 경우
            #slam_points.append([tx, -tz]) # [X: 동, Y: 북] = [ SLAM_X, -SLAM_Z]
            slam_points.append([-tz, tx]) # [X: 동, Y: 북] = [-SLAM_Z, SLAM_X]
slam_points = np.array(slam_points)

# 4️. SLAM-지도 정합
def align_slam_to_gps(slam_points, gps_points):
    slam_mean = np.mean(slam_points, axis=0)
    gps_mean = np.mean(gps_points, axis=0)
    slam_centered = slam_points - slam_mean
    gps_centered = gps_points - gps_mean
    H = slam_centered.T @ gps_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T
    scale = np.sum(S) / np.sum(slam_centered ** 2)
    T = gps_mean - scale * (R @ slam_mean)
    theta = math.atan2(R[1, 0], R[0, 0])
    return scale, theta, T, R

slam_sample = slam_points[::len(slam_points)//len(waypoints_m)][:len(waypoints_m)]
gps_sample = waypoints_m[:len(slam_sample)]
scale, theta, T, R = align_slam_to_gps(slam_sample, gps_sample)

slam_aligned = scale * (R @ slam_points.T).T + T

# 5️. Yaw 계산 및 보정 (수정)

# 1. 정합 전 SLAM 경로의 Yaw 계산
vec_forward_slam_raw = np.diff(slam_points, axis=0)
yaw_slam_raw = np.unwrap(np.arctan2(vec_forward_slam_raw[:, 1], vec_forward_slam_raw[:, 0]))

# 2. 정합 후 SLAM 경로의 Yaw 계산 (참고용)
vec_forward_all = np.diff(slam_aligned, axis=0)
yaw_slam_aligned = np.unwrap(np.arctan2(vec_forward_all[:, 1], vec_forward_all[:, 0]))

# 3. 지도 경로의 Yaw 계산
vec_map = np.diff(waypoints_m, axis=0)
yaw_map = np.unwrap(np.arctan2(vec_map[:, 1], vec_map[:, 0]))

#  모든 Yaw 배열의 길이를 가장 짧은 배열에 맞춤.
min_len = min(len(yaw_slam_aligned), len(yaw_map)) 

# 이 시점에서 모든 배열의 길이를 min_len으로 자릅니다.
yaw_slam_aligned = yaw_slam_aligned[:min_len]
yaw_map = yaw_map[:min_len] 
# =======================================================

# 3. SLAM Yaw 보정 (초기 오프셋 문제 해결 시도)
OFFSET_LEN = 100 
actual_offset_len = min(min_len, OFFSET_LEN) # 현재 12

# 오프셋 계산
offset = np.median(yaw_map[:actual_offset_len] - yaw_slam_aligned[:actual_offset_len]) 
yaw_slam_compensated = yaw_slam_aligned + offset
# yaw_slam_compensated의 길이는 min_len (12)입니다.

# 4. 스무딩 (SLAM의 빠른 흔들림 제거)
yaw_diff = np.diff(yaw_slam_compensated) 
yaw_diff[np.abs(np.degrees(yaw_diff)) > 10] = 0
yaw_slam_filtered = np.cumsum(np.insert(yaw_diff, 0, yaw_slam_compensated[0]))

# Low-pass filter (지도 Yaw와 블렌딩)
alpha = 0.05
# 여기서 두 배열 (길이 12 vs 길이 12)이 블렌딩됩니다.
yaw_blend = (1 - alpha) * yaw_slam_filtered + alpha * yaw_map 

window = 5
yaw_smooth = np.convolve(yaw_blend, np.ones(window)/window, mode="same")
yaw_slam_corrected = yaw_smooth # 최종 Yaw Current
#  HUD 영상 오버레이 루프 

cap = cv2.VideoCapture(r".\ARHUD\data\test1_with_traj.mp4")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
slam_len = len(slam_aligned)

h, w = 480, 640
center = (w // 2, h - 60)
length = 100
yaw_error_prev = 0
max_up, max_down = 8, 40
frame_idx = 0

print("영상 프레임 수:", total_frames)
print("SLAM 경로 수:", slam_len)
print("-" * 40)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임 비율로 SLAM 인덱스 매핑
    ratio = frame_idx / total_frames
    slam_idx = int(ratio * (slam_len - 1))

    # 현재 위치 및 목표 위치 계산
    cam_pos = slam_aligned[slam_idx]
    distances = np.linalg.norm(waypoints_m - cam_pos, axis=1)
    nearest_idx = np.argmin(distances)
    next_idx = min(nearest_idx + 1, len(waypoints_m) - 1)
    target_pos = waypoints_m[next_idx]

    # Yaw 계산
    yaw_current = yaw_slam_corrected[min(slam_idx, len(yaw_slam_corrected) - 1)]
    vec_target = target_pos - cam_pos
    yaw_target = math.atan2(vec_target[1], vec_target[0])
    yaw_error = (yaw_target - yaw_current + math.pi) % (2 * math.pi) - math.pi
    yaw_error_deg_raw = math.degrees(yaw_error)
    
    #  카메라(운전자) 기준으로 부호 보정
    yaw_error_deg_raw = -yaw_error_deg_raw


    # error 제한
    delta = yaw_error_deg_raw - yaw_error_prev
    if delta > max_up:
        yaw_error_deg = yaw_error_prev + max_up
    elif delta < -max_down:
        yaw_error_deg = yaw_error_prev - max_down
    else:
        yaw_error_deg = yaw_error_deg_raw
    yaw_error_prev = yaw_error_deg
    
    
    #  터미널 출력 (SLAM 인덱스 기준)
    print(f"SLAM Frame {slam_idx:3d}/{slam_len:3d} | Yaw Error: {yaw_error_deg:7.2f}°")

    # HUD 렌더링
    overlay = frame.copy()
    end_point = (
        int(center[0] + length * math.sin(math.radians(yaw_error_deg))),
        int(center[1] - length * math.cos(math.radians(yaw_error_deg)))
    )
    color = (0, 255, 0) if abs(yaw_error_deg) < 10 else (0, 0, 255)
    cv2.arrowedLine(overlay, center, end_point, color, 4, tipLength=0.3)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    cv2.putText(frame, f"{yaw_error_deg:+.1f}°", (center[0] - 40, center[1] - 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Frame {frame_idx}/{total_frames}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("HUD Simulation (Video Overlay)", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
