import numpy as np
import cv2
import os
from glob import glob
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R, Slerp

# ============================================================
# 1️. SLAM Pose 로드 및 보간
# ============================================================
def load_slam_poses(slam_path):
    """ORB-SLAM3 KeyFrameTrajectory.txt 전체 Pose 로드"""
    poses = []
    if not os.path.exists(slam_path):
        raise FileNotFoundError(f"X. SLAM Pose 파일이 없습니다: {slam_path}")

    with open(slam_path, "r") as f:
        for line in f:
            if line.startswith("#") or len(line.strip()) == 0:
                continue
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            timestamp = float(parts[0])
            tx, ty, tz = map(float, parts[1:4])
            qx, qy, qz, qw = map(float, parts[4:8])
            poses.append({
                "t": timestamp,
                "R": R.from_quat([qx, qy, qz, qw]),
                "T": np.array([[tx], [ty], [tz]])
            })
    print(f"> Loaded {len(poses)} SLAM poses")
    return poses


def interpolate_pose(poses, target_t):
    """주어진 target_t에 대해 SLAM pose 보간"""
    if target_t <= poses[0]["t"]:
        return poses[0]["R"], poses[0]["T"]
    if target_t >= poses[-1]["t"]:
        return poses[-1]["R"], poses[-1]["T"]

    for i in range(len(poses) - 1):
        t0, t1 = poses[i]["t"], poses[i + 1]["t"]
        if t0 <= target_t <= t1:
            ratio = (target_t - t0) / (t1 - t0)
            key_rots = R.from_quat([
                poses[i]["R"].as_quat(),
                poses[i + 1]["R"].as_quat()
            ])
            slerp = Slerp([0, 1], key_rots)
            R_interp = slerp([ratio])[0]
            T_interp = poses[i]["T"] * (1 - ratio) + poses[i + 1]["T"] * ratio
            return R_interp, T_interp
    return poses[-1]["R"], poses[-1]["T"]

# ============================================================
# 2️. Depth 매칭
# ============================================================
def find_closest_depth(depth_folder, frame_idx):
    """현재 프레임에 가장 가까운 depth 파일 경로 반환"""
    depth_files = sorted(glob(os.path.join(depth_folder, "depth_*.png")))
    depth_indices = [int(os.path.basename(f).split("_")[1].split(".")[0]) for f in depth_files]
    if not depth_indices:
        raise FileNotFoundError(f"X. Depth 파일이 없습니다: {depth_folder}")
    closest = min(depth_indices, key=lambda x: abs(x - frame_idx))
    return os.path.join(depth_folder, f"depth_{closest:04d}.png")

# ============================================================
# 3️. 3D Projection (timestamp 포함)
# ============================================================
def project_to_3d(centerline_path, depth_folder, slam_path, output_path):
    print("> Loading data")
    poses = load_slam_poses(slam_path)
    fx, fy = 1464.357, 1463.089
    cx, cy = 961.935, 529.971

    print(" 중심선 좌표 불러오는 중")
    with open(centerline_path, "r") as f:
        lines = f.readlines()

    frame_centers = []
    current = []
    for line in lines:
        if line.startswith("# Frame"):
            if current:
                frame_centers.append(np.array(current))
                current = []
            continue
        parts = line.strip().split()
        if len(parts) == 2:
            x, y = map(float, parts)
            current.append([x, y])
    if current:
        frame_centers.append(np.array(current))
    print(f"> 총 {len(frame_centers)} 프레임 로드 완료")

    total_frames = len(frame_centers)
    timestamps = np.linspace(poses[0]["t"], poses[-1]["t"], total_frames)

    points_world = []
    points_timestamps = []

    for i, (frame_coords, t) in enumerate(zip(frame_centers, timestamps)):
        if len(frame_coords) == 0:
            continue

        depth_path = find_closest_depth(depth_folder, i)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        if depth_img is None:
            print(f"warn Depth 이미지 불러오기 실패: {depth_path}")
            continue
        depth_img /= 255.0
        depth_img *= 50.0  # 최대 50m까지

        R_interp, T_interp = interpolate_pose(poses, t)

        for (x, y) in frame_coords:
            if 0 <= int(y) < depth_img.shape[0] and 0 <= int(x) < depth_img.shape[1]:
                Z = depth_img[int(y), int(x)]
                if Z <= 0:
                    continue
                X = (x - cx) * Z / fx
                Y = (y - cy) * Z / fy
                pt_cam = np.array([[X], [Y], [Z]])
                pt_world = R_interp.as_matrix() @ pt_cam + T_interp

                points_world.append(pt_world.flatten())
                points_timestamps.append(t)  # timestamp 추가

        if i % 50 == 0:
            print(f"Frame {i}/{total_frames} 변환 중...")

    points_world = np.array(points_world)
    points_timestamps = np.array(points_timestamps).reshape(-1, 1)
    points_with_time = np.hstack([points_timestamps, points_world])

    np.savetxt(output_path, points_with_time, fmt="%.6f")
    print(f"> Saved timestamped 3D points → {output_path}, shape={points_with_time.shape}")

# ============================================================
# 4️. 실행부
# ============================================================
if __name__ == "__main__":
    base = "./ARHUD/data"

    project_to_3d(
        centerline_path=os.path.join(base, "video_centerlines_ridge.txt"),
        depth_folder=os.path.join(base, "depth_frames"),
        slam_path=os.path.join(base, "KeyFrameTrajectory.txt"),
        output_path=os.path.join(base, "points_3d_full_timestamped.txt"),
    )
