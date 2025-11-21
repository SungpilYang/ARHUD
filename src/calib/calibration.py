import cv2
import numpy as np
import glob
import yaml
import os

# === 사용자 설정 ===
pattern_size = (10, 7)   # 내부 코너 수 (11x8 체커보드면 10x7)
image_path = r".\ARHUD\data\camera_calibration\calib_images\*.jpg"
## resize_to = (1920, 1080)  # 너무 큰 사진일 경우 리사이즈

# === 준비 ===
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

objpoints = []  # 3D
imgpoints = []  # 2D

images = glob.glob(image_path)
print(f" 이미지 파일 개수: {len(images)}")

if len(images) == 0:
    raise FileNotFoundError("X 이미지가 없습니다. 경로를 확인하세요!")

# === 체커보드 탐지 ===
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"warn 이미지 로드 실패: {fname}")
        continue

    ## img = cv2.resize(img, resize_to)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    print(f"{os.path.basename(fname)} → {'✅ 성공' if ret else '❌ 실패'}")

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)
        cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
        cv2.imshow('Detected Corners', img)
        cv2.waitKey(300)

cv2.destroyAllWindows()

# === 보정 ===
if len(objpoints) == 0:
    print("X 코너를 인식한 이미지가 없습니다. 조명이나 패턴을 확인하세요.")
    exit()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

print("\n >보정 완료!")
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)

# === 결과 저장 ===
data = {'camera_matrix': mtx.tolist(), 'dist_coeff': dist.tolist()}
with open('camera_calibration.yaml', 'w') as f:
    yaml.dump(data, f)
print("📄 camera_calibration.yaml 파일로 결과 저장됨.")


sample = cv2.imread(images[0])
h, w = sample.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
dst = cv2.undistort(sample, mtx, dist, None, newcameramtx)

cv2.imshow('Before', sample)
cv2.imshow('After (Undistorted)', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
