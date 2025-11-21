# ARHUD

#  AR-HUD Navigation System  
### *Monocular Camera 기반 증강현실 내비게이션 시스템*  
(2025-2 Media project Project )
<img width="1920" height="1080" alt="떵피리 (1)" src="https://github.com/user-attachments/assets/608046f6-2291-422f-b4e3-1423d5f90058" />


본 프로젝트는 단일 카메라 영상만을 이용해  
**ORB-SLAM3 기반 Monocular Visual SLAM →Lane Segmentation → Centerline Extraction → Depth Estimation → 3D Projection → AR HUD 렌더링**까지 전 과정을 구현한 AR Heads-Up Navigation Pipeline입니다.

---
##  Key Features

### 1. YOLOP 기반 Lane & Drivable Area Segmentation
- 실도로 영상에서 차선/도로를 분할
- Morphology + Sobel + Ridge Regression + EMA smoothing으로 중심선 안정화

<img width="430" height="270" alt="road_segmented_segformer" src="https://github.com/user-attachments/assets/5957e47d-4d7f-410a-8b3b-14c8963c0c12" />
<img width="430" height="270" alt="road_center_spline_trimmed2" src="https://github.com/user-attachments/assets/8960ced1-f5e7-4e50-b27b-20543b122de9" />


### 2. MiDaS Depth Estimation
- Monocular depth map 생성  
- Frame index 기반 depth 매칭
<img width="430" height="270" alt="depth_0005" src="https://github.com/user-attachments/assets/a611c2a7-0780-44c7-8e0c-bad9cd1251de" />
<img width="430" height="270" alt="depth_0560" src="https://github.com/user-attachments/assets/499615cc-3598-4f6c-a4ae-aa638e04e91a" />



### 3. ORB-SLAM3 Pose Tracking (Ubuntu)
- KeyFrameTrajectory 기반 카메라 위치/자세 수집  
- Timestamp 기반 Slerp pose interpolation

<img width="430" height="270" alt="ORB-SLAM3 Current Frame_screenshot_18 11 2025" src="https://github.com/user-attachments/assets/fda93b61-b41d-42ef-a644-6ad25f782c68" />
<img width="430" height="270" alt="SLAM_Trajectory3" src="https://github.com/user-attachments/assets/6451e1ff-43d9-4664-a512-d17d5bc0aa50" />


###  4. 2D Centerline → 3D World 좌표 변환
- Intrinsic K + depth + pose로 3D back-projection  
- Timestamp 포함하여 정교한 time-synchronized reconstruction 수행
<img width="430" height="350" alt="points_3d_timestamp_scatter" src="https://github.com/user-attachments/assets/36a40494-0b47-40ae-b59d-c16c8152218f" />

### 5. SLAM Heading → Map Heading 정합
- Heading mismatch 보정  
- Low-pass filter + smooth yaw correction 적용

### 6. OpenGL 기반 AR-HUD 렌더링
- 방향 화살표, 가이던스 표시  
- 차량 전방 화면에 HUD 오버레이

<img width="430" height="270" alt="fianl_image2" src="https://github.com/user-attachments/assets/99ba0171-e242-4ec4-9c94-ba0ff9adf283" />
<img width="430" height="270" alt="fianl_image5" src="https://github.com/user-attachments/assets/0d3092e2-9fca-44df-95c0-e528c97de408" />

---

## 📁 Project Structure

```
ARHUD/
│
├── data/
│ ├── camera_calibration/ # 체커보드 캘리브레이션 데이터
│ ├── ORB_SLAM3/ # SLAM 출력 데이터 (Trajectory 등)
│ ├── test1.mp4 # 원본 테스트 영상
│ ├── test1_with_traj.mp4 # SLAM trajectory overlay 영상
│ ├── points_3d_full_timestamped.txt# 최종 3D point cloud 결과
│ ├── video_centerlines_ridge.txt # YOLOP 기반 중심선 결과
│ ├── video_centerlines_polyfit.txt # Polyfit 실험 결과
│ └── road_test*.png # 테스트용 이미지
│
├── src/
│ ├── calib/
│ │ └── calibration.py #체커보드 캘리브레이션 코드
│ │
│ ├── perception/
│ │ ├── depth_extract.py   # 프레임별 depth 추출 코드
│ │ ├── lane_center_extractor_ridge_ema.py  #도로 중심선 추출 코드
│ │ ├── lane_center_extractor_polyfit_ver.py #도로 중심선 추출 코드 (test용)
│ │ ├── road_segmetation_yolop_test.py #도로, 차선 추출 
│ │ └── road_segmetation_nvidia_test.py #도로, 차선 추출 (test 용)
│ │
│ ├── projection/
│ │ ├── centerline_to_3d.py # 차선 3D 포인트 클라우드 변환
│ │ └── 3D_scatter.py # 3D 시각화
│ │
│ ├── slam_postprocess/
│ │ ├── create_timestamps.py
│ │ ├── DirOveray.py # SLAM 경로 시각화
│ │ ├── make_orbslam_yaml.py #slam 결과 변환 
│ │ └── show_trajectory.py #SLAM 경로 시각화
│ │
│ └── rendering/
│ ├── render_opengl.cpp  #AR-HUD 렌더링 코드 (shader 포함)
│
└── README.md



##  Videos (GitHub Releases)

### 1️. SLAM Trajectory Overlay  
파일: `test1_with_traj.mp4`  
설명: ORB-SLAM3으로 추출한 카메라 트래젝토리를 원본 영상에 오버레이한 버전.

 https://github.com/USERNAME/ARHUD/releases/download/v1/test1_with_traj.mp4


### 2. Raw Test Footage  
파일: `test1.mp4`  
설명: 촬영 원본.

 https://github.com/USERNAME/ARHUD/releases/download/v1/test1.mp4


### 3. Final AR-HUD Projection Result  
파일: `final_result.mp4`  
설명: 차량 전방 영상에 HUD 내비게이션 화살표를 AR로 렌더링한 최종 결과.

 https://github.com/USERNAME/ARHUD/releases/download/v1/final_result.mp4

---

##  Pipeline Overview
Camera Frame
↓
ORB-SLAM3 Pose (KeyFrameTrajectory.txt)
↓ Pose Interpolation (Slerp)
↓
YOLOP Lane Segmentation
↓
Centerline Extraction (Ridge + EMA)
↓
MiDaS Depth Estimation
↓
3D Projection (pixel + depth + pose)
↓
SLAM→Map Heading Correction
↓
OpenGL Rendering (AR-HUD Overlay)

##  Tech Stack

### **Computer Vision**
- PyTorch (YOLOP)
- MiDaS Depth v3.1
- OpenCV 4.x

### **SLAM**
- ORB-SLAM3 (Monocular)

### **Graphics**
- OpenGL + GLFW + GLAD
- glm

### **Language**
- Python 3
- C++17

---

##  Setup & Run

### 1. Clone Repository
```bash
git clone https://github.com/USERNAME/ARHUD.git
