import numpy as np
import matplotlib.pyplot as plt

# 파일 경로 설정
path = "KeyFrameTrajectory.txt"

# 주석(#) 제외하고 데이터 읽기
data = np.loadtxt(path, comments='#')

# 각 열 분리
timestamps = data[:, 0]
tx, ty, tz = data[:, 1], data[:, 2], data[:, 3]

# 3D 궤적 시각화
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(tx, ty, tz, label='Camera trajectory')
ax.scatter(tx[0], ty[0], tz[0], color='green', label='Start')
ax.scatter(tx[-1], ty[-1], tz[-1], color='red', label='End')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()
