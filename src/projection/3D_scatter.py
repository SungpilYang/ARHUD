import numpy as np
import matplotlib.pyplot as plt

# =====================
# 1) 데이터 로드
# =====================
path = "./ARHUD/data/points_3d_full_timestamped.txt"  # 네 경로로 수정
data = np.loadtxt(path)

print("Loaded:", data.shape)

# =====================
# 2) 컬럼 파싱
# =====================
t = data[:, 0]      # timestamp
X = data[:, 1]
Y = data[:, 2]
Z = data[:, 3]

# =====================
# 3) 컬러 매핑 (timestamp → 색)
# =====================
# timestamp normalized
t_norm = (t - t.min()) / (t.max() - t.min())

# =====================
# 4) 샘플링 (포인트가 너무 많다면)
# =====================
max_points = 200000
if len(X) > max_points:
    idx = np.random.choice(len(X), max_points, replace=False)
    X, Y, Z, t_norm = X[idx], Y[idx], Z[idx], t_norm[idx]

# =====================
# 5) 3D Scatter Plot
# =====================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

p = ax.scatter(X, Y, Z, c=t_norm, cmap='turbo', s=1)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Reconstruction with Timestamp Coloring")

fig.colorbar(p, ax=ax, shrink=0.6, label="Time (normalized)")

# =====================
# 6) 저장
# =====================
save_path = ".ARHUD/points_3d_timestamp_scatter2.png"
plt.savefig(save_path, dpi=300)
plt.show()

print("Saved:", save_path)
