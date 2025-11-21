import yaml

with open('camera_calibration.yaml', 'r') as f:
    data = yaml.safe_load(f)

mtx = data['camera_matrix']
dist = data['dist_coeff'][0]

out = {
    'Camera.fx': mtx[0][0],
    'Camera.fy': mtx[1][1],
    'Camera.cx': mtx[0][2],
    'Camera.cy': mtx[1][2],
    'Camera.k1': dist[0],
    'Camera.k2': dist[1],
    'Camera.p1': dist[2],
    'Camera.p2': dist[3],
    'Camera.k3': dist[4],
    'Camera.width': 1920,
    'Camera.height': 1080,
    'Camera.fps': 30.0,
    'Camera.model': 'PINHOLE',

    'ThDepth': 40.0
}

with open('orbslam3_camera.yaml', 'w') as f:
    yaml.dump(out, f)

print("> ORB-SLAM3용 YAML 변환 완료: orbslam3_camera.yaml")
