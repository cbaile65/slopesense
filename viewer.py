import open3d as o3d
import os
import numpy as np

# ----------------------------
# 1. Set the path to your PLY file
# ----------------------------
home = os.path.expanduser("~")
ply_path = os.path.join(home, "Downloads", "multidefect.ply")

if not os.path.exists(ply_path):
    raise FileNotFoundError(f"PLY file not found: {ply_path}")

# ----------------------------
# 2. Load the point cloud
# ----------------------------
pcd = o3d.io.read_point_cloud(ply_path)
print("Point cloud loaded:")
print(f"Number of points: {len(pcd.points)}")

# ----------------------------
# 3. Create a horizontal "shadow"
# ----------------------------
shadow_z = -1  # <-- change this value to move shadow up/down
shadow = o3d.geometry.PointCloud()
shadow_points = np.asarray(pcd.points).copy()
shadow_points[:, 2] = shadow_z  # flatten all points onto horizontal plane
shadow.points = o3d.utility.Vector3dVector(shadow_points)

# Change the shadow to lighter gray
shadow.paint_uniform_color([0.7, 0.7, 0.7])  # light gray shadow

# ----------------------------
# 4. Visualize both
# ----------------------------
o3d.visualization.draw_geometries(
    [pcd, shadow],
    window_name="Point Cloud with Horizontal Shadow",
    width=800,
    height=600,
    point_show_normal=False,
    mesh_show_wireframe=False
)
