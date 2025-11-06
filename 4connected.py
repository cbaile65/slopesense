import os
import numpy as np
import open3d as o3d
from scipy.interpolate import griddata
from scipy.ndimage import sobel
import matplotlib.pyplot as plt


def pointcloud_to_grid(points, grid_res=0.5):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    xi = np.arange(x.min(), x.max(), grid_res)
    yi = np.arange(y.min(), y.max(), grid_res)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata((x, y), z, (X, Y), method='linear')
    return X, Y, Z


def compute_slope_map(z, cell_size=1.0):
    dzdx = sobel(z, axis=1, mode='nearest') / (8.0 * cell_size)
    dzdy = sobel(z, axis=0, mode='nearest') / (8.0 * cell_size)
    slope_rad = np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2))
    slope_deg = np.degrees(slope_rad)
    return slope_deg


def main():
    ply_path = os.path.join(os.path.expanduser("~"), "Downloads", "defect.ply")

    print(f"Loading point cloud from: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    print(f"Loaded {len(points)} points.")

    X, Y, Z = pointcloud_to_grid(points, grid_res=0.5)
    Z = np.nan_to_num(Z, nan=np.nanmean(Z))

    cell_size = X[0, 1] - X[0, 0]
    slope = compute_slope_map(Z, cell_size=cell_size)

    # --- Display height and slope maps with axes showing number of original points ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Map grid to number of points in the point cloud
    num_points_x = len(np.unique(points[:, 0]))
    num_points_y = len(np.unique(points[:, 1]))
    extent = [0, num_points_x, 0, num_points_y]  # [x_min, x_max, y_min, y_max]

    im0 = axs[0].imshow(Z, cmap='gray', origin='lower', extent=extent)
    axs[0].set_title('Height Map (Z)')

    im1 = axs[1].imshow(slope, cmap='inferno', origin='lower', extent=extent)
    axs[1].set_title('Slope (degrees)')
    fig.colorbar(im1, ax=axs[1], fraction=0.046)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
