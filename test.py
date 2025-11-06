import os
import numpy as np
import open3d as o3d
from scipy.interpolate import griddata
from scipy.ndimage import sobel, label
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def pointcloud_to_grid(points, grid_res=0.5):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    xi = np.arange(x.min(), x.max(), grid_res)
    yi = np.arange(y.min(), y.max(), grid_res)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata((x, y), z, (X, Y), method='linear')
    return X, Y, Z


def compute_slope_map(Z, cell_size=1.0):
    dzdx = sobel(Z, axis=1, mode='nearest') / (8.0 * cell_size)
    dzdy = sobel(Z, axis=0, mode='nearest') / (8.0 * cell_size)
    slope_rad = np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2))
    slope_deg = np.degrees(slope_rad)
    return slope_deg


def mask_indents(Z, tolerance=0.01):
    """Mask points that are below the fitted plane by more than 'tolerance'."""
    X_idx, Y_idx = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
    valid_mask = ~np.isnan(Z)
    XY = np.column_stack([X_idx[valid_mask], Y_idx[valid_mask]])
    Z_flat = Z[valid_mask]

    reg = LinearRegression().fit(XY, Z_flat)
    Z_plane = reg.predict(np.column_stack([X_idx.ravel(), Y_idx.ravel()])).reshape(Z.shape)

    indent_mask = Z < (Z_plane - tolerance)
    return indent_mask


def main():
    ply_path = os.path.join(os.path.expanduser("~"), "Downloads", "defect.ply")
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    print(f"Loaded {len(points)} points.")

    X, Y, Z = pointcloud_to_grid(points, grid_res=0.5)
    Z = np.nan_to_num(Z, nan=np.nanmean(Z))

    cell_size = X[0, 1] - X[0, 0]
    slope = compute_slope_map(Z, cell_size=cell_size)

    # --- Detect shallow regions --- DOES NOT WORK PROPERLY *FIX*
    #**also need to generate another test point cloud with multiple defects
    shallow_mask = slope < 4.0
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]])
    labeled, num_shallow_regions = label(shallow_mask, structure=structure)
    print(f"Detected {num_shallow_regions} shallow regions.")

    # --- Detect indent points ---
    indent_mask = mask_indents(Z, tolerance=0.01)

    # --- Display ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    extent = [0, len(np.unique(points[:, 0])), 0, len(np.unique(points[:, 1]))]

    # Height map with red indent overlay
    axs[0].imshow(Z, cmap='gray', origin='lower', extent=extent)
    axs[0].set_title('Height Map (Z)')
    red_mask = np.zeros((*indent_mask.shape, 4))
    red_mask[indent_mask] = [1, 0, 0, 0.5]  # Red with 50% alpha
    axs[0].imshow(red_mask, origin='lower', extent=extent)

    # Slope map
    im1 = axs[1].imshow(slope, cmap='inferno', origin='lower', extent=extent)
    axs[1].set_title('Slope (degrees)')
    fig.colorbar(im1, ax=axs[1], fraction=0.046)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
