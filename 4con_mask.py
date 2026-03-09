import os
import numpy as np
import open3d as o3d
from scipy.interpolate import griddata
from scipy.ndimage import sobel
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def pointcloud_to_grid(points, grid_res=0.5):
    """Convert scattered point-cloud data into a regular X-Y grid."""
    # Split point cloud into x, y, and z arrays
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Create evenly-spaced x and y ranges for the grid
    xi = np.arange(x.min(), x.max(), grid_res)
    yi = np.arange(y.min(), y.max(), grid_res)

    # Create 2D grid of coordinates
    X, Y = np.meshgrid(xi, yi)

    # Interpolate Z values from the point cloud onto the grid
    Z = griddata((x, y), z, (X, Y), method='linear')
    return X, Y, Z


def compute_slope_map(Z, cell_size=1.0):
    """Compute slope (in degrees) from the height map."""
    # Estimate change in Z along X and Y using Sobel filters
    dzdx = sobel(Z, axis=1, mode='nearest') / (8.0 * cell_size)
    dzdy = sobel(Z, axis=0, mode='nearest') / (8.0 * cell_size)

    # Convert rise/run into slope angle
    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    slope_deg = np.degrees(slope_rad)
    return slope_deg


def mask_indents(Z, tolerance=0.01):
    """Find areas that dip below the fitted plane by more than 'tolerance'."""
    # Build grids of index positions (X and Y pixel indices)
    X_idx, Y_idx = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))

    # Keep only valid Z values (ignore NaNs)
    valid_mask = ~np.isnan(Z)
    XY = np.column_stack([X_idx[valid_mask], Y_idx[valid_mask]])
    Z_flat = Z[valid_mask]

    # Fit a best-fit plane to the surface
    reg = LinearRegression().fit(XY, Z_flat)

    # Predict what the plane height should be at each grid location
    Z_plane = reg.predict(np.column_stack([X_idx.ravel(), Y_idx.ravel()])).reshape(Z.shape)

    # Points lower than the plane by more than the tolerance are indents
    indent_mask = Z < (Z_plane - tolerance)
    return indent_mask


def main():
    # Load point cloud from the Downloads folder
    ply_path = os.path.join(os.path.expanduser("~"), "Downloads", "defect.ply")
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    print(f"Loaded {len(points)} points.")

    # Convert raw point cloud to a uniform grid
    X, Y, Z = pointcloud_to_grid(points, grid_res=0.5)

    # Replace missing grid values (NaN) with the average height
    Z = np.nan_to_num(Z, nan=np.nanmean(Z))

    # Compute distance between grid cells for slope calculation
    cell_size = X[0, 1] - X[0, 0]

    # Compute slope map
    slope = compute_slope_map(Z, cell_size=cell_size)

    # Detect areas that are lower than the fitted plane (possible indent)
    indent_mask = mask_indents(Z, tolerance=0.01)

    # --- Plot results ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot ranges based on unique X and Y positions
    extent = [0, len(np.unique(points[:, 0])), 0, len(np.unique(points[:, 1]))]

    # Show height map
    axs[0].imshow(Z, cmap='gray', origin='lower', extent=extent)
    axs[0].set_title('Height Map (Z)')

    # Overlay indent mask in transparent red
    red_mask = np.zeros((*indent_mask.shape, 4))
    red_mask[indent_mask] = [1, 0, 0, 0.5]
    axs[0].imshow(red_mask, origin='lower', extent=extent)

    # Show slope map
    im1 = axs[1].imshow(slope, cmap='inferno', origin='lower', extent=extent)
    axs[1].set_title('Slope (degrees)')
    fig.colorbar(im1, ax=axs[1], fraction=0.046)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

    #Thomas
