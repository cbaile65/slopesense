import pyrealsense2 as rs
import numpy as np
import cv2

# ----------------- Settings -----------------
W, H, FPS = 640, 480, 30
TIMEOUT_MS = 10000
DEPTH_STACK_N = 18

Z_MIN_M, Z_MAX_M = 0.15, 2.00
BAND_HALF_M = 0.02
SMOOTH_K = 7

FG_MIN_AREA = 3000
INTERIOR_ERODE_PX = 31

MIN_DEFECT_AREA_PX = 50
POST_CLOSE_K = 5

AUTO_SIGMA_MULT = 1.9
MIN_DEFECT_MM = 0.6
MAX_DEFECT_MM = 3.0

LOCAL_TREND_SIGMA = 31
RESIDUAL_BLUR_K = 5

PERSIST_N = 20
PERSIST_MIN_HITS = 15
PERSIST_DILATE_K = 5

MM_PER_INCH = 25.4
MIN_BBOX_W_IN = 3.0
MIN_BBOX_H_IN = 3.0

SHOW_DIVOTS = True


# ----------------- Helpers -----------------
def largest_component(mask_u8, min_area=500):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num <= 1: return np.zeros_like(mask_u8)
    best_i = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    if stats[best_i, cv2.CC_STAT_AREA] < min_area: return np.zeros_like(mask_u8)

    out = np.zeros_like(mask_u8)
    x, y, w, h = stats[best_i, cv2.CC_STAT_LEFT], stats[best_i, cv2.CC_STAT_TOP], stats[best_i, cv2.CC_STAT_WIDTH], \
        stats[best_i, cv2.CC_STAT_HEIGHT]
    out[y:y + h, x:x + w][labels[y:y + h, x:x + w] == best_i] = 255
    return out


def filter_components(mask_u8, min_area):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    out = np.zeros_like(mask_u8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[
                i, cv2.CC_STAT_HEIGHT]
            out[y:y + h, x:x + w][labels[y:y + h, x:x + w] == i] = 255
    return out


def robust_sigma(vals):
    vals = vals[np.isfinite(vals)]
    if vals.size < 20: return 0.0
    return 1.4826 * np.median(np.abs(vals - np.median(vals)))


def make_interior_mask(fg_mask_u8, erode_px):
    if erode_px < 3: return fg_mask_u8.copy()
    return cv2.erode(fg_mask_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px, erode_px)), iterations=1)


def draw_mask_outline(img, mask_u8, color, thickness=2):
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, thickness)


def fit_quadratic_surface(depth, mask):
    ys, xs = np.where(mask)
    if len(xs) < 1200:
        return np.array([0, 0, 0, 0, 0, float(np.median(depth[mask]))], dtype=np.float32) if np.any(mask) else np.zeros(
            6, dtype=np.float32)
    x, y, z = xs.astype(np.float32), ys.astype(np.float32), depth[ys, xs].astype(np.float32)
    coeffs, *_ = np.linalg.lstsq(np.stack([x * x, y * y, x * y, x, y, np.ones_like(x)], axis=1), z, rcond=None)
    return coeffs.astype(np.float32)


def quadratic_surface_img(coeffs, h, w):
    a, b, c, d, e, f = coeffs
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return a * X * X + b * Y * Y + c * X * Y + d * X + e * Y + f


def masked_local_trend(img, mask, sigma):
    mask_f = mask.astype(np.float32)
    num = cv2.GaussianBlur((img * mask_f).astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    den = cv2.GaussianBlur(mask_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return num / (den + 1e-6)


def filter_components_physical(mask_u8, depth_s, fx, fy, min_w_in, min_h_in):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    out = np.zeros_like(mask_u8)
    min_w_mm, min_h_mm = min_w_in * MM_PER_INCH, min_h_in * MM_PER_INCH

    for i in range(1, num):
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[
            i, cv2.CC_STAT_HEIGHT]
        if stats[i, cv2.CC_STAT_AREA] <= 0: continue

        comp_mask = (labels[y:y + h, x:x + w] == i)
        z_vals = depth_s[y:y + h, x:x + w][comp_mask]
        z_vals = z_vals[(z_vals > 0) & np.isfinite(z_vals)]
        if z_vals.size == 0: continue

        z_med_mm = float(np.median(z_vals)) * 1000.0
        if (w * z_med_mm / fx) >= min_w_mm and (h * z_med_mm / fy) >= min_h_mm:
            out[y:y + h, x:x + w][comp_mask] = 255
    return out


# ----------------- Main Class -----------------
class DefectDetector:
    def __init__(self):
        # Start only RGB and Depth on the primary pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
        config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)

        self.profile = self.pipeline.start(config)

        # Expose the raw device so AutoLeveler can grab the IMU sensor separately!
        self.device = self.profile.get_device()
        self.depth_scale = self.device.first_depth_sensor().get_depth_scale()

        intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.fx, self.fy = float(intr.fx), float(intr.fy)

        self.align = rs.align(rs.stream.color)
        self.k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.k3 = np.ones((3, 3), np.uint8)
        self.kclose = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (POST_CLOSE_K, POST_CLOSE_K))
        self.kvote = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (PERSIST_DILATE_K, PERSIST_DILATE_K))

        self.depth_stack = np.zeros((DEPTH_STACK_N, H, W), dtype=np.float32)
        self.depth_count = 0
        self.div_history = np.zeros((PERSIST_N, H, W), dtype=np.uint8)
        self.div_count = 0

        print("Warming up camera...")
        for _ in range(30):
            self.pipeline.wait_for_frames(TIMEOUT_MS)
        print("Warmup complete.")

        self.ref_depth = None

    def reset_depth(self):
        self.ref_depth = None
        self.depth_count = 0
        self.div_count = 0

    def get_frame(self):
        try:
            frames = self.pipeline.wait_for_frames(1000)
        except RuntimeError:
            return None

        frames = self.align.process(frames)
        cf, df = frames.get_color_frame(), frames.get_depth_frame()
        if not cf or not df: return None

        color = np.asanyarray(cf.get_data()).copy()
        depth_m = np.asanyarray(df.get_data()).astype(np.float32) * self.depth_scale

        del cf, df, frames

        self.depth_stack[self.depth_count % DEPTH_STACK_N] = depth_m
        self.depth_count += 1
        if self.depth_count < DEPTH_STACK_N: return color.copy()

        depth_med = np.median(self.depth_stack, axis=0).astype(np.float32)
        depth_s = cv2.GaussianBlur(depth_med, (SMOOTH_K, SMOOTH_K), 0) if SMOOTH_K >= 3 else depth_med

        # --- Automatic Center Sampling for Reference Depth ---
        if self.ref_depth is None:
            cy, cx = H // 2, W // 2
            half_box = 20  # Grabs a 40x40 pixel square right in the middle

            center_depths = depth_med[cy - half_box: cy + half_box, cx - half_box: cx + half_box]
            valid_depths = center_depths[(center_depths > Z_MIN_M) & (center_depths < Z_MAX_M)]

            if valid_depths.size > 0:
                self.ref_depth = float(np.median(valid_depths))

        fg_u8 = np.zeros((H, W), dtype=np.uint8)
        if self.ref_depth is not None:
            band = (depth_s > Z_MIN_M) & (depth_s < Z_MAX_M) & (np.abs(depth_s - self.ref_depth) <= BAND_HALF_M)
            fg_u8[band] = 255
            fg_u8 = largest_component(
                cv2.morphologyEx(cv2.morphologyEx(fg_u8, cv2.MORPH_OPEN, self.k5, iterations=1), cv2.MORPH_CLOSE,
                                 self.k5, iterations=2), FG_MIN_AREA)

        fg_mask = fg_u8 > 0
        highlight = color.copy()

        # If it can't find the foreground mask, just return the raw image
        if not np.any(fg_mask):
            return highlight

        interior_u8 = make_interior_mask(fg_u8, INTERIOR_ERODE_PX)
        interior_mask = interior_u8 > 0
        if np.count_nonzero(interior_mask) < 1500:
            interior_u8, interior_mask = fg_u8.copy(), fg_mask

        coeffs = fit_quadratic_surface(depth_s, interior_mask)
        resid_local_s = cv2.GaussianBlur(
            (depth_s - quadratic_surface_img(coeffs, H, W)) - masked_local_trend(
                depth_s - quadratic_surface_img(coeffs, H, W), interior_mask, LOCAL_TREND_SIGMA),
            (RESIDUAL_BLUR_K, RESIDUAL_BLUR_K), 0)

        sigma = robust_sigma(resid_local_s[interior_mask])
        div_u8 = filter_components(cv2.morphologyEx(cv2.morphologyEx(((interior_mask & (
                resid_local_s >= np.clip(AUTO_SIGMA_MULT * sigma, MIN_DEFECT_MM / 1000.0,
                                         MAX_DEFECT_MM / 1000.0))).astype(np.uint8) * 255), cv2.MORPH_OPEN, self.k3,
                                                                     iterations=1), cv2.MORPH_CLOSE, self.kclose,
                                                    iterations=1), MIN_DEFECT_AREA_PX)

        self.div_history[self.div_count % PERSIST_N] = (cv2.dilate(div_u8, self.kvote, iterations=1) > 0).astype(
            np.uint8)
        self.div_count += 1

        div_mask = np.sum(self.div_history[:min(self.div_count, PERSIST_N)],
                          axis=0) >= PERSIST_MIN_HITS if self.div_count > 0 else div_u8 > 0
        div_final_u8 = filter_components_physical(
            filter_components((div_mask.astype(np.uint8) * 255), MIN_DEFECT_AREA_PX), depth_s, self.fx, self.fy,
            MIN_BBOX_W_IN, MIN_BBOX_H_IN)

        overlay = highlight.copy()
        if SHOW_DIVOTS: overlay[div_final_u8 > 0] = (0, 0, 255)
        highlight = cv2.addWeighted(overlay, 0.45, highlight, 0.55, 0)

        # Draw only the edge outlines
        draw_mask_outline(highlight, fg_u8, (0, 255, 255), 1)
        draw_mask_outline(highlight, interior_u8, (0, 255, 0), 2)

        return highlight

    def stop(self):
        try:
            self.pipeline.stop()
        except:
            pass