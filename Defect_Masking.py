import pyrealsense2 as rs
import numpy as np
import cv2, time, math

# ----------------- Settings -----------------
W, H, FPS = 640, 480, 30
TIMEOUT_MS = 10000
DEPTH_STACK_N = 18

# Safety depth gate
Z_MIN_M, Z_MAX_M = 0.15, 2.00

# Foreground segmentation around clicked basin depth
BAND_HALF_M = 0.02

# Smoothing of stacked depth
SMOOTH_K = 7

# IMU fusion
ALPHA = 0.98

# Basin mask settings
FG_MIN_AREA = 3000
INTERIOR_ERODE_PX = 31

# Defect cleanup
MIN_DEFECT_AREA_PX = 50
POST_CLOSE_K = 5

# Thresholding
AUTO_SIGMA_MULT = 1.9
MIN_DEFECT_MM = 0.6
MAX_DEFECT_MM = 3.0

# Local detrending
LOCAL_TREND_SIGMA = 31

# Residual smoothing
RESIDUAL_BLUR_K = 5

# Persistence filter
PERSIST_N = 20
PERSIST_MIN_HITS = 15
PERSIST_DILATE_K = 5

# Physical blob filtering
MM_PER_INCH = 25.4
MIN_BBOX_W_IN = 3.0
MIN_BBOX_H_IN = 3.0

# Visualization
SHOW_DIVOTS = True


# ----------------- Helpers -----------------
def accel_to_roll_pitch(ax, ay, az):
    roll = math.atan2(ay, az)
    pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az))
    return roll, pitch


def list_motion_profiles(dev):
    prof = {"gyro": [], "accel": []}
    for s in dev.query_sensors():
        for p in s.get_stream_profiles():
            sp = p.as_stream_profile()
            try:
                st, fmt, fps = sp.stream_type(), sp.format(), sp.fps()
            except Exception:
                continue
            if st == rs.stream.gyro:
                prof["gyro"].append((fps, fmt))
            if st == rs.stream.accel:
                prof["accel"].append((fps, fmt))
    for k in prof:
        prof[k] = sorted(list({x for x in prof[k]}), key=lambda t: t[0])
    return prof


def start_with_best_imu():
    ctx = rs.context()
    devs = ctx.query_devices()
    if len(devs) == 0:
        raise RuntimeError("No RealSense device found.")
    dev = devs[0]

    base = rs.config()
    base.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    base.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)

    profiles = list_motion_profiles(dev)
    gyros = sorted(profiles["gyro"], key=lambda x: -x[0])
    accs = sorted(profiles["accel"], key=lambda x: -x[0])

    pipeline = rs.pipeline()

    if not gyros or not accs:
        return pipeline, pipeline.start(base), False

    last_err = None
    for gf, gfmt in gyros:
        for af, afmt in accs:
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
            cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
            cfg.enable_stream(rs.stream.gyro, gfmt, gf)
            cfg.enable_stream(rs.stream.accel, afmt, af)
            try:
                return pipeline, pipeline.start(cfg), True
            except RuntimeError as e:
                last_err = e
                try:
                    pipeline.stop()
                except:
                    pass
                pipeline = rs.pipeline()

    print("IMU start failed -> video only. Last error:", last_err)
    pipeline = rs.pipeline()
    return pipeline, pipeline.start(base), False


def largest_component(mask_u8, min_area=500):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num <= 1:
        return np.zeros_like(mask_u8)
    best_i, best_area = 0, 0
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > best_area:
            best_area, best_i = area, i
    if best_area < min_area:
        return np.zeros_like(mask_u8)

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
    if vals.size < 20:
        return 0.0
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    return 1.4826 * mad


def make_interior_mask(fg_mask_u8, erode_px):
    if erode_px < 3:
        return fg_mask_u8.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px, erode_px))
    return cv2.erode(fg_mask_u8, k, iterations=1)


def draw_mask_outline(img, mask_u8, color, thickness=2):
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, thickness)


def fit_quadratic_surface(depth, mask):
    ys, xs = np.where(mask)
    if len(xs) < 1200:
        if np.any(mask):
            z0 = float(np.median(depth[mask]))
            return np.array([0, 0, 0, 0, 0, z0], dtype=np.float32)
        return np.zeros(6, dtype=np.float32)

    x = xs.astype(np.float32)
    y = ys.astype(np.float32)
    z = depth[ys, xs].astype(np.float32)

    A = np.stack([x * x, y * y, x * y, x, y, np.ones_like(x)], axis=1)
    coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)
    return coeffs.astype(np.float32)


def quadratic_surface_img(coeffs, h, w):
    a, b, c, d, e, f = coeffs
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    return a * X * X + b * Y * Y + c * X * Y + d * X + e * Y + f


def masked_local_trend(img, mask, sigma):
    mask_f = mask.astype(np.float32)
    num = cv2.GaussianBlur((img * mask_f).astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    den = cv2.GaussianBlur(mask_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return num / (den + 1e-6)


def filter_components_physical(mask_u8, depth_s, fx, fy, min_w_in, min_h_in):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    out = np.zeros_like(mask_u8)

    min_w_mm = min_w_in * MM_PER_INCH
    min_h_mm = min_h_in * MM_PER_INCH

    for i in range(1, num):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area <= 0:
            continue

        slice_labels = labels[y:y + h, x:x + w]
        slice_depth = depth_s[y:y + h, x:x + w]

        comp_mask = (slice_labels == i)
        z_vals = slice_depth[comp_mask]
        z_vals = z_vals[np.isfinite(z_vals)]
        z_vals = z_vals[z_vals > 0]

        if z_vals.size == 0:
            continue

        z_med_m = float(np.median(z_vals))
        z_med_mm = z_med_m * 1000.0

        width_mm = (w * z_med_mm) / fx
        height_mm = (h * z_med_mm) / fy

        if width_mm >= min_w_mm and height_mm >= min_h_mm:
            out[y:y + h, x:x + w][comp_mask] = 255

    return out


# ----------------- Main Class for GUI Integration -----------------
class DefectDetector:
    def __init__(self):
        self.pipeline, self.profile, self.imu_enabled = start_with_best_imu()

        dev = self.profile.get_device()
        self.depth_scale = dev.first_depth_sensor().get_depth_scale()

        color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        self.fx, self.fy = float(intr.fx), float(intr.fy)

        try:
            print("Camera:", dev.get_info(rs.camera_info.name))
            print("USB:", dev.get_info(rs.camera_info.usb_type_descriptor))
            print("FW:", dev.get_info(rs.camera_info.firmware_version))
        except:
            pass
        print("Depth scale:", self.depth_scale, "m/unit | IMU:", self.imu_enabled)
        print(f"Intrinsics fx={self.fx:.2f}, fy={self.fy:.2f}")

        self.align = rs.align(rs.stream.color)
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole = rs.hole_filling_filter()

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

        self.click = {"x": W // 2, "y": H // 2}
        self.ref_depth = None

        self.roll = self.pitch = self.yaw = 0.0
        self.last_t = time.time()
        self.last_acc = None

    def register_click(self, x, y):
        self.click["x"] = int(np.clip(x, 0, W - 1))
        self.click["y"] = int(np.clip(y, 0, H - 1))
        self.reset_depth()

    def reset_depth(self):
        self.ref_depth = None
        self.depth_count = 0
        self.div_count = 0

    def get_frame(self):
        try:
            # Safe 1000ms block since Tkinter runs on the main thread
            frames = self.pipeline.wait_for_frames(1000)
        except RuntimeError:
            return None

        if self.imu_enabled:
            now = time.time()
            dt = max(1e-4, now - self.last_t)
            self.last_t = now

            gf = frames.first_or_default(rs.stream.gyro)
            af = frames.first_or_default(rs.stream.accel)

            if af:
                a = af.as_motion_frame().get_motion_data()
                self.last_acc = (a.x, a.y, a.z)

            if gf:
                g = gf.as_motion_frame().get_motion_data()
                self.roll += g.x * dt
                self.pitch += g.y * dt
                self.yaw += g.z * dt

            if self.last_acc is not None:
                ax, ay, az = self.last_acc
                ar, ap = accel_to_roll_pitch(ax, ay, az)
                self.roll = ALPHA * self.roll + (1 - ALPHA) * ar
                self.pitch = ALPHA * self.pitch + (1 - ALPHA) * ap

        frames = self.align.process(frames)
        cf = frames.get_color_frame()
        df = frames.get_depth_frame()
        if not cf or not df:
            return None

        # Detach array references from pyrealsense internal memory explicitly
        color = np.asanyarray(cf.get_data()).copy()
        depth_m = np.asanyarray(df.get_data()).astype(np.float32) * self.depth_scale

        # IMPORTANT: Explicitly delete C++ buffer references to prevent GC locking
        del cf
        del df
        del frames

        self.depth_stack[self.depth_count % DEPTH_STACK_N] = depth_m
        self.depth_count += 1

        if self.depth_count < DEPTH_STACK_N:
            return color.copy()

        depth_med = np.median(self.depth_stack, axis=0).astype(np.float32)

        if SMOOTH_K >= 3 and SMOOTH_K % 2 == 1:
            depth_s = cv2.GaussianBlur(depth_med, (SMOOTH_K, SMOOTH_K), 0)
        else:
            depth_s = depth_med

        x, y = self.click["x"], self.click["y"]
        click_depth = float(depth_med[y, x])

        if self.ref_depth is None and Z_MIN_M < click_depth < Z_MAX_M:
            self.ref_depth = click_depth

        valid = (depth_s > Z_MIN_M) & (depth_s < Z_MAX_M)
        fg_u8 = np.zeros((H, W), dtype=np.uint8)

        if self.ref_depth is not None:
            band = valid & (np.abs(depth_s - self.ref_depth) <= BAND_HALF_M)
            fg_u8[band] = 255

            fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_OPEN, self.k5, iterations=1)
            fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_CLOSE, self.k5, iterations=2)
            fg_u8 = largest_component(fg_u8, min_area=FG_MIN_AREA)

        fg_mask = fg_u8 > 0
        highlight = color.copy()

        if not np.any(fg_mask):
            cv2.circle(highlight, (x, y), 7, (0, 255, 255), -1)
            return highlight

        interior_u8 = make_interior_mask(fg_u8, INTERIOR_ERODE_PX)
        interior_mask = interior_u8 > 0

        if np.count_nonzero(interior_mask) < 1500:
            interior_u8 = fg_u8.copy()
            interior_mask = fg_mask

        coeffs = fit_quadratic_surface(depth_s, interior_mask)
        surf = quadratic_surface_img(coeffs, H, W)

        resid_global = depth_s - surf
        local_trend = masked_local_trend(resid_global, interior_mask, LOCAL_TREND_SIGMA)
        resid_local = resid_global - local_trend

        resid_local_s = cv2.GaussianBlur(
            resid_local.astype(np.float32),
            (RESIDUAL_BLUR_K, RESIDUAL_BLUR_K),
            0
        )

        vals = resid_local_s[interior_mask]
        sigma = robust_sigma(vals)

        auto_thr_m = AUTO_SIGMA_MULT * sigma
        auto_thr_m = np.clip(auto_thr_m, MIN_DEFECT_MM / 1000.0, MAX_DEFECT_MM / 1000.0)

        div_mask_now = interior_mask & (resid_local_s >= auto_thr_m)
        div_u8 = (div_mask_now.astype(np.uint8) * 255)

        div_u8 = cv2.morphologyEx(div_u8, cv2.MORPH_OPEN, self.k3, iterations=1)
        div_u8 = cv2.morphologyEx(div_u8, cv2.MORPH_CLOSE, self.kclose, iterations=1)
        div_u8 = filter_components(div_u8, MIN_DEFECT_AREA_PX)

        div_vote_u8 = cv2.dilate(div_u8, self.kvote, iterations=1)

        self.div_history[self.div_count % PERSIST_N] = (div_vote_u8 > 0).astype(np.uint8)
        self.div_count += 1

        if self.div_count > 0:
            valid_history = min(self.div_count, PERSIST_N)
            div_votes = np.sum(self.div_history[:valid_history], axis=0)
            div_mask = div_votes >= PERSIST_MIN_HITS
        else:
            div_mask = div_u8 > 0

        div_final_u8 = filter_components((div_mask.astype(np.uint8) * 255), MIN_DEFECT_AREA_PX)

        div_final_u8 = filter_components_physical(
            div_final_u8,
            depth_s,
            self.fx,
            self.fy,
            MIN_BBOX_W_IN,
            MIN_BBOX_H_IN
        )

        div_mask = div_final_u8 > 0

        overlay = highlight.copy()
        if SHOW_DIVOTS:
            overlay[div_mask] = (0, 0, 255)

        highlight = cv2.addWeighted(overlay, 0.45, highlight, 0.55, 0)

        draw_mask_outline(highlight, fg_u8, (0, 255, 255), 1)
        draw_mask_outline(highlight, interior_u8, (0, 255, 0), 2)

        cv2.circle(highlight, (x, y), 6, (0, 255, 255), -1)

        return highlight

    def stop(self):
        try:
            self.pipeline.stop()
        except:
            pass