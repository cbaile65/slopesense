import pyrealsense2 as rs
import numpy as np
import cv2, time, math
from collections import deque

# ----------------- Settings -----------------
W, H, FPS = 640, 480, 30
TIMEOUT_MS = 10000
DEPTH_STACK_N = 10

# Detection (meters)
DIVOT_MIN_M, DIVOT_MAX_M = 0.002, 0.008   # start wide; tighten to 0.005 later
MIN_BLOB_AREA_PX = 120                    # bump up if speckle persists

# Foreground segmentation (depth band around clicked surface)
BAND_HALF_M = 0.06                        # +/- 6 cm around clicked depth

# Depth gate (safety)
Z_MIN_M, Z_MAX_M = 0.15, 1.20

# Smoothing
SMOOTH_K = 7  # odd

# IMU fusion
ALPHA = 0.98

# ----------------- Helpers -----------------
def accel_to_roll_pitch(ax, ay, az):
    roll = math.atan2(ay, az)
    pitch = math.atan2(-ax, math.sqrt(ay*ay + az*az))
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
            if st == rs.stream.gyro:  prof["gyro"].append((fps, fmt))
            if st == rs.stream.accel: prof["accel"].append((fps, fmt))
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
    accs  = sorted(profiles["accel"], key=lambda x: -x[0])

    pipeline = rs.pipeline()

    # If no IMU, just start video
    if not gyros or not accs:
        return pipeline, pipeline.start(base), False

    # Try combinations (prefer high fps)
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
                try: pipeline.stop()
                except: pass
                pipeline = rs.pipeline()

    print("IMU start failed -> video only. Last error:", last_err)
    pipeline = rs.pipeline()
    return pipeline, pipeline.start(base), False

def fit_plane(depth, mask):
    ys, xs = np.where(mask)
    if len(xs) < 800:
        return 0.0, 0.0, float(np.median(depth[mask])) if np.any(mask) else 0.0
    z = depth[ys, xs].astype(np.float32)
    A = np.stack([xs.astype(np.float32), ys.astype(np.float32), np.ones_like(z)], axis=1)
    (a, b, c), *_ = np.linalg.lstsq(A, z, rcond=None)
    return float(a), float(b), float(c)

def plane_img(a, b, c, h, w):
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    return a*X + b*Y + c

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
    out[labels == best_i] = 255
    return out

def filter_components(mask_u8, min_area):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    out = np.zeros_like(mask_u8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out

# ----------------- Main -----------------
def main():
    pipeline, profile, imu_enabled = start_with_best_imu()

    dev = profile.get_device()
    depth_scale = dev.first_depth_sensor().get_depth_scale()
    try:
        print("Camera:", dev.get_info(rs.camera_info.name))
        print("USB:", dev.get_info(rs.camera_info.usb_type_descriptor))
        print("FW:", dev.get_info(rs.camera_info.firmware_version))
    except: pass
    print("Depth scale:", depth_scale, "m/unit | IMU:", imu_enabled)

    align = rs.align(rs.stream.color)
    spatial, temporal, hole = rs.spatial_filter(), rs.temporal_filter(), rs.hole_filling_filter()

    # Warmup
    print("Warming up...")
    for _ in range(30):
        pipeline.wait_for_frames(TIMEOUT_MS)
    print("Warmup complete.")

    depth_stack = deque(maxlen=DEPTH_STACK_N)

    # click state
    click = {"x": W//2, "y": H//2}
    ref_depth = None

    def on_mouse(e, x, y, *_):
        nonlocal ref_depth
        if e == cv2.EVENT_LBUTTONDOWN:
            if x >= W: x -= W
            click["x"], click["y"] = int(np.clip(x, 0, W-1)), int(np.clip(y, 0, H-1))
            ref_depth = None  # will be set from stabilized depth

    win = "RealSense (RGB | Defect Highlight)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)

    # IMU state
    roll = pitch = yaw = 0.0
    last_t = time.time()
    last_acc = None

    try:
        while True:
            frames = pipeline.wait_for_frames(TIMEOUT_MS)

            # IMU update
            if imu_enabled:
                now = time.time()
                dt = max(1e-4, now - last_t)
                last_t = now

                gf = frames.first_or_default(rs.stream.gyro)
                af = frames.first_or_default(rs.stream.accel)

                if af:
                    a = af.as_motion_frame().get_motion_data()
                    last_acc = (a.x, a.y, a.z)

                if gf:
                    g = gf.as_motion_frame().get_motion_data()
                    roll += g.x * dt
                    pitch += g.y * dt
                    yaw += g.z * dt

                if last_acc is not None:
                    ax, ay, az = last_acc
                    ar, ap = accel_to_roll_pitch(ax, ay, az)
                    roll = ALPHA*roll + (1-ALPHA)*ar
                    pitch = ALPHA*pitch + (1-ALPHA)*ap

            # Video + depth
            frames = align.process(frames)
            cf = frames.get_color_frame()
            df = frames.get_depth_frame()
            if not cf or not df:
                continue

            df = hole.process(temporal.process(spatial.process(df))).as_depth_frame()

            color = np.asanyarray(cf.get_data())
            depth_m = np.asanyarray(df.get_data()).astype(np.float32) * depth_scale

            depth_stack.append(depth_m)
            if len(depth_stack) < DEPTH_STACK_N:
                right = color.copy()
                cv2.putText(right, f"Building depth stack... ({len(depth_stack)}/{DEPTH_STACK_N})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
                cv2.imshow(win, np.hstack((color, right)))
                k = cv2.waitKey(1) & 0xFF
                if k in (ord('q'), 27): break
                continue

            depth_med = np.median(np.stack(depth_stack, 0), 0).astype(np.float32)
            if SMOOTH_K >= 3 and SMOOTH_K % 2 == 1:
                depth_s = cv2.GaussianBlur(depth_med, (SMOOTH_K, SMOOTH_K), 0)
            else:
                depth_s = depth_med

            x, y = click["x"], click["y"]
            click_depth = float(depth_med[y, x])

            # Set reference depth from stabilized click point
            if ref_depth is None and Z_MIN_M < click_depth < Z_MAX_M:
                ref_depth = click_depth

            # Foreground mask = pixels near clicked depth (band) + within global gate
            valid = (depth_s > Z_MIN_M) & (depth_s < Z_MAX_M)
            fg = np.zeros((H, W), dtype=np.uint8)
            if ref_depth is not None:
                band = valid & (np.abs(depth_s - ref_depth) <= BAND_HALF_M)
                fg[band] = 255

                # Keep only largest connected region (the part)
                fg = largest_component(fg, min_area=2000)

            fg_mask = fg > 0

            # If we don't have a good fg yet, just show guidance
            highlight = color.copy()
            if not np.any(fg_mask):
                cv2.putText(highlight, "Click on the PART surface to lock foreground mask",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
                cv2.circle(highlight, (x, y), 7, (0,255,255), -1)
                cv2.imshow(win, np.hstack((color, highlight)))
                k = cv2.waitKey(1) & 0xFF
                if k in (ord('q'), 27): break
                if k == ord('r'):
                    depth_stack.clear()
                    ref_depth = None
                continue

            # Plane fit ONLY on the segmented part
            a, b, c = fit_plane(depth_s, fg_mask)
            plane = plane_img(a, b, c, H, W)
            resid = depth_s - plane  # deeper-than-plane => positive

            # Divot mask on residual + foreground
            div = fg_mask & (resid >= DIVOT_MIN_M) & (resid <= DIVOT_MAX_M)
            div_u8 = (div.astype(np.uint8) * 255)

            # Morph cleanup + blob filter
            k3 = np.ones((3,3), np.uint8)
            div_u8 = cv2.morphologyEx(div_u8, cv2.MORPH_OPEN, k3, iterations=1)
            div_u8 = cv2.dilate(div_u8, k3, iterations=1)
            div_u8 = filter_components(div_u8, MIN_BLOB_AREA_PX)
            div_mask = div_u8 > 0

            # Overlay
            overlay = highlight.copy()
            overlay[div_mask] = (0,0,255)
            highlight = cv2.addWeighted(overlay, 0.45, highlight, 0.55, 0)

            # Draw click + text
            cv2.circle(color, (x, y), 6, (0,255,255), -1)
            cv2.circle(highlight, (x, y), 6, (0,255,255), -1)

            cv2.putText(color, f"Click depth (median {DEPTH_STACK_N}): {click_depth:.3f} m",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2, cv2.LINE_AA)

            if imu_enabled:
                rd, pd, yd = roll*180/math.pi, pitch*180/math.pi, yaw*180/math.pi
                cv2.putText(color, f"Roll:{rd:+.1f}  Pitch:{pd:+.1f}  Yaw:{yd:+.1f} (yaw drifts)",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0,255,255), 2, cv2.LINE_AA)

            cv2.putText(highlight,
                        f"Divots(resid): {int(DIVOT_MIN_M*1000)}–{int(DIVOT_MAX_M*1000)} mm | band=±{int(BAND_HALF_M*100)}cm | blob>={MIN_BLOB_AREA_PX}px",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0,255,255), 2, cv2.LINE_AA)

            cv2.imshow(win, np.hstack((color, highlight)))

            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):
                break
            if k == ord('r'):
                depth_stack.clear()
                ref_depth = None

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
