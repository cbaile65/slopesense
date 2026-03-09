import pyrealsense2 as rs
import numpy as np
import cv2

W, H, FPS = 640, 480, 30

# Residual visualization range (mm)
RESID_MIN_MM = -10.0
RESID_MAX_MM = +10.0

# Patch size for reference depth averaging (odd)
PATCH_R = 3  # radius -> (2*PATCH_R+1)^2 pixels, e.g. 7x7

cursor = [W//2, H//2]
roi = None  # (x, y, w, h)

# ---- ROI selection state ----
dragging = False
roi_start = (0, 0)
roi_end = (0, 0)

def clamp(v, lo, hi):
    return int(max(lo, min(hi, v)))

def mouse(event, x, y, flags, param):
    global dragging, roi_start, roi_end, cursor, roi

    # window is 2W wide (RGB | residual), map x into [0..W-1]
    if x >= W:
        x -= W

    x = clamp(x, 0, W-1)
    y = clamp(y, 0, H-1)

    cursor[0], cursor[1] = x, y

    # ROI draw with right mouse button drag
    if event == cv2.EVENT_RBUTTONDOWN:
        dragging = True
        roi_start = (x, y)
        roi_end = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        roi_end = (x, y)

    elif event == cv2.EVENT_RBUTTONUP:
        dragging = False
        x0, y0 = roi_start
        x1, y1 = roi_end
        x_min, x_max = sorted([x0, x1])
        y_min, y_max = sorted([y0, y1])
        w = max(1, x_max - x_min)
        h = max(1, y_max - y_min)
        roi = (x_min, y_min, w, h)

def patch_mean(depth_m, x, y, r):
    h, w = depth_m.shape
    x0 = max(0, x - r)
    x1 = min(w, x + r + 1)
    y0 = max(0, y - r)
    y1 = min(h, y + r + 1)
    patch = depth_m[y0:y1, x0:x1]
    # ignore zeros (invalid depth)
    valid = patch[patch > 0]
    if valid.size == 0:
        return 0.0
    return float(valid.mean())

# ---- RealSense setup ----
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
profile = pipeline.start(config)

depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
align = rs.align(rs.stream.color)

cv2.namedWindow("view")
cv2.setMouseCallback("view", mouse)

ref_depth = None  # meters

print("\nControls:")
print("  Right-click drag to draw ROI around phantom")
print("  c = confirm ROI (lock it)")
print("  x = clear ROI")
print("  SPACE = set reference depth (patch average at cursor)")
print("  r = reset reference depth")
print("  ESC = quit\n")

roi_locked = False

try:
    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        cf = frames.get_color_frame()
        df = frames.get_depth_frame()
        if not cf or not df:
            continue

        color = np.asanyarray(cf.get_data())
        depth = np.asanyarray(df.get_data()).astype(np.float32)
        depth_m = depth * depth_scale

        # Choose working region
        if roi is not None and roi_locked:
            rx, ry, rw, rh = roi
            color_roi = color[ry:ry+rh, rx:rx+rw].copy()
            depth_roi = depth_m[ry:ry+rh, rx:rx+rw].copy()
            cx = cursor[0] - rx
            cy = cursor[1] - ry
            # clamp cursor into ROI coords
            cx = clamp(cx, 0, rw-1)
            cy = clamp(cy, 0, rh-1)
        else:
            color_roi = color.copy()
            depth_roi = depth_m.copy()
            cx, cy = cursor

        # compute current depth at cursor (patch mean)
        d = patch_mean(depth_roi, cx, cy, PATCH_R)

        # residual in mm
        if ref_depth is None or ref_depth <= 0:
            resid_mm = np.zeros_like(depth_roi, dtype=np.float32)
        else:
            resid_mm = (depth_roi - ref_depth) * 1000.0

        # clamp residual to fixed range for visibility
        resid_mm_clamped = np.clip(resid_mm, RESID_MIN_MM, RESID_MAX_MM)

        # map fixed mm range -> 0..255
        vis = ((resid_mm_clamped - RESID_MIN_MM) / (RESID_MAX_MM - RESID_MIN_MM) * 255.0).astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)

        # draw cursor
        cv2.circle(color_roi, (cx, cy), 6, (0, 255, 255), -1)
        cv2.circle(vis, (cx, cy), 6, (0, 255, 255), -1)

        # draw ROI rectangle if not locked (or locked but want to show)
        show_color = color.copy()
        if roi is not None:
            rx, ry, rw, rh = roi
            cv2.rectangle(show_color, (rx, ry), (rx+rw, ry+rh), (0, 255, 255), 2)
            if dragging:
                x0, y0 = roi_start
                x1, y1 = roi_end
                cv2.rectangle(show_color, (x0, y0), (x1, y1), (0, 255, 255), 2)

        # overlay text
        if roi is None:
            msg_roi = "ROI: right-click drag, then press 'c' to lock"
        elif not roi_locked:
            msg_roi = "ROI drawn. Press 'c' to lock, or adjust with right-drag"
        else:
            msg_roi = f"ROI locked: {roi}"

        cv2.putText(show_color, f"Depth@cursor (patch): {d:.3f} m", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if ref_depth is not None:
            cv2.putText(show_color, f"Reference depth: {ref_depth:.3f} m", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(show_color, msg_roi, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        cv2.putText(vis, f"Residual mm (clamped): {RESID_MIN_MM:.0f}..{RESID_MAX_MM:.0f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        # display side-by-side (resize ROI view to W,H so window is consistent)
        left = show_color
        right = vis
        if roi_locked and roi is not None:
            # scale ROI views back to full display size for easy viewing
            right = cv2.resize(right, (W, H), interpolation=cv2.INTER_NEAREST)
            # keep left as full frame with ROI box
            left = show_color

        combined = np.hstack((left, right))
        cv2.imshow("view", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord('c'):
            roi_locked = True if roi is not None else False
        if key == ord('x'):
            roi = None
            roi_locked = False
        if key == ord('r'):
            ref_depth = None
        if key == 32:  # SPACE
            # set reference based on patch mean at cursor in the active region
            if d > 0:
                ref_depth = d
                print("Reference depth set to:", ref_depth)

finally:
    pipeline.stop()
    cv2.destroyAllWindows()