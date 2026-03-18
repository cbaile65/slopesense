import requests
import time
import cv2
import numpy as np
import pyrealsense2 as rs

# =========================================================
# PI CONNECTION / MOTOR PINS
# =========================================================
PI_URL = "http://192.168.5.2:5000"

MOTOR_A_UP_PIN = 24
MOTOR_A_DOWN_PIN = 23
MOTOR_B_FORWARD_PIN = 17
MOTOR_B_BACKWARD_PIN = 27

# =========================================================
# SETTINGS (EDIT THESE FIRST)
# =========================================================
TARGET_DISTANCE_M = 1.50
DISTANCE_TOL_M = 0.03

TARGET_Y_FRAC = 0.50
Y_TOL_PX = 15
Y_LOCK_FRAMES = 3

STARTUP_WAIT_S = 1.0
DRAIN_LOCK_FRAMES = 8
DRAIN_LOCK_PX = 20

# =========================================================
# CAMERA
# =========================================================
W = 640
H = 480
FPS = 30

# =========================================================
# DRAIN DETECTION SETTINGS
# =========================================================
GLOBAL_MIN_RADIUS = 10
GLOBAL_MAX_RADIUS = 35
LOCAL_MIN_RADIUS = 10
LOCAL_MAX_RADIUS = 35
LOCAL_SEARCH_BOX = 70

DIST_ROI_HALF = 22
LOOP_DELAY = 0.03

# =========================================================
# HTTP MOTOR CONTROL
# =========================================================
def set_pin(pin, state):
    try:
        requests.get(f"{PI_URL}/pin/{pin}/{state.lower()}", timeout=2)
        return True
    except:
        return False

def stop_pair(pin_a, pin_b):
    set_pin(pin_a, "off")
    set_pin(pin_b, "off")

def stop_all():
    stop_pair(MOTOR_A_UP_PIN, MOTOR_A_DOWN_PIN)
    stop_pair(MOTOR_B_FORWARD_PIN, MOTOR_B_BACKWARD_PIN)

def hold_direction(on_pin, off_pin):
    set_pin(on_pin, "on")
    set_pin(off_pin, "off")

def hold_height_up():
    hold_direction(MOTOR_A_UP_PIN, MOTOR_A_DOWN_PIN)

def hold_height_down():
    hold_direction(MOTOR_A_DOWN_PIN, MOTOR_A_UP_PIN)

def stop_height():
    stop_pair(MOTOR_A_UP_PIN, MOTOR_A_DOWN_PIN)

def hold_motor_b_forward():
    hold_direction(MOTOR_B_FORWARD_PIN, MOTOR_B_BACKWARD_PIN)

def hold_motor_b_backward():
    hold_direction(MOTOR_B_BACKWARD_PIN, MOTOR_B_FORWARD_PIN)

def stop_motor_b():
    stop_pair(MOTOR_B_FORWARD_PIN, MOTOR_B_BACKWARD_PIN)

# =========================================================
# CAMERA SETUP
# =========================================================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)

profile = pipeline.start(config)
align = rs.align(rs.stream.color)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

# =========================================================
# HELPERS
# =========================================================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def median_valid_depth(depth_img, x1, y1, x2, y2):
    roi = depth_img[y1:y2, x1:x2]
    valid = roi[roi > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid) * depth_scale)

def get_basin_distance(depth_frame):
    depth_img = np.asanyarray(depth_frame.get_data())
    h, w = depth_img.shape

    cx = w // 2
    cy = int(h * 0.68)

    x1 = clamp(cx - DIST_ROI_HALF, 0, w - 1)
    x2 = clamp(cx + DIST_ROI_HALF, 0, w - 1)
    y1 = clamp(cy - DIST_ROI_HALF, 0, h - 1)
    y2 = clamp(cy + DIST_ROI_HALF, 0, h - 1)

    return median_valid_depth(depth_img, x1, y1, x2, y2), (x1, y1, x2, y2)

# =========================================================
# DRAIN DETECTION
# Idea:
# - drain should be dark
# - drain should be surrounded by bright/white tub material
# - reject random dark spots on floor / shadows
# =========================================================
def circle_masks(shape, x, y, r):
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)

    center_mask = dist <= r * 0.45
    ring_mask = (dist >= r * 0.70) & (dist <= r * 1.15)
    outer_mask = (dist >= r * 1.35) & (dist <= r * 2.20)

    return center_mask, ring_mask, outer_mask

def score_circle(gray, x, y, r):
    h, w = gray.shape

    if x - 3 * r < 0 or x + 3 * r >= w or y - 3 * r < 0 or y + 3 * r >= h:
        return -999999

    center_mask, ring_mask, outer_mask = circle_masks(gray.shape, x, y, r)

    center_vals = gray[center_mask]
    ring_vals = gray[ring_mask]
    outer_vals = gray[outer_mask]

    if center_vals.size == 0 or ring_vals.size == 0 or outer_vals.size == 0:
        return -999999

    center_mean = float(np.mean(center_vals))
    ring_mean = float(np.mean(ring_vals))
    outer_mean = float(np.mean(outer_vals))

    # Drain center should be dark
    center_dark_score = 255.0 - center_mean

    # Drain ring is often dark too
    ring_dark_score = max(0.0, 200.0 - ring_mean)

    # Outer area should be bright/white tub material
    outer_bright_score = outer_mean

    # Strong bonus if outer area is clearly brighter than center
    contrast_score = outer_mean - center_mean

    # Slight bonus for realistic drain size
    size_bonus = 0.0
    if 12 <= r <= 28:
        size_bonus = 25.0

    # Prefer points inside the tub area and not too low in frame
    location_bonus = 0.0
    if y < int(H * 0.82):
        location_bonus += 20.0
    if x > int(W * 0.35):
        location_bonus += 10.0

    # Hard reject if surrounding area is not bright enough
    if outer_mean < 120:
        return -999999

    # Hard reject if center is not dark enough
    if center_mean > 110:
        return -999999

    score = (
        center_dark_score * 2.2 +
        ring_dark_score * 0.8 +
        outer_bright_score * 1.0 +
        contrast_score * 2.0 +
        size_bonus +
        location_bonus
    )

    return score

def auto_find_drain(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    # Search only upper tub area, not bottom floor area
    y1 = 0
    y2 = int(H * 0.82)
    x1 = 0
    x2 = W

    roi = gray[y1:y2, x1:x2]

    circles = cv2.HoughCircles(
        roi,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=100,
        param2=16,
        minRadius=GLOBAL_MIN_RADIUS,
        maxRadius=GLOBAL_MAX_RADIUS
    )

    if circles is None:
        return None

    circles = np.round(circles[0]).astype(int)

    best = None
    best_score = -999999

    for cx, cy, r in circles:
        x = cx + x1
        y = cy + y1

        s = score_circle(gray, x, y, r)
        if s > best_score:
            best_score = s
            best = (x, y, r)

    return best

def find_drain_local(img, last_x, last_y):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    box = LOCAL_SEARCH_BOX
    x1 = max(0, last_x - box)
    x2 = min(W, last_x + box)
    y1 = max(0, last_y - box)
    y2 = min(H, last_y + box)

    roi = gray[y1:y2, x1:x2]

    if roi.shape[0] < 20 or roi.shape[1] < 20:
        return None

    circles = cv2.HoughCircles(
        roi,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=100,
        param2=14,
        minRadius=LOCAL_MIN_RADIUS,
        maxRadius=LOCAL_MAX_RADIUS
    )

    if circles is None:
        return None

    circles = np.round(circles[0]).astype(int)

    best = None
    best_score = -999999

    for cx, cy, r in circles:
        x = cx + x1
        y = cy + y1

        s = score_circle(gray, x, y, r)

        # Prefer candidates near previous drain location
        dist_penalty = np.sqrt((x - last_x) ** 2 + (y - last_y) ** 2) * 2.0
        s = s - dist_penalty

        if s > best_score:
            best_score = s
            best = (x, y, r)

    return best

# =========================================================
# STATE
# =========================================================
tracked_x = None
tracked_y = None
tracked_r = None

candidate_x = None
candidate_y = None
candidate_r = None
candidate_count = 0

stage = 0
startup_time = time.time()
y_locked_frames = 0

# =========================================================
# MAIN
# =========================================================
window_name = "centering"
cv2.namedWindow(window_name)

print("Starting...")
print("Stage 0 = wait")
print("Stage 1 = search and lock drain")
print("Stage 2 = align with drain line")
print("Stage 3 = set height")
print("Stage 4 = hold")
print("Press q to quit, r to reset")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_img = np.asanyarray(color_frame.get_data())
        display_img = color_img.copy()

        h, w = color_img.shape[:2]
        target_y = int(h * TARGET_Y_FRAC)
        basin_distance_m, dist_roi = get_basin_distance(depth_frame)

        status = ""

        # -------------------------------------------------
        # STAGE 0: wait
        # -------------------------------------------------
        if stage == 0:
            stop_all()
            status = "waiting before search"

            if time.time() - startup_time >= STARTUP_WAIT_S:
                stage = 1

        # -------------------------------------------------
        # STAGE 1: global search and lock
        # -------------------------------------------------
        elif stage == 1:
            stop_all()

            drain = auto_find_drain(color_img)

            if drain is None:
                candidate_x = None
                candidate_y = None
                candidate_r = None
                candidate_count = 0
                status = "searching for drain"
            else:
                x, y, r = drain

                if candidate_x is None:
                    candidate_x = x
                    candidate_y = y
                    candidate_r = r
                    candidate_count = 1
                else:
                    dist = np.sqrt((x - candidate_x) ** 2 + (y - candidate_y) ** 2)

                    if dist <= DRAIN_LOCK_PX:
                        candidate_x = int((candidate_x + x) / 2)
                        candidate_y = int((candidate_y + y) / 2)
                        candidate_r = int((candidate_r + r) / 2)
                        candidate_count += 1
                    else:
                        candidate_x = x
                        candidate_y = y
                        candidate_r = r
                        candidate_count = 1

                status = f"locking drain {candidate_count}/{DRAIN_LOCK_FRAMES}"

                if candidate_count >= DRAIN_LOCK_FRAMES:
                    tracked_x = candidate_x
                    tracked_y = candidate_y
                    tracked_r = candidate_r
                    y_locked_frames = 0
                    stage = 2
                    status = "drain locked"

        # -------------------------------------------------
        # STAGE 2: align with drain line
        # -------------------------------------------------
        elif stage == 2:
            stop_height()

            drain = None
            if tracked_x is not None and tracked_y is not None:
                drain = find_drain_local(color_img, tracked_x, tracked_y)

            if drain is not None:
                tracked_x, tracked_y, tracked_r = drain
                y_error = tracked_y - target_y

                if abs(y_error) <= Y_TOL_PX:
                    y_locked_frames += 1
                    stop_motor_b()
                    status = f"in line {y_locked_frames}/{Y_LOCK_FRAMES}"
                else:
                    y_locked_frames = 0

                    # Swap these if your direction is backwards
                    if y_error < 0:
                        hold_motor_b_forward()
                        status = "aligning forward"
                    else:
                        hold_motor_b_backward()
                        status = "aligning backward"

                if y_locked_frames >= Y_LOCK_FRAMES:
                    stop_motor_b()
                    stage = 3
                    status = "drain line aligned"
            else:
                stop_motor_b()
                status = "local drain track lost"

        # -------------------------------------------------
        # STAGE 3: set height
        # -------------------------------------------------
        elif stage == 3:
            stop_motor_b()

            drain = None
            if tracked_x is not None and tracked_y is not None:
                drain = find_drain_local(color_img, tracked_x, tracked_y)

            if drain is not None:
                tracked_x, tracked_y, tracked_r = drain

            if basin_distance_m is None:
                stop_height()
                status = "no basin depth"
            else:
                height_error = TARGET_DISTANCE_M - basin_distance_m

                if abs(height_error) <= DISTANCE_TOL_M:
                    stop_height()
                    stage = 4
                    status = "height locked"
                else:
                    if height_error > 0:
                        hold_height_down()
                        status = "moving down"
                    else:
                        hold_height_up()
                        status = "moving up"

        # -------------------------------------------------
        # STAGE 4: hold
        # -------------------------------------------------
        else:
            stop_all()

            drain = None
            if tracked_x is not None and tracked_y is not None:
                drain = find_drain_local(color_img, tracked_x, tracked_y)

            if drain is not None:
                tracked_x, tracked_y, tracked_r = drain

            status = "done"

        # -------------------------------------------------
        # DRAW
        # -------------------------------------------------
        cv2.line(display_img, (0, target_y), (w, target_y), (0, 255, 255), 2)
        cv2.rectangle(display_img,
                      (0, target_y - Y_TOL_PX),
                      (w, target_y + Y_TOL_PX),
                      (0, 200, 255), 1)

        if candidate_x is not None and stage == 1:
            cv2.circle(display_img, (candidate_x, candidate_y), 7, (0, 0, 255), -1)
            cv2.circle(display_img, (candidate_x, candidate_y), max(12, candidate_r), (0, 0, 255), 2)

        if tracked_x is not None and tracked_y is not None:
            cv2.circle(display_img, (tracked_x, tracked_y), 7, (0, 255, 0), -1)
            if tracked_r is not None:
                cv2.circle(display_img, (tracked_x, tracked_y), tracked_r, (0, 255, 0), 2)

        dx1, dy1, dx2, dy2 = dist_roi
        cv2.rectangle(display_img, (dx1, dy1), (dx2, dy2), (255, 255, 0), 2)

        cv2.putText(display_img, f"Stage: {stage}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display_img, status, (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if tracked_x is not None and tracked_y is not None:
            cv2.putText(display_img, f"Tracked x,y: {tracked_x}, {tracked_y}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if basin_distance_m is not None:
            cv2.putText(display_img, f"Basin distance: {basin_distance_m:.3f} m", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_img, f"Height error: {TARGET_DISTANCE_M - basin_distance_m:.3f} m", (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(window_name, display_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            stop_all()
            tracked_x = None
            tracked_y = None
            tracked_r = None
            candidate_x = None
            candidate_y = None
            candidate_r = None
            candidate_count = 0
            stage = 0
            startup_time = time.time()
            y_locked_frames = 0
            print("[user] Reset.")

        time.sleep(LOOP_DELAY)

except KeyboardInterrupt:
    print("\nInterrupted by user.")

finally:
    stop_all()
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Stopped safely.")