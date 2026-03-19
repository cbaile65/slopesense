import requests
import time
import cv2
import numpy as np
import threading

# =========================================================
# PI CONNECTION / MOTOR PINS
# =========================================================
PI_URL = "http://192.168.5.2:5000"

# Swapped these pins so UP physically moves UP and DOWN physically moves DOWN
MOTOR_A_UP_PIN = 23
MOTOR_A_DOWN_PIN = 24
MOTOR_B_FORWARD_PIN = 17
MOTOR_B_BACKWARD_PIN = 27

# =========================================================
# SETTINGS
# =========================================================
TARGET_DISTANCE_M = 1.25
DISTANCE_TOL_M = 0.10

TARGET_Y_FRAC = 0.50
Y_TOL_PX = 15
Y_LOCK_FRAMES = 3

STARTUP_WAIT_S = 1.0
DRAIN_LOCK_FRAMES = 8
DRAIN_LOCK_PX = 20

W = 640
H = 480

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
pin_states = {}


def set_pin(pin, state):
    if pin_states.get(pin) == state:
        return True

    try:
        requests.get(f"{PI_URL}/pin/{pin}/{state.lower()}", timeout=0.5)
        pin_states[pin] = state
        return True
    except Exception:
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
# HELPERS
# =========================================================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def median_valid_depth(depth_img, x1, y1, x2, y2):
    roi = depth_img[y1:y2, x1:x2]
    valid = roi[roi > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def get_basin_distance(depth_img):
    h, w = depth_img.shape
    cx = w // 2
    cy = int(h * 0.68)
    x1 = clamp(cx - DIST_ROI_HALF, 0, w - 1)
    x2 = clamp(cx + DIST_ROI_HALF, 0, w - 1)
    y1 = clamp(cy - DIST_ROI_HALF, 0, h - 1)
    y2 = clamp(cy + DIST_ROI_HALF, 0, h - 1)
    return median_valid_depth(depth_img, x1, y1, x2, y2), (x1, y1, x2, y2)


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

    center_dark_score = 255.0 - center_mean
    ring_dark_score = max(0.0, 200.0 - ring_mean)
    outer_bright_score = outer_mean
    contrast_score = outer_mean - center_mean

    size_bonus = 25.0 if 12 <= r <= 28 else 0.0
    location_bonus = 0.0
    if y < int(H * 0.82):
        location_bonus += 20.0
    if x > int(W * 0.35):
        location_bonus += 10.0

    if outer_mean < 120 or center_mean > 110:
        return -999999

    return (
        center_dark_score * 2.2
        + ring_dark_score * 0.8
        + outer_bright_score * 1.0
        + contrast_score * 2.0
        + size_bonus
        + location_bonus
    )


def auto_find_drain(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    y1, y2, x1, x2 = 0, int(H * 0.82), 0, W
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

    best, best_score = None, -999999
    for cx, cy, r in np.round(circles[0]).astype(int):
        x, y = cx + x1, cy + y1
        s = score_circle(gray, x, y, r)
        if s > best_score:
            best_score, best = s, (x, y, r)

    return best


def find_drain_local(img, last_x, last_y):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    box = LOCAL_SEARCH_BOX
    x1, x2 = max(0, last_x - box), min(W, last_x + box)
    y1, y2 = max(0, last_y - box), min(H, last_y + box)
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

    best, best_score = None, -999999
    for cx, cy, r in np.round(circles[0]).astype(int):
        x, y = cx + x1, cy + y1
        s = score_circle(gray, x, y, r) - (np.sqrt((x - last_x) ** 2 + (y - last_y) ** 2) * 2.0)
        if s > best_score:
            best_score, best = s, (x, y, r)

    return best


# =========================================================
# PASSIVE DRAIN WATCHER (NO MOTOR MOTION)
# =========================================================
class DrainWatcher:
    def __init__(self, camera):
        self.camera = camera
        self.is_running = False
        self.thread = None
        self.callback = None
        self.status_callback = None

        self.raw_color = None
        self.display_frame = None

        self.drain_present = False

        # Current locked drain state
        self.locked_drain_ready = False
        self.locked_drain = None

        # One-shot event for GUI workflow
        self.new_lock_event = False

    def start(self, callback=None, status_callback=None):
        self.callback = callback
        self.status_callback = status_callback

        self.locked_drain_ready = False
        self.locked_drain = None
        self.new_lock_event = False
        self._update_drain_present(False)

        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self):
        was_running = self.is_running
        self.is_running = False
        self.locked_drain_ready = False
        self.locked_drain = None
        self.new_lock_event = False
        self._update_drain_present(False)

        if was_running and self.callback:
            try:
                self.callback("stopped")
            except Exception:
                pass

    def consume_new_lock_event(self):
        if self.new_lock_event:
            self.new_lock_event = False
            return True
        return False

    def _update_drain_present(self, drain_present):
        drain_present = bool(drain_present)
        changed = drain_present != self.drain_present
        self.drain_present = drain_present

        if self.status_callback and changed:
            try:
                self.status_callback(drain_present)
            except Exception:
                pass

    def _run(self):
        print("\n--- Passive Drain Watcher Started ---")

        tracked_x, tracked_y, tracked_r = None, None, None
        candidate_x, candidate_y, candidate_r, candidate_count = None, None, None, 0
        startup_time = time.time()

        # how many frames of misses before fully abandoning old tracking
        miss_count = 0
        MISS_RESET_FRAMES = 5

        try:
            while self.is_running:
                if self.raw_color is None:
                    time.sleep(LOOP_DELAY)
                    continue

                color_img = self.raw_color.copy()
                display_img = color_img.copy()
                status = ""
                drain_seen_this_frame = False

                if time.time() - startup_time < STARTUP_WAIT_S:
                    status = "waiting before search"

                else:
                    drain = None

                    # Try local tracking first if we had a locked drain
                    if tracked_x is not None and tracked_y is not None:
                        drain = find_drain_local(color_img, tracked_x, tracked_y)

                    # If local failed, fall back to full global search immediately
                    if drain is None:
                        drain = auto_find_drain(color_img)

                    if drain is None:
                        miss_count += 1
                        status = f"searching for drain ({miss_count})"

                        if miss_count >= MISS_RESET_FRAMES:
                            tracked_x, tracked_y, tracked_r = None, None, None
                            candidate_x, candidate_y, candidate_r, candidate_count = None, None, None, 0
                            self.locked_drain_ready = False
                            self.locked_drain = None

                    else:
                        miss_count = 0
                        drain_seen_this_frame = True
                        x, y, r = drain

                        if candidate_x is None:
                            candidate_x, candidate_y, candidate_r, candidate_count = x, y, r, 1
                        else:
                            if np.sqrt((x - candidate_x) ** 2 + (y - candidate_y) ** 2) <= DRAIN_LOCK_PX:
                                candidate_x = int((candidate_x + x) / 2)
                                candidate_y = int((candidate_y + y) / 2)
                                candidate_r = int((candidate_r + r) / 2)
                                candidate_count += 1
                            else:
                                candidate_x, candidate_y, candidate_r, candidate_count = x, y, r, 1

                        status = f"locking drain {candidate_count}/{DRAIN_LOCK_FRAMES}"

                        if candidate_count >= DRAIN_LOCK_FRAMES:
                            was_locked = self.locked_drain_ready

                            tracked_x, tracked_y, tracked_r = candidate_x, candidate_y, candidate_r
                            self.locked_drain_ready = True
                            self.locked_drain = (tracked_x, tracked_y, tracked_r)

                            # Fire one-shot event only when this is a fresh lock
                            if not was_locked:
                                self.new_lock_event = True

                            status = "drain locked"

                self._update_drain_present(drain_seen_this_frame)

                # If drain is gone now, let future re-locks count as fresh events
                if not drain_seen_this_frame:
                    self.locked_drain_ready = False

                if candidate_x is not None:
                    cv2.circle(display_img, (candidate_x, candidate_y), 7, (0, 0, 255), -1)
                    cv2.circle(display_img, (candidate_x, candidate_y), max(12, candidate_r), (0, 0, 255), 2)

                if tracked_x is not None and tracked_y is not None:
                    cv2.circle(display_img, (tracked_x, tracked_y), 7, (0, 255, 0), -1)
                    if tracked_r is not None:
                        cv2.circle(display_img, (tracked_x, tracked_y), tracked_r, (0, 255, 0), 2)

                cv2.putText(display_img, "Watcher", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display_img, status, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                self.display_frame = display_img
                time.sleep(LOOP_DELAY)

        except Exception as e:
            print(f"[DrainWatcher Error] {e}")
        finally:
            self.is_running = False
            self.locked_drain_ready = False
            self.locked_drain = None
            self.new_lock_event = False
            self._update_drain_present(False)


# =========================================================
# MAIN CLASS FOR GUI
# =========================================================
class AutoDrainer:
    def __init__(self, camera):
        self.camera = camera
        self.is_running = False
        self.thread = None
        self.callback = None
        self.status_callback = None
        self.stop_on_first_drain = False
        self.raw_color = None
        self.raw_depth = None
        self.display_frame = None
        self.drain_present = False

    def start(self, callback=None, status_callback=None, stop_on_first_drain=False):
        self.callback = callback
        self.status_callback = status_callback
        self.stop_on_first_drain = stop_on_first_drain
        self._update_drain_present(False)

        if not self.is_running:
            self.is_running = True
            global pin_states
            pin_states = {}
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self):
        was_running = self.is_running
        self.is_running = False
        stop_all()
        self._update_drain_present(False)

        if was_running and self.callback:
            self.callback("stopped")

    def _update_drain_present(self, drain_present):
        drain_present = bool(drain_present)
        changed = drain_present != self.drain_present
        self.drain_present = drain_present

        if self.status_callback and changed:
            try:
                self.status_callback(drain_present)
            except Exception:
                pass

    def _run(self):
        print("\n--- Auto-Drain Centering Started ---")

        tracked_x, tracked_y, tracked_r = None, None, None
        candidate_x, candidate_y, candidate_r, candidate_count = None, None, None, 0
        stage, y_locked_frames = 0, 0
        startup_time = time.time()

        try:
            while self.is_running:
                if self.raw_color is None or self.raw_depth is None:
                    time.sleep(LOOP_DELAY)
                    continue

                color_img = self.raw_color.copy()
                depth_img = self.raw_depth.copy()
                display_img = color_img.copy()

                h, w = color_img.shape[:2]
                target_y = int(h * TARGET_Y_FRAC)
                basin_distance_m, dist_roi = get_basin_distance(depth_img)
                status = ""
                drain_seen_this_frame = False

                # --- STAGE 0 ---
                if stage == 0:
                    stop_all()
                    status = "waiting before search"
                    if time.time() - startup_time >= STARTUP_WAIT_S:
                        stage = 1

                # --- STAGE 1 ---
                elif stage == 1:
                    stop_all()
                    drain = auto_find_drain(color_img)

                    if drain is None:
                        candidate_x, candidate_y, candidate_r, candidate_count = None, None, None, 0
                        status = "searching for drain"
                    else:
                        drain_seen_this_frame = True
                        x, y, r = drain

                        if candidate_x is None:
                            candidate_x, candidate_y, candidate_r, candidate_count = x, y, r, 1
                        else:
                            if np.sqrt((x - candidate_x) ** 2 + (y - candidate_y) ** 2) <= DRAIN_LOCK_PX:
                                candidate_x = int((candidate_x + x) / 2)
                                candidate_y = int((candidate_y + y) / 2)
                                candidate_r = int((candidate_r + r) / 2)
                                candidate_count += 1
                            else:
                                candidate_x, candidate_y, candidate_r, candidate_count = x, y, r, 1

                        status = f"locking drain {candidate_count}/{DRAIN_LOCK_FRAMES}"

                        if candidate_count >= DRAIN_LOCK_FRAMES:
                            tracked_x, tracked_y, tracked_r = candidate_x, candidate_y, candidate_r
                            y_locked_frames = 0
                            stage = 2
                            status = "drain locked"

                            if self.stop_on_first_drain:
                                self._update_drain_present(True)
                                cv2.putText(
                                    display_img,
                                    status,
                                    (20, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (255, 255, 255),
                                    2
                                )
                                self.display_frame = display_img
                                print("\n--- Drain Found: stopping centering early by request ---")
                                break

                # --- STAGE 2 ---
                elif stage == 2:
                    stop_height()
                    drain = find_drain_local(color_img, tracked_x, tracked_y) if tracked_x is not None else None

                    if drain is not None:
                        drain_seen_this_frame = True
                        tracked_x, tracked_y, tracked_r = drain
                        y_error = tracked_y - target_y

                        if abs(y_error) <= Y_TOL_PX:
                            y_locked_frames += 1
                            stop_motor_b()
                            status = f"in line {y_locked_frames}/{Y_LOCK_FRAMES}"
                        else:
                            y_locked_frames = 0
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

                # --- STAGE 3 ---
                elif stage == 3:
                    stop_motor_b()
                    drain = find_drain_local(color_img, tracked_x, tracked_y) if tracked_x is not None else None

                    if drain is not None:
                        drain_seen_this_frame = True
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
                            # If target distance is greater than current (error > 0), the camera needs to back away (move UP).
                            if height_error > 0:
                                hold_height_up()
                                status = "moving up"
                            # If target distance is less than current (error < 0), the camera is too far away (move DOWN).
                            else:
                                hold_height_down()
                                status = "moving down"

                # --- STAGE 4 ---
                else:
                    stop_all()
                    drain = find_drain_local(color_img, tracked_x, tracked_y) if tracked_x is not None else None

                    if drain is not None:
                        drain_seen_this_frame = True
                        tracked_x, tracked_y, tracked_r = drain

                    status = "done"
                    print("\n--- Auto-Drain Centering Finished! ---")

                    cv2.putText(
                        display_img,
                        status,
                        (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )
                    self.display_frame = display_img
                    break

                self._update_drain_present(drain_seen_this_frame)

                # --- DRAW ---
                cv2.line(display_img, (0, target_y), (w, target_y), (0, 255, 255), 2)
                cv2.rectangle(display_img, (0, target_y - Y_TOL_PX), (w, target_y + Y_TOL_PX), (0, 200, 255), 1)

                if candidate_x is not None and stage == 1:
                    cv2.circle(display_img, (candidate_x, candidate_y), 7, (0, 0, 255), -1)
                    cv2.circle(display_img, (candidate_x, candidate_y), max(12, candidate_r), (0, 0, 255), 2)

                if tracked_x is not None and tracked_y is not None:
                    cv2.circle(display_img, (tracked_x, tracked_y), 7, (0, 255, 0), -1)
                    if tracked_r is not None:
                        cv2.circle(display_img, (tracked_x, tracked_y), tracked_r, (0, 255, 0), 2)

                dx1, dy1, dx2, dy2 = dist_roi
                cv2.rectangle(display_img, (dx1, dy1), (dx2, dy2), (255, 255, 0), 2)

                cv2.putText(display_img, f"Stage: {stage}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display_img, status, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if tracked_x is not None and tracked_y is not None:
                    cv2.putText(
                        display_img,
                        f"Tracked x,y: {tracked_x}, {tracked_y}",
                        (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

                if basin_distance_m is not None:
                    cv2.putText(
                        display_img,
                        f"Basin distance: {basin_distance_m:.3f} m",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )
                    cv2.putText(
                        display_img,
                        f"Height error: {TARGET_DISTANCE_M - basin_distance_m:.3f} m",
                        (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

                self.display_frame = display_img
                time.sleep(LOOP_DELAY)

        except Exception as e:
            print(f"[AutoDrain Error] {e}")
        finally:
            self.stop()