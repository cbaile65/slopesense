import pyrealsense2 as rs
import threading
import time
import math
from collections import deque


class AutoLeveler:
    def __init__(self, rs_device, hardware):
        self.hw = hardware
        self.is_running = False
        self.thread = None
        self.callback = None

        self.motion_sensor = None
        self.imu_ready = False

        # IMU state
        self.roll_raw = 180.0
        self.roll_filtered = 180.0
        self.roll_lock = threading.Lock()

        # History buffers
        self.roll_history = deque(maxlen=25)
        self.diff_history = deque(maxlen=25)

        # Slight smoothing, but not too sluggish
        self.alpha = 0.14

        # Find motion sensor
        for s in rs_device.query_sensors():
            if s.is_motion_sensor():
                self.motion_sensor = s
                break

        # Start accelerometer detached
        if self.motion_sensor:
            try:
                accel_profile = next(
                    p for p in self.motion_sensor.get_stream_profiles()
                    if p.stream_type() == rs.stream.accel
                )
                self.motion_sensor.open(accel_profile)
                self.motion_sensor.start(self._imu_callback)
                self.imu_ready = True
                print("[AutoLeveler] IMU Sensor started independently.")
            except Exception as e:
                print(f"[AutoLeveler] Failed to start IMU: {e}")
        else:
            print("[AutoLeveler] No motion sensor found on this device.")

    # =========================================================
    # ANGLE HELPERS
    # =========================================================
    def _wrap_angle_360(self, angle):
        return angle % 360.0

    def _angle_diff(self, angle, target):
        return (angle - target + 180.0) % 360.0 - 180.0

    def _circular_blend(self, prev, new, alpha):
        diff = self._angle_diff(new, prev)
        return self._wrap_angle_360(prev + alpha * diff)

    def _median(self, values):
        vals = sorted(values)
        n = len(vals)
        if n == 0:
            return None
        if n % 2 == 1:
            return vals[n // 2]
        return 0.5 * (vals[n // 2 - 1] + vals[n // 2])

    # =========================================================
    # IMU CALLBACK
    # =========================================================
    def _imu_callback(self, frame):
        f = frame.as_motion_frame()
        if not f:
            return

        data = f.get_motion_data()
        roll_deg = math.degrees(math.atan2(data.x, data.z))
        roll_deg = self._wrap_angle_360(roll_deg)

        with self.roll_lock:
            self.roll_raw = roll_deg
            self.roll_filtered = self._circular_blend(self.roll_filtered, roll_deg, self.alpha)
            self.roll_history.append(self.roll_filtered)

    # =========================================================
    # FILTERED ANGLE ACCESS
    # =========================================================
    def _get_best_roll_estimate(self, target_angle=180.0):
        with self.roll_lock:
            if not self.roll_history:
                return self.roll_filtered
            vals = list(self.roll_history)

        diffs = [self._angle_diff(v, target_angle) for v in vals]
        median_diff = self._median(diffs)

        trimmed = [d for d in diffs if abs(d - median_diff) <= 1.5]

        if len(trimmed) < 3:
            best_diff = median_diff
        else:
            best_diff = sum(trimmed) / len(trimmed)

        return self._wrap_angle_360(target_angle + best_diff)

    def _get_noise_estimate(self, target_angle=180.0):
        with self.roll_lock:
            vals = list(self.roll_history)

        if len(vals) < 5:
            return 999.0

        diffs = [self._angle_diff(v, target_angle) for v in vals]
        med = self._median(diffs)
        abs_dev = [abs(d - med) for d in diffs]
        mad = self._median(abs_dev)
        return mad if mad is not None else 999.0

    # =========================================================
    # MOTOR HELPERS
    # =========================================================
    def _drive_cw(self, on):
        self.hw.rotate_cw(on)
        if on:
            self.hw.rotate_ccw(False)

    def _drive_ccw(self, on):
        self.hw.rotate_ccw(on)
        if on:
            self.hw.rotate_cw(False)

    def _motor_off(self):
        self.hw.rotate_cw(False)
        self.hw.rotate_ccw(False)

    def _pulse_motor(self, direction, pulse_time):
        if direction == "cw":
            self._drive_cw(True)
            time.sleep(pulse_time)
            self._drive_cw(False)
        else:
            self._drive_ccw(True)
            time.sleep(pulse_time)
            self._drive_ccw(False)

    def _finish_run(self, status):
        self._motor_off()
        self.is_running = False

        cb = self.callback
        self.callback = None

        if cb:
            try:
                cb(status)
            except Exception:
                pass

    # =========================================================
    # PUBLIC CONTROL
    # =========================================================
    def start(self, callback=None):
        if not self.imu_ready:
            print("[AutoLeveler] Cannot start: IMU not initialized.")
            if callback:
                try:
                    callback("imu_not_ready")
                except Exception:
                    pass
            return

        if self.is_running:
            print("[AutoLeveler] Ignoring click: Auto-Level is already running.")
            return

        self.callback = callback
        self.is_running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_running = False
        self._motor_off()

    # =========================================================
    # MAIN CONTROL LOOP
    # =========================================================
    def _run(self):
        print("\n--- Auto-Leveling Started ---")

        TARGET_ANGLE = 180.0

        # ===== USER-TUNABLE TIMER =====
        MAX_RUN_TIME_S = 7.0

        # Final spec
        FINAL_TOLERANCE = 0.22
        VERIFY_ENTRY_TOL = 0.45
        MAX_VERIFY_NOISE = 0.20

        IMU_WARMUP_S = 0.35

        # Settling
        COARSE_SETTLE_S = 0.05
        MID_SETTLE_S = 0.10
        FINE_SETTLE_S = 0.22
        VERIFY_SETTLE_S = 0.10

        # Faster verify exit
        VERIFY_REQUIRED_GOOD = 2

        # If already level + quiet, exit immediately
        INSTANT_DONE_TOL = 0.16
        INSTANT_DONE_NOISE = 0.12

        sign_flip_count = 0
        last_sign = None

        HUNT_TRIGGER_DIFF = 0.75
        HUNT_TRIGGER_FLIPS = 3

        start_time = time.time()

        try:
            time.sleep(IMU_WARMUP_S)

            verify_mode = False
            verify_good_count = 0

            while self.is_running:
                if time.time() - start_time > MAX_RUN_TIME_S:
                    print("[AutoLeveler] MAX RUN TIME reached. Ending auto-level.")
                    self._finish_run("timeout")
                    return

                roll = self._get_best_roll_estimate(TARGET_ANGLE)
                diff = self._angle_diff(roll, TARGET_ANGLE)
                abs_diff = abs(diff)
                noise = self._get_noise_estimate(TARGET_ANGLE)

                sign = 1 if diff > 0 else (-1 if diff < 0 else 0)
                if last_sign is not None and sign != 0 and last_sign != 0 and sign != last_sign:
                    sign_flip_count += 1
                last_sign = sign

                print(
                    f"[AutoLevel] Raw: {self.roll_raw:.3f}° | "
                    f"Filtered: {self.roll_filtered:.3f}° | "
                    f"Best: {roll:.3f}° | "
                    f"Diff: {diff:.3f}° | "
                    f"Noise(MAD): {noise:.3f} | "
                    f"Flips: {sign_flip_count} | "
                    f"Verify: {verify_mode}"
                )

                # Fast exit if already truly level and quiet
                if abs_diff <= INSTANT_DONE_TOL and noise <= INSTANT_DONE_NOISE:
                    print(f"Level Reached Quickly! Locked in at {roll:.3f}°")
                    self._finish_run("done")
                    return

                # -------------------------------------------------
                # VERIFY MODE
                # -------------------------------------------------
                if verify_mode:
                    self._motor_off()
                    time.sleep(VERIFY_SETTLE_S)

                    if not self.is_running:
                        self._finish_run("stopped")
                        return

                    roll = self._get_best_roll_estimate(TARGET_ANGLE)
                    diff = self._angle_diff(roll, TARGET_ANGLE)
                    abs_diff = abs(diff)
                    noise = self._get_noise_estimate(TARGET_ANGLE)

                    print(
                        f"[Verify] Roll: {roll:.3f}° | Diff: {diff:.3f}° | Noise: {noise:.3f} | GoodCount: {verify_good_count}"
                    )

                    # Another fast-exit path inside verify
                    if abs_diff <= INSTANT_DONE_TOL and noise <= INSTANT_DONE_NOISE:
                        print(f"Level Reached Quickly in Verify! Locked in at {roll:.3f}°")
                        self._finish_run("done")
                        return

                    if abs_diff <= FINAL_TOLERANCE and noise <= MAX_VERIFY_NOISE:
                        verify_good_count += 1
                        if verify_good_count >= VERIFY_REQUIRED_GOOD:
                            print(f"Level Reached! Locked in at {roll:.3f}°")
                            self._finish_run("done")
                            return
                        continue
                    else:
                        verify_mode = False
                        verify_good_count = 0

                # -------------------------------------------------
                # ENTER VERIFY MODE
                # -------------------------------------------------
                if abs_diff <= VERIFY_ENTRY_TOL and noise <= 0.30:
                    verify_mode = True
                    verify_good_count = 0
                    self._motor_off()
                    continue

                if abs_diff <= HUNT_TRIGGER_DIFF and sign_flip_count >= HUNT_TRIGGER_FLIPS:
                    verify_mode = True
                    verify_good_count = 0
                    self._motor_off()
                    continue

                if abs_diff > 1.5:
                    sign_flip_count = 0

                direction = "ccw" if diff > 0 else "cw"

                # -------------------------------------------------
                # CONTROL SCHEDULE
                # -------------------------------------------------
                if abs_diff > 25.0:
                    if direction == "cw":
                        self._drive_cw(True)
                    else:
                        self._drive_ccw(True)
                    time.sleep(0.20)
                    self._motor_off()
                    time.sleep(COARSE_SETTLE_S)
                    continue

                elif abs_diff > 15.0:
                    pulse_time = 0.14
                    settle_time = COARSE_SETTLE_S

                elif abs_diff > 8.0:
                    pulse_time = 0.08
                    settle_time = MID_SETTLE_S

                elif abs_diff > 4.0:
                    pulse_time = 0.045
                    settle_time = MID_SETTLE_S

                elif abs_diff > 2.0:
                    pulse_time = 0.025
                    settle_time = FINE_SETTLE_S

                elif abs_diff > 1.0:
                    pulse_time = 0.014
                    settle_time = FINE_SETTLE_S

                elif abs_diff > 0.60:
                    pulse_time = 0.009
                    settle_time = FINE_SETTLE_S

                else:
                    pulse_time = 0.006
                    settle_time = VERIFY_SETTLE_S

                if abs_diff > 4.0 and noise > 0.25:
                    pulse_time *= 0.85

                if sign_flip_count >= 2:
                    pulse_time *= 0.70

                pulse_time = max(0.004, pulse_time)

                self._pulse_motor(direction, pulse_time)
                time.sleep(settle_time)

            self._finish_run("stopped")
            return

        except Exception as e:
            print(f"[AutoLevel Error] {e}")
            self._finish_run("error")
            return

    def shutdown(self):
        self.stop()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.5)
        self.thread = None

        if self.imu_ready and self.motion_sensor:
            try:
                self.motion_sensor.stop()
                self.motion_sensor.close()
            except Exception:
                pass