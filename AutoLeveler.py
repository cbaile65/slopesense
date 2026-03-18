import pyrealsense2 as rs
import threading
import time
import math


class AutoLeveler:
    def __init__(self, rs_device, hardware):
        self.hw = hardware
        self.is_running = False
        self.thread = None
        self.roll = 180.0  # Default safe fallback

        self.motion_sensor = None
        self.imu_ready = False

        # 1. Find the Motion Module on the RealSense device
        for s in rs_device.query_sensors():
            if s.is_motion_sensor():
                self.motion_sensor = s
                break

        # 2. Start the Accelerometer completely detached
        if self.motion_sensor:
            try:
                accel_profile = next(p for p in self.motion_sensor.get_stream_profiles()
                                     if p.stream_type() == rs.stream.accel)
                self.motion_sensor.open(accel_profile)
                self.motion_sensor.start(self._imu_callback)
                self.imu_ready = True
                print("[AutoLeveler] IMU Sensor started independently.")
            except Exception as e:
                print(f"[AutoLeveler] Failed to start IMU: {e}")
        else:
            print("[AutoLeveler] No motion sensor found on this device.")

    def _imu_callback(self, frame):
        """Asynchronous callback that updates the roll angle instantly."""
        f = frame.as_motion_frame()
        if f:
            data = f.get_motion_data()
            roll_deg = math.degrees(math.atan2(data.x, data.z))
            self.roll = roll_deg % 360.0

    def start(self, callback=None):
        if not self.imu_ready:
            print("[AutoLeveler] Cannot start: IMU not initialized.")
            return

        # Prevent multiple threads from spawning and locking the button
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        else:
            print("[AutoLeveler] Ignoring click: Auto-Level is already running.")

    def stop(self):
        self.is_running = False
        self.hw.rotate_cw(False)
        self.hw.rotate_ccw(False)

    def _run(self):
        print("\n--- Auto-Leveling Started ---")
        TARGET_ANGLE = 180.0
        TOLERANCE = 0.8

        # Record when we started for the Watchdog
        start_time = time.time()

        try:
            while self.is_running:

                # --- WATCHDOG TIMEOUT ---
                # If we've been trying to level for 8 seconds, the motor is physically stuck.
                # Break the loop so the button doesn't get permanently locked out.
                if time.time() - start_time > 8.0:
                    print("[AutoLeveler] TIMEOUT: Motor stalled or took too long. Shutting down.")
                    break

                # 1. Math
                diff = self.roll - TARGET_ANGLE
                abs_diff = abs(diff)

                print(f"[AutoLevel] Roll: {self.roll:.2f}° | Diff: {diff:.2f}°")

                # 2. Check if we hit the target
                if abs_diff <= TOLERANCE:
                    print(f"Level Reached! Locked in at {self.roll:.2f}°")
                    break  # Exits loop, safely triggers 'finally' block

                # 3. Set Direction
                if diff > 0:
                    self.hw.rotate_ccw(True)
                    self.hw.rotate_cw(False)
                else:
                    self.hw.rotate_cw(True)
                    self.hw.rotate_ccw(False)

                # 4. Multi-Stage Proportional Braking
                if abs_diff <= 1.5:
                    # ZONE 1: Increased from 0.03 to 0.06 to overcome static friction
                    time.sleep(0.06)
                    self.hw.rotate_cw(False)
                    self.hw.rotate_ccw(False)
                    time.sleep(0.3)

                elif abs_diff <= 5.0:
                    # ZONE 2: Medium taps
                    time.sleep(0.12)
                    self.hw.rotate_cw(False)
                    self.hw.rotate_ccw(False)
                    time.sleep(0.25)

                else:
                    # ZONE 3: Far away, move continuously
                    time.sleep(0.05)

        except Exception as e:
            print(f"[AutoLevel Error] {e}")

        finally:
            # SAFETY NET: This ALWAYS runs, even if it crashes or times out,
            # ensuring motors turn off and the button works next time.
            self.stop()

    def shutdown(self):
        """Safely powers down the motion sensor when the app closes."""
        self.stop()
        if self.imu_ready and self.motion_sensor:
            try:
                self.motion_sensor.stop()
                self.motion_sensor.close()
            except:
                pass