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

        # 2. Start the Accelerometer completely detached from the Color/Depth pipeline
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
        """Asynchronous callback that updates the roll angle instantly when the IMU moves."""
        f = frame.as_motion_frame()
        if f:
            data = f.get_motion_data()
            # Calculate unbounded 360 degree tilt
            roll_deg = math.degrees(math.atan2(data.x, data.z))
            self.roll = roll_deg % 360.0

    def start(self):
        if not self.imu_ready:
            print("[AutoLeveler] Cannot start: IMU not initialized.")
            return

        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self):
        self.is_running = False
        self.hw.rotate_cw(False)
        self.hw.rotate_ccw(False)

    def _run(self):
        print("\n--- Auto-Leveling Started ---")
        IMU_OFFSET = 0.0
        TARGET_ANGLE = 180.0 + IMU_OFFSET

        TOLERANCE = 0.5
        BRAKE_ZONE = 3.0

        while self.is_running:
            try:
                # self.roll is automatically updated by the background callback
                diff = self.roll - TARGET_ANGLE
                abs_diff = abs(diff)

                # 1. Check if we hit the target
                if abs_diff <= TOLERANCE:
                    print(f"Level Reached! Locked in at {self.roll:.2f}°")
                    self.stop()
                    break

                # 2. Set Direction
                if diff > 0:
                    self.hw.rotate_ccw(True)
                    self.hw.rotate_cw(False)
                else:
                    self.hw.rotate_cw(True)
                    self.hw.rotate_ccw(False)

                # 3. Movement Logic (Continuous vs Pulse Braking)
                if abs_diff <= BRAKE_ZONE:
                    time.sleep(0.05)
                    self.hw.rotate_cw(False)
                    self.hw.rotate_ccw(False)
                    time.sleep(0.25)
                else:
                    time.sleep(0.05)

            except Exception as e:
                print(f"[AutoLevel Error] {e}")
                self.stop()
                break

    def shutdown(self):
        """Safely powers down the motion sensor when the app closes."""
        self.stop()
        if self.imu_ready and self.motion_sensor:
            try:
                self.motion_sensor.stop()
                self.motion_sensor.close()
            except:
                pass