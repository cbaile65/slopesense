import threading
import time


class AutoLeveler:
    def __init__(self, camera, hardware):
        self.camera = camera
        self.hw = hardware
        self.is_running = False
        self.thread = None
        self.callback = None

    def start(self, callback=None):
        if not self.is_running:
            self.is_running = True
            self.callback = callback
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self):
        self.is_running = False
        self.hw.rotate_cw(False)
        self.hw.rotate_ccw(False)

    def _run(self):
        TOLERANCE = 0.5  # Degrees of allowed error for Roll
        print("[AutoLeveler] Started...")

        while self.is_running:
            try:
                # Ask the camera for angles
                roll, pitch = self.camera.get_current_angles()
                print(f"[AutoLeveler] Current Roll: {roll:.2f}°")
            except Exception as e:
                print(f"[AutoLeveler] ERROR reading camera angles: {e}")
                self.stop()
                if self.callback:
                    self.callback("error")
                return

            # Logic for Roll using the Servo
            if roll > TOLERANCE:
                self.hw.rotate_cw(True)
                self.hw.rotate_ccw(False)
            elif roll < -TOLERANCE:
                self.hw.rotate_ccw(True)
                self.hw.rotate_cw(False)
            else:
                print("[AutoLeveler] Level Achieved!")
                self.hw.rotate_cw(False)
                self.hw.rotate_ccw(False)
                self.is_running = False

                if self.callback:
                    self.callback("success")
                return

            time.sleep(0.05)  # Check IMU at 20Hz