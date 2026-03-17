import DC_Motor_Control
import Servo_Control
import threading
import time

# Hardware Pins
MOTOR_A_PIN1 = 17  # Forward
MOTOR_A_PIN2 = 27  # Backward
MOTOR_B_PIN1 = 24  # Up
MOTOR_B_PIN2 = 23  # Down
SERVO_PIN = 18  # CW / CCW


class HardwareManager:
    def __init__(self):
        # State variables for positional servo sweeping
        self.servo_angle = 90
        self.servo_direction = 0
        self.servo_running = True

        self.servo_thread = threading.Thread(target=self._servo_loop, daemon=True)
        self.servo_thread.start()

        self.stop_all()

    def _servo_loop(self):
        """Continuously runs in the background to sweep the servo fluidly."""
        while self.servo_running:
            if self.servo_direction != 0:
                # Fluidity fix: Step by exactly 1 degree at a very fast rate (50 times a second).
                # This keeps the servo motor constantly tracking a moving target, eliminating the "stop/start" jitter.
                new_angle = self.servo_angle + self.servo_direction

                # Clamp the angle so it cannot exceed 0 or 180 degrees
                new_angle = max(0, min(180, new_angle))

                if new_angle != self.servo_angle:
                    self.servo_angle = new_angle

                    # Fire the HTTP request in a tiny disposable background thread.
                    # This guarantees the sweeping loop keeps perfect time (0.02s)
                    # and doesn't hiccup while waiting for Wi-Fi responses.
                    threading.Thread(target=self._send_servo_cmd, args=(self.servo_angle,), daemon=True).start()

            # Update rate: 0.02s = 50 updates a second (50 degrees per second sweep speed).
            time.sleep(0.02)

    def _send_servo_cmd(self, angle):
        """Helper function to cleanly send the command and ignore dropped packets."""
        try:
            Servo_Control.set_angle(SERVO_PIN, angle)
        except:
            pass

    def move_forward(self, start):
        if start:
            DC_Motor_Control.set_pin(MOTOR_A_PIN2, "off")  # Safety interlock
            DC_Motor_Control.set_pin(MOTOR_A_PIN1, "on")
        else:
            DC_Motor_Control.set_pin(MOTOR_A_PIN1, "off")

    def move_backward(self, start):
        if start:
            DC_Motor_Control.set_pin(MOTOR_A_PIN1, "off")  # Safety interlock
            DC_Motor_Control.set_pin(MOTOR_A_PIN2, "on")
        else:
            DC_Motor_Control.set_pin(MOTOR_A_PIN2, "off")

    def move_up(self, start):
        if start:
            DC_Motor_Control.set_pin(MOTOR_B_PIN2, "off")  # Safety interlock
            DC_Motor_Control.set_pin(MOTOR_B_PIN1, "on")
        else:
            DC_Motor_Control.set_pin(MOTOR_B_PIN1, "off")

    def move_down(self, start):
        if start:
            DC_Motor_Control.set_pin(MOTOR_B_PIN1, "off")  # Safety interlock
            DC_Motor_Control.set_pin(MOTOR_B_PIN2, "on")
        else:
            DC_Motor_Control.set_pin(MOTOR_B_PIN2, "off")

    def rotate_cw(self, start):
        if start:
            self.servo_direction = -1
        else:
            if self.servo_direction == -1:
                self.servo_direction = 0

    def rotate_ccw(self, start):
        if start:
            self.servo_direction = 1
        else:
            if self.servo_direction == 1:
                self.servo_direction = 0

    def stop_all(self):
        """Emergency stop for all hardware"""
        self.servo_direction = 0
        DC_Motor_Control.set_pin(MOTOR_A_PIN1, "off")
        DC_Motor_Control.set_pin(MOTOR_A_PIN2, "off")
        DC_Motor_Control.set_pin(MOTOR_B_PIN1, "off")
        DC_Motor_Control.set_pin(MOTOR_B_PIN2, "off")

    def shutdown(self):
        """Kills the background thread before the program exits"""
        self.servo_running = False
        self.stop_all()