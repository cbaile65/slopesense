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
        """Continuously runs in the background to sweep the servo while a button is held."""
        while self.servo_running:
            if self.servo_direction != 0:
                # Smoothing fix: Network requests over Wi-Fi take time.
                # If we spam 20 requests a second, the queue clogs and the servo stutters.
                # We increased the step size (4 deg) and lowered the refresh rate (10 Hz).
                new_angle = self.servo_angle + (self.servo_direction * 4)

                # Clamp the angle so it cannot exceed 0 or 180 degrees
                new_angle = max(0, min(180, new_angle))

                # Only send a command if the angle actually changed
                if new_angle != self.servo_angle:
                    self.servo_angle = new_angle

                    # Wrap in try/except so network spikes don't crash the sweeping thread
                    try:
                        Servo_Control.set_angle(SERVO_PIN, self.servo_angle)
                    except:
                        pass

            # Update rate: 0.1s = 10 updates a second. Much easier on the Pi's network.
            time.sleep(0.1)

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
            self.servo_direction = -1  # Swapped Direction
        else:
            if self.servo_direction == -1:
                self.servo_direction = 0

    def rotate_ccw(self, start):
        if start:
            self.servo_direction = 1  # Swapped Direction
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