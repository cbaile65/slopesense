import motor_control
import time

print("Starting motor Hardware Test...")

# The GPIO pin your relay is connected to
MOTOR_PIN = 17

try:
    while True:
        # Turn ON
        print(f"Commanding Pin {MOTOR_PIN} ON...")
        motor_control.set_pin(MOTOR_PIN, "on")
        time.sleep(3)  # Motor runs for 3 seconds

        # Turn OFF
        print(f"Commanding Pin {MOTOR_PIN} OFF...")
        motor_control.set_pin(MOTOR_PIN, "off")
        time.sleep(3)  # Motor rests for 3 seconds before repeating

# Safety catch for when you hit Stop
except KeyboardInterrupt:
    print("\nTest interrupted! Engaging shutdown...")
    motor_control.set_pin(MOTOR_PIN, "off")