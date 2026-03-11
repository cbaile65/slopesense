import requests

# The dedicated IP of your Raspberry Pi
PI_URL = "http://192.168.5.2:5000"


def set_angle(pin, angle):
    """
    Sends an HTTP request to the Pi to set a servo to a specific angle (0-180).
    """
    # Safety catch: Ensure the angle is between 0 and 180 degrees
    if not (0 <= angle <= 180):
        print(f"[servo_control] Error: Angle {angle} is out of bounds. Must be 0-180.")
        return False

    # Notice the URL now uses '/servo/' instead of '/pin/'
    url = f"{PI_URL}/servo/{pin}/{angle}"

    try:
        response = requests.get(url, timeout=3)
        print(f"[servo_control] Success! Servo on Pin {pin} set to {angle}°.")
        return True

    except requests.exceptions.RequestException as e:
        print(f"[servo_control] Error: Could not reach the Pi. Details: {e}")
        return False