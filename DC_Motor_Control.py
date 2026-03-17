import requests

# The dedicated IP of your Raspberry Pi on the new network
PI_URL = "http://192.168.5.2:5000"


def set_pin(pin, state):
    """
    Sends an HTTP request to the Pi to set a specific pin 'on' or 'off'.
    """
    state = state.lower()
    url = f"{PI_URL}/pin/{pin}/{state}"

    try:
        response = requests.get(url, timeout=3)
        print(f"[motor_control] Success! Pin {pin} set to '{state}'.")
        return True

    except requests.exceptions.RequestException as e:
        print(f"[motor_control] Error: Could not reach the Pi. Details: {e}")
        return False