import requests
import time

# IP address of your Raspberry Pi and the port Flask is running on
PI_URL = "http://192.168.5.2:5000"

def set_remote_pin(pin, state):
    """Sends an HTTP GET request to the Pi to toggle a pin."""
    try:
        # This hits the URL we defined in the Pi's Flask script
        response = requests.get(f"{PI_URL}/pin/{pin}/{state}")
        print(f"Response from Pi: {response.json()}")
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to the Raspberry Pi at {PI_URL}")

print("Starting Master Control...")

try:
    # Main logic loop running on the Mac
    while True:
        print("Commanding Pin 17 ON")
        set_remote_pin(17, "on")
        time.sleep(1)

        print("Commanding Pin 17 OFF")
        set_remote_pin(17, "off")
        time.sleep(1)

except KeyboardInterrupt:
    print("\nExiting master program...")