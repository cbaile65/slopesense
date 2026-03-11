import motor_control
import servo_control

# Define your hardware pins
MOTOR_PIN = 17
SERVO_PIN = 18

print("=======================================")
print(" 🎛️ SlopeSense Live Hardware Control 🎛️")
print("=======================================")
print("Commands:")
print(" - Type 'on' or 'off' to control the motor.")
print(" - Type a number (0-180) to move the servo.")
print(" - Type 'quit' or 'exit' to stop the program.")
print("=======================================\n")

try:
    while True:
        # Wait for the user to type something and press Enter
        # .strip().lower() ensures accidental spaces or capital letters don't break it
        user_input = input("Enter command: ").strip().lower()

        # 1. Check if the user wants to quit
        if user_input in ['quit', 'exit']:
            print("Exiting interactive mode...")
            break  # This breaks us out of the infinite loop

        # 2. Check if the user typed a motor command
        elif user_input in ['on', 'off']:
            motor_control.set_pin(MOTOR_PIN, user_input)

        # 3. Check if the user typed a number (for the servo)
        elif user_input.isdigit():
            angle = int(user_input)
            servo_control.set_angle(SERVO_PIN, angle)

        # 4. Catch invalid typos
        else:
            print("⚠️ Invalid command. Please type 'on', 'off', or a number between 0 and 180.")

except KeyboardInterrupt:
    print("\nProgram interrupted by Stop button.")

finally:
    # This 'finally' block runs no matter how the script closes (typing quit or hitting Stop)
    # It guarantees your hardware doesn't get left in a dangerous state
    print("\nEngaging safety shutdown...")
    motor_control.set_pin(MOTOR_PIN, "off")
    # Uncomment the next line if you want the servo to auto-center when you quit
    # servo_control.set_angle(SERVO_PIN, 90)
    print("Hardware safed. Goodbye.")