from controller import Robot

# Time step
TIME_STEP = 64

# Create the Robot instance
robot = Robot()

# Initialize the motors
wheels = []
wheel_names = ['wheel1','wheel2', 'wheel3', 'wheel4']

for name in wheel_names:
    motor = robot.getDevice(name)
    motor.setPosition(float('inf'))  # Infinite position for velocity control
    motor.setVelocity(0.0)
    wheels.append(motor)

# Set forward velocity
forward_speed = 3.0  # Adjust speed as needed

# Main loop
while robot.step(TIME_STEP) != -1:
    for wheel in wheels:
        wheel.setVelocity(forward_speed)
