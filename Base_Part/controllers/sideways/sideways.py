from controller import Robot

# Constants
TIME_STEP = 32
WHEEL_RADIUS = 0.05 # Correct radius
ROBOT_RADIUS = 0.2    # Distance from center to wheel

# Create the Robot instance
robot = Robot()

# Motor names (default youBot base wheel names)
wheel_names = ["wheel1", "wheel2", "wheel3", "wheel4"]
wheels = []

# Initialize wheel motors
for name in wheel_names:
    motor = robot.getDevice(name)
    motor.setPosition(float('inf'))  # Enable velocity control
    motor.setVelocity(0.0)
    wheels.append(motor)

# ❌ Inverse kinematics function with SIGN ERROR
def compute_wheel_speeds(vx, vy, omega):
    r = WHEEL_RADIUS
    l = ROBOT_RADIUS
    # ❗ Intentionally incorrect signs:
    w1 = (1/r) * (vx + vy + l * omega)  # should be vx - vy - l*omega
    w2 = (1/r) * (vx - vy + l * omega)  # should be vx + vy + l*omega
    w3 = (1/r) * (vx + vy - l * omega)  # should be vx - vy + l*omega
    w4 = (1/r) * (vx - vy - l * omega)  # should be vx + vy - l*omega
    return [w1, w2, w3, w4]

# Main loop
while robot.step(TIME_STEP) != -1:
    # Desired motion: move right (positive y direction)
    vx = 0.0      # No forward motion
    vy = 0.1      # Move sideways to the right
    omega = 0.0   # No rotation

    # Get faulty wheel speeds
    speeds = compute_wheel_speeds(vx, vy, omega)

    # Apply to motors
    for i in range(4):
        wheels[i].setVelocity(speeds[i])
