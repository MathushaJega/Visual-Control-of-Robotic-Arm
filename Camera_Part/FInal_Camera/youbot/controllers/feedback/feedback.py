from controller import Robot, Supervisor
import math

# Constants
TIME_STEP = 32
MAX_SPEED = 6.28
WHEEL_RADIUS = 0.05
HALF_WHEELBASE_LENGTH = 0.235
HALF_WHEELBASE_WIDTH = 0.15

# Target position (X, Y)
target_pos = [-0.57, 0.35]
stop_offset = 0.3  # Distance to stop before reaching the target

# PID gains
Kp = 2.0
Ki = 0.01
Kd = 0.3

# Initialize supervisor and robot node
robot = Supervisor()
robot_node = robot.getFromDef("YOUBOT")
if robot_node is None:
    print("Error: Could not find robot node with DEF name 'YOUBOT'.")
    exit(1)

translation_field = robot_node.getField("translation")
rotation_field = robot_node.getField("rotation")

# Initialize motors
wheel_names = ["wheel1", "wheel2", "wheel3", "wheel4"]
wheels = []
for name in wheel_names:
    motor = robot.getDevice(name)
    motor.setPosition(float('inf'))
    motor.setVelocity(0.0)
    wheels.append(motor)

def compute_wheel_velocities(vx, vy, omega):
    l = HALF_WHEELBASE_LENGTH
    w = HALF_WHEELBASE_WIDTH
    r = WHEEL_RADIUS
    return [
        (vx + vy + (l + w) * omega) / r,  # Front left
        (vx - vy - (l + w) * omega) / r,  # Front right
        (vx - vy + (l + w) * omega) / r,  # Rear left
        (vx + vy - (l + w) * omega) / r   # Rear right
    ]

# PID variables
integral = [0.0, 0.0]
prev_error = [0.0, 0.0]
reached = False

while robot.step(TIME_STEP) != -1:
    pos = translation_field.getSFVec3f()
    rotation = rotation_field.getSFRotation()

    # Get yaw (robot orientation)
    axis = rotation[:3]
    angle = rotation[3]
    if axis[2] < 0:
        angle = -angle
    robot_theta = angle

    # Front-center position of robot
    center_x = pos[0] + HALF_WHEELBASE_LENGTH * math.cos(robot_theta)
    center_y = pos[1] + HALF_WHEELBASE_LENGTH * math.sin(robot_theta)

    # Compute angle to target
    goal_angle = math.atan2(target_pos[1] - center_y, target_pos[0] - center_x)

    # Adjust the target to stop early
    adjusted_target = [
        target_pos[0] - stop_offset * math.cos(goal_angle),
        target_pos[1] - stop_offset * math.sin(goal_angle)
    ]

    # Compute error
    error = [adjusted_target[0] - center_x, adjusted_target[1] - center_y]
    distance = math.hypot(error[0], error[1])

    # Stop if within tolerance
    if distance < 0.05:
        if not reached:
            print("Target offset reached. Stopping.")
            reached = True
        for motor in wheels:
            motor.setVelocity(0.0)
        continue

    # PID calculations
    integral[0] += error[0] * TIME_STEP / 1000.0
    integral[1] += error[1] * TIME_STEP / 1000.0
    derivative = [
        (error[0] - prev_error[0]) / (TIME_STEP / 1000.0),
        (error[1] - prev_error[1]) / (TIME_STEP / 1000.0)
    ]
    prev_error = error[:]

    # Calculate control velocities
    vx = Kp * error[0] + Ki * integral[0] + Kd * derivative[0]
    vy = Kp * error[1] + Ki * integral[1] + Kd * derivative[1]

    # Orientation correction (optional, could set omega = 0.0)
    angle_diff = math.atan2(math.sin(goal_angle - robot_theta), math.cos(goal_angle - robot_theta))
    omega = 1.0 * angle_diff

    # Compute and apply wheel velocities
    speeds = compute_wheel_velocities(vx, vy, omega)
    speeds = [max(min(s, MAX_SPEED), -MAX_SPEED) for s in speeds]
    for i in range(4):
        wheels[i].setVelocity(speeds[i])
