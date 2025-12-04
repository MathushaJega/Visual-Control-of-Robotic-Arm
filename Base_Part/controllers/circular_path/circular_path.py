# youbot_circle_path.py
from controller import Supervisor
import math

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
RADIUS = 0.1        # 1 cm = 0.01 m
SPEED  = 2.0         # wheel speed [rad/s]
TIME_STEP = 32

# ------------------------------------------------------------
# Helper
# ------------------------------------------------------------
def set_wheel_speeds(robot, v_left, v_right):
    for wheel in ['wheel1', 'wheel2', 'wheel3', 'wheel4']:
        m = robot.getDevice(wheel)
        m.setPosition(float('inf'))   # velocity control
        if wheel in ['wheel1', 'wheel3']:  # left wheels
            m.setVelocity(v_left)
        else:                          # right wheels
            m.setVelocity(v_right)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
robot = Supervisor()

# Get robot node and its initial position (circle center)
youbot_node = robot.getFromDef("YOUBOT")   # Ensure your YouBot node has DEF YOUBOT
translation_field = youbot_node.getField("translation")
init_pos = translation_field.getSFVec3f()  # [x, y, z]
center_x, center_y = init_pos[0], init_pos[2]

print(f"Initial Position (center of circle): X={center_x:.4f}, Y={center_y:.4f}")

# Wheel setup
wheels = []
for wheel_name in ['wheel1', 'wheel2', 'wheel3', 'wheel4']:
    motor = robot.getDevice(wheel_name)
    motor.setPosition(float('inf'))
    motor.setVelocity(0.0)
    wheels.append(motor)

# YouBot wheel radius and axle length (approx)
WHEEL_RADIUS = 0.047   # 4.7 cm
AXLE_LENGTH  = 0.3     # 30 cm between left and right wheels

# Compute wheel speeds for circular motion
# v = ω * R ,  Δv between wheels = 2 * RADIUS / AXLE_LENGTH * v
linear_speed = 0.05   # m/s (slow to follow small circle)
omega = linear_speed / RADIUS
v_left  = (linear_speed - (AXLE_LENGTH/2.0) * omega) / WHEEL_RADIUS
v_right = (linear_speed + (AXLE_LENGTH/2.0) * omega) / WHEEL_RADIUS

print(f"Wheel speeds: left={v_left:.3f}, right={v_right:.3f}")

# ------------------------------------------------------------
# Loop
# ------------------------------------------------------------
while robot.step(TIME_STEP) != -1:
    # Move wheels for circular trajectory
    for motor in [wheels[0], wheels[2]]:  # left
        motor.setVelocity(v_left)
    for motor in [wheels[1], wheels[3]]:  # right
        motor.setVelocity(v_right)

    # Optional: track current position
    pos = translation_field.getSFVec3f()
    print(f"Robot Pos: X={pos[0]:.4f}, Y={pos[2]:.4f}")
