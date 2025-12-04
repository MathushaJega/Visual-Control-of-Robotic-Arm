from controller import Supervisor
import math

TIME_STEP = 32
MAX_SPEED = 6.28
WHEEL_RADIUS = 0.029
ROBOT_RADIUS = 0.2

# Initialize Supervisor
supervisor = Supervisor()

# Get robot node by DEF name
robot_node = supervisor.getFromDef("MY_ROBOT")

# Initialize wheels
wheel_names = ["wheel1", "wheel2", "wheel3", "wheel4"]
wheels = []
for name in wheel_names:
    motor = supervisor.getDevice(name)
    motor.setPosition(float('inf'))
    motor.setVelocity(0.0)
    wheels.append(motor)

def compute_wheel_speeds(vx, vy, omega):
    r = WHEEL_RADIUS
    l = ROBOT_RADIUS
    w1 = (1/r) * (vx - vy - l * omega)
    w2 = (1/r) * (vx + vy + l * omega)
    w3 = (1/r) * (vx - vy + l * omega)
    w4 = (1/r) * (vx + vy - l * omega)
    return [w1, w2, w3, w4]

# Target and obstacles
goal_x, goal_y = 2.0, 1.73
obstacles = [(0.68, 1.59), (1.64, 0.92)]
avoid_radius = 0.4
avoid_strength = 0.6

while supervisor.step(TIME_STEP) != -1:
    # Get robot position from supervisor node translation field
    pos = robot_node.getField("translation").getSFVec3f()
    x, z = pos[0], pos[2]  # Webots Y is up axis, so horizontal plane is X-Z

    # Calculate vector to goal
    dx = goal_x - x
    dy = goal_y - z
    dist = math.hypot(dx, dy)

    if dist < 0.05:
        vx = vy = omega = 0.0
    else:
        vx = 1.0 * dx
        vy = 1.0 * dy
        omega = 0.0

    # Obstacle avoidance with repulsive velocity
    for ox, oy in obstacles:
        ox_dist = ox - x
        oy_dist = oy - z
        obs_distance = math.hypot(ox_dist, oy_dist)
        if obs_distance < avoid_radius:
            repulsion = avoid_strength * (1.0/obs_distance - 1.0/avoid_radius)
            repulsion = max(repulsion, 0)
            angle = math.atan2(oy_dist, ox_dist)
            vx -= repulsion * math.cos(angle)
            vy -= repulsion * math.sin(angle)

    # Calculate wheel speeds and apply
    speeds = compute_wheel_speeds(vx, vy, omega)
    for i in range(4):
        speed = max(-MAX_SPEED, min(MAX_SPEED, speeds[i]))
        wheels[i].setVelocity(speed)
