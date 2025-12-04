from controller import Robot
import math
import numpy as np
import cv2

# === Init Robot ===
robot = Robot()
TIME_STEP = int(robot.getBasicTimeStep())

# === Initialize Cameras ===
left_cam = robot.getDevice("left_camera")
right_cam = robot.getDevice("right_camera")
left_cam.enable(TIME_STEP)
right_cam.enable(TIME_STEP)

width = left_cam.getWidth()
height = left_cam.getHeight()
fov = left_cam.getFov()
f = (width / 2) / math.tan(fov / 2)
B = 0.1  # Baseline (distance between cameras)

# === Camera Utility ===
def get_camera_image(camera):
    img = camera.getImage()
    img_np = np.frombuffer(img, np.uint8).reshape((height, width, 4))
    return img_np[:, :, :3]

def detect_red_center(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 100, 100])
    upper = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 300:
            x, y, w, h = cv2.boundingRect(largest)
            cx = x + w // 2
            cy = y + h // 2
            return (cx, cy)
    return None

# === PID Motion Setup ===
wheel_names = ["wheel1", "wheel2", "wheel3", "wheel4"]
wheels = []
for name in wheel_names:
    motor = robot.getDevice(name)
    motor.setPosition(float('inf'))
    motor.setVelocity(0.0)
    wheels.append(motor)

WHEEL_RADIUS = 0.05
HALF_WHEELBASE_LENGTH = 0.235
HALF_WHEELBASE_WIDTH = 0.15
MAX_SPEED = 6.28

def compute_wheel_velocities(vx, vy, omega):
    l = HALF_WHEELBASE_LENGTH
    w = HALF_WHEELBASE_WIDTH
    r = WHEEL_RADIUS
    return [
        (vx + vy + (l + w) * omega) / r,  # front left
        (vx - vy - (l + w) * omega) / r,  # front right
        (vx - vy + (l + w) * omega) / r,  # rear left
        (vx + vy - (l + w) * omega) / r   # rear right
    ]

# === PID Controller Variables ===
Kp = 2.0
Ki = 0.01
Kd = 0.3
integral = [0.0, 0.0]
prev_error = [0.0, 0.0]

target_found = False
target_pos = None

# === Main Loop ===
while robot.step(TIME_STEP) != -1:
    left_img = get_camera_image(left_cam)
    right_img = get_camera_image(right_cam)

    l_center = detect_red_center(left_img)
    r_center = detect_red_center(right_img)

    if l_center and r_center:
        lx, ly = l_center
        rx, ry = r_center
        disparity = abs(lx - rx)

        if disparity != 0:
            Z = (f * B) / disparity
            cx = width / 2
            cy = height / 2
            X = (lx - cx) * Z / f
            Y = (ly - cy) * Z / f

            target_pos = (X, Y)
            target_found = True
            print(f"[INFO] Red object position in robot frame: ({X:.2f}, {Y:.2f})")

    # === PID Navigation ===
    if target_found and target_pos:
        error = [target_pos[0], target_pos[1]]
        distance = math.hypot(*error)
        if distance < 0.1:
            for motor in wheels:
                motor.setVelocity(0.0)
            print("[INFO] Target reached.")
            break

        # PID calculations
        integral[0] += error[0] * TIME_STEP / 1000.0
        integral[1] += error[1] * TIME_STEP / 1000.0
        derivative = [
            (error[0] - prev_error[0]) / (TIME_STEP / 1000.0),
            (error[1] - prev_error[1]) / (TIME_STEP / 1000.0)
        ]
        prev_error = error[:]

        vx = Kp * error[0] + Ki * integral[0] + Kd * derivative[0]
        vy = Kp * error[1] + Ki * integral[1] + Kd * derivative[1]
        omega = 0  # Optional: you can include orientation control

        speeds = compute_wheel_velocities(vx, vy, omega)
        speeds = [max(min(s, MAX_SPEED), -MAX_SPEED) for s in speeds]

        for i in range(4):
            wheels[i].setVelocity(speeds[i])
