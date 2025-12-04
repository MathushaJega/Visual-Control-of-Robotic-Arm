from controller import Robot, Camera
import numpy as np
import cv2
import time
import math

# Initialize robot
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Camera setup
left_cam = robot.getDevice("left_camera")
right_cam = robot.getDevice("right_camera")
left_cam.enable(timestep)
right_cam.enable(timestep)

# Camera parameters
WIDTH = left_cam.getWidth()
HEIGHT = left_cam.getHeight()
FOV = np.deg2rad(68)
BASELINE = 0.2
FOCAL_LENGTH = WIDTH / (2 * np.tan(FOV / 2))

# Stereo matcher
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
stereo.setPreFilterType(1)
stereo.setPreFilterSize(5)
stereo.setPreFilterCap(31)
stereo.setTextureThreshold(10)
stereo.setUniquenessRatio(15)

# Blob detector parameters
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 100
params.maxArea = 10000
detector = cv2.SimpleBlobDetector_create(params)

def get_cv_image(webots_image):
    img = np.frombuffer(webots_image, np.uint8).reshape((HEIGHT, WIDTH, 4))
    return np.ascontiguousarray(img[:, :, :3])

def compute_depth(left_img, right_img):
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    depth = np.zeros_like(disparity)
    valid_pixels = disparity > 0
    depth[valid_pixels] = (BASELINE * FOCAL_LENGTH) / disparity[valid_pixels]
    return depth, disparity

def detect_objects(image, depth_map):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = image.copy()
    boxes = []
    vector_image = np.zeros_like(image)
    vector_text = ""

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            top_left = (x, y)
            boxes.append(top_left)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(output, top_left, 5, (0, 0, 255), -1)
            cv2.putText(output, f"({x},{y})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    if len(boxes) == 2:
        pt1, pt2 = boxes
        cv2.line(output, pt1, pt2, (255, 0, 0), 2)
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        distance = math.sqrt(dx**2 + dy**2)
        mid_x, mid_y = (pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2
        vector_text = f"dx={dx}, dy={dy}, dist={distance:.1f}"
        cv2.putText(output, vector_text, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    keypoints = detector.detect(gray)
    object_data = []

    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        size = int(kp.size)
        x1, x2 = max(0, x - size), min(WIDTH, x + size)
        y1, y2 = max(0, y - size), min(HEIGHT, y + size)
        region = depth_map[y1:y2, x1:x2]
        valid_depths = region[region > 0]

        if len(valid_depths) > 0:
            median_depth = np.median(valid_depths)
            object_data.append({
                'position': (x, y),
                'size': size,
                'depth': median_depth
            })

    return object_data, output, vector_image

def draw_object_info(image, objects):
    for obj in objects:
        x, y = obj['position']
        size = obj['size']
        depth = obj['depth']
        cv2.circle(image, (x, y), size, (0, 255, 0), 2)
        cv2.putText(image, f"{depth:.2f}m", (x - 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(image, f"Focal: {FOCAL_LENGTH:.1f}px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"Baseline: {BASELINE}m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"Objects: {len(objects)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Main loop
try:
    while robot.step(timestep) != -1:
        left_img = get_cv_image(left_cam.getImage())
        right_img = get_cv_image(right_cam.getImage())
        depth_map, disparity_map = compute_depth(left_img, right_img)
        objects, processed_img, vector_img = detect_objects(left_img, depth_map)
        draw_object_info(processed_img, objects)
        disp_vis = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imshow('Left Camera + Objects', processed_img)
        cv2.imshow('Disparity', disp_vis)
        cv2.imshow('Depth Map', depth_vis)
        cv2.imshow('Object Distance Vector', vector_img)
        if cv2.waitKey(1) == 27:
            break
except Exception as e:
    print(f"Error: {e}")
finally:
    cv2.destroyAllWindows()
    print("Controller stopped")
