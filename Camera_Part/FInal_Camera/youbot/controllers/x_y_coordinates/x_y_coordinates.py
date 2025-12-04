```python
from controller import Robot, Camera, Display
import cv2
import numpy as np

# Initialize the robot
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Initialize cameras (assuming 'camera_left' and 'camera_right' in youBot)
try:
    left_camera = robot.getDevice("camera_left")
    right_camera = robot.getDevice("camera_right")
    left_camera.enable(timestep)
    right_camera.enable(timestep)
except AttributeError as e:
    print(f"Error: Camera not found. Check camera names in the world file. Error: {e}")
    exit(1)

# Verify camera initialization
if not left_camera or not right_camera:
    print("Error: One or both cameras could not be initialized. Check world file.")
    exit(1)

# Initialize display for side-by-side visualization
try:
    display = robot.getDevice("display")
    display_width = left_camera.getWidth() * 2  # Double width for side-by-side
    display_height = left_camera.getHeight()
    display.setOpacity(1.0)
except AttributeError:
    print("Error: Display node not found. Ensure 'display' is added to the youBot.")
    exit(1)

# Color ranges for detection (in HSV)
color_ranges = {
    "Red":       [(np.array([0, 100, 100]), np.array([10, 255, 255]))],
    "Green":     [(np.array([40, 50, 50]), np.array([80, 255, 255]))],
    "Blue":      [(np.array([100, 100, 100]), np.array([140, 255, 255]))],
    "LightBlue": [(np.array([85, 50, 100]), np.array([105, 255, 255]))],
    "Pink":      [(np.array([140, 50, 100]), np.array([170, 255, 255]))],
    "Brown":     [(np.array([10, 50, 20]), np.array([20, 255, 200]))],
    "Yellow":    [(np.array([20, 100, 100]), np.array([30, 255, 255]))]
}

# BGR colors for annotations
color_bgr = {
    "Red": (0, 0, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0),
    "LightBlue": (255, 255, 0),
    "Pink": (255, 0, 255),
    "Brown": (42, 42, 165),
    "Yellow": (0, 255, 255)
}

# Function to align right image to left image using feature matching
def align_images(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None:
        print("Warning: No features detected in one or both images.")
        return img2, 0.0
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    if len(matches) < 10:
        print(f"Warning: Only {len(matches)} matches found, using minimal alignment.")
        return img2, 0.0
    
    y_shifts = [kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1] for m in matches[:10]]
    avg_shift = np.mean(y_shifts)
    
    M = np.float32([[1, 0, 0], [0, 1, avg_shift]])
    aligned_img = cv2.warpAffine(img2, M, (img2.shape[1], img2.shape[0]))
    return aligned_img, avg_shift

# Function to detect and draw color objects
def detect_and_draw(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    centers = []

    for color_name, ranges in color_ranges.items():
        for lower, upper in ranges:
            mask = cv2.inRange(hsv, lower, upper)
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 500:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cx, cy = x + w // 2, y + h // 2
                    centers.append((color_name, cx, cy))
                    cv2.rectangle(image, (x, y), (x + w, y + h), color_bgr[color_name], 2)
                    cv2.circle(image, (cx, cy), 5, (255, 255, 255), -1)
                    cv2.putText(image, f"{color_name} ({cx}, {cy})", (x, y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr[color_name], 1)
    return centers, image

# Main loop
while robot.step(timestep) != -1:
    # Get images from cameras
    left_data = left_camera.getImage()
    right_data = right_camera.getImage()
    
    if left_data is None or right_data is None:
        print("Error: Failed to retrieve image from one or both cameras.")
        continue
    
    # Convert Webots image data to OpenCV format (BGRA to BGR)
    try:
        left_img = np.frombuffer(left_data, np.uint8).reshape((left_camera.getHeight(), left_camera.getWidth(), 4))
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGRA2BGR)
        right_img = np.frombuffer(right_data, np.uint8).reshape((right_camera.getHeight(), right_camera.getWidth(), 4))
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGRA2BGR)
    except ValueError as e:
        print(f"Error: Failed to process camera images. Check resolution. Error: {e}")
        continue
    
    # Align right image to left
    right_img_aligned, detected_shift = align_images(left_img, right_img)
    print(f"Detected Y-shift: {detected_shift:.2f} pixels")
    
    # Detect and annotate objects
    left_centers, left_annotated = detect_and_draw(left_img)
    right_centers, right_annotated = detect_and_draw(right_img_aligned)
    
    # Convert annotated images to BGRA for Webots display
    left_annotated_bgra = cv2.cvtColor(left_annotated, cv2.COLOR_BGR2BGRA)
    right_annotated_bgra = cv2.cvtColor(right_annotated, cv2.COLOR_BGR2BGRA)
    
    # Display images side by side
    side_by_side = np.zeros((display_height, display_width, 4), dtype=np.uint8)
    side_by_side[:, :left_camera.getWidth(), :] = left_annotated_bgra
    side_by_side[:, left_camera.getWidth():, :] = right_annotated_bgra
    
    try:
        image_ref = display.imageNew(side_by_side.tobytes(), Display.BGRA)
        display.imagePaste(image_ref, 0, 0)
        display.imageDelete(image_ref)
    except Exception as e:
        print(f"Error: Failed to display images. Error: {e}")
        continue
    
    # Print Y-coordinate differences
    print("\n🔍 Y-Coordinate Differences (after alignment):\n")
    for color in color_ranges.keys():
        l = [pt for pt in left_centers if pt[0] == color]
        r = [pt for pt in right_centers if pt[0] == color]
        if l and r:
            for lpt, rpt in zip(l, r):
                y_diff = abs(lpt[2] - rpt[2])
                print(f"{color}: Left Y = {lpt[2]}, Right Y = {rpt[2]}, |ΔY| = {y_diff:.1f}")
        else:
            print(f"{color}: Not detected in both images")