from controller import Robot, Camera
import numpy as np
import cv2
import math
import csv

# === STEP 1: Calibration & Affine Transform ===

# Calibration pixel points (left camera)
pixel_points = np.array([
    [306, 240],  # Red
    [306, 143],  # Red 2
    [222, 242],  # Green
    [308, 333],  # Blue
    [446, 331],  # LightBlue
    [439, 156],  # Pink
    [453, 240],  # Brown
    [306, 47],  # Yellow
])

# Corresponding Webots world coordinates (ground truth)
true_world_coords = np.array([
     [0.000, 0.000],     # Red
    [0.370, 0.000],     # Red 2
    [-0.010, 0.330],    # Green
    [-0.39, 0.00],    # Blue
    [-0.37, -0.58],     # LightBlue
    [0.34, -0.53],    # Pink
    [0.00, -0.6],     # Brown
    [0.78, 0.00],     # Yellow
])

# Fit 2D affine transform: A @ [lx, ly, 1] = [Xw, Yw]
X = np.hstack((pixel_points, np.ones((pixel_points.shape[0], 1))))
A, _, _, _ = np.linalg.lstsq(X, true_world_coords, rcond=None)
A = A.T  # Shape: (2, 3)

def opencv_to_webots_affine(lx, ly):
    pixel = np.array([lx, ly, 1])
    return A @ pixel  # → (Xw, Yw)

# === INIT ===
robot = Robot()
timestep = int(robot.getBasicTimeStep())

left_cam = robot.getDevice("left_camera")
right_cam = robot.getDevice("right_camera")
left_cam.enable(timestep)
right_cam.enable(timestep)

width = left_cam.getWidth()
height = left_cam.getHeight()
fov = left_cam.getFov()
f = (width / 2) / math.tan(fov / 2)  # Focal length in pixels
B = 0.1  # Baseline in meters
camera_height = 1.5  # meters from floor

# === Utility Functions ===
def get_camera_image(camera):
    img = camera.getImage()
    img_np = np.frombuffer(img, np.uint8).reshape((height, width, 4))
    return img_np[:, :, :3]

def align_images(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    if des1 is None or des2 is None:
        return img2, 0
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:10]
    if len(matches) < 5:
        return img2, 0
    y_shifts = [kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1] for m in matches]
    avg_shift = np.mean(y_shifts)
    M = np.float32([[1, 0, 0], [0, 1, avg_shift]])
    aligned_img = cv2.warpAffine(img2, M, (img2.shape[1], img2.shape[0]))
    return aligned_img, avg_shift

color_ranges = {
    "Red":       [(np.array([0, 100, 100]), np.array([10, 255, 255]))],
    "Green":     [(np.array([40, 50, 50]), np.array([80, 255, 255]))],
    "Blue":      [(np.array([100, 100, 100]), np.array([140, 255, 255]))],
    "LightBlue": [(np.array([85, 50, 100]), np.array([105, 255, 255]))],
    "Pink":      [(np.array([145, 60, 100]), np.array([170, 255, 255]))],
    "Brown":     [(np.array([10, 120, 50]), np.array([20, 255, 200]))],
    "Yellow":    [(np.array([20, 150, 150]), np.array([35, 255, 255]))],
    "Purple":    [(np.array([135, 150, 60]), np.array([145, 255, 180]))],        
}

def detect_colors(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    centers = []
    for color_name, ranges in color_ranges.items():
        for lower, upper in ranges:
            mask = cv2.inRange(hsv, lower, upper)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 50:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cx, cy = x + w // 2, y + h // 2
                    centers.append((color_name, cx, cy))
    return centers

# === MAIN LOOP ===
detected_points = []
while robot.step(timestep) != -1:
    left_img = get_camera_image(left_cam)
    right_img = get_camera_image(right_cam)
    right_img_aligned, shift = align_images(left_img, right_img)

    print(f"\n[INFO] Detected vertical Y-shift: {shift:.2f} pixels")
    left_centers = detect_colors(left_img)
    right_centers = detect_colors(right_img_aligned)
    print("🎯 Detected Object Coordinates:")

    cx = width / 2
    cy = height / 2
    display_img = left_img.copy()

    for color in color_ranges.keys():
        l = [pt for pt in left_centers if pt[0] == color]
        r = [pt for pt in right_centers if pt[0] == color]
    
        if color == "Green" and l and r:
            green_candidates = []
    
            for lpt, rpt in zip(l, r):
                lx, ly = lpt[1], lpt[2]
                rx, ry = rpt[1], rpt[2]
                disparity = abs(lx - rx)
                if disparity != 0:
                    Z_raw = (f * B) / disparity
                    Z = 0.2510 * Z_raw + 0.9486
                    X = (lx - cx) * Z / f
                    Y = (ly - cy) * Z / f
                    Xw, Yw = opencv_to_webots_affine(lx, ly)
                    Z_floor = camera_height - Z
    
                    green_candidates.append({
                        "lx": lx, "ly": ly, "rx": rx, "ry": ry,
                        "Xw": Xw, "Yw": Yw, "Z": Z, "Z_floor": Z_floor,
                        "Z_raw": Z_raw
                    })
    
            if len(green_candidates) == 2:
                g1, g2 = green_candidates
    
                # Heuristic: gripper is closer to camera → higher Z_floor
                if g1["Z_floor"] > g2["Z_floor"]:
                    gripper, green_box = g1, g2
                else:
                    gripper, green_box = g2, g1
    
                for label, obj in [("Gripper", gripper), ("GreenBox", green_box)]:
                    lx, ly = obj["lx"], obj["ly"]
                    Xw, Yw, Z = obj["Xw"], obj["Yw"], obj["Z"]
                    label_text = f"{label} ({Xw:.2f}, {Yw:.2f}, {Z:.2f})"
                    cv2.circle(display_img, (lx, ly), 10, (0, 255, 0), -1)
                    cv2.putText(display_img, label_text, (lx + 10, ly - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    print(f"📍 {label}: Xw={Xw:.3f}, Yw={Yw:.3f}, Z={Z:.3f}, Height from Floor={obj['Z_floor']:.3f} m")
    
            elif len(green_candidates) == 1:
                obj = green_candidates[0]
                lx, ly = obj["lx"], obj["ly"]
                Xw, Yw, Z = obj["Xw"], obj["Yw"], obj["Z"]
                label_text = f"Green? ({Xw:.2f}, {Yw:.2f}, {Z:.2f})"
                cv2.circle(display_img, (lx, ly), 10, (0, 255, 255), -1)
                cv2.putText(display_img, label_text, (lx + 10, ly - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                print(f"⚠️ One green object found: Xw={Xw:.3f}, Yw={Yw:.3f}, Z={Z:.3f}")
            else:
                print("❌ No valid green detections")
    
        elif l and r:
            for lpt, rpt in zip(l, r):
                lx, ly = lpt[1], lpt[2]
                rx, ry = rpt[1], rpt[2]
                disparity = abs(lx - rx)
                if disparity != 0:
                    Z_raw = (f * B) / disparity
                    Z = 0.2510 * Z_raw + 0.9486
                    X = (lx - cx) * Z / f
                    Y = (ly - cy) * Z / f
                    Xw, Yw = opencv_to_webots_affine(lx, ly)
                    Z_floor = camera_height - Z
    
                    detected_points.append((color, lx, ly, rx, ry, Xw, Yw))
    
                    print(f" - {color}: L=({lx},{ly}), R=({rx},{ry}), Δx={disparity}, "
                          f"Camera XY: ({X:.3f}, {Y:.3f}), Depth Z={Z:.3f} m → "
                          f"Webots 3D: Xw={Xw:.3f}, Yw={Yw:.3f}, Zw={Z:.3f}, Height from Floor={Z_floor:.3f} m")
    
                    label = f"{color} ({Xw:.2f},{Yw:.2f},{Z:.2f})m"
                    cv2.circle(display_img, (lx, ly), 10, (0, 255, 0), -1)
                    cv2.putText(display_img, label, (lx + 10, ly - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    print(f" - {color}: Disparity is zero — cannot compute depth.")
        else:
            print(f" - {color}: Not detected in both images")

    cv2.imshow("Detected Colors and Webots Coordinates", display_img)
    if cv2.waitKey(1) == 27:  # ESC
        break

# Save detected points to CSV
with open("detected_points.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Color", "lx", "ly", "rx", "ry", "Xw", "Yw"])
    for row in detected_points:
        writer.writerow(row)

cv2.destroyAllWindows()
