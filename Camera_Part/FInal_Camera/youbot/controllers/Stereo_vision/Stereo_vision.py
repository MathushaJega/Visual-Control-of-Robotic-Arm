from controller import Robot, Camera
import numpy as np
import cv2
import time

# Initialize robot
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Camera setup
left_cam = robot.getDevice("left_camera")
right_cam = robot.getDevice("right_camera")
left_cam.enable(timestep)
right_cam.enable(timestep)

# Camera parameters (tuned for your setup)
BASELINE = 0.2  # Measured distance between cameras in meters
FOCAL_LENGTH = 205.47  # Your calculated focal length in pixels
WIDTH = left_cam.getWidth()
HEIGHT = left_cam.getHeight()

# Stereo matcher configuration (optimized parameters)
stereo = cv2.StereoBM_create(
    numDisparities=96,    # 64 works well for 71° FOV (divisible by 16)
    blockSize=21          # Medium smoothing for general use
)

# Object detection setup (using OpenCV's simple blob detector)
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 100  # Adjust based on your scene
params.maxArea = 10000
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False
detector = cv2.SimpleBlobDetector_create(params)

def get_cv_image(webots_image):
    """Convert Webots image to writable OpenCV format"""
    img = np.frombuffer(webots_image, np.uint8).reshape((HEIGHT, WIDTH, 4))
    return np.ascontiguousarray(img[:, :, :3])  # Makes array writable and removes alpha

def compute_depth(left_img, right_img):
    """Full depth calculation pipeline"""
    # Convert to grayscale
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    
    # Compute disparity
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    
    # Calculate depth (in meters)
    depth = np.zeros_like(disparity)
    valid_pixels = disparity > 0
    depth[valid_pixels] = (BASELINE * FOCAL_LENGTH) / disparity[valid_pixels]
    
    return depth, disparity

def detect_objects(image, depth_map):
    """Detect objects and return their positions and depths"""
    # Convert to grayscale for blob detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect blobs (objects)
    keypoints = detector.detect(gray)
    
    object_data = []
    
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        size = int(kp.size)
        
        # Get depth in a region around the keypoint
        x1, x2 = max(0, x-size), min(WIDTH, x+size)
        y1, y2 = max(0, y-size), min(HEIGHT, y+size)
        region = depth_map[y1:y2, x1:x2]
        
        # Calculate median depth to avoid outliers
        valid_depths = region[region > 0]
        if len(valid_depths) > 0:
            median_depth = np.median(valid_depths)
            object_data.append({
                'position': (x, y),
                'size': size,
                'depth': median_depth
            })
    
    return object_data

def draw_object_info(image, objects):
    """Draw bounding boxes and depth information on the image"""
    for obj in objects:
        x, y = obj['position']
        size = obj['size']
        depth = obj['depth']
        
        # Draw bounding circle
        cv2.circle(image, (x, y), size, (0, 255, 0), 2)
        
        # Draw depth text
        cv2.putText(image, f"{depth:.2f}m", (x - 20, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # Display object count
    cv2.putText(image, f"Objects: {len(objects)}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Main loop
try:
    while robot.step(timestep) != -1:
        # Get and convert images
        left_img = get_cv_image(left_cam.getImage())
        right_img = get_cv_image(right_cam.getImage())
        
        # Compute depth
        depth_map, disparity_map = compute_depth(left_img, right_img)
        
        # Detect objects and their depths
        objects = detect_objects(left_img, depth_map)
        
        # Create display image with object info
        display_img = left_img.copy()
        draw_object_info(display_img, objects)
        
        # Add general info overlay
        cv2.putText(display_img, f"Focal: {FOCAL_LENGTH:.1f}px", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_img, f"Baseline: {BASELINE}m", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_img, f"Time: {robot.getTime():.1f}s", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Visualization (normalized)
        disp_vis = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # Display results
        cv2.imshow('Left Camera with Object Depths', display_img)
        cv2.imshow('Disparity', disp_vis)
        cv2.imshow('Depth (m)', depth_vis)
        
        if cv2.waitKey(1) == 27:  # Exit on ESC
            break

except Exception as e:
    print(f"Error: {str(e)}")
finally:
    cv2.destroyAllWindows()
    print("Controller stopped")