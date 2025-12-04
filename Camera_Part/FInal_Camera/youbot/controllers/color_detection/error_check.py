import numpy as np

# === STEP 1: Calibration Data ===

# Pixel coordinates (from left camera image)
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

# Ground truth Webots world coordinates (Xw, Yw)
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

colors = ["Red", "Red 2", "Green", "Blue", "LightBlue", "Pink", "Brown", "Yellow"]

# === STEP 2: Fit Affine Transform ===

N = pixel_points.shape[0]
X = np.hstack((pixel_points, np.ones((N, 1))))  # Add column of ones for affine

# Solve for affine matrix A (2x3)
A, _, _, _ = np.linalg.lstsq(X, true_world_coords, rcond=None)
A = A.T  # So that A @ [lx, ly, 1] gives [Xw, Yw]

# === STEP 3: Define Mapping Function ===

def opencv_to_webots_affine(lx, ly):
    pixel = np.array([lx, ly, 1])
    return A @ pixel  # Returns (Xw, Yw)

# === STEP 4: Evaluate and Report Errors ===

print("📊 Affine Transform Error Report:\n")
total_error = 0

for i in range(N):
    lx, ly = pixel_points[i]
    x_true, y_true = true_world_coords[i]

    x_pred, y_pred = opencv_to_webots_affine(lx, ly)

    x_error = abs(x_pred - x_true)
    y_error = abs(y_pred - y_true)
    euc_error = np.sqrt(x_error**2 + y_error**2)
    total_error += euc_error

    print(f"🔹 {colors[i]:10} | X error: {x_error:.4f} m | Y error: {y_error:.4f} m | Euclidean error: {euc_error:.4f} m")

mean_error = total_error / N
print(f"\n📌 Mean Euclidean Error (Affine): {mean_error:.4f} m")
