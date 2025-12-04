# ---------------- YouBot Alignment + A* Path Planning ----------------
from controller import Supervisor
import math
import heapq

# ---------------- Configuration ----------------
TARGET_POINT = [0.98, 0, 0.00225]  # [X, Y, Z]
ANGLE_TOL = 0.04          # rad (~2.3 deg)
KP_YAW = 3.0
OMEGA_MAX = 1.5
OMEGA_MIN = 0.15
STABLE_STEPS = 6

# YouBot geometry & wheel limits
WHEEL_RADIUS = 0.05
HALF_L = 0.235
HALF_W = 0.15
MAX_WHEEL_VEL = 14.81

# Grid settings
FLOOR_SIZE = 5.0
GRID_RES = 0.5
GRID_SIZE = int(FLOOR_SIZE / GRID_RES)

# Manually set obstacles (x, y) in meters
obstacles = [
    [0,0]
]

# ---------------- YouBot Setup ----------------
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

node = robot.getFromDef("YOUBOT")
if node is None:
    print("ERROR: DEF YOUROBOT not found.")
    raise SystemExit

translation_field = node.getField("translation")
rotation_field = node.getField("rotation")

wheel_names = ["wheel1", "wheel2", "wheel3", "wheel4"]
wheels = []
for n in wheel_names:
    m = robot.getDevice(n)
    if m is None:
        print(f"ERROR: Device '{n}' not found.")
        raise SystemExit
    m.setPosition(float("inf"))
    m.setVelocity(0.0)
    wheels.append(m)

# ---------------- Utility Functions ----------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def set_body_yaw_rate(omega):
    r = WHEEL_RADIUS
    a = (HALF_L + HALF_W)
    w_fl = -(a * omega) / r
    w_fr = +(a * omega) / r
    w_rl = -(a * omega) / r
    w_rr = +(a * omega) / r
    wheels[0].setVelocity(clamp(w_fl, -MAX_WHEEL_VEL, MAX_WHEEL_VEL))
    wheels[1].setVelocity(clamp(w_fr, -MAX_WHEEL_VEL, MAX_WHEEL_VEL))
    wheels[2].setVelocity(clamp(w_rl, -MAX_WHEEL_VEL, MAX_WHEEL_VEL))
    wheels[3].setVelocity(clamp(w_rr, -MAX_WHEEL_VEL, MAX_WHEEL_VEL))

def stop_wheels():
    for m in wheels:
        m.setVelocity(0.0)

def axis_angle_to_R(ax, ay, az, ang):
    n = math.sqrt(ax*ax + ay*ay + az*az)
    if n < 1e-9:
        return [[1,0,0],[0,1,0],[0,0,1]]
    x, y, z = ax/n, ay/n, az/n
    c = math.cos(ang)
    s = math.sin(ang)
    C = 1 - c
    return [
        [c + x*x*C, x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C, y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C]
    ]

def mat_vec(R, v):
    return [
        R[0][0]*v[0] + R[0][1]*v[1] + R[0][2]*v[2],
        R[1][0]*v[0] + R[1][1]*v[1] + R[1][2]*v[2],
        R[2][0]*v[0] + R[2][1]*v[1] + R[2][2]*v[2],
    ]

def wrap_pi(a):
    return (a + math.pi) % (2*math.pi) - math.pi

def detect_world_up_index(R):
    local_up = [0,0,1]
    up_world = mat_vec(R, local_up)
    comps = [abs(up_world[0]), abs(up_world[1]), abs(up_world[2])]
    return comps.index(max(comps))

def current_yaw_and_target_angle():
    p = translation_field.getSFVec3f()
    rot = rotation_field.getSFRotation()
    R = axis_angle_to_R(rot[0], rot[1], rot[2], rot[3])
    up_idx = detect_world_up_index(R)
    ground_axes = [0,1,2]; ground_axes.remove(up_idx)
    g0, g1 = ground_axes
    f_world = mat_vec(R, [1,0,0])
    cur_yaw = math.atan2(f_world[g1], f_world[g0])
    tgt = list(TARGET_POINT)
    if tgt[up_idx] is None: tgt[up_idx] = p[up_idx]
    d0 = tgt[g0] - p[g0]
    d1 = tgt[g1] - p[g1]
    tgt_ang = math.atan2(d1, d0)
    return cur_yaw, tgt_ang

# ---------------- Grid Functions ----------------
def real_to_grid(pos):
    gx = int(math.floor((pos[0] + FLOOR_SIZE/2)/GRID_RES))
    gy = int(math.floor((pos[1] + FLOOR_SIZE/2)/GRID_RES))
    gx = max(0, min(GRID_SIZE-1, gx))
    gy = max(0, min(GRID_SIZE-1, gy))
    return gy, gx  # row, col

def generate_grid(obstacles):
    grid = [[1 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    for ob in obstacles:
        row, col = real_to_grid(ob)
        grid[row][col] = 0
    return grid

def print_grid(grid):
    print("Grid (0 = blocked, 1 = free):")
    for row in grid:
        print(row)

# ---------------- A* Classes & Functions ----------------
class Cell:
    def __init__(self):
        self.parent_i = 0
        self.parent_j = 0
        self.f = float('inf')
        self.g = float('inf')
        self.h = 0

def is_valid(row, col):
    return 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE

def is_unblocked(grid, row, col):
    return grid[row][col] == 1

def is_destination(row, col, dest):
    return row == dest[0] and col == dest[1]

def calculate_h_value(row, col, dest):
    return ((row - dest[0])**2 + (col - dest[1])**2)**0.5

def trace_path(cell_details, dest):
    print("Path:")
    path = []
    row, col = dest
    while not (cell_details[row][col].parent_i == row and cell_details[row][col].parent_j == col):
        path.append((row, col))
        row, col = cell_details[row][col].parent_i, cell_details[row][col].parent_j
    path.append((row, col))
    path.reverse()
    for i in path: print("->", i, end=" ")
    print()

def a_star_search(grid, src, dest):
    if not is_valid(*src) or not is_valid(*dest):
        print("Source or destination invalid"); return
    if not is_unblocked(grid, *src) or not is_unblocked(grid, *dest):
        print("Source or destination blocked"); return
    if is_destination(*src, dest):
        print("Already at destination"); return

    closed_list = [[False]*GRID_SIZE for _ in range(GRID_SIZE)]
    cell_details = [[Cell() for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    i,j = src
    cell_details[i][j].f = 0; cell_details[i][j].g = 0; cell_details[i][j].h = 0
    cell_details[i][j].parent_i = i; cell_details[i][j].parent_j = j

    open_list = []
    heapq.heappush(open_list, (0.0, i, j))
    found_dest = False

    while open_list:
        p = heapq.heappop(open_list)
        i,j = p[1], p[2]
        closed_list[i][j] = True
        directions = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
        for dir in directions:
            new_i, new_j = i + dir[0], j + dir[1]
            if is_valid(new_i,new_j) and is_unblocked(grid,new_i,new_j) and not closed_list[new_i][new_j]:
                if is_destination(new_i,new_j,dest):
                    cell_details[new_i][new_j].parent_i = i
                    cell_details[new_i][new_j].parent_j = j
                    print("Destination found")
                    trace_path(cell_details, dest)
                    return
                g_new = cell_details[i][j].g + 1.0
                h_new = calculate_h_value(new_i,new_j,dest)
                f_new = g_new + h_new
                if cell_details[new_i][new_j].f == float('inf') or cell_details[new_i][new_j].f > f_new:
                    heapq.heappush(open_list, (f_new,new_i,new_j))
                    cell_details[new_i][new_j].f = f_new
                    cell_details[new_i][new_j].g = g_new
                    cell_details[new_i][new_j].h = h_new
                    cell_details[new_i][new_j].parent_i = i
                    cell_details[new_i][new_j].parent_j = j
    print("Failed to find destination")

# ---------------- Main Loop ----------------
aligned_printed = False
stable = 0

while robot.step(timestep) != -1:
    cur_yaw, tgt_ang = current_yaw_and_target_angle()
    err = wrap_pi(tgt_ang - cur_yaw)
    if abs(err) <= ANGLE_TOL:
        stable += 1
        set_body_yaw_rate(0.0)
        if not aligned_printed:
            print("Alignment correct")
            aligned_printed = True
        if stable >= STABLE_STEPS:
            print("Ready for Path Planning")
            stop_wheels()
            break
    else:
        aligned_printed = False
        stable = 0
        omega_cmd = clamp(KP_YAW * abs(err), OMEGA_MIN, OMEGA_MAX)
        set_body_yaw_rate(omega_cmd)

# ---------------- After Alignment: Path Planning ----------------
grid = generate_grid(obstacles)
print_grid(grid)

robot_pos = translation_field.getSFVec3f()
src = real_to_grid([robot_pos[0], robot_pos[1]])
dest = real_to_grid([TARGET_POINT[0], TARGET_POINT[1]])

print(f"Robot initial cell: {src}")
print(f"Target cell: {dest}")

a_star_search(grid, src, dest)
