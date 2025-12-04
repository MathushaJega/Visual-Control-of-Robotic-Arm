# rotate_and_go_with_obstacle_check_with_astar.py
from controller import Supervisor
import math, heapq
import numpy as np

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
TARGET_POINT = [-1.54554, -1.34145, 0.0199568]   # [X, Y, Z]
ANGLE_TOL = 0.04
KP_YAW    = 3.0
OMEGA_MAX = 1.5
OMEGA_MIN = 0.15
STABLE_STEPS = 6

# Forward motion
V_FORWARD = 0.15
STOP_DIST = 0.5
POS_TOL = 0.05

# YouBot geometry
WHEEL_RADIUS = 0.05
HALF_L = 0.24
HALF_W = 0.1 
MAX_WHEEL_VEL = 14.81

# Example obstacles (ground plane coords)
OBSTACLES = [
   
]

# Floor/Map setup
FLOOR_SIZE = 5.0       # meters (one side of the square floor)
GRID_RES   =0.5        # meters per cell
GRID_SIZE  = int(FLOOR_SIZE / GRID_RES)

# ------------------------------------------------------------
# Controller setup
# ------------------------------------------------------------
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

node = robot.getFromDef("YOUBOT")
if node is None:
    print("ERROR: DEF YOUBOT not found.")
    raise SystemExit

translation_field = node.getField("translation")
rotation_field    = node.getField("rotation")

wheel_names = ["wheel1", "wheel2", "wheel3", "wheel4"]
wheels = []
for n in wheel_names:
    m = robot.getDevice(n)
    m.setPosition(float("inf"))
    m.setVelocity(0.0)
    wheels.append(m)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def clamp(v, lo, hi): return max(lo, min(hi, v))

def set_body_velocity(vx, vy, omega):
    r = WHEEL_RADIUS
    a = (HALF_L + HALF_W)
    w_fl = (1/r) * (vx - vy - a*omega)
    w_fr = (1/r) * (vx + vy + a*omega)
    w_rl = (1/r) * (vx + vy - a*omega)
    w_rr = (1/r) * (vx - vy + a*omega)
    wheels[0].setVelocity(clamp(w_fl, -MAX_WHEEL_VEL, MAX_WHEEL_VEL))
    wheels[1].setVelocity(clamp(w_fr, -MAX_WHEEL_VEL, MAX_WHEEL_VEL))
    wheels[2].setVelocity(clamp(w_rl, -MAX_WHEEL_VEL, MAX_WHEEL_VEL))
    wheels[3].setVelocity(clamp(w_rr, -MAX_WHEEL_VEL, MAX_WHEEL_VEL))

def stop_wheels():
    for m in wheels: m.setVelocity(0.0)

def axis_angle_to_R(ax, ay, az, ang):
    n = math.sqrt(ax*ax + ay*ay + az*az)
    if n < 1e-9: return [[1,0,0],[0,1,0],[0,0,1]]
    x,y,z = ax/n, ay/n, az/n
    c,s = math.cos(ang), math.sin(ang); C=1-c
    return [
        [c+x*x*C, x*y*C-z*s, x*z*C+y*s],
        [y*x*C+z*s, c+y*y*C, y*z*C-x*s],
        [z*x*C-y*s, z*y*C+x*s, z*z*C+c]
    ]

def mat_vec(R, v):
    return [R[0][0]*v[0]+R[0][1]*v[1]+R[0][2]*v[2],
            R[1][0]*v[0]+R[1][1]*v[1]+R[1][2]*v[2],
            R[2][0]*v[0]+R[2][1]*v[1]+R[2][2]*v[2]]

def wrap_pi(a): return (a+math.pi)%(2*math.pi)-math.pi

def detect_world_up_index(R):
    local_up=[0,0,1]; up_world=mat_vec(R,local_up)
    comps=[abs(up_world[0]),abs(up_world[1]),abs(up_world[2])]
    return comps.index(max(comps))

def current_yaw_and_target_angle():
    p=translation_field.getSFVec3f()
    rot=rotation_field.getSFRotation()
    R=axis_angle_to_R(rot[0],rot[1],rot[2],rot[3])
    up_idx=detect_world_up_index(R)
    ground_axes=[0,1,2]; ground_axes.remove(up_idx)
    g0,g1=ground_axes
    f_world=mat_vec(R,[1,0,0])
    cur_yaw=math.atan2(f_world[g1],f_world[g0])
    tgt=list(TARGET_POINT)
    d0,d1=tgt[g0]-p[g0], tgt[g1]-p[g1]
    tgt_ang=math.atan2(d1,d0)
    dist=math.sqrt(d0**2+d1**2)
    return cur_yaw,tgt_ang,dist,g0,g1,p,tgt

# ---------- Geometry helpers ----------
def line_from_points(p1,p2):
    x1,y1=p1; x2,y2=p2
    A=y1-y2; B=x2-x1; C=x1*y2-x2*y1
    return A,B,C

def parallel_line(A,B,C,pt):
    x,y=pt; return A,B,-(A*x+B*y)

def intersect(L1,L2):
    A1,B1,C1=L1; A2,B2,C2=L2
    D=A1*B2-A2*B1
    if abs(D)<1e-9: return None
    return [(B1*C2-B2*C1)/D,(C1*A2-C2*A1)/D]

def point_in_poly(pt,poly):
    x,y=pt; inside=False; n=len(poly)
    for i in range(n):
        x1,y1=poly[i]; x2,y2=poly[(i+1)%n]
        if ((y1>y)!=(y2>y)) and (x<(x2-x1)*(y-y1)/(y2-y1+1e-9)+x1):
            inside=not inside
    return inside

def corridor_quad_for_segment(start, target):
    sx, sy = start
    tx, ty = target
    vx, vy = tx - sx, ty - sy
    L = math.hypot(vx, vy)
    if L < 1e-9: return None
    ux, uy = vx/L, vy/L
    nx, ny = -uy, ux
    left_front  = [sx + nx*HALF_W + ux*HALF_L, sy + ny*HALF_W + uy*HALF_L]
    right_front = [sx - nx*HALF_W + ux*HALF_L, sy - ny*HALF_W + uy*HALF_L]
    L_ct   = line_from_points([sx, sy], [tx, ty])
    L_left = parallel_line(*L_ct, left_front)
    L_right= parallel_line(*L_ct, right_front)
    L_front= line_from_points(left_front, right_front)
    L_tgt  = parallel_line(*L_front, [tx, ty])
    quad = [intersect(L_left, L_front),
            intersect(L_front, L_right),
            intersect(L_right, L_tgt),
            intersect(L_tgt, L_left)]
    if any(v is None for v in quad): return None
    return quad

# ------------------------------------------------------------
# A* Helpers
# ------------------------------------------------------------
def world_to_grid(x, y):
    gx = int((x + FLOOR_SIZE/2) / GRID_RES)
    gy = int((y + FLOOR_SIZE/2) / GRID_RES)
    return gx, gy

def grid_to_world(gx, gy):
    x = gx * GRID_RES - FLOOR_SIZE/2 + GRID_RES/2
    y = gy * GRID_RES - FLOOR_SIZE/2 + GRID_RES/2
    return x, y

def build_grid(obstacles):
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for ox, oy in obstacles:
        gx, gy = world_to_grid(ox, oy)
        if 0 <= gx < grid.shape[0] and 0 <= gy < grid.shape[1]:
            grid[gx, gy] = 1
    return grid

def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def a_star(grid, start, goal):
    open_set = []
    heapq.heappush(open_set, (0+heuristic(start, goal), 0, start, None))
    came_from = {}
    g_score = {start: 0}
    while open_set:
        _, cost, current, parent = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while parent:
                path.append(parent)
                parent = came_from[parent]
            return path[::-1]
        if current in came_from: continue
        came_from[current] = parent
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            neigh = (current[0]+dx, current[1]+dy)
            if not (0 <= neigh[0] < grid.shape[0] and 0 <= neigh[1] < grid.shape[1]): continue
            if grid[neigh[0], neigh[1]] == 1: continue
            new_cost = cost + 1
            if neigh not in g_score or new_cost < g_score[neigh]:
                g_score[neigh] = new_cost
                heapq.heappush(open_set, (new_cost+heuristic(neigh, goal), new_cost, neigh, current))
    return None

# ------------------------------------------------------------
# Main loop (A* path planning for entire path)
# ------------------------------------------------------------
stable=0
phase="rotate"
waypoints=[]
wp_index=0

while robot.step(timestep)!=-1:
    cur_yaw,tgt_ang,dist,g0,g1,p,tgt=current_yaw_and_target_angle()
    err=wrap_pi(tgt_ang-cur_yaw)
    robot_x, robot_y = p[g0], p[g1]

    if phase=="rotate":
        if abs(err)<=ANGLE_TOL:
            stable+=1
            set_body_velocity(0,0,0)
            if stable>=STABLE_STEPS:
                # === Always use A* ===
                grid = build_grid(OBSTACLES)
                start = world_to_grid(robot_x, robot_y)
                goal  = world_to_grid(tgt[g0], tgt[g1])
                path = a_star(grid, start, goal)
                if path is None:
                    print("No path found by A* → stopping robot")
                    stop_wheels()
                    break
                waypoints = [grid_to_world(px, py) for px, py in path]

                # Adjust final waypoint to STOP_DIST before target
                final_wp_x, final_wp_y = waypoints[-1]
                dx_t, dy_t = tgt[g0]-final_wp_x, tgt[g1]-final_wp_y
                d_t = math.hypot(dx_t, dy_t)
                if d_t > STOP_DIST:
                    ratio = (d_t - STOP_DIST) / d_t
                    final_wp_x += dx_t * ratio
                    final_wp_y += dy_t * ratio
                    waypoints[-1] = [final_wp_x, final_wp_y]

                print("Planned path with STOP_DIST:", waypoints)
                phase = "follow_path"
                wp_index = 0
        else:
            stable=0
            omega_cmd=clamp(KP_YAW*abs(err),OMEGA_MIN,OMEGA_MAX)
            set_body_velocity(0,0,omega_cmd)

    elif phase=="follow_path":
        if wp_index >= len(waypoints):
            # Final approach
            dx, dy = tgt[g0]-robot_x, tgt[g1]-robot_y
            dist_to_target = math.hypot(dx, dy)
            if dist_to_target <= STOP_DIST:
                print(f"Reached safe distance ({STOP_DIST} m) from target")
                stop_wheels()
                break
            else:
                tgt_ang = math.atan2(dy, dx)
                err = wrap_pi(tgt_ang - cur_yaw)
                if abs(err) > ANGLE_TOL:
                    omega_cmd = clamp(KP_YAW*abs(err), OMEGA_MIN, OMEGA_MAX)
                    set_body_velocity(0, 0, omega_cmd)
                else:
                    speed = V_FORWARD * min(1.0, (dist_to_target-STOP_DIST)/0.5)
                    speed = max(speed, 0.05)
                    set_body_velocity(speed, 0, 0)
        else:
            wx, wy = waypoints[wp_index]
            dx, dy = wx - robot_x, wy - robot_y
            dist_wp = math.hypot(dx, dy)
            if dist_wp < 0.1:
                wp_index += 1
            else:
                tgt_ang = math.atan2(dy, dx)
                err = wrap_pi(tgt_ang - cur_yaw)
                if abs(err) > ANGLE_TOL:
                    omega_cmd = clamp(KP_YAW*abs(err), OMEGA_MIN, OMEGA_MAX)
                    set_body_velocity(0, 0, omega_cmd)
                else:
                    set_body_velocity(V_FORWARD, 0, 0)
