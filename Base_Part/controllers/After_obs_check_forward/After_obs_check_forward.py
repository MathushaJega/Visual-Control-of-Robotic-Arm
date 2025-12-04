# rotate_and_go_full_corrected.py
from controller import Supervisor
import math

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
TARGET_POINT = [-0.734052, -1.33508, 0.102613]  # [X, Y, Z]
ANGLE_TOL = 0.04
KP_YAW    = 3.0
OMEGA_MAX = 1.5
OMEGA_MIN = 0.15
STABLE_STEPS = 6

V_FORWARD = 0.15
STOP_DIST = 0.5

WHEEL_RADIUS = 0.05
MAX_WHEEL_VEL = 14.81

# Example obstacles
OBSTACLES = [
    [-0.765473,-0.840557], [-1.06114,-1.04976]
]

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

# Front wheels (dynamic)
fl_node = robot.getFromDef("WHEEL2")  # front-left
fr_node = robot.getFromDef("WHEEL1")  # front-right
if not fl_node or not fr_node:
    print("ERROR: Front wheel nodes not found.")
    raise SystemExit

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def clamp(v, lo, hi): return max(lo, min(hi, v))

def set_body_velocity(vx, vy, omega):
    a = 0.24 + 0.1  # HALF_L + HALF_W
    r = WHEEL_RADIUS
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
    return cur_yaw, tgt_ang, dist, g0, g1, p, tgt

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

def is_point_in_quadrilateral(point, quad_points):
    x, y = point
    n = len(quad_points)
    inside = False
    p1x, p1y = quad_points[0]
    for i in range(n + 1):
        p2x, p2y = quad_points[i % n]
        if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y:
                xinters = (y - p1y)*(p2x - p1x)/(p2y - p1y) + p1x
            if p1x == p2x or x <= xinters:
                inside = not inside
        p1x, p1y = p2x, p2y
    return inside

# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------
stable = 0
phase = "rotate"
path_clear = False

while robot.step(timestep) != -1:
    cur_yaw, tgt_ang, dist, g0, g1, p, tgt = current_yaw_and_target_angle()
    err = wrap_pi(tgt_ang - cur_yaw)

    if phase == "rotate":
        if abs(err) <= ANGLE_TOL:
            stable += 1
            set_body_velocity(0,0,0)
            if stable >= STABLE_STEPS:
                robot_x, robot_y = p[g0], p[g1]

                # ---------- Get front wheel positions ----------
                fl_pos = fl_node.getField("translation").getSFVec3f()
                fr_pos = fr_node.getField("translation").getSFVec3f()
                left_front_wheel  = [fl_pos[g0], fl_pos[g1]]
                right_front_wheel = [fr_pos[g0], fr_pos[g1]]
                target_pt = [tgt[g0], tgt[g1]]

                # ---------- Quadrilateral lines ----------
                L_ct    = line_from_points([robot_x, robot_y], target_pt)
                L_left  = parallel_line(*L_ct, left_front_wheel)
                L_right = parallel_line(*L_ct, right_front_wheel)
                L_front = line_from_points(left_front_wheel, right_front_wheel)
                L_tgt   = parallel_line(*L_front, target_pt)

                # ---------- Quadrilateral corners ----------
                poly = [
                    intersect(L_left,  L_front),   # Front-Left
                    intersect(L_front, L_right),   # Front-Right
                    intersect(L_right, L_tgt),     # Target-Right
                    intersect(L_tgt,   L_left)     # Target-Left
                ]
                quad_points = [(pt[0], pt[1]) for pt in poly]

                # Print quadrilateral corners
                print("Quadrilateral corners (in order):")
                labels = ["Front-Left", "Front-Right", "Target-Right", "Target-Left"]
                for label, corner in zip(labels, quad_points):
                    print(f"{label}: {corner}")

                # Print line equations
                lines_info = [
                    ("Line through left front wheel (parallel to center→target)", L_left),
                    ("Line through right front wheel (parallel to center→target)", L_right),
                    ("Line through front wheels", L_front),
                    ("Line through target (parallel to front wheels)", L_tgt)
                ]
                for name, L in lines_info:
                    A,B,C = L
                    if abs(B) > 1e-9:
                        m = -A/B
                        c = -C/B
                        print(f"{name}: y = {m:.3f}x + {c:.3f}")
                    else:
                        x_val = -C/A
                        print(f"{name}: x = {x_val:.3f}")

                # ---------- Obstacle check ----------
                blocked = False
                for obs in OBSTACLES:
                    if is_point_in_quadrilateral(obs, quad_points):
                        print(f"Obstacle {obs} is INSIDE the quadrilateral")
                        blocked = True
                    else:
                        print(f"Obstacle {obs} is OUTSIDE the quadrilateral")

                if blocked:
                    print("no path → stopping robot")
                    stop_wheels()
                    break
                else:
                    print("clear path → moving towards target")
                    path_clear = True
                    phase = "move"
        else:
            stable = 0
            omega_cmd = clamp(KP_YAW*abs(err), OMEGA_MIN, OMEGA_MAX)
            set_body_velocity(0, 0, omega_cmd)

    elif phase == "move" and path_clear:
        if dist <= STOP_DIST:
            print(f"Stopping {STOP_DIST} m before target")
            stop_wheels()
            break
        speed = V_FORWARD * min(1.0, (dist-STOP_DIST)/0.5)
        speed = max(speed, 0.05)
        set_body_velocity(speed, 0, 0)
