# rotate_and_go_with_obstacle_check_and_wheels.py
from controller import Supervisor
import math

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
TARGET_POINT = [-0.734271, -1.33518, 0.102613]   # [X, Y, Z]
ANGLE_TOL = 0.04
KP_YAW    = 3.0
OMEGA_MAX = 1.5
OMEGA_MIN = 0.15
STABLE_STEPS = 6

# Forward motion
V_FORWARD = 0.15
STOP_DIST = 0.5

# YouBot geometry
WHEEL_RADIUS = 0.05
HALF_L = 0.24
HALF_W = 0.1 
MAX_WHEEL_VEL = 14.81

# Example obstacles (ground plane coords)
OBSTACLES = [
    [-1.0953,-0.885395], [-0.369469, -1.81375]
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

def point_in_poly(pt,poly):
    x,y=pt; inside=False; n=len(poly)
    for i in range(n):
        x1,y1=poly[i]; x2,y2=poly[(i+1)%n]
        if ((y1>y)!=(y2>y)) and (x<(x2-x1)*(y-y1)/(y2-y1+1e-9)+x1):
            inside=not inside
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

                left_front_wheel  = [robot_x + HALF_W, robot_y - HALF_L]
                right_front_wheel = [robot_x - HALF_W, robot_y - HALF_L]

                # ---------- Quadrilateral computation ----------
                target_pt = [tgt[g0], tgt[g1]]
                L_ct    = line_from_points([robot_x, robot_y], target_pt)
                L_left  = parallel_line(*L_ct, left_front_wheel)
                L_right = parallel_line(*L_ct, right_front_wheel)
                L_front = line_from_points(left_front_wheel, right_front_wheel)
                L_tgt   = parallel_line(*L_front, target_pt)

                poly = [
                    intersect(L_left,  L_front),
                    intersect(L_front, L_right),
                    intersect(L_right, L_tgt),
                    intersect(L_tgt,   L_left)
                ]

                # Print corners
                for i, pt in enumerate(poly):
                    print(f"Corner P{i+1}: ({pt[0]:.4f}, {pt[1]:.4f})")

                # Print line equations
                def print_line_eq(name,L):
                    A,B,C=L
                    if abs(B) > 1e-9:
                        print(f"{name}: y = {-A/B:.4f} x + {-C/B:.4f}")
                    else:
                        print(f"{name}: x = {-C/A:.4f}")
                print_line_eq("Line center->target", L_ct)
                print_line_eq("Line through left front (parallel)", L_left)
                print_line_eq("Line through right front (parallel)", L_right)
                print_line_eq("Line through front wheels", L_front)
                print_line_eq("Line through target (parallel)", L_tgt)

                # Check obstacles
                blocked = False
                for obs in OBSTACLES:
                    if point_in_poly(obs, poly):
                        print("Obstacle inside quadrilateral:", obs)
                        blocked = True

                if blocked:
                    print("no path → stopping robot")
                    stop_wheels()
                    break
                else:
                    print("clear path → moving towards target")
                    print("Robot center:", [robot_x, robot_y])
                    print("Left front wheel:", left_front_wheel)
                    print("Right front wheel:", right_front_wheel)
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
