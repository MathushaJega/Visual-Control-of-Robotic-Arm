# rotation_only_align_anticlockwise.py
# Rotate a Webots YouBot in place until its front is aligned with a 3D target point.
# The robot always rotates anticlockwise.

from controller import Supervisor
import math

# ------------------------------------------------------------
# Configuration (EDIT THESE)
# ------------------------------------------------------------
TARGET_POINT = [0.98,0,0.00225 ] #[X, Y, Z]  <-- change as needed
ANGLE_TOL = 0.04          # rad (~2.3 deg)
KP_YAW    = 3.0           # P gain
OMEGA_MAX = 1.5           # rad/s
OMEGA_MIN = 0.15          # minimum spin to overcome stiction
STABLE_STEPS = 6

# YouBot geometry & wheel limits
WHEEL_RADIUS = 0.05
HALF_L = 0.235
HALF_W = 0.15
MAX_WHEEL_VEL = 14.81

# ------------------------------------------------------------
# Controller setup
# ------------------------------------------------------------
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

node = robot.getFromDef("YOUBOT")
if node is None:
    print("ERROR: DEF YOUROBOT not found. Give your robot this DEF name.")
    raise SystemExit

translation_field = node.getField("translation")
rotation_field    = node.getField("rotation")

wheel_names = ["wheel1", "wheel2", "wheel3", "wheel4"]
wheels = []
for n in wheel_names:
    m = robot.getDevice(n)
    if m is None:
        print(f"ERROR: Device '{n}' not found. Check wheel names.")
        raise SystemExit
    m.setPosition(float("inf"))
    m.setVelocity(0.0)
    wheels.append(m)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def set_body_yaw_rate(omega):
    r = WHEEL_RADIUS
    a = (HALF_L + HALF_W)
    w_fl = -(a * omega) / r
    w_fr =  +(a * omega) / r
    w_rl = -(a * omega) / r
    w_rr =  +(a * omega) / r

    w_fl = clamp(w_fl, -MAX_WHEEL_VEL, MAX_WHEEL_VEL)
    w_fr = clamp(w_fr, -MAX_WHEEL_VEL, MAX_WHEEL_VEL)
    w_rl = clamp(w_rl, -MAX_WHEEL_VEL, MAX_WHEEL_VEL)
    w_rr = clamp(w_rr, -MAX_WHEEL_VEL, MAX_WHEEL_VEL)

    wheels[0].setVelocity(w_fl)
    wheels[1].setVelocity(w_fr)
    wheels[2].setVelocity(w_rl)
    wheels[3].setVelocity(w_rr)

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
        [c + x*x*C,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, z*z*C + c*(0)]
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
    local_up = [0, 0, 1]
    up_world = mat_vec(R, local_up)
    comps = [abs(up_world[0]), abs(up_world[1]), abs(up_world[2])]
    return comps.index(max(comps))

def current_yaw_and_target_angle():
    p = translation_field.getSFVec3f()
    rot = rotation_field.getSFRotation()
    R = axis_angle_to_R(rot[0], rot[1], rot[2], rot[3])
    up_idx = detect_world_up_index(R)
    ground_axes = [0, 1, 2]
    ground_axes.remove(up_idx)
    g0, g1 = ground_axes

    f_world = mat_vec(R, [1, 0, 0])
    f0, f1 = f_world[g0], f_world[g1]
    cur_yaw = math.atan2(f1, f0)

    tgt = list(TARGET_POINT)
    if tgt[up_idx] is None:
        tgt[up_idx] = p[up_idx]

    d0 = tgt[g0] - p[g0]
    d1 = tgt[g1] - p[g1]
    tgt_ang = math.atan2(d1, d0)

    return cur_yaw, tgt_ang

# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------
aligned_printed = False
stable = 0

while robot.step(timestep) != -1:
    cur_yaw, tgt_ang = current_yaw_and_target_angle()
    err = wrap_pi(tgt_ang - cur_yaw)

    if abs(err) <= ANGLE_TOL:
        stable += 1
        set_body_yaw_rate(0.0)
        if not aligned_printed:
            print("Alignment is correct")
            aligned_printed = True
        if stable >= STABLE_STEPS:
            print("Ready")
            stop_wheels()
            break
    else:
        aligned_printed = False
        stable = 0
        # Force anticlockwise rotation
        omega_cmd = clamp(KP_YAW * abs(err), OMEGA_MIN, OMEGA_MAX)
        set_body_yaw_rate(omega_cmd)
