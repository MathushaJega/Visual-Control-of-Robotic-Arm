# rotate_and_go_astar_final.py
from controller import Supervisor
import math
import heapq

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
TARGET_POINT = [-1.54554, -1.34145, 0.181125]  # [X,Y,Z]
ANGLE_TOL = 0.04          # rad
KP_YAW    = 3.0
OMEGA_MAX = 1.5
OMEGA_MIN = 0.15
STABLE_STEPS = 6

# Forward motion
V_FORWARD = 0.15
STOP_DIST = 0.2

# YouBot geometry & wheel limits
WHEEL_RADIUS = 0.05
HALF_L = 0.235
HALF_W = 0.15
MAX_WHEEL_VEL = 14.81

# Floor/grid settings
FLOOR_SIZE = 5.0       # meters (square floor)
GRID_RES   = 0.5       # meters per cell
GRID_SIZE  = int(FLOOR_SIZE / GRID_RES)

# Robot footprint
ROBOT_RADIUS = 0.28
CLEARANCE = 0.02

# Optional manual obstacles
OBSTACLES = []  # e.g., [[-0.47,-0.82], [-0.37,-1.81]]

# ------------------------------------------------------------
# Supervisor & devices
# ------------------------------------------------------------
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

node = robot.getFromDef("YOUBOT")
if node is None:
    print("ERROR: DEF YOUBOT not found.")
    raise SystemExit

translation_field = node.getField("translation")
rotation_field    = node.getField("rotation")

wheel_names = ["wheel1","wheel2","wheel3","wheel4"]
wheels=[]
for n in wheel_names:
    m = robot.getDevice(n)
    if m is None:
        print(f"ERROR: Device '{n}' not found.")
        raise SystemExit
    m.setPosition(float("inf"))
    m.setVelocity(0.0)
    wheels.append(m)

def clamp(v,lo,hi):
    return max(lo,min(hi,v))

def set_body_velocity(vx,vy,omega):
    r = WHEEL_RADIUS
    a = HALF_L + HALF_W
    w_fl = (vx - vy - a*omega)/r
    w_fr = (vx + vy + a*omega)/r
    w_rl = (vx + vy - a*omega)/r
    w_rr = (vx - vy + a*omega)/r
    wheels[0].setVelocity(clamp(w_fl,-MAX_WHEEL_VEL,MAX_WHEEL_VEL))
    wheels[1].setVelocity(clamp(w_fr,-MAX_WHEEL_VEL,MAX_WHEEL_VEL))
    wheels[2].setVelocity(clamp(w_rl,-MAX_WHEEL_VEL,MAX_WHEEL_VEL))
    wheels[3].setVelocity(clamp(w_rr,-MAX_WHEEL_VEL,MAX_WHEEL_VEL))

def stop_wheels():
    for m in wheels: m.setVelocity(0.0)

# ------------------ rotation & orientation math ------------------
def axis_angle_to_R(ax, ay, az, ang):
    n = math.sqrt(ax*ax + ay*ay + az*az)
    if n<1e-9: return [[1,0,0],[0,1,0],[0,0,1]]
    x,y,z = ax/n, ay/n, az/n
    c = math.cos(ang)
    s = math.sin(ang)
    C = 1-c
    return [
        [c + x*x*C,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, z*z*C + c]
    ]

def mat_vec(R,v):
    return [R[0][0]*v[0]+R[0][1]*v[1]+R[0][2]*v[2],
            R[1][0]*v[0]+R[1][1]*v[1]+R[1][2]*v[2],
            R[2][0]*v[0]+R[2][1]*v[1]+R[2][2]*v[2]]

def wrap_pi(a): return (a + math.pi) % (2*math.pi) - math.pi

def detect_world_up_index(R):
    local_up=[0,0,1]
    up_world = mat_vec(R,local_up)
    comps=[abs(up_world[0]),abs(up_world[1]),abs(up_world[2])]
    return comps.index(max(comps))

def current_yaw_and_target_angle():
    p = translation_field.getSFVec3f()
    rot = rotation_field.getSFRotation()
    R = axis_angle_to_R(rot[0],rot[1],rot[2],rot[3])
    up_idx = detect_world_up_index(R)
    ground_axes=[0,1,2]; ground_axes.remove(up_idx)
    g0,g1 = ground_axes
    f_world = mat_vec(R,[1,0,0])
    cur_yaw = math.atan2(f_world[g1],f_world[g0])
    tgt = list(TARGET_POINT)
    if tgt[up_idx] is None: tgt[up_idx] = p[up_idx]
    d0 = tgt[g0]-p[g0]; d1=tgt[g1]-p[g1]
    tgt_ang = math.atan2(d1,d0)
    dist = math.hypot(d0,d1)
    return cur_yaw,tgt_ang,dist,g0,g1,p,tgt

# ------------------ grid / occupancy helpers ------------------
def world_to_grid(x,y):
    i = int((x+FLOOR_SIZE/2)/GRID_RES)
    j = int((y+FLOOR_SIZE/2)/GRID_RES)
    i = max(0,min(GRID_SIZE-1,i))
    j = max(0,min(GRID_SIZE-1,j))
    return i,j

def grid_to_world(i,j):
    x = -FLOOR_SIZE/2 + (i+0.5)*GRID_RES
    y = -FLOOR_SIZE/2 + (j+0.5)*GRID_RES
    return x,y

def in_bounds(i,j): return 0<=i<GRID_SIZE and 0<=j<GRID_SIZE

# ------------------ obstacles ------------------
def find_scene_obstacles():
    found=[]
    root = robot.getRoot()
    children_field=root.getField("children")
    if children_field:
        for k in range(children_field.getCount()):
            n=children_field.getMFNode(k)
            if n:
                d = n.getDef()
                if d and (d.upper().startswith("BALL") or d.upper().startswith("OBSTACLE")):
                    t = n.getField("translation").getSFVec3f()
                    found.append([t[0],t[1]])
    return found if found else OBSTACLES[:]

def build_occupancy(obstacles_world,inflate_radius):
    occ = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    inflate_cells = int(math.ceil(inflate_radius/GRID_RES))
    for ox,oy in obstacles_world:
        ci,cj = world_to_grid(ox,oy)
        for di in range(-inflate_cells,inflate_cells+1):
            for dj in range(-inflate_cells,inflate_cells+1):
                i=ci+di; j=cj+dj
                if not in_bounds(i,j): continue
                wx,wy = grid_to_world(i,j)
                if math.hypot(wx-ox,wy-oy)<=inflate_radius+1e-9:
                    occ[i][j]=1
    return occ

def bresenham_line(i0,j0,i1,j1):
    points=[]
    dx=abs(i1-i0); dy=abs(j1-j0)
    x=i0; y=j0
    sx=1 if i1>=i0 else -1
    sy=1 if j1>=j0 else -1
    if dy<=dx:
        err=dx//2
        while x!=i1:
            points.append((x,y))
            err-=dy
            if err<0: y+=sy; err+=dx
            x+=sx
        points.append((i1,j1))
    else:
        err=dy//2
        while y!=j1:
            points.append((x,y))
            err-=dx
            if err<0: x+=sx; err+=dy
            y+=sy
        points.append((i1,j1))
    return points

def straight_path_clear(p0,p1,occ):
    i0,j0 = world_to_grid(p0[0],p0[1])
    i1,j1 = world_to_grid(p1[0],p1[1])
    for i,j in bresenham_line(i0,j0,i1,j1):
        if occ[i][j]==1: return False
    return True

# ------------------ A* ------------------
def astar(start_w,goal_w,occ):
    si,sj = world_to_grid(start_w[0],start_w[1])
    gi,gj = world_to_grid(goal_w[0],goal_w[1])
    occ[si][sj]=0; occ[gi][gj]=0
    open_heap=[]; heapq.heappush(open_heap,(0.0,(si,sj),None))
    came_from={}; gscore={(si,sj):0.0}; closed=set()
    neighbors=[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    def h(i,j):
        xw,yw = grid_to_world(i,j)
        gx,gy = grid_to_world(gi,gj)
        return math.hypot(gx-xw,gy-yw)
    while open_heap:
        f,(i,j),parent = heapq.heappop(open_heap)
        if (i,j) in closed: continue
        came_from[(i,j)] = parent
        if (i,j)==(gi,gj):
            path=[]; cur=(i,j)
            while cur:
                path.append(grid_to_world(cur[0],cur[1]))
                cur=came_from.get(cur,None)
            path.reverse(); return path
        closed.add((i,j))
        for di,dj in neighbors:
            ni=i+di; nj=j+dj
            if not in_bounds(ni,nj) or occ[ni][nj]==1: continue
            ng=gscore[(i,j)]+math.hypot(di,dj)*GRID_RES
            if ng<gscore.get((ni,nj),float("inf"))-1e-9:
                gscore[(ni,nj)]=ng
                heapq.heappush(open_heap,(ng+h(ni,nj),(ni,nj),(i,j)))
    return None

# ------------------ Path following ------------------
def move_along_path(path,current_pos):
    if not path or len(path)<2:
        stop_wheels(); return True
    next_wp=path[1]
    dx=next_wp[0]-current_pos[0]
    dy=next_wp[1]-current_pos[1]
    dist=math.hypot(dx,dy)
    tgt_ang=math.atan2(dy,dx)
    rot=rotation_field.getSFRotation()
    R=axis_angle_to_R(rot[0],rot[1],rot[2],rot[3])
    up_idx=detect_world_up_index(R)
    ground_axes=[0,1,2]; ground_axes.remove(up_idx)
    g0,g1 = ground_axes
    f_world=mat_vec(R,[1,0,0])
    cur_yaw=math.atan2(f_world[g1],f_world[g0])
    err_ang=wrap_pi(tgt_ang-cur_yaw)
    if abs(err_ang)>ANGLE_TOL:
        omega_cmd=clamp(KP_YAW*abs(err_ang),OMEGA_MIN,OMEGA_MAX)
        set_body_velocity(0,0,omega_cmd)
        return False
    speed = V_FORWARD * min(1.0, dist/0.5)
    speed = max(speed,0.05)
    set_body_velocity(speed,0,0)
    if dist<0.05: path.pop(0)
    return False

# ------------------ Main loop ------------------
aligned_printed=False; stable=0; phase="rotate"; current_path=None

while robot.step(timestep)!=-1:
    cur_yaw,tgt_ang,dist,g0,g1,p,tgt = current_yaw_and_target_angle()
    err=wrap_pi(tgt_ang-cur_yaw)

    if phase=="rotate":
        if abs(err)<=ANGLE_TOL:
            stable+=1; set_body_velocity(0,0,0)
            if not aligned_printed: print("Alignment is correct"); aligned_printed=True
            if stable>=STABLE_STEPS:
                print("Ready to move forward / plan path"); phase="move"
        else:
            aligned_printed=False; stable=0
            omega_cmd=clamp(KP_YAW*abs(err),OMEGA_MIN,OMEGA_MAX)
            set_body_velocity(0,0,omega_cmd)

    elif phase=="move":
        stop_point=[p[0]+(tgt[0]-p[0])*(1.0-(STOP_DIST/dist)),
                    p[1]+(tgt[1]-p[1])*(1.0-(STOP_DIST/dist))]
        obstacles_world=find_scene_obstacles()
        occ=build_occupancy(obstacles_world,ROBOT_RADIUS+CLEARANCE)
        si,sj=world_to_grid(p[0],p[1])
        gi,gj=world_to_grid(stop_point[0],stop_point[1])
        # Straight path check
        if straight_path_clear(p,stop_point,occ):
            if dist<=STOP_DIST:
                print(f"Stopping {STOP_DIST} m before target")
                stop_wheels(); break
            speed = V_FORWARD * min(1.0, (dist-STOP_DIST)/0.5)
            speed = max(speed,0.05)
            set_body_velocity(speed,0,0)
        else:
            print("Straight path blocked. Running A* planner.")
            if current_path is None:
                path = astar([p[0],p[1]], stop_point, occ)
                if path is None:
                    print("No path found by planner. Stopping.")
                    stop_wheels(); break
                current_path=path
            finished=move_along_path(current_path,[p[0],p[1]])
            if finished:
                print(f"Reached stop point before target")
                stop_wheels(); break
