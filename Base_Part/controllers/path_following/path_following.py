# ---------------- YouBot Alignment + A* Path Planning + Final Orientation ----------------
from controller import Supervisor
import math
import heapq

# ---------------- Configuration ----------------
TARGET_POINT = [0.75, 0.25, 0.00225]  # [X, Y, Z]
ANGLE_TOL = 0.04
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

# Obstacles
obstacles = [[0.875865,-0.77313]]

# ---------------- YouBot Setup ----------------
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
node = robot.getFromDef("YOUBOT")
if node is None:
    raise SystemExit("ERROR: DEF YOUROBOT not found.")

translation_field = node.getField("translation")
rotation_field = node.getField("rotation")

wheel_names = ["wheel1","wheel2","wheel3","wheel4"]
wheels = []
for n in wheel_names:
    m = robot.getDevice(n)
    if m is None: raise SystemExit(f"ERROR: Device '{n}' not found.")
    m.setPosition(float('inf'))
    m.setVelocity(0.0)
    wheels.append(m)

# ---------------- Utility Functions ----------------
def clamp(v, lo, hi): return max(lo, min(hi, v))
def set_body_yaw_rate(omega):
    r = WHEEL_RADIUS; a = (HALF_L+HALF_W)
    w = [-(a*omega)/r, (a*omega)/r, -(a*omega)/r, (a*omega)/r]
    for i in range(4): wheels[i].setVelocity(clamp(w[i],-MAX_WHEEL_VEL,MAX_WHEEL_VEL))
def stop_wheels(): [w.setVelocity(0.0) for w in wheels]
def axis_angle_to_R(ax,ay,az,ang):
    n = math.sqrt(ax*ax+ay*ay+az*az)
    if n<1e-9: return [[1,0,0],[0,1,0],[0,0,1]]
    x,y,z = ax/n,ay/n,az/n; c,s=math.cos(ang),math.sin(ang); C=1-c
    return [[c+x*x*C, x*y*C-z*s, x*z*C+y*s],[y*x*C+z*s, c+y*y*C, y*z*C-x*s],[z*x*C-y*s, z*y*C+x*s, c+z*z*C]]
def mat_vec(R,v): return [R[0][0]*v[0]+R[0][1]*v[1]+R[0][2]*v[2], R[1][0]*v[0]+R[1][1]*v[1]+R[1][2]*v[2], R[2][0]*v[0]+R[2][1]*v[1]+R[2][2]*v[2]]
def wrap_pi(a): return (a+math.pi)%(2*math.pi)-math.pi
def detect_world_up_index(R): return max(range(3), key=lambda i: abs(mat_vec(R,[0,0,1])[i]))
def current_yaw_and_target_angle():
    p=translation_field.getSFVec3f(); rot=rotation_field.getSFRotation()
    R=axis_angle_to_R(rot[0],rot[1],rot[2],rot[3])
    up_idx=detect_world_up_index(R)
    ground_axes=[0,1,2]; ground_axes.remove(up_idx)
    g0,g1=ground_axes
    f_world=mat_vec(R,[1,0,0])
    cur_yaw=math.atan2(f_world[g1],f_world[g0])
    tgt=list(TARGET_POINT)
    if tgt[up_idx] is None: tgt[up_idx]=p[up_idx]
    dx,dy=tgt[g0]-p[g0], tgt[g1]-p[g1]
    return cur_yaw, math.atan2(dy,dx)

# ---------------- Grid ----------------
def real_to_grid(pos):
    gx=int(math.floor((pos[0]+FLOOR_SIZE/2)/GRID_RES))
    gy=int(math.floor((pos[1]+FLOOR_SIZE/2)/GRID_RES))
    return max(0,min(GRID_SIZE-1,gy)), max(0,min(GRID_SIZE-1,gx))
def generate_grid(obstacles):
    g=[[1]*GRID_SIZE for _ in range(GRID_SIZE)]
    for ob in obstacles: row,col=real_to_grid(ob); g[row][col]=0
    return g
def print_grid(grid):
    print("Grid (0=blocked,1=free):"); [print(row) for row in grid]

# ---------------- A* Path Planning ----------------
class Cell: 
    def __init__(self): self.parent_i=0; self.parent_j=0; self.f=float('inf'); self.g=float('inf'); self.h=0
def is_valid(row,col): return 0<=row<GRID_SIZE and 0<=col<GRID_SIZE
def is_unblocked(grid,row,col): return grid[row][col]==1
def is_destination(row,col,dest): return row==dest[0] and col==dest[1]
def calculate_h_value(row,col,dest): return math.hypot(row-dest[0],col-dest[1])

def a_star_get_path(grid, src, dest):
    if not is_valid(*src) or not is_valid(*dest): return None
    if not is_unblocked(grid,*src) or not is_unblocked(grid,*dest): return None
    if is_destination(*src,dest): return [src]

    closed_list=[[False]*GRID_SIZE for _ in range(GRID_SIZE)]
    cell_details=[[Cell() for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    i,j=src; cell_details[i][j].f=0; cell_details[i][j].g=0; cell_details[i][j].h=0; cell_details[i][j].parent_i=i; cell_details[i][j].parent_j=j
    open_list=[]; heapq.heappush(open_list,(0.0,i,j))

    while open_list:
        p=heapq.heappop(open_list); i,j=p[1],p[2]
        if closed_list[i][j]: continue
        closed_list[i][j]=True
        directions=[(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
        for d in directions:
            ni,nj=i+d[0],j+d[1]
            if is_valid(ni,nj) and is_unblocked(grid,ni,nj) and not closed_list[ni][nj]:
                g_new=cell_details[i][j].g+1.0
                h_new=calculate_h_value(ni,nj,dest)
                f_new=g_new+h_new
                if cell_details[ni][nj].f==float('inf') or cell_details[ni][nj].f>f_new:
                    heapq.heappush(open_list,(f_new,ni,nj))
                    cell_details[ni][nj].f=f_new; cell_details[ni][nj].g=g_new; cell_details[ni][nj].h=h_new
                    cell_details[ni][nj].parent_i=i; cell_details[ni][nj].parent_j=j
                # Stop one cell before destination
                if is_destination(ni,nj,dest):
                    path=[]; row,col=i,j
                    while not (cell_details[row][col].parent_i==row and cell_details[row][col].parent_j==col):
                        path.append((row,col))
                        row,col=cell_details[row][col].parent_i, cell_details[row][col].parent_j
                    path.append((row,col)); path.reverse()
                    return path
    return None

# ---------------- Inverse Kinematics ----------------
def compute_wheel_velocities(vx,vy,omega):
    r=WHEEL_RADIUS; a=HALF_L+HALF_W
    w=[(vx-vy-a*omega)/r,(vx+vy+a*omega)/r,(vx+vy-a*omega)/r,(vx-vy+a*omega)/r]
    return [clamp(wi,-MAX_WHEEL_VEL,MAX_WHEEL_VEL) for wi in w]
def move_robot(vx,vy,omega):
    wv=compute_wheel_velocities(vx,vy,omega)
    for i in range(4): wheels[i].setVelocity(wv[i])
def stop_all(): [w.setVelocity(0.0) for w in wheels]
def cell_to_world(cell):
    row,col=cell; x=(col+0.5)*GRID_RES-FLOOR_SIZE/2; y=(row+0.5)*GRID_RES-FLOOR_SIZE/2; return x,y

# ---------------- Follow Path ----------------
def follow_waypoints(waypoints):
    if not waypoints: return
    max_lin=MAX_WHEEL_VEL*WHEEL_RADIUS*0.6; kp_lin=0.8; kp_ang=2.0
    print("Following path with", len(waypoints), "waypoints.")
    for idx,cell in enumerate(waypoints):
        wx,wy=cell_to_world(cell)
        print(f"Waypoint {idx}: grid {cell} -> world x={wx:.3f}, y={wy:.3f}")
        while robot.step(timestep)!=-1:
            p=translation_field.getSFVec3f(); rot=rotation_field.getSFRotation()
            R=axis_angle_to_R(rot[0],rot[1],rot[2],rot[3])
            dx=wx-p[0]; dy=wy-p[1]; dist=math.hypot(dx,dy)
            if dist<=GRID_RES*0.25:
                stop_all(); break
            vx_world=kp_lin*dx; vy_world=kp_lin*dy
            vmag=math.hypot(vx_world,vy_world)
            if vmag>max_lin: scale=max_lin/vmag; vx_world*=scale; vy_world*=scale
            vx_local=R[0][0]*vx_world+R[1][0]*vy_world
            vy_local=R[0][1]*vx_world+R[1][1]*vy_world
            cur_yaw,_=current_yaw_and_target_angle(); err_yaw=wrap_pi(math.atan2(dy,dx)-cur_yaw)
            move_robot(vx_local,vy_local,clamp(kp_ang*err_yaw,-OMEGA_MAX,OMEGA_MAX))

# ---------------- Main ----------------
# Initial alignment
aligned_printed=False; stable=0
while robot.step(timestep)!=-1:
    cur_yaw,tgt_ang=current_yaw_and_target_angle()
    err=wrap_pi(tgt_ang-cur_yaw)
    if abs(err)<=ANGLE_TOL:
        stable+=1; set_body_yaw_rate(0.0)
        if not aligned_printed: print("Alignment correct"); aligned_printed=True
        if stable>=STABLE_STEPS: stop_wheels(); break
    else:
        aligned_printed=False; stable=0
        set_body_yaw_rate(clamp(KP_YAW*abs(err),OMEGA_MIN,OMEGA_MAX))

# Generate grid and path
grid=generate_grid(obstacles); print_grid(grid)
robot_pos=translation_field.getSFVec3f()
src=real_to_grid([robot_pos[0], robot_pos[1]])
dest=real_to_grid([TARGET_POINT[0], TARGET_POINT[1]])
path_cells=a_star_get_path(grid, src, dest)

# Follow path
if path_cells:
    follow_waypoints(path_cells)

# ---------------- Final Orientation Adjustment ----------------
# Rotate anticlockwise to face target at the end
final_cur_yaw, final_tgt_yaw = current_yaw_and_target_angle()
final_err = wrap_pi(final_tgt_yaw - final_cur_yaw)
while robot.step(timestep) != -1:
    if abs(final_err) <= ANGLE_TOL:
        stop_all()
        print("Final orientation aligned with target")
        break
    omega_cmd = clamp(KP_YAW * abs(final_err), OMEGA_MIN, OMEGA_MAX)
    set_body_yaw_rate(omega_cmd)
    final_cur_yaw, final_tgt_yaw = current_yaw_and_target_angle()
    final_err = wrap_pi(final_tgt_yaw - final_cur_yaw)
