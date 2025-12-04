from controller import Supervisor
import math, heapq

# -------------------------------------------------
#   Configuration
# -------------------------------------------------
ANGLE_TOL   = 0.04
KP_YAW      = 3.0
OMEGA_MAX   = 1.5
OMEGA_MIN   = 0.15
STABLE_STEPS= 6

WHEEL_RADIUS= 0.05
HALF_L      = 0.235
HALF_W      = 0.15
MAX_WHEEL_VEL = 14.81

FLOOR_SIZE  = 5.0          # meters
GRID_RES    = 0.5          # meters per cell
GRID_SIZE   = int(FLOOR_SIZE / GRID_RES)

# -------------------------------------------------
#   Webots robot setup
# -------------------------------------------------
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
node = robot.getFromDef("YOUBOT")
if node is None: raise SystemExit("DEF YOUBOT not found.")
translation_field = node.getField("translation")
rotation_field    = node.getField("rotation")

wheels=[]
for n in ["wheel1","wheel2","wheel3","wheel4"]:
    m=robot.getDevice(n)
    if m is None: raise SystemExit(f"Device {n} not found.")
    m.setPosition(float('inf')); m.setVelocity(0.0)
    wheels.append(m)

# -------------------------------------------------
#   Scene-tree utilities
# -------------------------------------------------
def iter_root_children():
    root = robot.getRoot()
    children = root.getField("children")
    for i in range(children.getCount()):
        yield children.getMFNode(i)

def node_def_name(node):
    try: return node.getDef()
    except: return None

def node_translation(node):
    try:
        f = node.getField("translation")
        return f.getSFVec3f() if f else None
    except: return None

def find_first_node_translation_by_prefix(prefixes):
    for node in iter_root_children():
        defname = node_def_name(node)
        if not defname: continue
        for pref in prefixes:
            if defname.startswith(pref):
                t = node_translation(node)
                if t: return [t[0], t[1], t[2]]
    return None

def find_all_node_translations_by_prefix(prefixes):
    found=[]
    for node in iter_root_children():
        defname=node_def_name(node)
        if not defname: continue
        for pref in prefixes:
            if defname.startswith(pref):
                t=node_translation(node)
                if t: found.append([t[0], t[1], t[2]])
                break
    return found

def get_dynamic_obstacles():
    obs_nodes = find_all_node_translations_by_prefix(["BALL2","BALL3"])
    return [[t[0], t[1]] for t in obs_nodes]

TARGET_POINT = find_first_node_translation_by_prefix(["BALL1"])
if not TARGET_POINT: raise SystemExit("Target BALL1 not found.")
print("Target from scene tree:", TARGET_POINT)

# -------------------------------------------------
#   Low-level robot helpers
# -------------------------------------------------
def clamp(v,lo,hi): return max(lo,min(hi,v))

def set_body_yaw_rate(omega):
    r=WHEEL_RADIUS; a=(HALF_L+HALF_W)
    w=[-(a*omega)/r,(a*omega)/r, -(a*omega)/r,(a*omega)/r]
    for i in range(4): wheels[i].setVelocity(clamp(w[i],-MAX_WHEEL_VEL,MAX_WHEEL_VEL))

def stop_wheels(): [w.setVelocity(0.0) for w in wheels]

def axis_angle_to_R(ax,ay,az,ang):
    n=math.sqrt(ax*ax+ay*ay+az*az)
    if n<1e-9: return [[1,0,0],[0,1,0],[0,0,1]]
    x,y,z=ax/n,ay/n,az/n; c,s=math.cos(ang),math.sin(ang); C=1-c
    return [[c+x*x*C, x*y*C-z*s, x*z*C+y*s],
            [y*x*C+z*s, c+y*y*C, y*z*C-x*s],
            [z*x*C-y*s, z*y*C+x*s, c+z*z*C]]

def mat_vec(R,v): return [R[0][0]*v[0]+R[0][1]*v[1]+R[0][2]*v[2],
                          R[1][0]*v[0]+R[1][1]*v[1]+R[1][2]*v[2],
                          R[2][0]*v[0]+R[2][1]*v[1]+R[2][2]*v[2]]

def wrap_pi(a): return (a+math.pi)%(2*math.pi)-math.pi

def detect_world_up_index(R):
    return max(range(3), key=lambda i: abs(mat_vec(R,[0,0,1])[i]))

def current_yaw_and_target_angle():
    p = translation_field.getSFVec3f()
    rot = rotation_field.getSFRotation()
    R = axis_angle_to_R(rot[0],rot[1],rot[2],rot[3])
    up_idx = detect_world_up_index(R)
    ground_axes=[0,1,2]; ground_axes.remove(up_idx)
    g0,g1 = ground_axes
    f_world = mat_vec(R,[1,0,0])
    cur_yaw = math.atan2(f_world[g1],f_world[g0])
    tgt=list(TARGET_POINT)
    dx,dy = tgt[g0]-p[g0], tgt[g1]-p[g1]
    return cur_yaw, math.atan2(dy,dx)

def compute_wheel_velocities(vx,vy,omega):
    r=WHEEL_RADIUS; a=HALF_L+HALF_W
    w=[(vx-vy-a*omega)/r,(vx+vy+a*omega)/r,
       (vx+vy-a*omega)/r,(vx-vy+a*omega)/r]
    return [clamp(wi,-MAX_WHEEL_VEL,MAX_WHEEL_VEL) for wi in w]

def move_robot(vx,vy,omega):
    wv = compute_wheel_velocities(vx,vy,omega)
    for i in range(4): wheels[i].setVelocity(wv[i])

def stop_all(): [w.setVelocity(0.0) for w in wheels]

# -------------------------------------------------
#   Grid helpers (footprint-aware)
# -------------------------------------------------
def real_to_grid(pos):
    gx=int(math.floor((pos[0]+FLOOR_SIZE/2)/GRID_RES))
    gy=int(math.floor((pos[1]+FLOOR_SIZE/2)/GRID_RES))
    return max(0,min(GRID_SIZE-1,gy)), max(0,min(GRID_SIZE-1,gx))

def grid_to_world(cell):
    row,col = cell
    x = (col+0.5)*GRID_RES - FLOOR_SIZE/2
    y = (row+0.5)*GRID_RES - FLOOR_SIZE/2
    return x,y

def make_grid(obs_xy):
    g = [[1]*GRID_SIZE for _ in range(GRID_SIZE)]
    footprint_cells = int(math.ceil(max(HALF_L,HALF_W)*2/GRID_RES))
    for ob in obs_xy:
        r,c = real_to_grid(ob)
        # mark surrounding cells as blocked according to robot footprint
        for dr in range(-footprint_cells//2, footprint_cells//2+1):
            for dc in range(-footprint_cells//2, footprint_cells//2+1):
                rr,cc = r+dr, c+dc
                if 0 <= rr < GRID_SIZE and 0 <= cc < GRID_SIZE:
                    g[rr][cc]=0
    return g

# -------------------------------------------------
#   Minimal D* Lite Implementation
# -------------------------------------------------
class DStarLite:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal  = goal
        self.km    = 0
        self.g     = {}
        self.rhs   = {}
        self.U     = []
        self.moves = [(0,1),(0,-1),(1,0),(-1,0),
                      (1,1),(1,-1),(-1,1),(-1,-1)]
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                self.g[(r,c)]   = float('inf')
                self.rhs[(r,c)] = float('inf')
        self.rhs[goal] = 0
        heapq.heappush(self.U,(self.calculate_key(goal),goal))
        self.last = start
        self.compute_shortest_path()

    def heuristic(self,u,v):
        return math.hypot(u[0]-v[0],u[1]-v[1])

    def calculate_key(self,s):
        return (min(self.g[s],self.rhs[s]) + self.heuristic(self.start,s) + self.km,
                min(self.g[s],self.rhs[s]))

    def update_vertex(self,u):
        if u != self.goal:
            self.rhs[u] = min([self.g[n]+self.heuristic(u,n)
                               for n in self.neighbors(u)
                               if self.is_free(n)] or [float('inf')])
        inU = [x for x in self.U if x[1]==u]
        if inU:
            self.U = [x for x in self.U if x[1]!=u]
            heapq.heapify(self.U)
        if self.g[u] != self.rhs[u]:
            heapq.heappush(self.U,(self.calculate_key(u),u))

    def neighbors(self,u):
        for d in self.moves:
            v=(u[0]+d[0],u[1]+d[1])
            if 0<=v[0]<GRID_SIZE and 0<=v[1]<GRID_SIZE:
                yield v

    def is_free(self,cell):
        r,c=cell
        return self.grid[r][c]==1

    def compute_shortest_path(self):
        while self.U and (self.U[0][0] < self.calculate_key(self.start) or
              self.rhs[self.start] != self.g[self.start]):
            k_old,s = heapq.heappop(self.U)
            k_new   = self.calculate_key(s)
            if k_old < k_new:
                heapq.heappush(self.U,(k_new,s))
            elif self.g[s] > self.rhs[s]:
                self.g[s] = self.rhs[s]
                for n in self.neighbors(s):
                    self.update_vertex(n)
            else:
                g_old = self.g[s]
                self.g[s] = float('inf')
                self.update_vertex(s)
                for n in self.neighbors(s):
                    self.update_vertex(n)

    def update_grid(self,new_grid):
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if new_grid[r][c] != self.grid[r][c]:
                    self.grid[r][c] = new_grid[r][c]
                    self.update_vertex((r,c))
        self.km += self.heuristic(self.last,self.start)
        self.last = self.start
        self.compute_shortest_path()

    def path(self):
        if self.g[self.start] == float('inf'): return None
        path=[self.start]
        s=self.start
        while s!=self.goal:
            nbrs=[n for n in self.neighbors(s) if self.is_free(n)]
            if not nbrs: return None
            s=min(nbrs,key=lambda n:self.g[n]+self.heuristic(s,n))
            path.append(s)
        return path

# -------------------------------------------------
#   D* Lite planning and execution
# -------------------------------------------------
robot_pos = translation_field.getSFVec3f()
start = real_to_grid([robot_pos[0], robot_pos[1]])
goal  = real_to_grid([TARGET_POINT[0], TARGET_POINT[1]])

grid = make_grid(get_dynamic_obstacles())
planner = DStarLite(grid, start, goal)
planner.last = start  # needed for km update

print("Initial plan length:", len(planner.path() or []))

while robot.step(timestep) != -1:
    robot_pos = translation_field.getSFVec3f()
    planner.start = real_to_grid([robot_pos[0], robot_pos[1]])

    new_grid = make_grid(get_dynamic_obstacles())
    planner.update_grid(new_grid)

    path_cells = planner.path()
    
    # Always assume path exists
    
    if len(path_cells) > 2:  # stop one cell before last
        wx,wy = grid_to_world(path_cells[1])
        p = translation_field.getSFVec3f()
        dx,dy = wx - p[0], wy - p[1]
        dist = math.hypot(dx,dy)
        if dist <= GRID_RES*0.25:
            continue
        kp_lin=0.8; max_lin=MAX_WHEEL_VEL*WHEEL_RADIUS*0.6
        vx_world = kp_lin*dx; vy_world = kp_lin*dy
        vmag = math.hypot(vx_world, vy_world)
        if vmag>max_lin:
            s = max_lin/vmag; vx_world*=s; vy_world*=s
        rot = rotation_field.getSFRotation()
        R = axis_angle_to_R(rot[0],rot[1],rot[2],rot[3])
        vx_local = R[0][0]*vx_world + R[1][0]*vy_world
        vy_local = R[0][1]*vx_world + R[1][1]*vy_world
        cur_yaw,_ = current_yaw_and_target_angle()
        err_yaw = wrap_pi(math.atan2(dy,dx) - cur_yaw)
        move_robot(vx_local,vy_local,clamp(2.0*err_yaw,-OMEGA_MAX,OMEGA_MAX))
    else:
        stop_all()
        print("Reached one cell before goal.")
        break

# -------------------------------------------------
#   Final alignment
# -------------------------------------------------
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
