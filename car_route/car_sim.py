"""
pp_gap_sim.py
Pure Pursuit + Beam-gap local avoidance 演示

运行:
    python car_route/pp_gap_sim.py

说明:
- 生成椭圆赛道的中线 waypoints 作为全局路径。
- 每帧发射多束 beam（-90..+90 deg）测距。
- 通过阈值 safe_dist 判定 beam 是否free，找到连续 free 区间（gaps）。
- 目标方向优先为 pure pursuit 的目标点方向；若前方有障碍，从 gaps 中选择最接近该方向的 gap center。
- 生成 steer = clamp(Kp * angle_diff) 并做一阶滤波。
- Debug 模式（按 D）会打印/可视化更多信息；按 R 重置；按 T 暂停。
- 如果 car_route/best_pp_params.npy 存在，会尝试加载作为参数: [Kp, safe_dist, lookahead_steps, beta, beam_step_deg, lookahead_wp]
"""

import pygame, math, numpy as np, random, os, sys
pygame.init()

# ---------------- config / load params ----------------
WIDTH, HEIGHT = 900, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pure Pursuit + Gap-based Avoidance Demo")
clock = pygame.time.Clock()

# track parameters (ellipse)
CX, CY = WIDTH//2, HEIGHT//2
A_OUT, B_OUT = 300, 220
A_IN, B_IN = 210, 160

# default control params (can be overridden by npy)
DEFAULT_PARAMS = {
    'Kp': 0.9,            # steering gain
    'safe_dist': 60.0,    # beam distance threshold to consider free
    'lookahead_steps': 10,# index steps ahead on waypoint array for pure pursuit
    'beta': 0.75,         # smoothing
    'beam_step_deg': 6.0, # beam angular resolution
    'lookahead_wp': 60.0, # pure pursuit lookahead distance (pixels)
}

PARAMS_FILE = "car_route/best_pp_params.npy"
if os.path.exists(PARAMS_FILE):
    try:
        arr = np.load(PARAMS_FILE)
        # accept array shape 6
        if arr.size >= 6:
            params = {
                'Kp': float(arr[0]),
                'safe_dist': float(arr[1]),
                'lookahead_steps': int(round(arr[2])),
                'beta': float(arr[3]),
                'beam_step_deg': float(arr[4]),
                'lookahead_wp': float(arr[5])
            }
            print("Loaded params from", PARAMS_FILE, params)
        else:
            params = DEFAULT_PARAMS.copy()
    except Exception as e:
        print("Failed load params:", e)
        params = DEFAULT_PARAMS.copy()
else:
    params = DEFAULT_PARAMS.copy()

# ---------------- track / waypoints ----------------
def generate_midline(samples=360):
    mid_a = (A_IN + A_OUT) / 2.0
    mid_b = (B_IN + B_OUT) / 2.0
    pts = []
    for k in range(samples):
        ang = 2*math.pi*k/samples
        x = CX + mid_a*math.cos(ang)
        y = CY + mid_b*math.sin(ang)
        pts.append((x,y))
    return pts

WAYPOINTS = generate_midline(samples=720)

def point_in_track(x,y):
    dx, dy = x-CX, y-CY
    val_out = (dx/A_OUT)**2 + (dy/B_OUT)**2
    val_in = (dx/A_IN)**2 + (dy/B_IN)**2
    return (val_in >= 1.0) and (val_out <= 1.0)

# ---------------- Car ----------------
class Car:
    def __init__(self, x, y, heading=0.0):
        self.x = x
        self.y = y
        self.h = heading
        self.speed = 2.0
        self.trail = []
        self.alive = True
    def step(self, steer):
        steer = max(-math.radians(30), min(math.radians(30), steer))
        self.h += steer * 0.08
        self.x += math.cos(self.h) * self.speed
        self.y += math.sin(self.h) * self.speed
        self.trail.append((int(self.x), int(self.y)))
        if len(self.trail)>1000: self.trail.pop(0)
    def draw(self, surf):
        if len(self.trail)>1:
            pygame.draw.lines(surf, (0,200,0), False, self.trail, 2)
        pygame.draw.circle(surf, (255,0,0), (int(self.x), int(self.y)), 6)
        ax = self.x + 12*math.cos(self.h); ay = self.y + 12*math.sin(self.h)
        pygame.draw.line(surf, (255,255,255), (int(self.x),int(self.y)), (int(ax),int(ay)),2)

# ---------------- sensor / gap detection / pure pursuit ----------------
def cast_beams(x,y,heading,beam_step_deg, max_range=120.0):
    beams = []
    half = 90
    for a in np.arange(-half, half+0.1, beam_step_deg):
        ang = heading + math.radians(a)
        d = ray_distance(x,y,ang, max_range=max_range)
        beams.append((a, d))
    return beams

def ray_distance(x,y,ang, max_range=120.0, step=3.0):
    dist=0.0
    dx, dy = math.cos(ang), math.sin(ang)
    while dist < max_range:
        px = x + dx*dist; py = y + dy*dist
        if not point_in_track(px,py):
            return dist
        # check obstacles
        for o in obstacles:
            if o.collidepoint(px,py):
                return dist
        dist += step
    return max_range

def detect_gaps(beams, safe_dist):
    # beams: list of (angle_deg, dist)
    free = [1 if d>safe_dist else 0 for (_,d) in beams]
    gaps = []
    n = len(free)
    i=0
    while i<n:
        if free[i]==1:
            j=i
            while j<n and free[j]==1:
                j+=1
            # gap from i..j-1
            angles = [beams[k][0] for k in range(i,j)]
            center = (angles[0] + angles[-1]) / 2.0
            width = angles[-1] - angles[0] + 1e-6
            gaps.append({'start':i,'end':j-1,'center':center,'width':width})
            i=j
        else:
            i+=1
    return gaps

def find_pursuit_target(car, waypoints, lookahead_dist):
    # find the waypoint at distance >= lookahead_dist along the path ahead of the car
    # naive: find nearest index, then step forward until point is >= lookahead_dist
    best_idx = 0
    best_d2 = 1e12
    for i,(wx,wy) in enumerate(waypoints):
        d2 = (wx-car.x)**2 + (wy-car.y)**2
        if d2 < best_d2:
            best_d2 = d2; best_idx = i
    # step forward to reach lookahead dist
    idx = best_idx
    acc = 0.0
    samples = len(waypoints)
    while acc < lookahead_dist:
        nxt = (idx+1) % samples
        dx = waypoints[nxt][0]-waypoints[idx][0]
        dy = waypoints[nxt][1]-waypoints[idx][1]
        seg = math.hypot(dx,dy)
        acc += seg
        idx = nxt
        if acc > 10000: break
    return waypoints[idx], idx

# ---------------- world / obstacles ----------------
def generate_obstacles(n, track, forbidden_point=None, min_dist=60):
    obs=[]
    for _ in range(n):
        attempts=0
        while True:
            attempts+=1
            x = random.randint(track.cx-track.a_out+20, track.cx+track.a_out-20)
            y = random.randint(track.cy-track.b_out+20, track.cy+track.b_out-20)
            if not track.point_in_track(x,y):
                continue
            if forbidden_point and math.hypot(x-forbidden_point[0], y-forbidden_point[1])<min_dist:
                if attempts>300: break
                continue
            obs.append(pygame.Rect(int(x-10),int(y-10),20,20))
            break
    return obs

# ---------------- draw utilities ----------------
def draw_track(surf):
    pygame.draw.ellipse(surf, (90,90,90), (CX-A_OUT, CY-B_OUT, 2*A_OUT, 2*B_OUT))
    pygame.draw.ellipse(surf, (0,0,0), (CX-A_IN, CY-B_IN, 2*A_IN, 2*B_IN))
    # draw midline small dots
    for (wx,wy) in WAYPOINTS[::8]:
        pygame.draw.circle(surf, (120,120,255), (int(wx),int(wy)), 2)

# ---------------- main demo ----------------
track = type("T",(object,),{"cx":CX,"cy":CY,"a_out":A_OUT,"b_out":B_OUT,"a_in":A_IN,"b_in":B_IN,"point_in_track":point_in_track})()
obstacles = []
car = None
paused = False
debug = False

def restart():
    global car, obstacles, prev_steer
    start_x = CX + A_IN + 10
    start_y = CY
    car = Car(start_x, start_y, heading=0.0)
    obstacles.clear()
    # generate obstacles away from start
    for _ in range(12):
        while True:
            angle = random.uniform(0,2*math.pi)
            a = random.uniform(A_IN+20, A_OUT-20)
            b = random.uniform(B_IN+20, B_OUT-20)
            x = CX + a*math.cos(angle)
            y = CY + b*math.sin(angle)
            if point_in_track(x,y) and math.hypot(x-start_x,y-start_y)>80:
                obstacles.append(pygame.Rect(int(x-10),int(y-10),20,20))
                break
    prev_steer = 0.0
    print("Restart: obstacles", len(obstacles))

restart()

font = pygame.font.SysFont(None,22)
running = True
while running:
    dt = clock.tick(60)/1000.0
    for ev in pygame.event.get():
        if ev.type==pygame.QUIT:
            running=False
        if ev.type==pygame.KEYDOWN:
            if ev.key==pygame.K_r: restart()
            if ev.key==pygame.K_t: paused = not paused
            if ev.key==pygame.K_d:
                debug = not debug
                print("Debug", debug)
        if ev.type==pygame.MOUSEBUTTONDOWN:
            if ev.button==1:
                restart()

    if not paused and car.alive:
        # beams
        beam_step = params['beam_step_deg']
        beams = cast_beams(car.x, car.y, car.h, beam_step, max_range=120.0)
        gaps = detect_gaps(beams, params['safe_dist'])
        # pure pursuit target
        target_pt, target_idx = find_pursuit_target(car, WAYPOINTS, params['lookahead_wp'])
        target_angle = math.atan2(target_pt[1]-car.y, target_pt[0]-car.x)
        angle_to_wp = (target_angle - car.h + math.pi)%(2*math.pi) - math.pi
        # decide desired angle: if no front obstacle -> go to waypoint
        front_blocked = any(d < params['safe_dist'] for (a,d) in beams if abs(a) <= 20)
        chosen_dir = angle_to_wp
        chosen_gap = None
        if front_blocked and gaps:
            # find gap center closest to angle_to_wp (in degrees)
            best = None; best_diff=1e9
            for g in gaps:
                g_ang = math.radians(g['center'])
                diff = abs(((g_ang - angle_to_wp + math.pi)%(2*math.pi) - math.pi))
                if diff < best_diff:
                    best_diff = diff; best = g
            chosen_gap = best
            if chosen_gap:
                chosen_dir = math.radians(chosen_gap['center'])
        # compute steer
        diff = (chosen_dir - 0.0)  # relative angle from heading after normalizing below
        # convert angle_to_heading (we already used angle differences)
        desired = math.atan2(math.sin(chosen_dir), math.cos(chosen_dir))
        angle_diff = (desired - 0.0 + math.pi)%(2*math.pi) - math.pi
        # but angle_to_wp was relative to car.h: so better compute diff relative to car.h
        # recompute correctly:
        desired_global = (car.h + chosen_dir)
        angle_diff = (desired_global - car.h + math.pi)%(2*math.pi) - math.pi
        # simpler: angle_diff = chosen_dir (already relative)
        angle_diff = chosen_dir
        raw = params['Kp'] * angle_diff
        raw = max(-math.radians(30), min(math.radians(30), raw))
        steer = prev_steer * params['beta'] + raw * (1-params['beta'])
        prev_steer = steer

        car.step(steer)

        # collision
        if not point_in_track(car.x, car.y):
            car.alive = False
        for o in obstacles:
            if o.collidepoint(car.x, car.y):
                car.alive = False

    # draw
    screen.fill((0,0,0))
    draw_track(screen)
    for o in obstacles:
        pygame.draw.rect(screen, (255,200,0), o)
    # draw target waypoint
    tx, ty = WAYPOINTS[0]
    # draw beams
    if car:
        beams = cast_beams(car.x, car.y, car.h, params['beam_step_deg'], max_range=120.0)
        for i,(a,d) in enumerate(beams):
            ang = car.h + math.radians(a)
            ex = car.x + math.cos(ang)*d
            ey = car.y + math.sin(ang)*d
            color = (0,200,0) if d>params['safe_dist'] else (200,60,60)
            pygame.draw.line(screen, color, (int(car.x),int(car.y)), (int(ex),int(ey)), 2)
            if debug and i%2==0:
                midx = car.x + math.cos(ang)*min(d,params['safe_dist'])
                midy = car.y + math.sin(ang)*min(d,params['safe_dist'])
                txt = font.render(f"{int(d)}", True, (200,200,200))
                screen.blit(txt, (int(midx), int(midy)))
    # draw chosen waypoint and gap
    if car and car.alive:
        tpt, tidx = find_pursuit_target(car, WAYPOINTS, params['lookahead_wp'])
        pygame.draw.circle(screen, (0,0,255), (int(tpt[0]), int(tpt[1])), 6)
        if 'chosen_gap' in locals() and chosen_gap:
            center_ang = math.radians(chosen_gap['center'])
            ex = car.x + math.cos(car.h + center_ang)*80
            ey = car.y + math.sin(car.h + center_ang)*80
            pygame.draw.circle(screen, (255,0,255), (int(ex), int(ey)), 6)
        car.draw(screen)
    # HUD
    txt = font.render("R:restart  T:pause  D:debug", True, (200,200,200))
    screen.blit(txt, (10,10))
    paramtxt = font.render(f"Kp={params['Kp']:.2f} safe={params['safe_dist']:.0f} beta={params['beta']:.2f} beam={params['beam_step_deg']:.1f}", True, (200,200,200))
    screen.blit(paramtxt, (10,34))
    if not car.alive:
        t = font.render("Car crashed! Press R to restart", True, (255,0,0))
        screen.blit(t, (WIDTH//2-140, 20))
    pygame.display.flip()

pygame.quit()