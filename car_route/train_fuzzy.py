"""
pp_gap_train.py
简单的 GA 用来调优 Pure Pursuit + Gap-Avoidance 参数

参数向量（长度 6）:
 [Kp, safe_dist, lookahead_steps, beta, beam_step_deg, lookahead_wp]

注意:
- 训练在 CPU 上运行，默认 generations=40, pop_size=24。可调小做快速测试。
- 运行后保存： car_route/best_pp_params.npy
- 评估时会随机生成障碍布局并计算平均得分（距离+稳定性-碰撞惩罚-偏航惩罚）

运行:
    python car_route/pp_gap_train.py
"""

import numpy as np, random, math, time, os
from copy import deepcopy

# ---- GA hyperparams ----
POP_SIZE = 24
GENERATIONS = 40
MUT_RATE = 0.18
MUT_SCALE = 0.15
ELITE = 2
EPISODES = 6
EPISODE_STEPS = 600

# ---- bounds for params ----
# [Kp, safe_dist, lookahead_steps, beta, beam_step_deg, lookahead_wp]
BOUNDS = [
    (0.3, 2.0),   # Kp
    (30.0, 120.0),# safe_dist
    (2, 30),      # lookahead_steps (int)
    (0.5, 0.95),  # beta
    (3.0, 12.0),  # beam_step_deg
    (20.0, 120.0) # lookahead_wp
]

# ---- world utilities copied from sim (lightweight) ----
CX, CY = 900//2, 700//2
A_OUT, B_OUT = 300, 220
A_IN, B_IN = 210, 160

def point_in_track(x,y):
    dx, dy = x-CX, y-CY
    val_out = (dx/A_OUT)**2 + (dy/B_OUT)**2
    val_in = (dx/A_IN)**2 + (dy/B_IN)**2
    return (val_in >= 1.0) and (val_out <= 1.0)

def ray_distance(x,y,ang, obstacles, max_range=120.0, step=3.0):
    dist=0.0
    dx, dy = math.cos(ang), math.sin(ang)
    while dist < max_range:
        px = x + dx*dist; py = y + dy*dist
        if not point_in_track(px,py):
            return dist
        for ox,oy,s in obstacles:
            if abs(px-ox)<=s/2 and abs(py-oy)<=s/2:
                return dist
        dist += step
    return max_range

def generate_obstacles(n=10, seed=None):
    rng = random.Random(seed)
    obs=[]
    for _ in range(n):
        attempts=0
        while True:
            attempts+=1
            angle = rng.uniform(0,2*math.pi)
            a = rng.uniform(A_IN+20, A_OUT-20)
            b = rng.uniform(B_IN+20, B_OUT-20)
            x = CX + a*math.cos(angle)
            y = CY + b*math.sin(angle)
            size = rng.uniform(15,25)
            if point_in_track(x,y):
                obs.append((x,y,size))
                break
            if attempts>300: break
    return obs

def cast_beams_nums(x,y,heading,beam_step_deg, obstacles, max_range=120.0):
    beams=[]
    half=90
    a = -half
    while a<=half+1e-6:
        ang = heading + math.radians(a)
        d = ray_distance(x,y,ang,obstacles, max_range=max_range)
        beams.append((a,d))
        a += beam_step_deg
    return beams

def detect_gaps_nums(beams, safe_dist):
    free = [1 if d>safe_dist else 0 for (_,d) in beams]
    gaps=[]
    n=len(free); i=0
    while i<n:
        if free[i]==1:
            j=i
            while j<n and free[j]==1: j+=1
            angles=[beams[k][0] for k in range(i,j)]
            center=(angles[0]+angles[-1])/2.0
            width=angles[-1]-angles[0]+1e-9
            gaps.append({'start':i,'end':j-1,'center':center,'width':width})
            i=j
        else:
            i+=1
    return gaps

def find_pursuit_target_simple(x,y, waypoints, lookahead_dist):
    best_idx=0; best_d2=1e12
    for i,(wx,wy) in enumerate(waypoints):
        d2=(wx-x)**2+(wy-y)**2
        if d2<best_d2:
            best_d2=d2; best_idx=i
    idx=best_idx; acc=0.0; samples=len(waypoints)
    while acc<lookahead_dist:
        nxt=(idx+1)%samples
        dx = waypoints[nxt][0]-waypoints[idx][0]
        dy = waypoints[nxt][1]-waypoints[idx][1]
        seg = math.hypot(dx,dy); acc+=seg; idx=nxt
        if acc>10000: break
    return waypoints[idx], idx

WAYPOINTS = [(CX + ((A_IN+A_OUT)/2.0)*math.cos(2*math.pi*k/360), CY + ((B_IN+B_OUT)/2.0)*math.sin(2*math.pi*k/360)) for k in range(360)]

# ---- evaluation of a parameter vector ----
def evaluate(params, seed=None):
    Kp, safe_dist, lookahead_steps, beta, beam_step_deg, lookahead_wp = params
    lookahead_steps = int(round(lookahead_steps))
    total = 0.0
    rng = random.Random(seed)
    for ep in range(EPISODES):
        # random start
        angle0 = rng.uniform(0,2*math.pi)
        r0 = rng.uniform(A_IN+20, A_OUT-20)
        x = CX + r0*math.cos(angle0)
        y = CY + r0*math.sin(angle0)
        heading = rng.uniform(-math.pi, math.pi)
        obstacles = generate_obstacles(n=10, seed=rng.randint(0,100000))
        alive = True
        score = 0.0
        prev_steer = 0.0
        for step in range(EPISODE_STEPS):
            beams = cast_beams_nums(x,y,heading,beam_step_deg, obstacles, max_range=120.0)
            gaps = detect_gaps_nums(beams, safe_dist)
            target_pt, tidx = find_pursuit_target_simple(x,y, WAYPOINTS, lookahead_wp)
            target_angle = math.atan2(target_pt[1]-y, target_pt[0]-x)
            angle_to_wp = (target_angle - heading + math.pi)%(2*math.pi) - math.pi
            front_blocked = any(d<safe_dist for (a,d) in beams if abs(a)<=20)
            chosen_dir = angle_to_wp
            if front_blocked and gaps:
                best=None; bd=1e9
                for g in gaps:
                    g_ang = math.radians(g['center'])
                    diff = abs(((g_ang - angle_to_wp + math.pi)%(2*math.pi) - math.pi))
                    if diff < bd:
                        bd = diff; best=g
                if best: chosen_dir = math.radians(best['center'])
            # steer
            raw = Kp * chosen_dir
            raw = max(-math.radians(30), min(math.radians(30), raw))
            steer = prev_steer * beta + raw * (1-beta)
            prev_steer = steer
            # step forward
            heading += steer * 0.08
            x += math.cos(heading)*2.0
            y += math.sin(heading)*2.0
            # collision
            if not point_in_track(x,y):
                alive=False; score -= 150; break
            for ox,oy,s in obstacles:
                if abs(x-ox)<=s/2+5 and abs(y-oy)<=s/2+5:
                    alive=False; score -= 150; break
            if not alive: break
            # reward: forward progress + closeness to midline
            dx = x - CX; dy = y - CY
            d_center = math.hypot(dx,dy)
            track_center = (A_IN + A_OUT)/2.0
            step_reward = 1.0 + max(0, 1 - abs(d_center - track_center)/track_center)
            # small penalty on large absolute heading diff to encourage facing forward
            step_reward -= 0.05 * abs(angle_to_wp)
            score += step_reward
        total += score
    return total/EPISODES

# ---- GA ----
def random_individual():
    vec = []
    for (lo,hi) in BOUNDS:
        vec.append(random.uniform(lo,hi))
    return np.array(vec)

def mutate(child):
    for i in range(len(child)):
        if random.random() < MUT_RATE:
            lo,hi = BOUNDS[i]
            span = hi-lo
            child[i] += random.gauss(0, MUT_SCALE*span)
            child[i] = max(lo, min(hi, child[i]))
    return child

def crossover(a,b):
    return 0.5*(a+b)

def run():
    pop = [random_individual() for _ in range(POP_SIZE)]
    fitness = np.zeros(POP_SIZE)
    best = None; best_f = -1e12
    for g in range(GENERATIONS):
        for i,ind in enumerate(pop):
            fitness[i] = evaluate(ind, seed=g*100+i)
        idxs = np.argsort(fitness)
        # log
        print(f"Gen {g+1}/{GENERATIONS} best={fitness[idxs[-1]]:.1f} mean={fitness.mean():.1f}")
        if fitness[idxs[-1]] > best_f:
            best_f = fitness[idxs[-1]]; best = pop[idxs[-1]].copy()
        # elitism
        new_pop = [pop[idxs[-1-i]].copy() for i in range(ELITE)]
        while len(new_pop) < POP_SIZE:
            a,b = random.sample(pop,2)
            child = crossover(a,b)
            child = mutate(child)
            new_pop.append(child)
        pop = new_pop
    # save best
    os.makedirs("car_route", exist_ok=True)
    np.save("car_route/best_pp_params.npy", best)
    np.savez("car_route/pp_train_meta.npz", best_fitness=best_f, gens=GENERATIONS)
    print("Saved best to car_route/best_pp_params.npy best_f=", best_f)
    return best

if __name__ == "__main__":
    start=time.time()
    best = run()
    print("Time:", time.time()-start)