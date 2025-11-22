"""
Lightweight training entry that invokes a GA to optimize PP params.
Training evaluation uses the FuzzyPID controller (deterministic FuzzyPID with base gains).
Run: python car_route/refactor/run_train.py
"""
import os
import sys
#确保可以导入本目录下的模块
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import random
import math
from track import generate_midline, point_in_track
from sensors import cast_beams, detect_gaps
from controller import FuzzyPID
from obstacle import generate_obstacles

# GA参数
POP_SIZE = 40 #种群大小
GENERATIONS = 60 #进化代数
MUT_RATE = 0.18 #变异概率
MUT_SCALE = 0.15 #变异幅度比例
ELITE = 2 #精英保留数量
EPISODES = 5 #每个个体评估的仿真次数
EPISODE_STEPS = 500 #每次仿真步数
# 轨道中线
WAYPOINTS = generate_midline(samples=360)

BOUNDS = [
    (0.3, 2.0),   # Kp转向比例增益
    (30.0, 120.0),# 安全距离
    (2, 30),      # 前视步数
    (0.5, 0.95),  # 转向平滑系数
    (3.0, 12.0),  # 激光角度步长
    (20.0, 120.0) # 路径点前视距离
]

from config import B_IN, B_OUT, CX, CY, A_IN, A_OUT

def random_individual():
    return np.array([random.uniform(lo, hi) for (lo, hi) in BOUNDS])
# 变异操作，幅度与参数范围成比例
def mutate(ind):
    for i in range(len(ind)):
        if random.random() < MUT_RATE:
            lo, hi = BOUNDS[i]
            span = hi - lo
            ind[i] += random.gauss(0, MUT_SCALE * span) # 高斯变异
            ind[i] = max(lo, min(hi, ind[i])) # 边界约束
    return ind
# 单点交叉
def crossover(a, b):
    return 0.5 * (a + b)
# 评估个体表现
def evaluate(params, seed=None):
    Kp, safe_dist, lookahead_steps, beta, beam_step_deg, lookahead_wp = params
    total = 0.0
    rng = random.Random(seed)
    for ep in range(EPISODES):
        angle0 = rng.uniform(0, 2 * math.pi)
        r0 = rng.uniform(A_IN + 20, A_OUT - 20)
        x = CX + r0 * math.cos(angle0)
        y = CY + r0 * math.sin(angle0)
        heading = rng.uniform(-math.pi, math.pi)
        # 随机生成障碍物
        obstacles = []
        obstacles = generate_obstacles(8, min_dist=100)
        # 初始化模糊PID控制器
        fpid = FuzzyPID(kp=Kp, ki=0.01, kd=0.05, integral_limit=1.0, max_err=math.pi)
        score = 0.0
        prev_steer = 0.0
        for step in range(EPISODE_STEPS):
            # 传感器扫描
            beams = cast_beams(x, y, heading, beam_step_deg, [], max_range=120.0)
            gaps = detect_gaps(beams, safe_dist)
            #   寻找路径目标点， 前方无障碍，跟踪路径点避障模式；前方有障碍，选择最接近目标方向的间隙
            target_idx = 0
            target_pt = WAYPOINTS[target_idx]
            target_angle = math.atan2(target_pt[1] - y, target_pt[0] - x)
            angle_to_wp = (target_angle - heading + math.pi) % (2 * math.pi) - math.pi
            front_blocked = any(d < safe_dist for (_, d) in beams if abs(_) <= 20)
            chosen_dir = angle_to_wp
            if front_blocked and gaps:
                best = None; bd = 1e9
                for g in gaps:
                    g_ang = math.radians(g['center'])
                    diff = abs(((g_ang - angle_to_wp + math.pi) % (2 * math.pi) - math.pi))
                    if diff < bd:
                        bd = diff; best = g
                if best:
                    chosen_dir = math.radians(best['center'])
            # 计算转向
            steer_raw = fpid.update(chosen_dir, dt=1.0/60.0)
            steer = max(-math.radians(30), min(math.radians(30), steer_raw))
            steer = prev_steer * beta + steer * (1 - beta)
            prev_steer = steer
            # 车辆状态更新
            heading += steer * 0.08
            x += math.cos(heading) * 2.0
            y += math.sin(heading) * 2.0
            # 碰撞惩罚
            if not point_in_track(x, y):
                score -= 150
                break
            # 计算奖励
            dx = x - CX; dy = y - CY
            d_center = math.hypot(dx, dy)
            track_center = (A_IN + A_OUT) / 2.0
            step_reward = 1.0 + max(0, 1 - abs(d_center - track_center) / track_center)
            step_reward -= 0.05 * abs(angle_to_wp)
            score += step_reward
        total += score
    return total / EPISODES

def run():
    pop = [random_individual() for _ in range(POP_SIZE)]
    fitness = [0.0] * POP_SIZE
    best = None
    best_f = -1e12
    for g in range(GENERATIONS):
        # 评估当前种群
        for i, ind in enumerate(pop):
            fitness[i] = evaluate(ind, seed=g * 100 + i)
        # 选择和生成下一代
        idxs = sorted(range(len(fitness)), key=lambda k: fitness[k])
        print(f"Gen {g+1}/{GENERATIONS} best={fitness[idxs[-1]]:.1f} mean={sum(fitness)/len(fitness):.1f}")
        # 更新最优
        if fitness[idxs[-1]] > best_f:
            best_f = fitness[idxs[-1]]; best = pop[idxs[-1]].copy()
        new_pop = [pop[idxs[-1 - i]].copy() for i in range(ELITE)] # 精英保留
        while len(new_pop) < POP_SIZE:
            a, b = random.sample(pop, 2) 
            child = crossover(a, b)
            child = mutate(child)
            new_pop.append(child)
        pop = new_pop
    os.makedirs('car_route', exist_ok=True)
    np.save('car_route/best_pp_params.npy', best)
    print('Saved car_route/best_pp_params.npy')

if __name__ == '__main__':
    run()