"""
Training entry: GA optimizes fuzzy-rule outputs and membership functions.

Usage:
    python run_train_fuzzy_rules.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import random
import math
import time
from track import generate_midline, point_in_track
from sensors import cast_beams, detect_gaps
from metrics import TrainingMetrics

# GA参数
POP_SIZE = 30 #种群大小
GENERATIONS = 80 #进化代数
MUT_RATE = 0.15 #变异概率
MUT_SCALE = 0.15 #变异幅度比例
ELITE = 5 #精英保留数量
EPISODES = 5 #每个个体评估的仿真次数
EPISODE_STEPS = 600 #每次仿真步数

WAYPOINTS = generate_midline(samples=360)

# 初始化参数
_default_safe_dist = (30.0 + 120.0) / 2.0        # 75.0
_default_lookahead_steps = int(round((2 + 30) / 2.0))  # 16
_default_beta = (0.5 + 0.95) / 2.0               # 0.725
_default_beam_step_deg = (3.0 + 12.0) / 2.0      # 7.5
_default_lookahead_wp = int(round((20.0 + 120.0) / 2.0)) # 70

from config import B_IN, B_OUT, CX, CY, A_IN, A_OUT

#基因编码长度
GENE_SIZE = 45

def random_individual():
    # 基因初始化为均匀分布在[0,1]
    return np.random.rand(GENE_SIZE)

def mutate(child):
    # 基因变异
    for i in range(len(child)):
        if random.random() < MUT_RATE:
            child[i] += random.gauss(0, MUT_SCALE)
        
            child[i] = max(0.0, min(1.0, child[i]))
    return child

def crossover(a, b):
    return 0.5 * (a + b)

class FuzzyPID:
    def __init__(self, gene, base_kp=0.9, base_ki=0.01, base_kd=0.05,
                 max_err=math.pi, max_der=5.0):

        assert len(gene) == GENE_SIZE
        self.gene = np.array(gene, dtype=float)

        self.base_kp = base_kp
        self.base_ki = base_ki
        self.base_kd = base_kd

        self.max_err = max_err
        self.max_der = max_der

        #  隶属度函数解析
        self.mf_e = np.zeros((3,3))
        self.mf_de = np.zeros((3,3))
        ptr = 0
        for i in range(3):
            vals = np.sort(self.gene[ptr:ptr+3])
            self.mf_e[i] = vals
            ptr += 3
        for i in range(3):
            vals = np.sort(self.gene[ptr:ptr+3])
            self.mf_de[i] = vals
            ptr += 3
        # 模糊规则解析
        self.rules = np.zeros((9,3))
        rule_raw = self.gene[18:45]
        for i in range(9):
            gkp = rule_raw[i*3 + 0]
            gki = rule_raw[i*3 + 1]
            gkd = rule_raw[i*3 + 2]
            dkp = (gkp * 2.0) - 1.0          # [-1,1]
            dki = (gki * 0.1) - 0.05        # [-0.05,0.05]
            dkd = (gkd * 1.0) - 0.5         # [-0.5,0.5]
            self.rules[i] = np.array([dkp, dki, dkd])

        # 初始化状态
        self.integral = 0.0
        self.prev_err = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_err = 0.0

    @staticmethod
    def tri_mf(x, a, b, c):
        #  三角形隶属度函数
        if x <= a or x >= c:
            return 0.0
        if x == b:
            return 1.0
        if x < b:
            denom = (b - a) if (b - a) != 0 else 1e-9
            return (x - a) / denom
        else:
            denom = (c - b) if (c - b) != 0 else 1e-9
            return (c - x) / denom

    def fuzzify_e(self, err_norm):
      
        return [self.tri_mf(err_norm, *self.mf_e[i]) for i in range(3)]

    def fuzzify_de(self, der_norm):
       
        return [self.tri_mf(der_norm, *self.mf_de[i]) for i in range(3)]

    def inference(self, e_mf, de_mf):
        # 模糊推理
        num = np.zeros(3)
        den = 0.0
        idx = 0
        for i in range(3):
            for j in range(3):
                w = e_mf[i] * de_mf[j]
                if w > 0:
                    num += w * self.rules[idx]
                    den += w
                idx += 1
        if den == 0.0:
            return 0.0, 0.0, 0.0
        out = num / den
        return out[0], out[1], out[2]

    def update(self, err, dt):
        # PID 更新
        der = (err - self.prev_err) / (dt if dt > 0 else 1e-6)

        # 归一化
        e_norm = max(-1, min(1, err / math.pi))
        de_norm = max(-1, min(1, der / 5.0))

        e_mf = self.fuzzify_e((e_norm + 1) / 2)
        de_mf = self.fuzzify_de((de_norm+1)/2)

        dKp, dKi, dKd = self.inference(e_mf, de_mf)

        
        Kp = self.base_kp + dKp
        Ki = self.base_ki + dKi
        Kd = self.base_kd + dKd


        self.integral += err * dt
        D = (err - self.prev_err)

        u = Kp * err + Ki * self.integral + Kd * D

        self.prev_err = err
        return u, Kp, Ki, Kd

# 纯追踪目标点选择
def find_pursuit_target(x, y, waypoints, lookahead_dist):
    best_idx = 0
    best_d2 = 1e12
    for i, (wx, wy) in enumerate(waypoints):
        d2 = (wx - x)**2 + (wy - y)**2
        if d2 < best_d2:
            best_d2 = d2
            best_idx = i

    idx = best_idx
    acc = 0.0
    n = len(waypoints)

    while acc < lookahead_dist:
        nxt = (idx + 1) % n
        dx = waypoints[nxt][0] - waypoints[idx][0]
        dy = waypoints[nxt][1] - waypoints[idx][1]
        seg = math.hypot(dx, dy)

        acc += seg
        idx = nxt

        if acc > 10000:
            break

    return waypoints[idx], idx

def choose_gap_or_target(angle_to_wp, gaps):
    if not gaps:
        return angle_to_wp
    best = None
    best_diff = float('inf')
    for g in gaps:
        g_ang = math.radians(g['center'])
        diff = abs(((g_ang - angle_to_wp + math.pi) % (2 * math.pi) - math.pi))
        if diff < best_diff:
            best_diff = diff
            best = g
    if best:
        return math.radians(best['center'])
    return angle_to_wp

#评估函数
def evaluate(gene, seed=None):
    safe_dist = _default_safe_dist
    lookahead_steps = _default_lookahead_steps
    beta = _default_beta
    beam_step_deg = _default_beam_step_deg
    lookahead_wp = _default_lookahead_wp

    total = 0.0
    rng = random.Random(seed)
    for ep in range(EPISODES):
        angle0 = rng.uniform(0, 2*math.pi)
        r0 = rng.uniform(A_IN+20, A_OUT-20)
        x = CX + r0 * math.cos(angle0)
        y = CY + r0 * math.sin(angle0)
        heading = rng.uniform(-math.pi, math.pi)

      
        obstacles = []
        for _ in range(10):
            attempts = 0
            while True:
                attempts += 1
                angle = rng.uniform(0, 2*math.pi)
                a = rng.uniform(A_IN+20, A_OUT-20)
                b = rng.uniform(B_IN+20, B_OUT+20) if hasattr(sys.modules.get('config'), 'B_OUT') else rng.uniform(B_IN+20, B_OUT+20)
                ox = CX + a * math.cos(angle)
                oy = CY + b * math.sin(angle)
                size = rng.uniform(15, 25)
                if point_in_track(ox, oy):
                    obstacles.append((ox, oy, size))
                    break
                if attempts > 300:
                    break

        score = 0.0
        prev_steer = 0.0

        
        controller = FuzzyPID(gene)
        controller.reset()

        for step in range(EPISODE_STEPS):
        
            beams = cast_beams(x, y, heading, beam_step_deg, [], max_range=120.0)
            gaps = detect_gaps(beams, safe_dist)

            
            target_pt = WAYPOINTS[0]
            target_angle = math.atan2(target_pt[1] - y, target_pt[0] - x)
            angle_to_wp = (target_angle - heading + math.pi) % (2*math.pi) - math.pi

            front_blocked = any(d < safe_dist for (a, d) in beams if abs(a) <= 20)
            chosen_dir = angle_to_wp
            if front_blocked and gaps:
                best = None; bd = 1e9
                for g in gaps:
                    g_ang = math.radians(g['center'])
                    diff = abs(((g_ang - angle_to_wp + math.pi) % (2*math.pi) - math.pi))
                    if diff < bd:
                        bd = diff; best = g
                if best:
                    chosen_dir = math.radians(best['center'])

       
            err = chosen_dir

         
            u, Kp, Ki, Kd = controller.update(err, dt=0.08)

            raw = u
            raw = max(-math.radians(30), min(math.radians(30), raw))
            steer = prev_steer * beta + raw * (1 - beta)
            prev_steer = steer

         
            heading += steer * 0.08
            x += math.cos(heading) * 2.0
            y += math.sin(heading) * 2.0

            # 碰撞惩罚
            if not point_in_track(x, y):
                score -= 150
                break
            for ox, oy, s in obstacles:
                if abs(x - ox) <= s/2 + 5 and abs(y - oy) <= s/2 + 5:
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
    metrics = TrainingMetrics()
    pop = [random_individual() for _ in range(POP_SIZE)]
    fitness = [0.0] * POP_SIZE
    best = None; best_f = -1e12
    for g in range(GENERATIONS):
        for i, ind in enumerate(pop):
            fitness[i] = evaluate(ind, seed=g*100+i)
      
        f_best = max(fitness)
        f_mean = float(np.mean(fitness))
        f_std = float(np.std(fitness))
        metrics.add_generation(g+1, f_best, f_mean, f_std)
        print(f"Gen {g+1}/{GENERATIONS} best={f_best:.1f} mean={f_mean:.1f} std={f_std:.2f}")
        if f_best > best_f:
            best_f = f_best; best = pop[int(np.argmax(fitness))].copy()
        # 生成下一代
        idxs = np.argsort(fitness)
        new_pop = [pop[idxs[-1 - i]].copy() for i in range(ELITE)]
        while len(new_pop) < POP_SIZE:
            a, b = random.sample(pop, 2)
            child = crossover(a, b)
            child = mutate(child)
            new_pop.append(child)
        pop = new_pop
    # 保存最优个体和训练数据 (路径与原脚本一致)
    if best is not None:
        np.save('car_track/best_pp_params.npy', best)
    metrics.save('car_track/pp_train_metrics.npz')
    metrics.plot(path_png='car_track/pp_train_final.png', show=False)
    print("Training finished. best fitness:", best_f)

if __name__ == "__main__":
    start = time.time()
    run()
    print("Elapsed", time.time()-start)
