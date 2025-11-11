"""
train_fuzzy.py (最终修正版)
- 与演示脚本保持一致的感知与控制逻辑：
  - nearest_point_ahead（优先选择车前方目标）
  - 模糊前向权重 = far + 0.5*mid
  - 自适应混合：根据前向安全程度在目标/模糊间动态混合
- 训练保存：./car_route/best_controller.npy 和 train_metadata.npz
"""

import math
import random
import numpy as np
import os
import time

# ================= 参数设置 =================
MAX_STEER = math.radians(30)
SENSOR_RANGE = 120.0
POP_SIZE = 36
MUT_RATE = 0.12
MUT_SCALE = 0.25
ELITE = 2
EVAL_EPISODES = 5
EPISODE_STEPS = 800

TRACK_A_OUT, TRACK_B_OUT = 250, 180
TRACK_A_IN, TRACK_B_IN = 180, 130
TRACK_CX, TRACK_CY = 450, 300

FUZZY_LABELS = ['near', 'mid', 'far']

# =================== 隶属函数 ===================
def tri_mf(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if (b - a) != 0 else 0.0
    return (c - x) / (c - b) if (c - b) != 0 else 0.0

FUZZY_POINTS = {
    'near': (0.0, 0.0, 0.5),
    'mid': (0.0, 0.5, 1.0),
    'far': (0.5, 1.0, 1.0),
}

def fuzzify_sensor(val):
    m = {}
    for k, (a, b, c) in FUZZY_POINTS.items():
        m[k] = tri_mf(val, a, b, c)
    return m

# =================== 椭圆赛道 ===================
class EllipseTrack:
    def __init__(self):
        self.cx, self.cy = TRACK_CX, TRACK_CY
        self.a_out, self.b_out = TRACK_A_OUT, TRACK_B_OUT
        self.a_in, self.b_in = TRACK_A_IN, TRACK_B_IN

    def inside_track(self, x, y):
        dx = x - self.cx
        dy = y - self.cy
        val_out = (dx / self.a_out) ** 2 + (dy / self.b_out) ** 2
        val_in = (dx / self.a_in) ** 2 + (dy / self.b_in) ** 2
        return val_in >= 1.0 and val_out <= 1.0

    def nearest_point_ahead(self, x, y, heading, samples=360, lookahead=14):
        mid_a = (self.a_in + self.a_out) / 2.0
        mid_b = (self.b_in + self.b_out) / 2.0
        best_idx = 0
        best_d2 = float('inf')
        pts = []
        for k in range(samples):
            ang = 2 * math.pi * k / samples
            px = self.cx + mid_a * math.cos(ang)
            py = self.cy + mid_b * math.sin(ang)
            pts.append((px, py))
            d2 = (px - x)**2 + (py - y)**2
            if d2 < best_d2:
                best_d2 = d2
                best_idx = k
        chosen_idx = best_idx
        best_abs = float('inf')
        for offs in range(0, lookahead+1):
            k = (best_idx + offs) % samples
            px, py = pts[k]
            target_angle = math.atan2(py - y, px - x)
            diff = (target_angle - heading + math.pi) % (2*math.pi) - math.pi
            if abs(diff) < math.pi/2:
                if abs(diff) < best_abs:
                    best_abs = abs(diff)
                    chosen_idx = k
        return chosen_idx

# =================== 世界 ===================
class World:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.track = EllipseTrack()
        self.obstacles = []
        self.generate_obstacles()

    def generate_obstacles(self):
        self.obstacles = []
        for _ in range(10):
            attempts = 0
            while True:
                attempts += 1
                angle = random.uniform(0, 2 * math.pi)
                a = random.uniform(self.track.a_in + 20, self.track.a_out - 20)
                b = random.uniform(self.track.b_in + 20, self.track.b_out - 20)
                x = self.track.cx + a * math.cos(angle)
                y = self.track.cy + b * math.sin(angle)
                size = random.uniform(15, 25)
                if self.track.inside_track(x, y):
                    self.obstacles.append((x, y, size))
                    break
                if attempts > 300:
                    break

    def query_distance(self, x, y, angle, max_range=SENSOR_RANGE):
        step = 3.0
        dist = 0.0
        dx, dy = math.cos(angle), math.sin(angle)
        while dist < max_range:
            px = x + dx * dist
            py = y + dy * dist
            if not self.track.inside_track(px, py):
                return dist
            for ox, oy, s in self.obstacles:
                if abs(px - ox) <= s / 2 and abs(py - oy) <= s / 2:
                    return dist
            dist += step
        return max_range

# =================== 车辆 ===================
class Car:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.heading = 0.0
        self.speed = 2.0

    def step(self, steer):
        steer = max(-MAX_STEER, min(MAX_STEER, steer))
        self.heading += steer * 0.08
        self.x += math.cos(self.heading) * self.speed
        self.y += math.sin(self.heading) * self.speed

    def collided(self, world):
        if not world.track.inside_track(self.x, self.y):
            return True
        for ox, oy, s in world.obstacles:
            if abs(self.x - ox) <= s / 2 + 5 and abs(self.y - oy) <= s / 2 + 5:
                return True
        return False

# =================== 模糊控制器（返回 steer, front_m） ===================
class FuzzyController:
    def __init__(self, rule_vector):
        self.rules = np.array(rule_vector, dtype=float)

    def decide(self, left_dist, front_dist, right_dist):
        ln = max(0.0, min(1.0, left_dist / SENSOR_RANGE))
        rn = max(0.0, min(1.0, right_dist / SENSOR_RANGE))
        fn = max(0.0, min(1.0, front_dist / SENSOR_RANGE))
        left_m = fuzzify_sensor(ln)
        right_m = fuzzify_sensor(rn)
        front_m = fuzzify_sensor(fn)
        front_weight = front_m['far'] + 0.5 * front_m['mid']
        out_num = 0.0
        out_den = 0.0
        for i, llabel in enumerate(FUZZY_LABELS):
            for j, rlabel in enumerate(FUZZY_LABELS):
                mu = left_m[llabel] * right_m[rlabel] * front_weight
                val = self.rules[i * 3 + j]
                out_num += mu * val
                out_den += mu
        steer = out_num / out_den if out_den > 1e-8 else 0.0
        steer = max(-MAX_STEER, min(MAX_STEER, steer))
        return steer, front_m

# =================== GA ===================
class GAOptimizer:
    def __init__(self):
        self.pop = [np.random.uniform(-MAX_STEER, MAX_STEER, 9) for _ in range(POP_SIZE)]
        self.fitness = np.zeros(POP_SIZE)
        self.generation = 0

    def evaluate_individual(self, ind_vector, seed=None):
        total = 0.0
        for e in range(EVAL_EPISODES):
            world = World(seed=(seed + e if seed else None))
            angle0 = random.uniform(0, 2*math.pi)
            r0 = random.uniform(world.track.a_in + 20, world.track.a_out - 20)
            x0 = world.track.cx + r0 * math.cos(angle0)
            y0 = world.track.cy + r0 * math.sin(angle0)
            car = Car(x0, y0)
            controller = FuzzyController(ind_vector)
            score = 0.0
            for _ in range(EPISODE_STEPS):
                left = world.query_distance(car.x, car.y, car.heading + math.radians(-45))
                right = world.query_distance(car.x, car.y, car.heading + math.radians(45))
                front = world.query_distance(car.x, car.y, car.heading)
                fuzzy, front_m = controller.decide(left, front, right)

                idx = world.track.nearest_point_ahead(car.x, car.y, car.heading, samples=360, lookahead=14)
                tx = world.track.cx + ((world.track.a_in + world.track.a_out)/2) * math.cos(2*math.pi*idx/360)
                ty = world.track.cy + ((world.track.b_in + world.track.b_out)/2) * math.sin(2*math.pi*idx/360)
                target_angle = math.atan2(ty - car.y, tx - car.x)
                diff = (target_angle - car.heading + math.pi) % (2*math.pi) - math.pi

                front_weight = front_m['far'] + 0.5 * front_m['mid']
                alpha_target = 1.0 - front_weight
                alpha_target = max(0.1, min(0.6, alpha_target))
                K_diff = 0.4

                steer = alpha_target * (K_diff * diff) + (1.0 - alpha_target) * fuzzy
                car.step(steer)
                if car.collided(world):
                    score -= 100
                    break
                dx = car.x - world.track.cx
                dy = car.y - world.track.cy
                d_center = math.sqrt(dx**2 + dy**2)
                track_center = (world.track.a_in + world.track.a_out)/2
                score += 1 + max(0, 1 - abs(d_center - track_center)/track_center)
            total += score
        return total / EVAL_EPISODES

    def epoch(self):
        for i, ind in enumerate(self.pop):
            self.fitness[i] = self.evaluate_individual(ind, seed=self.generation*100 + i)
        elite_idx = np.argsort(self.fitness)[-ELITE:][::-1]
        new_pop = [self.pop[idx].copy() for idx in elite_idx]
        while len(new_pop) < POP_SIZE:
            a, b = random.sample(self.pop, 2)
            child = 0.5*(a + b)
            for i in range(9):
                if random.random() < MUT_RATE:
                    child[i] += np.random.normal(0, MUT_SCALE*MAX_STEER)
            new_pop.append(np.clip(child, -MAX_STEER, MAX_STEER))
        self.pop = new_pop
        self.generation += 1

    def best_controller(self):
        return self.pop[int(np.argmax(self.fitness))]

# =================== 主程序 ===================
if __name__ == "__main__":
    ga = GAOptimizer()
    generations = 60
    best_overall = None
    best_fitness = -1e9
    start_time = time.time()
    for g in range(generations):
        ga.epoch()
        gen_best = np.max(ga.fitness)
        print(f"第 {g+1}/{generations} 代，最佳适应度 = {gen_best:.1f}")
        if gen_best > best_fitness:
            best_fitness = gen_best
            best_overall = ga.best_controller().copy()
    elapsed = time.time() - start_time
    os.makedirs("./car_route", exist_ok=True)
    if best_overall is not None:
        np.save("./car_route/best_controller.npy", best_overall)
        np.savez("./car_route/train_metadata.npz", best_fitness=best_fitness, generations=generations, elapsed_seconds=elapsed)
        print("✅ 训练完成，已保存最佳参数到 car_route/best_controller.npy")
        print(f"最佳适应度: {best_fitness:.2f}, 用时 {elapsed:.1f}s")
    else:
        print("训练失败：未找到最佳参数。")