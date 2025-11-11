import pygame
import numpy as np
import math
import sys
import os
import random
import time

# ==========================
# car_sim.py (最终修正版)
# - 稳健的 nearest_point_ahead（优先前方目标）
# - 模糊前向权重调整（far + 0.5*mid）
# - 自适应混合：根据前向安全程度在目标/模糊间动态混合
# - 平滑 steer（一阶滤波）避免瞬态 clamp 导致失控
# - 可视化射线，按 R 或鼠标左键重启，按 D 切换 debug 输出
# ==========================

pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("小车避障演示（椭圆赛道）")
clock = pygame.time.Clock()

SENSOR_RANGE = 120.0
MAX_STEER = math.radians(30)

best_params_path = "./car_route/best_controller.npy"
if os.path.exists(best_params_path):
    try:
        best_params = np.load(best_params_path)
        best_params = np.asarray(best_params, dtype=float).reshape(-1)
        if best_params.size != 9:
            print("警告：best_controller.npy 大小不是 9，使用随机参数代替。")
            best_params = np.random.uniform(-MAX_STEER, MAX_STEER, 9)
    except Exception as e:
        print("无法加载 best_controller.npy：", e)
        best_params = np.random.uniform(-MAX_STEER, MAX_STEER, 9)
else:
    print("未找到训练好的参数文件 best_controller.npy，使用随机参数演示。")
    best_params = np.random.uniform(-MAX_STEER, MAX_STEER, 9)

# ----------------------------
# 隶属函数（与训练脚本一致）
# ----------------------------
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
FUZZY_LABELS = ['near', 'mid', 'far']

def fuzzify_sensor_normalized(val):
    m = {}
    for k, (a, b, c) in FUZZY_POINTS.items():
        m[k] = tri_mf(val, a, b, c)
    return m

# ----------------------------
# 模糊控制器（返回 steer, front_m）
# ----------------------------
class FuzzyController:
    def __init__(self, rule_vector):
        self.rules = np.array(rule_vector, dtype=float).reshape(-1)
        if self.rules.size != 9:
            self.rules = np.random.uniform(-MAX_STEER, MAX_STEER, 9)

    def decide(self, left_dist, front_dist, right_dist):
        # 归一化
        ln = max(0.0, min(1.0, left_dist / SENSOR_RANGE))
        rn = max(0.0, min(1.0, right_dist / SENSOR_RANGE))
        fn = max(0.0, min(1.0, front_dist / SENSOR_RANGE))
        left_m = fuzzify_sensor_normalized(ln)
        right_m = fuzzify_sensor_normalized(rn)
        front_m = fuzzify_sensor_normalized(fn)
        # 改进：前向权重 = far + 0.5*mid，避免完全屏蔽模糊输出
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

# ----------------------------
# 椭圆赛道（新增 nearest_point_ahead）
# ----------------------------
class EllipseTrack:
    def __init__(self):
        self.cx, self.cy = WIDTH // 2, HEIGHT // 2
        self.a_out, self.b_out = 250, 180
        self.a_in, self.b_in = 180, 130

    def point_in_track(self, x, y):
        dx = x - self.cx
        dy = y - self.cy
        val_out = (dx / self.a_out) ** 2 + (dy / self.b_out) ** 2
        val_in = (dx / self.a_in) ** 2 + (dy / self.b_in) ** 2
        return val_in >= 1.0 and val_out <= 1.0

    def draw(self):
        pygame.draw.ellipse(screen, (80, 80, 80), (self.cx - self.a_out, self.cy - self.b_out, 2*self.a_out, 2*self.b_out))
        pygame.draw.ellipse(screen, (0, 0, 0), (self.cx - self.a_in, self.cy - self.b_in, 2*self.a_in, 2*self.b_in))

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

# ----------------------------
# 障碍物生成
# ----------------------------
def generate_obstacles(n, track, forbidden_point=None, min_dist=0):
    obs = []
    for _ in range(n):
        attempts = 0
        while True:
            attempts += 1
            x = np.random.randint(track.cx - track.a_out + 20, track.cx + track.a_out - 20)
            y = np.random.randint(track.cy - track.b_out + 20, track.cy + track.b_out - 20)
            if not track.point_in_track(x, y):
                if attempts > 300:
                    break
                continue
            if forbidden_point is not None:
                fx, fy = forbidden_point
                if math.hypot(x - fx, y - fy) < min_dist:
                    if attempts > 300:
                        break
                    continue
            obs.append(pygame.Rect(int(x - 10), int(y - 10), 20, 20))
            break
    return obs

def draw_obstacles(obstacles):
    for o in obstacles:
        pygame.draw.rect(screen, (255, 200, 0), o)

# ----------------------------
# 射线传感器（返回距离和检测点）
# ----------------------------
def query_distance(x, y, angle, track, obstacles, max_range=SENSOR_RANGE):
    step = 3.0
    dist = 0.0
    dx = math.cos(angle)
    dy = math.sin(angle)
    while dist < max_range:
        px = x + dx * dist
        py = y + dy * dist
        if not track.point_in_track(px, py):
            return dist, (px, py)
        for o in obstacles:
            if o.collidepoint(px, py):
                return dist, (px, py)
        dist += step
    return max_range, (x + dx * max_range, y + dy * max_range)

# ----------------------------
# 小车类
# ----------------------------
class Car:
    def __init__(self, x, y, angle=0.0):
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = 2.0
        self.trail = []
        self.alive = True

    def update(self, steer, track, obstacles):
        if not self.alive:
            return
        steer = max(-MAX_STEER, min(MAX_STEER, steer))
        self.angle += steer * 0.08
        self.x += self.speed * math.cos(self.angle)
        self.y += self.speed * math.sin(self.angle)
        self.trail.append((int(self.x), int(self.y)))
        if len(self.trail) > 500:
            self.trail.pop(0)
        if not track.point_in_track(self.x, self.y):
            self.alive = False
        for o in obstacles:
            if o.collidepoint(self.x, self.y):
                self.alive = False

    def draw(self):
        if len(self.trail) > 1:
            pygame.draw.lines(screen, (0, 255, 0), False, self.trail, 2)
        pygame.draw.circle(screen, (255, 0, 0), (int(self.x), int(self.y)), 6)
        ax = self.x + 14 * math.cos(self.angle)
        ay = self.y + 14 * math.sin(self.angle)
        pygame.draw.line(screen, (255, 255, 255), (int(self.x), int(self.y)), (int(ax), int(ay)), 2)

# ----------------------------
# 主程序与重启逻辑
# ----------------------------
track = EllipseTrack()
start_x = track.cx + track.a_in + 10
start_y = track.cy
start_angle = 0.0

prev_steer = 0.0  # 一阶滤波的上一值

def restart():
    global car, obstacles, controller, prev_steer
    min_dist = 70
    obstacles = generate_obstacles(12, track, forbidden_point=(start_x, start_y), min_dist=min_dist)
    car = Car(start_x, start_y, angle=start_angle)
    controller = FuzzyController(best_params)
    prev_steer = 0.0
    print(f"演示已重置：起点 = ({start_x:.1f}, {start_y:.1f})，障碍物数 = {len(obstacles)}")

restart()

try:
    font = pygame.font.SysFont('SimHei', 24)
except Exception:
    font = pygame.font.SysFont(None, 24)

debug_mode = False
running = True

# 调参常量
K_diff = 0.4   # target diff 的缩放因子
beta = 0.75    # 滤波系数（0..1），越大响应越平滑
alpha_min = 0.1
alpha_max = 0.6

while running:
    dt = clock.tick(60) / 1000.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                restart()
            if event.key == pygame.K_d:
                debug_mode = not debug_mode
                print("Debug 模式:", debug_mode)

    screen.fill((0, 0, 0))
    track.draw()
    draw_obstacles(obstacles)

    if car.alive:
        left_d, left_pt = query_distance(car.x, car.y, car.angle + math.radians(-45), track, obstacles)
        front_d, front_pt = query_distance(car.x, car.y, car.angle, track, obstacles)
        right_d, right_pt = query_distance(car.x, car.y, car.angle + math.radians(45), track, obstacles)

        fuzzy_steer, front_m = controller.decide(left_d, front_d, right_d)

        idx = track.nearest_point_ahead(car.x, car.y, car.angle, samples=360, lookahead=14)
        mid_a = (track.a_in + track.a_out) / 2.0
        mid_b = (track.b_in + track.b_out) / 2.0
        tx = track.cx + mid_a * math.cos(2*math.pi*idx/360)
        ty = track.cy + mid_b * math.sin(2*math.pi*idx/360)
        target_angle = math.atan2(ty - car.y, tx - car.x)
        diff = (target_angle - car.angle + math.pi) % (2*math.pi) - math.pi

        # 自适应混合系数：当前方安全（front_weight 大）时更信任目标，否则更信任模糊避障
        front_weight = front_m['far'] + 0.5 * front_m['mid']
        alpha_target = 1.0 - front_weight
        alpha_target = max(alpha_min, min(alpha_max, alpha_target))

        raw = alpha_target * (K_diff * diff) + (1.0 - alpha_target) * fuzzy_steer
        raw = max(-MAX_STEER, min(MAX_STEER, raw))

        # 平滑
        steer = prev_steer * beta + raw * (1 - beta)
        prev_steer = steer

        car.update(steer, track, obstacles)

        # 可视化射线
        pygame.draw.line(screen, (0, 180, 0), (int(car.x), int(car.y)), (int(left_pt[0]), int(left_pt[1])), 2)
        pygame.draw.line(screen, (0, 180, 0), (int(car.x), int(car.y)), (int(front_pt[0]), int(front_pt[1])), 2)
        pygame.draw.line(screen, (0, 180, 0), (int(car.x), int(car.y)), (int(right_pt[0]), int(right_pt[1])), 2)
        pygame.draw.circle(screen, (0, 200, 0), (int(left_pt[0]), int(left_pt[1])), 3)
        pygame.draw.circle(screen, (0, 200, 0), (int(front_pt[0]), int(front_pt[1])), 3)
        pygame.draw.circle(screen, (0, 200, 0), (int(right_pt[0]), int(right_pt[1])), 3)

        if debug_mode:
            print(f"sensors L/F/R: {left_d:.1f} {front_d:.1f} {right_d:.1f}")
            print(f"front_m: {front_m}, fuzzy_steer: {fuzzy_steer:.4f}, diff: {diff:.4f}, alpha_target: {alpha_target:.3f}, final: {steer:.4f}")
            print("rules:", controller.rules)
    else:
        text = font.render('小车已失败！ 按 R 重新开始，D 打印调试', True, (255, 0, 0))
        screen.blit(text, (WIDTH//2 - 260, 20))

    car.draw()
    pygame.display.flip()

pygame.quit()