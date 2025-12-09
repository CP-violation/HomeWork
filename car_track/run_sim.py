import os
import sys
import time
import math
import random
import numpy as np
import pygame
import matplotlib.pyplot as plt

# allow importing local modules
sys.path.insert(0, os.path.dirname(__file__))

from config import *
from monitor import PerformanceConsoleMonitor
from track import generate_midline, point_in_track
from vehicle import Car
from sensors import cast_beams, detect_gaps
from controller import find_pursuit_target, choose_gap_or_target, FuzzyPID
from viz import draw_track, draw_beams
from obstacle import generate_obstacles

pygame.init()
console_monitor = PerformanceConsoleMonitor()

# screen + clock
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 20)

# waypoints & params
WAYPOINTS = generate_midline(samples=720)
params = DEFAULT_PARAMS.copy()
if os.path.exists(PARAMS_FILE):
    try:
        arr = np.load(PARAMS_FILE)
        if arr.size >= 6:
            params.update({
                'beam_step_deg': 7.5,   # 激光束间隔
                'lookahead_wp': 70,     # Pure Pursuit前视点
                'safe_dist': 75.0,      # 避障安全距离
                'beta': 0.725,          # 指数平滑
                'Kp': 0.9,
                'Ki': 0.01,
                'Kd': 0.05
            })
            print("Loaded params:", params)
    except Exception:
        pass

# 模糊 PID（你原来的）
DEFAULT_GENES = np.full(45, 0.5)
fpid = FuzzyPID(genes=DEFAULT_GENES,
                kp=params.get('Kp', 0.9),
                ki=params.get('Ki', 0.01),
                kd=params.get('Kd', 0.05),
                )

# 传统 PID 控制器（新增）
class PIDController:
    def __init__(self, kp=25, ki=2, kd=0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.I = 0.0
        self.last_err = 0.0

    def reset(self):
        self.I = 0.0
        self.last_err = 0.0

    def update(self, err, dt):
        self.I += err * dt
        derivative = (err - self.last_err) / dt if dt > 0 else 0.0
        self.last_err = err
        return self.kp * err + self.ki * self.I + self.kd * derivative

# 仿真实体
car_fuzzy = None
car_pid = None
obstacles = []
prev_steer_fuzzy = 0.0
prev_steer_pid = 0.0

# 性能记录结构
perf = {
    'fuzzy': {
        'time': [],            # timestamps
        'heading_err': [],     # abs heading error (rad)
        'cross_err': [],       # cross-track distance (px)
        'steer': [],           # steering commands
        'collisions': 0,
        'offtrack': False,
        'lap_time': None,
        'lap_recorded': False,
        'prev_target_idx': None
    },
    'pid': {
        'time': [],
        'heading_err': [],
        'cross_err': [],
        'steer': [],
        'collisions': 0,
        'offtrack': False,
        'lap_time': None,
        'lap_recorded': False,
        'prev_target_idx': None
    }
}

# 帮助函数：最近中线点到车的距离（横向误差近似）
def cross_track_error(x, y, waypoints):
    # compute min Euclidean distance to waypoints
    arr = np.array(waypoints)
    dx = arr[:,0] - x
    dy = arr[:,1] - y
    d2 = dx*dx + dy*dy
    return float(np.sqrt(d2.min()))

# 初始化/重启
def restart():
    global car_fuzzy, car_pid, obstacles, prev_steer_fuzzy, prev_steer_pid
    start_x = CX + A_IN + 10
    start_y = CY
    car_fuzzy = Car(start_x, start_y, heading=0.0)
    car_pid = Car(start_x, start_y, heading=0.0)  # 下移一点避免完全重叠
    obstacles.clear()
    obstacles.extend(generate_obstacles(10, min_dist=80))
    prev_steer_fuzzy = 0.0
    prev_steer_pid = 0.0
    fpid.reset()
    pid_ctrl.reset()
    # reset perf
    for key in ('fuzzy','pid'):
        perf[key]['time'].clear()
        perf[key]['heading_err'].clear()
        perf[key]['cross_err'].clear()
        perf[key]['steer'].clear()
        perf[key]['collisions'] = 0
        perf[key]['offtrack'] = False
        perf[key]['lap_time'] = None
        perf[key]['lap_recorded'] = False
        perf[key]['prev_target_idx'] = None

# create PID instance
pid_ctrl = PIDController(kp=1.0, ki=0.01, kd=0.07)

# init
restart()
simulation_start_time = time.time()

running = True
paused = False
debug = False

while running:
    dt = clock.tick(60) / 1000.0
    now = time.time() - simulation_start_time

    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running = False
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_r:
                restart()
                simulation_start_time = time.time()
            if ev.key == pygame.K_t:
                paused = not paused
            if ev.key == pygame.K_d:
                debug = not debug

    if not paused:
        # ====== 更新：模糊PID车辆 ======
        if car_fuzzy.alive:
            beams = cast_beams(car_fuzzy.x, car_fuzzy.y, car_fuzzy.h, params['beam_step_deg'], obstacles, max_range=SENSOR_RANGE)
            gaps = detect_gaps(beams, params['safe_dist'])
            target_pt, target_idx = find_pursuit_target(car_fuzzy.x, car_fuzzy.y, WAYPOINTS, params['lookahead_wp'])
            target_angle = math.atan2(target_pt[1] - car_fuzzy.y, target_pt[0] - car_fuzzy.x)
            angle_to_wp = (target_angle - car_fuzzy.h + math.pi) % (2 * math.pi) - math.pi
            front_blocked = any(d < params['safe_dist'] for (_, d) in beams if abs(_) <= 20)
            chosen_dir = angle_to_wp
            if front_blocked and gaps:
                chosen_dir = choose_gap_or_target(angle_to_wp, gaps)

            steer_raw,Kp, Ki, Kd= fpid.update(chosen_dir, dt=dt)
            steer = max(-MAX_STEER_RAD, min(MAX_STEER_RAD, steer_raw))
            steer = prev_steer_fuzzy * params['beta'] + steer * (1 - params['beta'])
            prev_steer_fuzzy = steer
            car_fuzzy.step(steer)

            # stats
            perf['fuzzy']['time'].append(now)
            perf['fuzzy']['heading_err'].append(abs(chosen_dir))
            perf['fuzzy']['cross_err'].append(cross_track_error(car_fuzzy.x, car_fuzzy.y, WAYPOINTS))
            perf['fuzzy']['steer'].append(steer)

            # collisions / offtrack
            collided = False
            for o in obstacles:
                if o.collidepoint(car_fuzzy.x, car_fuzzy.y):
                    collided = True
            if collided:
                perf['fuzzy']['collisions'] += 1
                car_fuzzy.alive = False

            if not point_in_track(car_fuzzy.x, car_fuzzy.y):
                perf['fuzzy']['offtrack'] = True
                car_fuzzy.alive = False

            # lap detection using target_idx wrapping
            prev_idx = perf['fuzzy']['prev_target_idx']
            if prev_idx is None:
                perf['fuzzy']['prev_target_idx'] = target_idx
            else:
                # detect wrap-around (index decreased significantly)
                if (not perf['fuzzy']['lap_recorded']) and (target_idx < prev_idx):
                    perf['fuzzy']['lap_time'] = now
                    perf['fuzzy']['lap_recorded'] = True
                perf['fuzzy']['prev_target_idx'] = target_idx

        # ====== 更新：传统PID车辆 ======
        # ====== 更新：传统PID车辆（改为完全一致的避障逻辑） ======
        if car_pid.alive:
            beams = cast_beams(car_pid.x, car_pid.y, car_pid.h, params['beam_step_deg'], obstacles, max_range=SENSOR_RANGE)
            gaps = detect_gaps(beams, params['safe_dist'])

            # 计算Pure Pursuit目标角度（与蓝色车一致）
            target_pt2, target_idx2 = find_pursuit_target(car_pid.x, car_pid.y, WAYPOINTS, params['lookahead_wp'])
            wp_angle = math.atan2(target_pt2[1] - car_pid.y, target_pt2[0] - car_pid.x)

            # **方向误差 = 目标角 - 当前航向**
            angle_to_wp = (wp_angle - car_pid.h + math.pi) % (2 * math.pi) - math.pi

            # 判断前方是否阻挡
            front_blocked = any(d < params['safe_dist'] for a, d in beams if abs(a) <= 20)

            # ----------------- 与蓝色车完全一致 -----------------
            chosen_dir = angle_to_wp
            if front_blocked and gaps:
                chosen_dir = choose_gap_or_target(angle_to_wp, gaps)
            # -----------------------------------------------------

            # PID接收方向误差作为输入（关键修复）
            steer_raw = pid_ctrl.update(chosen_dir, dt=dt)

            # 饱和 + 滤波（平滑参数与蓝车一致为 beta）
            steer = max(-MAX_STEER_RAD, min(MAX_STEER_RAD, steer_raw))
            steer = prev_steer_pid * params['beta'] + steer * (1 - params['beta'])
            prev_steer_pid = steer

            car_pid.step(steer)

            # ================= 数据记录 =======================
            perf['pid']['time'].append(now)
            perf['pid']['heading_err'].append(abs(chosen_dir))
            perf['pid']['cross_err'].append(cross_track_error(car_pid.x, car_pid.y, WAYPOINTS))
            perf['pid']['steer'].append(steer)

            # 碰撞检查
            for o in obstacles:
                if o.collidepoint(car_pid.x, car_pid.y):
                    perf['pid']['collisions'] += 1
                    car_pid.alive = False

            if not point_in_track(car_pid.x, car_pid.y):
                perf['pid']['offtrack'] = True
                car_pid.alive = False

            # 圈计时
            prev_idx2 = perf['pid']['prev_target_idx']
            if prev_idx2 is None:
                perf['pid']['prev_target_idx'] = target_idx2
            else:
                if (not perf['pid']['lap_recorded']) and (target_idx2 < prev_idx2):
                    perf['pid']['lap_time'] = now
                    perf['pid']['lap_recorded'] = True
                perf['pid']['prev_target_idx'] = target_idx2


        # update console monitor for fuzzy car (optional)
        if car_fuzzy:
            console_monitor.update(car_fuzzy.x, car_fuzzy.y, car_fuzzy.h, now)

    # ====== 绘制 ======
    screen.fill((0, 0, 0))
    draw_track(screen, WAYPOINTS, CX, CY, A_OUT, B_OUT, A_IN, B_IN)

    # obstacles
    for o in obstacles:
        pygame.draw.rect(screen, (255, 200, 0), o)

    # draw beams & car visuals (we draw ourselves to control color)
    # fuzzy car: blue
    if car_fuzzy:
        beams_f = cast_beams(car_fuzzy.x, car_fuzzy.y, car_fuzzy.h, params['beam_step_deg'], obstacles, max_range=SENSOR_RANGE)
        draw_beams(screen, car_fuzzy, beams_f, params['safe_dist'])
        # trail
        if hasattr(car_fuzzy, 'trail') and len(car_fuzzy.trail) > 1:
            pygame.draw.lines(screen, (0, 200, 255), False, car_fuzzy.trail, 2)
        # body
        pygame.draw.circle(screen, (0, 128, 255), (int(car_fuzzy.x), int(car_fuzzy.y)), 6)
        # heading line
        ax = car_fuzzy.x + 12 * math.cos(car_fuzzy.h)
        ay = car_fuzzy.y + 12 * math.sin(car_fuzzy.h)
        pygame.draw.line(screen, (255, 255, 255), (int(car_fuzzy.x), int(car_fuzzy.y)), (int(ax), int(ay)), 2)

    # pid car: red
    if car_pid:
        beams_p = cast_beams(car_pid.x, car_pid.y, car_pid.h, params['beam_step_deg'], obstacles, max_range=SENSOR_RANGE)
        draw_beams(screen, car_pid, beams_p, params['safe_dist'])
        if hasattr(car_pid, 'trail') and len(car_pid.trail) > 1:
            pygame.draw.lines(screen, (255, 200, 200), False, car_pid.trail, 2)
        pygame.draw.circle(screen, (255, 80, 80), (int(car_pid.x), int(car_pid.y)), 6)
        ax2 = car_pid.x + 12 * math.cos(car_pid.h)
        ay2 = car_pid.y + 12 * math.sin(car_pid.h)
        pygame.draw.line(screen, (255, 255, 255), (int(car_pid.x), int(car_pid.y)), (int(ax2), int(ay2)), 2)

    # HUD
    hud1 = font.render(f"Fuzzy alive: {car_fuzzy.alive}  collisions: {perf['fuzzy']['collisions']}", True, (0, 200, 255))
    hud2 = font.render(f"PID alive: {car_pid.alive}    collisions: {perf['pid']['collisions']}", True, (255, 120, 120))
    hud3 = font.render("R restart  T pause/unpause  D debug", True, (200, 200, 200))
    screen.blit(hud1, (10, 8))
    screen.blit(hud2, (10, 28))
    screen.blit(hud3, (10, 50))

    if not car_fuzzy.alive and not car_pid.alive:
        t = font.render("Both cars dead. Press R to restart", True, (255, 0, 0))
        screen.blit(t, (WIDTH // 2 - 140, 20))
    elif not car_fuzzy.alive:
        t = font.render("Fuzzy car dead. Press R to restart", True, (0, 200, 255))
        screen.blit(t, (WIDTH // 2 - 140, 20))
    elif not car_pid.alive:
        t = font.render("PID car dead. Press R to restart", True, (255, 120, 120))
        screen.blit(t, (WIDTH // 2 - 140, 20))

    pygame.display.flip()

# 仿真结束，生成并显示性能分析图表
pygame.quit()

# --- 生成统计数据函数 ---
def summarize(name):
    rec = perf[name]
    total_time = rec['time'][-1] if rec['time'] else 0.0
    mean_heading = float(np.mean(rec['heading_err'])) if rec['heading_err'] else float('nan')
    mean_cross = float(np.mean(rec['cross_err'])) if rec['cross_err'] else float('nan')
    steer_arr = np.array(rec['steer']) if rec['steer'] else np.array([])
    # smoothness metric: RMSE of steering difference (delta steer)
    if steer_arr.size >= 2:
        ds = np.diff(steer_arr)
        smoothness = float(np.sqrt(np.mean(ds * ds)))
    else:
        smoothness = float('nan')
    collisions = rec['collisions']
    lap_time = rec['lap_time']
    return {
        'total_time': total_time,
        'mean_heading_err': mean_heading,
        'mean_cross_err': mean_cross,
        'smoothness': smoothness,
        'collisions': collisions,
        'lap_time': lap_time
    }

s_f = summarize('fuzzy')
s_p = summarize('pid')

print("\n=== Performance Summary ===")
print("Fuzzy PID:", s_f)
print("Normal PID:", s_p)

# 绘图对比： heading error over time, cross-track error over time, steer smoothness histogram
plt.figure(figsize=(12, 8))

# heading error over time
plt.subplot(2,2,1)
if perf['fuzzy']['time']:
    plt.plot(perf['fuzzy']['time'], perf['fuzzy']['heading_err'], label='Fuzzy PID')
if perf['pid']['time']:
    plt.plot(perf['pid']['time'], perf['pid']['heading_err'], label='Normal PID')
plt.xlabel('time (s)')
plt.ylabel('abs heading error (rad)')
plt.title('Heading error over time')
plt.legend()

# cross track error over time
plt.subplot(2,2,2)
if perf['fuzzy']['time']:
    plt.plot(perf['fuzzy']['time'], perf['fuzzy']['cross_err'], label='Fuzzy PID')
if perf['pid']['time']:
    plt.plot(perf['pid']['time'], perf['pid']['cross_err'], label='Normal PID')
plt.xlabel('time (s)')
plt.ylabel('cross-track error (px)')
plt.title('Cross-track error over time')
plt.legend()

# steering signals
plt.subplot(2,2,3)
if perf['fuzzy']['time']:
    plt.plot(perf['fuzzy']['time'], perf['fuzzy']['steer'], label='Fuzzy PID')
if perf['pid']['time']:
    plt.plot(perf['pid']['time'], perf['pid']['steer'], label='Normal PID')
plt.xlabel('time (s)')
plt.ylabel('steer (rad)')
plt.title('Steering command')
plt.legend()

# smoothness bar
plt.subplot(2,2,4)
vals = [s_f['smoothness'], s_p['smoothness']]
labels = ['Fuzzy', 'PID']
plt.bar(labels, vals)
plt.title('Steering smoothness (RMSE of delta-steer)')
plt.ylabel('RMSE (rad)')

plt.tight_layout()
plt.show()

# 打印更友好的摘要
def print_readable(name, stats):
    print(f"\n-- {name} --")
    print(f"Total sim time (recorded): {stats['total_time']:.2f} s")
    if stats['lap_time'] is not None:
        print(f"First lap time: {stats['lap_time']:.2f} s")
    else:
        print("First lap time: N/A")
    print(f"Mean heading error (rad): {stats['mean_heading_err']:.4f}")
    print(f"Mean cross-track error (px): {stats['mean_cross_err']:.2f}")
    print(f"Steering smoothness (RMSE delta): {stats['smoothness']:.4f}")
    print(f"Collision count: {stats['collisions']}")
    print(f"Off-track happened: {perf[name]['offtrack']}")

print_readable('fuzzy', s_f)
print_readable('pid', s_p)

# also print console monitor summary if desired
try:
    console_monitor.print_summary()
except Exception:
    pass
