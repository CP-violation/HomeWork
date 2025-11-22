
import os
import sys
#可以导入本目录下的模块
sys.path.insert(0, os.path.dirname(__file__))

import pygame
import math
import random
import numpy as np

from config import *
from track import generate_midline, point_in_track
from vehicle import Car
from sensors import cast_beams, detect_gaps
from controller import find_pursuit_target, choose_gap_or_target, FuzzyPID
from viz import draw_track, draw_beams
from obstacle import generate_obstacles
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
# 加载参数
WAYPOINTS = generate_midline(samples=720)
params = DEFAULT_PARAMS.copy()
if os.path.exists(PARAMS_FILE):
    try:
        arr = np.load(PARAMS_FILE)
        if arr.size >= 6:
            params.update({
                'Kp': float(arr[0]),
                'safe_dist': float(arr[1]),
                'lookahead_steps': int(round(arr[2])),
                'beta': float(arr[3]),
                'beam_step_deg': float(arr[4]),
                'lookahead_wp': float(arr[5])
            })
            print("Loaded params:", params)
    except Exception:
        pass

# 初始化模糊PID控制器
fpid = FuzzyPID(kp=params.get('pid_kp', 0.9),
                ki=params.get('pid_ki', 0.01),
                kd=params.get('pid_kd', 0.05),
                integral_limit=params.get('pid_integral_limit', 1.0),
                max_err=math.pi)

car = None
obstacles = []
prev_steer = 0.0
paused = False
debug = False


def restart():
    global car, obstacles, prev_steer
    start_x = CX + A_IN +10
    start_y = CY 
    car = Car(start_x, start_y, heading=0.0)
    obstacles.clear()
    obstacles=generate_obstacles(10, min_dist=80)
    prev_steer = 0.0
    fpid.reset()


restart()
font = pygame.font.SysFont(None, 22)

running = True
while running:
    dt = clock.tick(60) / 1000.0
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running = False
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_r:
                restart()
            if ev.key == pygame.K_t:
                paused = not paused
            if ev.key == pygame.K_d:
                debug = not debug
    if not paused and car.alive:
        beams = cast_beams(car.x, car.y, car.h, params['beam_step_deg'], obstacles, max_range=SENSOR_RANGE)
        gaps = detect_gaps(beams, params['safe_dist'])
        target_pt, _ = find_pursuit_target(car.x, car.y, WAYPOINTS, params['lookahead_wp'])
        target_angle = math.atan2(target_pt[1] - car.y, target_pt[0] - car.x)
        angle_to_wp = (target_angle - car.h + math.pi) % (2 * math.pi) - math.pi
        front_blocked = any(d < params['safe_dist'] for (_, d) in beams if abs(_) <= 20)
        chosen_dir = angle_to_wp
        if front_blocked and gaps:
            chosen_dir = choose_gap_or_target(angle_to_wp, gaps)

        # 计算转向
        steer_raw = fpid.update(chosen_dir, dt=dt)
        # 限制最大转向角
        steer = max(-MAX_STEER_RAD, min(MAX_STEER_RAD, steer_raw))
        # 平滑转向
        steer = prev_steer * params['beta'] + steer * (1 - params['beta'])
        prev_steer = steer

        car.step(steer)
        if not point_in_track(car.x, car.y):
            car.alive = False
        for o in obstacles:
            if o.collidepoint(car.x, car.y):
                car.alive = False
                
    screen.fill((0, 0, 0))
    draw_track(screen, WAYPOINTS, CX, CY, A_OUT, B_OUT, A_IN, B_IN)
    for o in obstacles:
        pygame.draw.rect(screen, (255, 200, 0), o)
    if car:
        beams = cast_beams(car.x, car.y, car.h, params['beam_step_deg'], obstacles, max_range=SENSOR_RANGE)
        draw_beams(screen, car, beams, params['safe_dist'])
        if car.alive:
            tpt, _ = find_pursuit_target(car.x, car.y, WAYPOINTS, params['lookahead_wp'])
            pygame.draw.circle(screen, (0, 0, 255), (int(tpt[0]), int(tpt[1])), 6)
            car.draw(screen)
    hud = font.render(f"Kp(base)={fpid.base_kp:.2f} Ki(base)={fpid.base_ki:.3f} Kd(base)={fpid.base_kd:.3f}", True, (200, 200, 200))
    screen.blit(hud, (10, 34))
    if not car.alive:
        t = font.render("Car crashed! Press R to restart", True, (255, 0, 0))
        screen.blit(t, (WIDTH // 2 - 140, 20))
    pygame.display.flip()

pygame.quit()