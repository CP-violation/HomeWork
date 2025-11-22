"""
绘制赛道和激光束
"""
import pygame
import math


def draw_track(surf, waypoints, cx, cy, a_out, b_out, a_in, b_in):
    pygame.draw.ellipse(surf, (90, 90, 90), (cx - a_out, cy - b_out, 2 * a_out, 2 * b_out))
    pygame.draw.ellipse(surf, (0, 0, 0), (cx - a_in, cy - b_in, 2 * a_in, 2 * b_in))
    for (wx, wy) in waypoints[::8]:
        pygame.draw.circle(surf, (120, 120, 255), (int(wx), int(wy)), 2)


def draw_beams(surf, car, beams, safe_dist):
    for (a, d) in beams:
        ang = car.h + math.radians(a)
        ex = car.x + math.cos(ang) * d
        ey = car.y + math.sin(ang) * d
        color = (0, 200, 0) if d > safe_dist else (200, 60, 60)
        pygame.draw.line(surf, color, (int(car.x), int(car.y)), (int(ex), int(ey)), 2)