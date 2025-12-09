"""
车辆的运动模型、碰撞检测和可视化功能
"""
import math
from config import MAX_STEER_RAD


class Car:
    def __init__(self, x, y, heading=0.0):
        self.x = x
        self.y = y
        self.h = heading
        self.speed = 2.0
        self.trail = []
        self.alive = True 
    def step(self, steer):
        steer = max(-MAX_STEER_RAD, min(MAX_STEER_RAD, steer))
        self.h += steer * 0.08 #直接累加转向角，模拟瞬时转向,转向系数控制转向灵敏度
        self.x += math.cos(self.h) * self.speed
        self.y += math.sin(self.h) * self.speed
        self.trail.append((int(self.x), int(self.y)))
        if len(self.trail) > 1000:
            self.trail.pop(0)
    #碰撞检测
    def bbox_collide(self, rects):
        for r in rects:
            if r.collidepoint(self.x, self.y):
                return True
        return False

    # def draw(self, surf):
    #     import pygame
    #     if len(self.trail) > 1:
    #         pygame.draw.lines(surf, (0, 200, 0), False, self.trail, 2)
    #     pygame.draw.circle(surf, (255, 0, 0), (int(self.x), int(self.y)), 6)
    #     ax = self.x + 12 * math.cos(self.h)
    #     ay = self.y + 12 * math.sin(self.h)
    #     pygame.draw.line(surf, (255, 255, 255), (int(self.x), int(self.y)), (int(ax), int(ay)), 2)
    # --------------- 修改后的 Car.draw() ----------------
    def draw(self, surf, color=(255,0,0), trail_color=(0,200,0)):
        import pygame, math

        # 轨迹
        if len(self.trail) > 1:
            pygame.draw.lines(surf, trail_color, False, self.trail, 2)

        # 小车主体
        pygame.draw.circle(surf, color, (int(self.x), int(self.y)), 6)

        # 朝向指示
        ax = self.x + 12 * math.cos(self.h)
        ay = self.y + 12 * math.sin(self.h)
        pygame.draw.line(surf, (255,255,255), (int(self.x),int(self.y)), (int(ax),int(ay)), 2)
