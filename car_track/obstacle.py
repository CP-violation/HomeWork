"""
障碍物生成
"""
import random
import math
import pygame
from track import point_in_track
from config import CX, CY, A_OUT, B_OUT, A_IN, B_IN


class Obstacle:
    def __init__(self, x, y, size=20):
        self.x = x
        self.y = y
        self.shape_type = "rect"
        self.size = size
        self.rect = pygame.Rect(int(x - size/2), int(y - size/2), size, size)
    def collidepoint(self, px, py):
            return self.rect.collidepoint(px, py)
    
    def draw(self, surface):
        """绘制障碍物"""
        pygame.draw.rect(surface, (255, 200, 0))


def generate_obstacles(n, forbidden_points=None, min_dist=100,   min_spacing=40):

    if forbidden_points is None:
        forbidden_points = []
    
    obstacles = []
    existing_positions = []  # 用于间距检查的现有障碍物位置
    
    for _ in range(n):
   
        # 1. 生成随机位置
        x, y = generate_valid_position()
        if x is None:  # 位置生成失败
            continue
        
        # 2. 安全区域检查
        if not check_safety_zone(x, y, forbidden_points, min_dist):
            continue
        
        # 3. 障碍物间距检查
        if not check_obstacle_spacing(x, y, existing_positions, min_spacing):
            continue
        
        # 4. 创建障碍物
        obstacle = Obstacle(x, y)
        obstacles.append(obstacle)
        existing_positions.append((x, y))  # 记录位置和尺寸用于间距检查

    return obstacles

def generate_valid_position():
    """生成有效的赛道内位置"""
    attempts = 0
    while attempts < 100:  # 位置生成的最大尝试次数
        x = random.randint(CX - A_OUT + 30, CX + A_OUT - 30)
        y = random.randint(CY - B_OUT + 30, CY + B_OUT - 30)
        
        if point_in_track(x, y):
            return x, y
        
        attempts += 1
    
    return None, None  # 位置生成失败


def check_safety_zone(x, y, forbidden_points, min_dist):
    """检查是否在安全区域外"""
    for fp in forbidden_points:
        if isinstance(fp, (list, tuple)) and len(fp) >= 2:
            fx, fy = fp[0], fp[1]
        else:
            fx, fy = fp.x, fp.y  # 假设是车辆对象
        
        distance = math.sqrt((x - fx)**2 + (y - fy)**2)
        if distance < min_dist:
            return False
    
    return True


def check_obstacle_spacing(x, y, existing_positions, min_spacing):
    """检查与现有障碍物的间距"""
    for ex, ey in existing_positions:
        # 考虑障碍物尺寸的中心距离
        center_distance = math.sqrt((x - ex)**2 + (y - ey)**2)
        esize = 20  # 假设所有障碍物尺寸相同
        # 简化的间距检查：中心距离 > 最小间距 + 平均半径
        if center_distance < min_spacing + (esize / 2):
            return False
    
    return True
