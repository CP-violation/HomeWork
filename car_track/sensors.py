"""
激光传感器模拟来检测周围环境和识别安全通道
射线距离检测、多波束扫描、间隙检测
"""
import math
from track import point_in_track

# 计算从(x, y)沿角度ang发射的射线与障碍物或赛道边界的距离
def ray_distance(x, y, ang, obstacles, max_range=120.0, step=3.0):
    dist = 0.0
    dx = math.cos(ang)
    dy = math.sin(ang)
    while dist < max_range:
        px = x + dx * dist
        py = y + dy * dist
        if not point_in_track(px, py):
            return dist
        for o in obstacles:  
            if o.collidepoint(px, py): #检查是否在矩形的内部
                return dist
        dist += step  
    return max_range

# 发射多束激光并测量距离
def cast_beams(x, y, heading, beam_step_deg, obstacles, max_range=120.0):
    beams = []
    half = 90
    a = -half
    while a <= half + 1e-6:
        ang = heading + math.radians(a)
        d = ray_distance(x, y, ang, obstacles, max_range=max_range)
        beams.append((a, d))  
        a += beam_step_deg 
    return beams

# 识别激光束中的安全间隙 
def detect_gaps(beams, safe_dist):
    # 安全区域标记
    free = [1 if d > safe_dist else 0 for (_, d) in beams]
    gaps = []
    n = len(free)
    i = 0
    # 查找连续的安全区域
    while i < n:
        if free[i] == 1:
            j = i
            while j < n and free[j] == 1:
                j += 1
            angles = [beams[k][0] for k in range(i, j)] 
            center = (angles[0] + angles[-1]) / 2.0
            width = angles[-1] - angles[0] + 1e-6
            gaps.append({'start': i, 'end': j - 1, 'center': center, 'width': width})
            i = j
        else:
            i += 1
    return gaps