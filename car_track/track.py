"""
轨迹工具：生成中线路径点和几何测试。
"""
from math import cos, sin, pi, atan2
from config import CX, CY, A_OUT, B_OUT, A_IN, B_IN

# 生成赛道中线路径点
# 椭圆参数方程: x = CX + a*cos(θ), y = CY + b*sin(θ)
def generate_midline(samples=360):
    mid_a = (A_IN + A_OUT) / 2.0
    mid_b = (B_IN + B_OUT) / 2.0
    pts = []
    for k in range(samples):
        ang = 2 * pi * k / samples 
        pts.append((CX + mid_a * cos(ang), CY + mid_b * sin(ang))) 
    return pts

# 赛道边界检测
def point_in_track(x, y):
    dx = x - CX
    dy = y - CY
    val_out = (dx / A_OUT) ** 2 + (dy / B_OUT) ** 2
    val_in = (dx / A_IN) ** 2 + (dy / B_IN) ** 2
    return (val_in >= 1.0) and (val_out <= 1.0)

  
#前向最近点搜索，Pure Pursuit路径跟踪
def nearest_point_ahead(x, y, heading, waypoints, samples=360, lookahead=28):
    # 找到最近的路径点索引
    best_idx = 0
    best_d2 = float('inf')
    n = len(waypoints)
    for i, (wx, wy) in enumerate(waypoints):
        d2 = (wx - x) ** 2 + (wy - y) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_idx = i
    # 在最近点附近搜索前向路径点
    chosen_idx = best_idx
    best_abs = float('inf')
    for offs in range(-lookahead, lookahead + 1):
        k = (best_idx + offs) % n
        px, py = waypoints[k]
        target_angle = atan2(py - y, px - x) #目标点相对于车辆的角度
        diff = (target_angle - heading + pi) % (2 * pi) - pi
        #选择车辆前方(90度)且角度差最小的点
        if abs(diff) < pi / 2 and abs(diff) < best_abs:
            best_abs = abs(diff)
            chosen_idx = k
    if best_abs == float('inf'):
        # 没有找到前向点，选择最接近的点
        best_abs2 = float('inf')
        for offs in range(-lookahead, lookahead + 1):
            k = (best_idx + offs) % n
            px, py = waypoints[k]
            target_angle = atan2(py - y, px - x)
            diff = (target_angle - heading + pi) % (2 * pi) - pi
            if abs(diff) < best_abs2:
                best_abs2 = abs(diff)
                chosen_idx = k
    return chosen_idx


'''
samples参数
小值(如180): 计算快，但路径粗糙

大值(如720): 路径平滑，但计算量增加

推荐值: 360-720，平衡精度和性能

lookahead参数
小值: 紧密跟踪路径，但可能振荡

大值: 平滑行驶，但跟踪精度降低

推荐值: 20-40，根据车辆速度调整
'''