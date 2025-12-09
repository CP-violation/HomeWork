import math

# -------------------------------
# 寻找 Pure Pursuit 前视点
# -------------------------------
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

# 在路径跟踪和避障间进行决策
# 如果没有检测到安全间隙，直接返回路径目标角度
# 如果存在安全间隙，选择最接近路径目标方向的间隙中心角度
# 使用角度差计算确保结果在[-π, π]范围内
def choose_gap_or_target(angle_to_wp, gaps):
    # 目标路径点相对于车辆当前朝向的角度差 
    if not gaps:
        return angle_to_wp
    best = None
    best_diff = float('inf')
    for g in gaps:
        g_ang = math.radians(g['center'])     #间隙中心角度转换为弧度
        diff = abs(((g_ang - angle_to_wp + math.pi) % (2 * math.pi) - math.pi))   #计算角度差
        if diff < best_diff:
            best_diff = diff
            best = g
    if best:
        return math.radians(best['center'])
    return angle_to_wp

import math
import numpy as np


class FuzzyPID:

    def __init__(self, genes,kp=1, ki=0.1, kd=0.1,max_err=math.pi, max_der=5.0):
        self.kp0 = kp
        self.ki0 = ki
        self.kd0 = kd
        self.max_err = max_err
        self.max_der = max_der
        # ----------- 拆分基因 -----------
        self.mf_e = genes[0:9].reshape((3, 3))      # 误差 e 的MF
        self.mf_de = genes[9:18].reshape((3, 3))    # de 的MF
        self.rules = genes[18:45].reshape((9, 3))   # 9条规则，每条3输出 Δkp Δki Δkd

        self.integral = 0.0
        self.prev_err = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_err = 0.0

    # ------------------------------
    # 三角形MF：由 GA 提供 a,b,c
    # ------------------------------
    @staticmethod
    def tri_mf(x, a, b, c):
        if x <= a or x >= c: return 0.0
        if x < b: return (x - a) / (b - a + 1e-6)
        return (c - x) / (c - b + 1e-6)

    # ------------------------------
    # 模糊化
    # ------------------------------
    def fuzzify(self, x, MF):
        out = []
        for i in range(3):
            a, b, c = MF[i]
            out.append(self.tri_mf(x, a, b, c))
        return out

    # ------------------------------
    # 推理
    # ------------------------------
    def inference(self, e_mf, de_mf):
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

        if den == 0:
            return 0, 0, 0
        return num / den

    # ------------------------------
    # PID 更新
    # ------------------------------
    def update(self, err, dt=0.08):

        der = (err - self.prev_err) / dt

        e_norm = max(-1, min(1, err / math.pi))
        de_norm = max(-1, min(1, der / 5.0))

        e_mf = self.fuzzify((e_norm + 1) / 2, self.mf_e)
        de_mf = self.fuzzify((de_norm + 1) / 2, self.mf_de)

        dKp, dKi, dKd = self.inference(e_mf, de_mf)

        
        Kp = self.kp0 + dKp
        Ki = self.ki0 + dKi
        Kd = self.kd0 + dKd


        self.integral += err * dt
        D = (err - self.prev_err)

        u = Kp * err + Ki * self.integral + Kd * D

        self.prev_err = err
        return u, Kp, Ki, Kd
