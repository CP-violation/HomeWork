"""
1.Pure Pursuit目标选择
2.基于间隙的期望方向决策
3.模糊PID控制器
"""
import math

# 在全局路径上寻找前视目标点
# 寻找最近路径点: 遍历所有路径点，找到距离车辆当前位置最近的点
# 沿路径前进: 从最近点开始，沿着路径累计距离，直到达到前视距离
# 返回目标点: 返回累计距离达到前视距离时的路径点
def find_pursuit_target(x, y, waypoints, lookahead_dist):
    best_idx = 0
    best_d2 = 1e12
    for i, (wx, wy) in enumerate(waypoints):
        d2 = (wx - x) ** 2 + (wy - y) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_idx = i
    idx = best_idx
    acc = 0.0
    n = len(waypoints)
    while acc < lookahead_dist:     #前视距离
        nxt = (idx + 1) % n         #循环路径
        dx = waypoints[nxt][0] - waypoints[idx][0]
        dy = waypoints[nxt][1] - waypoints[idx][1]
        seg = math.hypot(dx, dy)    #线段长度
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

# 模糊PID控制器
class FuzzyPID:

    def __init__(self, kp=0.9, ki=0.01, kd=0.05, integral_limit=1.0, max_err=math.pi):
        self.base_kp = kp
        self.base_ki = ki
        self.base_kd = kd
        self.integral_limit = abs(integral_limit)
        self.max_err = max_err
        self.integral = 0.0
        self.prev_err = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_err = 0.0

   
    @staticmethod
    # 三角隶属函数，定义模糊集的隶属度计算
    def _tri_mf(x, a, b, c):
        if x <= a or x >= c:
            return 0.0
        if x == b:
            return 1.0
        if x < b:
            return (x - a) / (b - a) if (b - a) != 0 else 0.0
        return (c - x) / (c - b) if (c - b) != 0 else 0.0
    # 模糊化误差，将归一化绝对误差映射到模糊集合，返回各模糊集的隶属度，取值范围[0,1]
    def _fuzzify_err(self, abs_err_norm):
        pts = {
            'near': (0.0, 0.0, 0.5),
            'mid' : (0.0, 0.5, 1.0),
            'far' : (0.5, 1.0, 1.0)
        }
        return {k: self._tri_mf(abs_err_norm, *pts[k]) for k in pts}
    # 基于模糊规则动态调整PID增益，根据模糊化结果调整PID增益
    def _scale_gains(self, err, err_rate):
        abs_err = min(abs(err), self.max_err)
        abs_err_norm = abs_err / self.max_err  # in [0,1]
        m = self._fuzzify_err(abs_err_norm)
        #   根据模糊集隶属度调整PID增益
    #小误差(near): 降低Kp(0.6)避免超调，提高Ki(1.2)消除静差，降低Kd(0.8)减少噪声影响
    #中等误差(mid): 使用基准增益(1.0)
    # 大误差(far): 提高Kp(1.5)加快响应，降低Ki(0.6)防止积分饱和，提高Kd(1.1)增强稳定性
        kp_scale = m['near'] * 0.6 + m['mid'] * 1.0 + m['far'] * 1.5
        ki_scale = m['near'] * 1.2 + m['mid'] * 1.0 + m['far'] * 0.6
        kd_scale = m['near'] * 0.8 + m['mid'] * 1.0 + m['far'] * 1.1
        # 当误差变化率较大时，适当增加微分增益以提高阻尼效果
        # kp_scale *= (1.0 + min(1.0, abs(err_rate) * 0.5))
        # ki_scale *= (1.0 - min(0.5, abs(err_rate) * 0.5))
        kd_scale *= (1.0 + min(1.0, abs(err_rate) * 0.5))
        return self.base_kp * kp_scale, self.base_ki * ki_scale, self.base_kd * kd_scale
    # PID控制器更新函数
    def update(self, err, dt=1.0/60.0):
        # 计算误差变化率
        err_rate = (err - self.prev_err) / (dt if dt > 0 else 1e-6)
        # 调整PID增益
        kp, ki, kd = self._scale_gains(err, err_rate)
        # 更新积分项
        self.integral += err * dt
        # 积分限幅
        if self.integral > self.integral_limit:
            self.integral = self.integral_limit
        elif self.integral < -self.integral_limit:
            self.integral = -self.integral_limit
        # PID
        p = kp * err
        i = ki * self.integral
        d = kd * err_rate
        out = p + i + d
        # 保存当前误差用于下一次计算
        self.prev_err = err
        return out