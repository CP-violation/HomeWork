"""
初始化参数
"""

WIDTH = 900
HEIGHT = 700

# 椭圆赛道中心坐标与长短轴
CX = WIDTH // 2 
CY = HEIGHT // 2
A_OUT, B_OUT = 300, 220
A_IN, B_IN = 210, 160

# 激光传感器范围
SENSOR_RANGE = 120.0
MAX_STEER_RAD = 0.5235987755982988  #30°角限制

# 控制器初始参数
DEFAULT_PARAMS = {
    'Kp': 0.9,                  # 比例增益（传统P控制器）
    'safe_dist': 60.0,          # 安全距离
    'lookahead_steps': 10,      # 前瞻步数
    'beta': 0.75,               # 指数平滑系数
    'beam_step_deg': 7.5,       # 激光束步进角度
    'lookahead_wp': 60.0,       # 前瞻路径点距离
    # 模糊PID参数
    'use_fuzzy_pid': True,
    'pid_kp': 1,        # PID比例增益
    'pid_ki': 0.1,       # PID积分增益
    'pid_kd': 0.1,       # PID微分增益
    # 'pid_integral_limit': 1.0,  # PID积分限制
}

# Paths
PARAMS_FILE = 'car_track/best_pp_params.npy'
