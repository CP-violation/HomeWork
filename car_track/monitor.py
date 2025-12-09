"""
车辆控制性能指标实时监控系统
实时输出横向偏差、航向角偏差和稳态误差
"""
import math
import time
import numpy as np
from collections import deque
import os
import sys

# 确保可以导入本地模块
sys.path.insert(0, os.path.dirname(__file__))

# 导入项目中的模块
from track import generate_midline, point_in_track
from config import CX, CY, A_IN, A_OUT, B_IN, B_OUT

class PerformanceConsoleMonitor:
    """控制台性能监控器"""
    
    def __init__(self, window_size=200):
        # 数据存储
        self.lateral_errors = deque(maxlen=window_size)  # 横向偏差历史
        self.heading_errors = deque(maxlen=window_size)  # 航向角偏差历史
        self.timestamps = deque(maxlen=window_size)      # 时间戳
        
        # 性能指标
        self.current_lateral_error = 0.0
        self.current_heading_error = 0.0
        self.steady_state_lateral_error = 0.0
        self.steady_state_heading_error = 0.0
        
        # 统计参数
        self.steady_state_start = 100  # 从第100步开始计算稳态
        self.update_interval = 0.5     # 控制台更新间隔(秒)
        self.last_update_time = 0
        
        # 路径信息
        self.waypoints = generate_midline(samples=360)
        self.track_center_radius = (A_IN + A_OUT) / 2.0  # 赛道中心线半径
        
        print("=" * 80)
        print("车辆控制性能指标监控系统")
        print("=" * 80)
        print(f"{'时间(s)':<10} {'横向偏差(px)':<15} {'航向角偏差(°)':<15} {'稳态横向误差':<15} {'稳态航向误差':<15}")
        print("-" * 80)
    
    def calculate_lateral_error(self, car_x, car_y):
        """计算横向偏差：车辆与参考路径的垂直距离"""
        # 计算车辆到赛道中心的距离
        dx = car_x - CX
        dy = car_y - CY
        distance_to_center = math.hypot(dx, dy)
        
        # 横向偏差 = 实际距离 - 理想中心线距离
        lateral_error = distance_to_center - self.track_center_radius
        
        # 确定偏差方向（内侧为负，外侧为正）
        # 通过比较实际位置与中心线椭圆的关系确定方向
        ideal_radius_x = self.track_center_radius * (dx / distance_to_center) if distance_to_center > 0 else 0
        ideal_radius_y = self.track_center_radius * (dy / distance_to_center) if distance_to_center > 0 else 0
        ideal_x = CX + ideal_radius_x
        ideal_y = CY + ideal_radius_y
        
        # 计算实际位置到理想位置的向量
        vec_x = car_x - ideal_x
        vec_y = car_y - ideal_y
        
        # 计算该点的法线方向（从中心指向外）
        normal_x = dx / distance_to_center if distance_to_center > 0 else 1
        normal_y = dy / distance_to_center if distance_to_center > 0 else 0
        
        # 通过点积确定方向
        dot_product = vec_x * normal_x + vec_y * normal_y
        
        # 如果点积为正，说明车辆在中心线外侧，偏差为正
        # 如果点积为负，说明车辆在中心线内侧，偏差为负
        lateral_error = abs(lateral_error) * (1 if dot_product > 0 else -1)
        
        return lateral_error
    
    def calculate_heading_error(self, car_x, car_y, car_heading):
        """计算航向角偏差：车辆航向与参考路径切线的夹角"""
        if not self.waypoints:
            return 0.0
        
        # 找到最近的路径点
        min_dist = float('inf')
        nearest_idx = 0
        for i, (wx, wy) in enumerate(self.waypoints):
            dist = math.hypot(car_x - wx, car_y - wy)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        # 计算路径切线方向（使用前后点）
        prev_idx = (nearest_idx - 5) % len(self.waypoints)  # 前一个点
        next_idx = (nearest_idx + 5) % len(self.waypoints)  # 后一个点
        
        prev_x, prev_y = self.waypoints[prev_idx]
        next_x, next_y = self.waypoints[next_idx]
        
        # 计算切线方向（从prev到next的方向）
        tangent_angle = math.atan2(next_y - prev_y, next_x - prev_x)
        
        # 计算航向角偏差（归一化到[-π, π]）
        heading_error = tangent_angle - car_heading
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi
        
        # 转换为角度
        heading_error_deg = math.degrees(heading_error)
        
        return heading_error_deg
    
    def update(self, car_x, car_y, car_heading, current_time):
        """更新性能指标"""
        # 计算当前偏差
        self.current_lateral_error = self.calculate_lateral_error(car_x, car_y)
        self.current_heading_error = self.calculate_heading_error(car_x, car_y, car_heading)
        
        # 记录历史数据
        self.lateral_errors.append(self.current_lateral_error)
        self.heading_errors.append(self.current_heading_error)
        self.timestamps.append(current_time)
        
        # 计算稳态误差（使用最近的数据）
        if len(self.lateral_errors) >= self.steady_state_start:
            recent_lateral = list(self.lateral_errors)[-50:]  # 最近50个点
            recent_heading = list(self.heading_errors)[-50:]  # 最近50个点
            
            self.steady_state_lateral_error = np.mean(recent_lateral)
            self.steady_state_heading_error = np.mean(recent_heading)
        
        # 控制台输出（限制更新频率）
        if current_time - self.last_update_time >= self.update_interval:
            self._print_metrics(current_time)
            self.last_update_time = current_time
    
    def _print_metrics(self, current_time):
        """在控制台输出性能指标"""
        # 使用颜色编码（在支持ANSI颜色的终端中显示）
        lateral_color = "\033[92m" if abs(self.current_lateral_error) < 20 else "\033[93m"  # 绿色/黄色
        heading_color = "\033[92m" if abs(self.current_heading_error) < 15 else "\033[93m"  # 绿色/黄色
        steady_lateral_color = "\033[92m" if abs(self.steady_state_lateral_error) < 10 else "\033[93m"  # 绿色/黄色
        steady_heading_color = "\033[92m" if abs(self.steady_state_heading_error) < 8 else "\033[93m"   # 绿色/黄色
        reset_color = "\033[0m"
        
        print(f"{current_time:<10.1f} "
              f"{lateral_color}{self.current_lateral_error:<15.2f}{reset_color} "
              f"{heading_color}{self.current_heading_error:<15.2f}{reset_color} "
              f"{steady_lateral_color}{self.steady_state_lateral_error:<15.2f}{reset_color} "
              f"{steady_heading_color}{self.steady_state_heading_error:<15.2f}{reset_color}")
    
    def get_summary(self):
        """获取性能摘要"""
        if len(self.lateral_errors) == 0:
            return {}
        
        lateral_errors_list = list(self.lateral_errors)
        heading_errors_list = list(self.heading_errors)
        
        return {
            'current_lateral_error': self.current_lateral_error,
            'current_heading_error': self.current_heading_error,
            'steady_state_lateral_error': self.steady_state_lateral_error,
            'steady_state_heading_error': self.steady_state_heading_error,
            'mean_lateral_error': np.mean(lateral_errors_list),
            'mean_heading_error': np.mean(heading_errors_list),
            'max_lateral_error': np.max(np.abs(lateral_errors_list)),
            'max_heading_error': np.max(np.abs(heading_errors_list)),
            'lateral_error_std': np.std(lateral_errors_list),
            'heading_error_std': np.std(heading_errors_list),
            'total_samples': len(self.lateral_errors)
        }
    
    def print_summary(self):
        """打印性能摘要报告"""
        summary = self.get_summary()
        
        print("\n" + "=" * 80)
        print("性能指标摘要报告")
        print("=" * 80)
        print(f"总样本数: {summary['total_samples']}")
        print(f"平均横向偏差: {summary['mean_lateral_error']:.2f} px")
        print(f"平均航向角偏差: {summary['mean_heading_error']:.2f} °")
        print(f"最大横向偏差: {summary['max_lateral_error']:.2f} px")
        print(f"最大航向角偏差: {summary['max_heading_error']:.2f} °")
        print(f"横向偏差标准差: {summary['lateral_error_std']:.2f} px")
        print(f"航向角偏差标准差: {summary['heading_error_std']:.2f} °")
        print(f"稳态横向误差: {summary['steady_state_lateral_error']:.2f} px")
        print(f"稳态航向误差: {summary['steady_state_heading_error']:.2f} °")
        print("=" * 80)


