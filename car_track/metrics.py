"""
TrainingMetrics: 记录 GA 训练过程中每代的 best/mean/worst/std，保存为 npz 并可绘图（png 或交互窗口）。
"""

import threading
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class TrainingMetrics:
    def __init__(self, history_len=None):
        self.generation = []
        self.best = []
        self.mean = []
        self.std = []
        self.history_len = history_len

    def add_generation(self, gen_idx, best_val, mean_val, std_val):
        self.generation.append(gen_idx)
        self.best.append(best_val)
        self.mean.append(mean_val)
        self.std.append(std_val)
        # optionally trim
        if self.history_len:
            self.generation = self.generation[-self.history_len:]
            self.best = self.best[-self.history_len:]
            self.mean = self.mean[-self.history_len:]
            self.std = self.std[-self.history_len:]

    def save(self, path_npz):
        np.savez(path_npz,
                 generation=np.array(self.generation),
                 best=np.array(self.best),
                 mean=np.array(self.mean),
                 std=np.array(self.std))

    def plot(self, path_png=None, show=True):
        plt.figure(figsize=(8,4))
        plt.plot(self.generation, self.best, label='best')
        plt.plot(self.generation, self.mean, label='mean')
        plt.fill_between(self.generation,
                         np.array(self.mean)-np.array(self.std),
                         np.array(self.mean)+np.array(self.std),
                         color='gray', alpha=0.2, label='mean ± std')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()
        plt.grid(True)
        if path_png:
            plt.savefig(path_png, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def start_live_plot(self, interval=1.0):
        # start a non-blocking matplotlib window that updates every interval seconds
        def runner():
            plt.ion()
            fig, ax = plt.subplots(figsize=(8,4))
            while getattr(runner, "running", True):
                ax.cla()
                if len(self.generation) > 0:
                    ax.plot(self.generation, self.best, label='best')
                    ax.plot(self.generation, self.mean, label='mean')
                    ax.fill_between(self.generation,
                                    np.array(self.mean)-np.array(self.std),
                                    np.array(self.mean)+np.array(self.std),
                                    color='gray', alpha=0.2)
                    ax.set_xlabel('Generation')
                    ax.set_ylabel('Fitness')
                    ax.legend()
                    ax.grid(True)
                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(interval)
            plt.close(fig)
        t = threading.Thread(target=runner, daemon=True)
        t.start()
        runner.running = True
        self._live_thread = t
        return t

    def stop_live_plot(self):
        if hasattr(self, "_live_thread"):
            self._live_thread._target.running = False

