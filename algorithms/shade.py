import numpy as np
import random
from .base_algorithm import BaseOptimizer

class SHADE(BaseOptimizer):
    def __init__(self, nPop, MaxIt, VarMin, VarMax, dim, CostFunc):
        super().__init__(nPop, MaxIt, VarMin, VarMax, dim, CostFunc)

        # SHADE 专属核心参数
        self.H = 100  # 历史记忆库大小
        self.p = 0.11  # p-best 变异比例

        # 初始化历史记忆库
        self.M_CR = np.ones(self.H) * 0.5
        self.M_F = np.ones(self.H) * 0.5

        self.archive = []  # 外部劣解存档
        self.k = 0  # 记忆库更新索引

    def optimize(self):
        # 1. 种群初始化
        self.X = np.random.uniform(self.VarMin, self.VarMax, (self.nPop, self.dim))
        self.scores = np.zeros(self.nPop)
        for i in range(self.nPop):
            # 🚨 修复 1：强制转换为 float。
            # CEC 某些函数可能会返回形如 [123.45] 的一维数组，不转成 float 会导致后续矩阵形状崩溃！
            self.scores[i] = float(self.CostFunc(self.X[i]))

        best_idx = np.argmin(self.scores)
        self.best_score = self.scores[best_idx]
        self.best_position = np.copy(self.X[best_idx])

        # 2. 核心迭代
        for t in range(self.MaxIt):
            S_CR = []
            S_F = []
            delta_f = []

            r_idx = np.random.randint(0, self.H, self.nPop)
            CR = np.random.normal(self.M_CR[r_idx], 0.1)
            CR = np.clip(CR, 0, 1)

            F = np.zeros(self.nPop)
            for i in range(self.nPop):
                while F[i] <= 0:
                    F[i] = self.M_F[r_idx[i]] + 0.1 * np.random.standard_cauchy()
                if F[i] > 1:
                    F[i] = 1

            sort_idxs = np.argsort(self.scores)
            p_num = max(int(self.nPop * self.p), 1)

            trial_X = np.zeros_like(self.X)

            # 🚨 修复 2：将 P_union_A 移出内层循环！
            # list(self.X) 生成的是 numpy array 的列表，与 archive 类型