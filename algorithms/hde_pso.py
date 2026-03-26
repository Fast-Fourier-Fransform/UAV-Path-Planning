import numpy as np
import math
from .base_algorithm import BaseOptimizer


class HDEPSO(BaseOptimizer):
    def __init__(self, nPop, MaxIt, VarMin, VarMax, dim, CostFunc):
        super().__init__(nPop, MaxIt, VarMin, VarMax, dim, CostFunc)

        # PSO 参数
        self.w_max, self.w_min = 0.9, 0.4
        self.c1, self.c2 = 2.0, 2.0
        self.v_limit = (self.VarMax - self.VarMin) * 0.2

        # DE 差分进化参数
        self.F_base = 0.5  # 基础缩放因子
        self.CR = 0.8  # 交叉概率

    def optimize(self):
        # 1. 初始化种群位置与速度
        self.X = np.random.uniform(self.VarMin, self.VarMax, (self.nPop, self.dim))
        self.V = np.random.uniform(-self.v_limit, self.v_limit, (self.nPop, self.dim))

        # 初始化适应度
        self.pbest_X = np.copy(self.X)
        self.pbest_scores = np.zeros(self.nPop)
        for i in range(self.nPop):
            self.pbest_scores[i] = self.CostFunc(self.X[i])

        best_idx = np.argmin(self.pbest_scores)
        self.best_position = np.copy(self.pbest_X[best_idx])
        self.best_score = self.pbest_scores[best_idx]

        # 2. 核心混合迭代
        for t in range(self.MaxIt):
            # 动态调整权重与 DE 缩放因子
            w = self.w_max - (self.w_max - self.w_min) * (t / self.MaxIt)
            amplitude = 0.4 * (1 - t / self.MaxIt)
            # 频率系数设为 4，整个迭代过程只完成 2 个完整周期的震荡，更利于局部深度剥削
            F = 0.5 + amplitude * math.cos(math.pi * (t / self.MaxIt) * 4)

            # =======================================================
            # 阶段 A：标准 PSO 速度与位置更新 (负责全局快速探索)
            # =======================================================
            r1 = np.random.rand(self.nPop, self.dim)
            r2 = np.random.rand(self.nPop, self.dim)

            self.V = w * self.V + self.c1 * r1 * (self.pbest_X - self.X) + self.c2 * r2 * (self.best_position - self.X)
            self.V = np.clip(self.V, -self.v_limit, self.v_limit)
            self.X = np.clip(self.X + self.V, self.VarMin, self.VarMax)

            # 评估 PSO 产生的新位置
            for i in range(self.nPop):
                cost = self.CostFunc(self.X[i])
                if cost < self.pbest_scores[i]:
                    self.pbest_scores[i] = cost
                    self.pbest_X[i] = np.copy(self.X[i])
                    if cost < self.best_score:
                        self.best_score = cost
                        self.best_position = np.copy(self.X[i])

            # =======================================================
            # 阶段 B：Pbest 差分进化 (DE) 阶段 (负责航迹平滑与防碰撞)
            # =======================================================
            # 为每个粒子随机选择 3 个互不相等的其他粒子索引
            idxs = np.array(
                [np.random.choice([j for j in range(self.nPop) if j != i], 3, replace=False) for i in range(self.nPop)])
            r1_idx, r2_idx, r3_idx = idxs[:, 0], idxs[:, 1], idxs[:, 2]

            # 1. 差分变异：利用 Pbest 库生成变异向量 (DE/pbest-to-pbest/1)
            mutant = self.pbest_X[r1_idx] + F * (self.pbest_X[r2_idx] - self.pbest_X[r3_idx])
            mutant = np.clip(mutant, self.VarMin, self.VarMax)

            # 2. 二项式交叉
            cross_points = np.random.rand(self.nPop, self.dim) < self.CR
            # 确保每个粒子至少有一个维度发生交叉
            for i in range(self.nPop):
                if not np.any(cross_points[i]):
                    cross_points[i, np.random.randint(0, self.dim)] = True

            trial_X = np.where(cross_points, mutant, self.pbest_X)

            # 3. 严格贪婪选择 (拒绝任何撞山劣解)
            for i in range(self.nPop):
                trial_cost = self.CostFunc(trial_X[i])
                if trial_cost < self.pbest_scores[i]:
                    self.pbest_X[i] = np.copy(trial_X[i])
                    self.pbest_scores[i] = trial_cost
                    if trial_cost < self.best_score:
                        self.best_score = trial_cost
                        self.best_position = np.copy(trial_X[i])

            # 记录当前迭代最优解
            self.curve[t] = self.best_score
            #print(f"HDE-PSO - Iteration {t + 1}: Best Cost = {self.best_score:.4f}")

        return self.best_position, self.best_score, self.curve