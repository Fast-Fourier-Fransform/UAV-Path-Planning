import numpy as np
from .base_algorithm import BaseOptimizer


class DE(BaseOptimizer):
    def __init__(self, nPop, MaxIt, VarMin, VarMax, dim, CostFunc, F=0.5, CR=0.7):
        super().__init__(nPop, MaxIt, VarMin, VarMax, dim, CostFunc)
        # 差分进化专属参数
        self.F = F  # 缩放因子 (Mutation factor)
        self.CR = CR  # 交叉概率 (Crossover rate)

    def optimize(self):
        # 初始化种群
        self.X = np.random.uniform(self.VarMin, self.VarMax, (self.nPop, self.dim))
        self.scores = np.zeros(self.nPop)
        for i in range(self.nPop):
            self.scores[i] = self.CostFunc(self.X[i])

        best_idx = np.argmin(self.scores)
        self.best_score = self.scores[best_idx]
        self.best_position = np.copy(self.X[best_idx])

        # 核心迭代
        for t in range(self.MaxIt):
            for i in range(self.nPop):
                # 1. 变异操作 (Mutation): 随机选择另外三个不同的个体
                idxs = [idx for idx in range(self.nPop) if idx != i]
                a, b, c = self.X[np.random.choice(idxs, 3, replace=False)]
                mutant = a + self.F * (b - c)
                mutant = np.clip(mutant, self.VarMin, self.VarMax)

                # 2. 交叉操作 (Crossover)
                cross_points = np.random.rand(self.dim) < self.CR
                # 确保至少有一个维度发生交叉
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, self.X[i])

                # 3. 选择操作 (Selection)
                trial_score = self.CostFunc(trial)
                if trial_score < self.scores[i]:
                    self.X[i] = trial
                    self.scores[i] = trial_score

                    # 更新全局最优
                    if trial_score < self.best_score:
                        self.best_score = trial_score
                        self.best_position = np.copy(trial)

            self.curve[t] = self.best_score
            #print(f"DE - Iteration {t + 1}: Best Cost = {self.best_score:.4f}")

        return self.best_position, self.best_score, self.curve