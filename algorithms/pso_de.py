import numpy as np
from .base_algorithm import BaseOptimizer


class PSODE(BaseOptimizer):
    """
    消融变体1: 传统的 PSO-DE 混合算法。
    DE 算子作用于粒子的当前位置 X，而不是历史最优 Pbest。
    """

    def __init__(self, nPop, MaxIt, VarMin, VarMax, dim, CostFunc):
        super().__init__(nPop, MaxIt, VarMin, VarMax, dim, CostFunc)
        self.w_max, self.w_min = 0.9, 0.4
        self.c1, self.c2 = 2.0, 2.0
        self.v_limit = (self.VarMax - self.VarMin) * 0.2
        self.F = 0.5
        self.CR = 0.8

    def optimize(self):
        self.X = np.random.uniform(self.VarMin, self.VarMax, (self.nPop, self.dim))
        self.V = np.random.uniform(-self.v_limit, self.v_limit, (self.nPop, self.dim))

        self.pbest_X = np.copy(self.X)
        self.pbest_scores = np.zeros(self.nPop)
        for i in range(self.nPop):
            self.pbest_scores[i] = self.CostFunc(self.X[i])

        best_idx = np.argmin(self.pbest_scores)
        self.best_position = np.copy(self.pbest_X[best_idx])
        self.best_score = self.pbest_scores[best_idx]

        for t in range(self.MaxIt):
            w = self.w_max - (self.w_max - self.w_min) * (t / self.MaxIt)

            # 1. PSO 阶段
            r1 = np.random.rand(self.nPop, self.dim)
            r2 = np.random.rand(self.nPop, self.dim)
            self.V = w * self.V + self.c1 * r1 * (self.pbest_X - self.X) + self.c2 * r2 * (self.best_position - self.X)
            self.V = np.clip(self.V, -self.v_limit, self.v_limit)
            self.X = np.clip(self.X + self.V, self.VarMin, self.VarMax)

            # 2. 传统 DE 阶段 (作用于当前位置 X)
            idxs = np.array(
                [np.random.choice([j for j in range(self.nPop) if j != i], 3, replace=False) for i in range(self.nPop)])
            r1_idx, r2_idx, r3_idx = idxs[:, 0], idxs[:, 1], idxs[:, 2]

            mutant = self.X[r1_idx] + self.F * (self.X[r2_idx] - self.X[r3_idx])
            mutant = np.clip(mutant, self.VarMin, self.VarMax)

            cross_points = np.random.rand(self.nPop, self.dim) < self.CR
            for i in range(self.nPop):
                if not np.any(cross_points[i]):
                    cross_points[i, np.random.randint(0, self.dim)] = True

            trial_X = np.where(cross_points, mutant, self.X)

            # 评估与更新
            for i in range(self.nPop):
                trial_cost = self.CostFunc(trial_X[i])
                # DE 选择操作直接决定 X 的去留
                if trial_cost < self.CostFunc(self.X[i]):
                    self.X[i] = np.copy(trial_X[i])
                    current_cost = trial_cost
                else:
                    current_cost = self.CostFunc(self.X[i])

                # 更新 Pbest 和 Gbest
                if current_cost < self.pbest_scores[i]:
                    self.pbest_X[i] = np.copy(self.X[i])
                    self.pbest_scores[i] = current_cost
                    if current_cost < self.best_score:
                        self.best_score = current_cost
                        self.best_position = np.copy(self.X[i])

            self.curve[t] = self.best_score

        return self.best_position, self.best_score, self.curve