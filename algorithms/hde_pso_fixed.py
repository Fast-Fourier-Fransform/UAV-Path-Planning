import numpy as np
from .base_algorithm import BaseOptimizer


class HDEPSO_Fixed(BaseOptimizer):
    """
    消融变体2: HDE-PSO 的固定参数版本。
    保留了 Pbest 记忆重组，但移除了 F 的自适应指数衰减机制。
    """

    def __init__(self, nPop, MaxIt, VarMin, VarMax, dim, CostFunc):
        super().__init__(nPop, MaxIt, VarMin, VarMax, dim, CostFunc)
        self.w_max, self.w_min = 0.9, 0.4
        self.c1, self.c2 = 2.0, 2.0
        self.v_limit = (self.VarMax - self.VarMin) * 0.2
        self.F_constant = 0.2  # 剥夺衰减机制，强制固定
        self.CR = 0.9

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
            F = self.F_constant  # <--- 消融点：固定 F 值

            # PSO 阶段
            r1 = np.random.rand(self.nPop, self.dim)
            r2 = np.random.rand(self.nPop, self.dim)
            self.V = w * self.V + self.c1 * r1 * (self.pbest_X - self.X) + self.c2 * r2 * (self.best_position - self.X)
            self.V = np.clip(self.V, -self.v_limit, self.v_limit)
            self.X = np.clip(self.X + self.V, self.VarMin, self.VarMax)

            for i in range(self.nPop):
                cost = self.CostFunc(self.X[i])
                if cost < self.pbest_scores[i]:
                    self.pbest_scores[i] = cost
                    self.pbest_X[i] = np.copy(self.X[i])
                    if cost < self.best_score:
                        self.best_score = cost
                        self.best_position = np.copy(self.X[i])

            # DE/Pbest 记忆重组阶段
            idxs = np.array(
                [np.random.choice([j for j in range(self.nPop) if j != i], 3, replace=False) for i in range(self.nPop)])
            r1_idx, r2_idx, r3_idx = idxs[:, 0], idxs[:, 1], idxs[:, 2]

            mutant = self.pbest_X[r1_idx] + F * (self.pbest_X[r2_idx] - self.pbest_X[r3_idx])
            mutant = np.clip(mutant, self.VarMin, self.VarMax)

            cross_points = np.random.rand(self.nPop, self.dim) < self.CR
            for i in range(self.nPop):
                if not np.any(cross_points[i]):
                    cross_points[i, np.random.randint(0, self.dim)] = True

            trial_X = np.where(cross_points, mutant, self.pbest_X)

            for i in range(self.nPop):
                trial_cost = self.CostFunc(trial_X[i])
                if trial_cost < self.pbest_scores[i]:
                    self.pbest_X[i] = np.copy(trial_X[i])
                    self.pbest_scores[i] = trial_cost
                    if trial_cost < self.best_score:
                        self.best_score = trial_cost
                        self.best_position = np.copy(trial_X[i])

            self.curve[t] = self.best_score

        return self.best_position, self.best_score, self.curve