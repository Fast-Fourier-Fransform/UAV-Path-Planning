"""
DCWPSO: Particle Swarm Optimization with Dynamic Inertia Weight Updating and Enhanced Learning Strategies
Reference: Chen et al. (2024).
"""
import numpy as np

class DCWPSO:
    """
    动态振荡惯性权重粒子群优化算法
    核心改进：动态振荡惯性权重、增强学习策略
    """
    def __init__(self, nPop, MaxIt, VarMin, VarMax, dim, CostFunc):
        self.nPop = nPop
        self.MaxIt = MaxIt
        self.VarMin = VarMin
        self.VarMax = VarMax
        self.dim = dim
        self.CostFunc = CostFunc

        self.w_max = 0.9
        self.w_min = 0.4
        self.c1 = 1.8
        self.c2 = 1.8
        self.oscillation_freq = 4

        self.best_score = float('inf')
        self.best_position = np.zeros(dim)

    def _enhanced_learning(self, pbest_X, i):
        """增强学习策略：随机向另一个优秀的个体学习"""
        idx = np.random.randint(0, self.nPop)
        while idx == i:
            idx = np.random.randint(0, self.nPop)
        return pbest_X[idx]

    def optimize(self):
        X = np.random.uniform(self.VarMin, self.VarMax, (self.nPop, self.dim))
        V = np.zeros((self.nPop, self.dim))

        pbest_X = np.copy(X)
        pbest_scores = np.array([self.CostFunc(x) for x in X])

        best_idx = np.argmin(pbest_scores)
        self.best_score = pbest_scores[best_idx]
        self.best_position = np.copy(pbest_X[best_idx])

        # 🌟 修复 1：将限速从 0.1 解除到 0.2，增强动态避障急转弯能力
        VelMax = 0.2 * (self.VarMax - self.VarMin)
        VelMin = -VelMax
        curve = np.zeros(self.MaxIt)

        for t in range(self.MaxIt):
            # 动态振荡权重
            w = (self.w_max - self.w_min) / 2 * np.cos(self.oscillation_freq * np.pi * t / self.MaxIt) + (self.w_max + self.w_min) / 2

            for i in range(self.nPop):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                r3 = np.random.rand(self.dim)

                exemplar = self._enhanced_learning(pbest_X, i)

                # 速度更新
                V[i] = (w * V[i] +
                        self.c1 * r1 * (pbest_X[i] - X[i]) +
                        self.c2 * r2 * (self.best_position - X[i]) +
                        0.5 * r3 * (exemplar - X[i]))

                V[i] = np.clip(V[i], VelMin, VelMax)
                X[i] = X[i] + V[i]

                # 🌟 修复 2：越界反弹机制！防止粒子在边界死循环摩擦
                is_outside = (X[i] < self.VarMin) | (X[i] > self.VarMax)
                V[i][is_outside] = -0.5 * V[i][is_outside]
                X[i] = np.clip(X[i], self.VarMin, self.VarMax)

                cost = self.CostFunc(X[i])

                if cost < pbest_scores[i]:
                    pbest_scores[i] = cost
                    pbest_X[i] = np.copy(X[i])

                    if cost < self.best_score:
                        self.best_score = cost
                        self.best_position = np.copy(X[i])

            curve[t] = self.best_score

        return self.best_position, self.best_score, curve