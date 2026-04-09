"""
MPSOGOA: Multi-Strategy Improved Particle Swarm Optimization with Gazelle Optimization Algorithm
Reference: Wang et al. (2024).
"""
import numpy as np

class MPSOGOA:
    """
    多策略改进粒子群优化算法
    核心改进：融合瞪羚优化算法机制、莱维飞行
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

        self.best_score = float('inf')
        self.best_position = np.zeros(dim)

    def optimize(self):
        X = np.random.uniform(self.VarMin, self.VarMax, (self.nPop, self.dim))
        V = np.zeros((self.nPop, self.dim))

        pbest_X = np.copy(X)
        pbest_scores = np.array([self.CostFunc(x) for x in X])

        best_idx = np.argmin(pbest_scores)
        self.best_score = pbest_scores[best_idx]
        self.best_position = np.copy(pbest_X[best_idx])

        # 🌟 修复 1：将限速解禁至 0.2
        VelMax = 0.2 * (self.VarMax - self.VarMin)
        VelMin = -VelMax
        curve = np.zeros(self.MaxIt)

        for t in range(self.MaxIt):
            w = self.w_max - (self.w_max - self.w_min) * (t / self.MaxIt)

            for i in range(self.nPop):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                # 速度更新
                V[i] = (w * V[i] +
                        self.c1 * r1 * (pbest_X[i] - X[i]) +
                        self.c2 * r2 * (self.best_position - X[i]))

                V[i] = np.clip(V[i], VelMin, VelMax)
                X[i] = X[i] + V[i]

                # 🌟 修复 2：越界反弹保护
                is_outside = (X[i] < self.VarMin) | (X[i] > self.VarMax)
                V[i][is_outside] = -0.5 * V[i][is_outside]

                # 瞪羚跳跃步 (莱维飞行变异)
                if np.random.rand() < 0.2:
                    # 🌟 修复 3：将跳跃步长系数从 0.01 提升到 0.05，确保真的能跳出雷达陷阱
                    step = np.random.standard_t(df=1.5, size=self.dim) * 0.05 * (self.VarMax - self.VarMin)
                    X[i] = X[i] + step

                    # 跳跃后再次保护边界
                    is_outside_jump = (X[i] < self.VarMin) | (X[i] > self.VarMax)
                    V[i][is_outside_jump] = -0.5 * V[i][is_outside_jump]

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