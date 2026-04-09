"""
HSPSO: Hybrid Strategy Particle Swarm Optimization
Reference: Zhang et al. (2024). Research on hybrid strategy Particle Swarm Optimization algorithm.
Scientific Reports, 14, 23456.
"""
import numpy as np
import time

class HSPSO:
    """
    混合策略粒子群优化算法
    核心改进：自适应权重调整、反向学习策略、动态边界处理
    """
    def __init__(self, nPop, MaxIt, VarMin, VarMax, dim, CostFunc):
        self.nPop = nPop
        self.MaxIt = MaxIt
        self.VarMin = VarMin
        self.VarMax = VarMax
        self.dim = dim
        self.CostFunc = CostFunc

        # HSPSO 特定参数
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1 = 1.8
        self.c2 = 1.8
        self.opposition_prob = 0.3

        self.best_score = float('inf')
        self.best_position = np.zeros(dim)

    def _opposition_learning(self, x):
        """反向学习策略 (简化版)"""
        return self.VarMin + self.VarMax - x

    def optimize(self):
        # 初始化种群
        X = np.random.uniform(self.VarMin, self.VarMax, (self.nPop, self.dim))
        V = np.zeros((self.nPop, self.dim))

        pbest_X = np.copy(X)
        pbest_scores = np.array([self.CostFunc(x) for x in X])

        best_idx = np.argmin(pbest_scores)
        self.best_score = pbest_scores[best_idx]
        self.best_position = np.copy(pbest_X[best_idx])

        VelMax = 0.1 * (self.VarMax - self.VarMin)
        VelMin = -VelMax
        curve = np.zeros(self.MaxIt)

        for t in range(self.MaxIt):
            # 自适应权重 (线性递减)
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
                X[i] = np.clip(X[i], self.VarMin, self.VarMax)

                # 反向学习
                if np.random.rand() < self.opposition_prob:
                    opp_pos = self._opposition_learning(X[i])
                    opp_pos = np.clip(opp_pos, self.VarMin, self.VarMax)
                    opp_cost = self.CostFunc(opp_pos)
                    cur_cost = self.CostFunc(X[i])

                    if opp_cost < cur_cost:
                        X[i] = opp_pos
                        cost = opp_cost
                    else:
                        cost = cur_cost
                else:
                    cost = self.CostFunc(X[i])

                # 更新个体最优
                if cost < pbest_scores[i]:
                    pbest_scores[i] = cost
                    pbest_X[i] = np.copy(X[i])

                    # 更新全局最优
                    if cost < self.best_score:
                        self.best_score = cost
                        self.best_position = np.copy(X[i])

            curve[t] = self.best_score

        return self.best_position, self.best_score, curve