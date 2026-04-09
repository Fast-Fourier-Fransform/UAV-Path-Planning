"""
VN-IPSO: Improved Particle Swarm Optimization Based on Variable Neighborhood Search
Reference: Li et al. (2024). Mathematics, 12(17), 2708.
"""
import numpy as np


class VNIPSO:
    """
    变邻域搜索改进粒子群优化算法
    核心改进：变邻域搜索策略、局部搜索增强
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

        self.n_neighborhoods = 5
        self.local_search_prob = 0.3

        self.best_score = float('inf')
        self.best_position = np.zeros(dim)

    def _variable_neighborhood_search(self, x, t, k):
        """简化的变邻域搜索：在当前位置附近随机扰动"""
        radius = (self.VarMax - self.VarMin) * 0.1 * (1 - t / self.MaxIt) * (k / self.n_neighborhoods)
        perturbation = np.random.uniform(-radius, radius, self.dim)
        new_x = x + perturbation
        return np.clip(new_x, self.VarMin, self.VarMax)

    def _local_search(self, best_pos, t):
        """简化的局部搜索：在全局最优附近进行高斯变异"""
        sigma = (self.VarMax - self.VarMin) * 0.05 * (1 - t / self.MaxIt)
        new_pos = best_pos + np.random.normal(0, sigma, self.dim)
        new_pos = np.clip(new_pos, self.VarMin, self.VarMax)
        cost = self.CostFunc(new_pos)
        return new_pos, cost

    def optimize(self):
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
            w = self.w_max - (self.w_max - self.w_min) * (t / self.MaxIt)

            for i in range(self.nPop):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                V[i] = (w * V[i] +
                        self.c1 * r1 * (pbest_X[i] - X[i]) +
                        self.c2 * r2 * (self.best_position - X[i]))
                V[i] = np.clip(V[i], VelMin, VelMax)
                X[i] = X[i] + V[i]
                X[i] = np.clip(X[i], self.VarMin, self.VarMax)

                # 变邻域搜索机制
                neighborhood_idx = np.random.randint(1, self.n_neighborhoods + 1)
                new_pos = self._variable_neighborhood_search(X[i], t, neighborhood_idx)
                new_cost = self.CostFunc(new_pos)

                if new_cost < pbest_scores[i]:
                    X[i] = np.copy(new_pos)
                    pbest_scores[i] = new_cost
                    pbest_X[i] = np.copy(new_pos)
                    if new_cost < self.best_score:
                        self.best_score = new_cost
                        self.best_position = np.copy(new_pos)
                else:
                    cost = self.CostFunc(X[i])
                    if cost < pbest_scores[i]:
                        pbest_scores[i] = cost
                        pbest_X[i] = np.copy(X[i])
                        if cost < self.best_score:
                            self.best_score = cost
                            self.best_position = np.copy(X[i])

            # 局部最优跳出机制
            if np.random.rand() < self.local_search_prob:
                new_best_pos, new_best_cost = self._local_search(self.best_position, t)
                if new_best_cost < self.best_score:
                    self.best_score = new_best_cost
                    self.best_position = np.copy(new_best_pos)

            curve[t] = self.best_score

        return self.best_position, self.best_score, curve