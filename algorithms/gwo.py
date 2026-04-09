import numpy as np
from .base_algorithm import BaseOptimizer


class GWO(BaseOptimizer):
    def optimize(self):
        # 1. 初始化种群
        self.X = np.random.uniform(self.VarMin, self.VarMax, (self.nPop, self.dim))
        self.scores = np.zeros(self.nPop)
        for i in range(self.nPop):
            self.scores[i] = self.CostFunc(self.X[i])

        # 2. 初始化 Alpha, Beta, Delta 狼的位置和分数
        sort_idx = np.argsort(self.scores)
        self.alpha_pos = np.copy(self.X[sort_idx[0]])
        self.alpha_score = self.scores[sort_idx[0]]

        self.beta_pos = np.copy(self.X[sort_idx[1]])
        self.beta_score = self.scores[sort_idx[1]]

        self.delta_pos = np.copy(self.X[sort_idx[2]])
        self.delta_score = self.scores[sort_idx[2]]

        # 3. 开始迭代
        for t in range(self.MaxIt):
            # 收敛因子 a 从 2 线性递减到 0
            a = 2.0 - t * (2.0 / self.MaxIt)

            for i in range(self.nPop):
                # Alpha 狼的引导
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = np.abs(C1 * self.alpha_pos - self.X[i])
                X1 = self.alpha_pos - A1 * D_alpha

                # Beta 狼的引导
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = np.abs(C2 * self.beta_pos - self.X[i])
                X2 = self.beta_pos - A2 * D_beta

                # Delta 狼的引导
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = np.abs(C3 * self.delta_pos - self.X[i])
                X3 = self.delta_pos - A3 * D_delta

                # 综合三只头狼的位置更新当前狼的位置
                self.X[i] = (X1 + X2 + X3) / 3.0
                self.X[i] = np.clip(self.X[i], self.VarMin, self.VarMax)

                # 评估适应度
                cost = self.CostFunc(self.X[i])
                self.scores[i] = cost

                # 严格的等级更新
                if cost < self.alpha_score:
                    self.delta_score, self.delta_pos = self.beta_score, np.copy(self.beta_pos)
                    self.beta_score, self.beta_pos = self.alpha_score, np.copy(self.alpha_pos)
                    self.alpha_score, self.alpha_pos = cost, np.copy(self.X[i])
                elif cost < self.beta_score:
                    self.delta_score, self.delta_pos = self.beta_score, np.copy(self.beta_pos)
                    self.beta_score, self.beta_pos = cost, np.copy(self.X[i])
                elif cost < self.delta_score:
                    self.delta_score, self.delta_pos = cost, np.copy(self.X[i])

            self.best_score = self.alpha_score
            self.best_position = np.copy(self.alpha_pos)
            self.curve[t] = self.best_score
            #print(f"GWO - Iteration {t + 1}: Best Cost = {self.best_score:.4f}")

        return self.best_position, self.best_score, self.curve