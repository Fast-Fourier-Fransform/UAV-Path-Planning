import numpy as np
from .base_algorithm import BaseOptimizer



class HEOMPSO(BaseOptimizer):
    def __init__(self, nPop, MaxIt, VarMin, VarMax, dim, CostFunc):
        super().__init__(nPop, MaxIt, VarMin, VarMax, dim, CostFunc)

        # 算法核心参数
        self.w_max, self.w_min = 0.9, 0.4
        self.c1, self.c2 = 2.0, 2.0

        # HEOM 专属停滞计数器参数
        self.stall_count = 0
        self.stall_limit = 15

        # 速度边界 (最大速度限制为搜索空间的 20%)
        self.v_limit = (self.VarMax - self.VarMin) * 0.2

    def optimize(self):
        # 1. 向量化初始化种群位置和速度
        self.X = np.random.uniform(self.VarMin, self.VarMax, (self.nPop, self.dim))
        self.V = np.random.uniform(-self.v_limit, self.v_limit, (self.nPop, self.dim))

        # 初始化适应度数组
        current_F = np.zeros(self.nPop)
        for i in range(self.nPop):
            current_F[i] = self.CostFunc(self.X[i])

        # 初始化个体最优和全局最优
        self.pbest_X = np.copy(self.X)
        self.pbest_F = np.copy(current_F)

        best_idx = np.argmin(self.pbest_F)
        self.best_position = np.copy(self.pbest_X[best_idx])
        self.best_score = self.pbest_F[best_idx]

        # 2. 开始核心迭代
        for t in range(self.MaxIt):
            # 线性递减惯性权重
            w = self.w_max - (self.w_max - self.w_min) * (t / self.MaxIt)
            r1 = np.random.rand(self.nPop, self.dim)
            r2 = np.random.rand(self.nPop, self.dim)

            # 速度与位置更新 (矩阵化运算，极其高效)
            self.V = w * self.V + self.c1 * r1 * (self.pbest_X - self.X) + self.c2 * r2 * (self.best_position - self.X)
            self.V = np.clip(self.V, -self.v_limit, self.v_limit)
            self.X = self.X + self.V
            self.X = np.clip(self.X, self.VarMin, self.VarMax)

            # ==========================================================
            # 3. HEOM 核心改进机制 (终端保护期：最后 15% 迭代关闭扰动)
            if t < int(self.MaxIt * 0.85):
                if self.stall_count > self.stall_limit:

                    # 策略 A：Gbest 自适应高斯变异 (引自 HCLDMS-PSO)
                    decay = (1.0 - t / self.MaxIt) ** 2
                    sigma = (self.VarMax - self.VarMin) * 0.1 * decay
                    mut_gbest = self.best_position + np.random.normal(0, sigma, self.dim)
                    mut_gbest = np.clip(mut_gbest, self.VarMin, self.VarMax)

                    mut_F = self.CostFunc(mut_gbest)
                    if mut_F < self.best_score:
                        self.best_position = np.copy(mut_gbest)
                        self.best_score = mut_F

                    # 策略 B：劣质粒子动态边界对立学习 (引自 LCPSO 改进)
                    sort_idx = np.argsort(self.pbest_F)
                    worst_idx = sort_idx[-int(self.nPop * 0.2):]  # 取最差的 20%

                    c_min = np.min(self.X, axis=0)
                    c_max = np.max(self.X, axis=0)

                    for idx in worst_idx:
                        # 动态包围盒内对立反转
                        self.X[idx] = c_min + c_max - self.X[idx]
                        self.X[idx] = np.clip(self.X[idx], self.VarMin, self.VarMax)
                        self.V[idx] = np.random.uniform(-self.v_limit, self.v_limit, self.dim)

                    self.stall_count = 0  # 触发完毕，重置停滞计数器
            # ==========================================================

            # 4. 评估适应度并更新最优解
            for i in range(self.nPop):
                cost = self.CostFunc(self.X[i])

                # 更新个体最优
                if cost < self.pbest_F[i]:
                    self.pbest_X[i] = np.copy(self.X[i])
                    self.pbest_F[i] = cost

                    # 更新全局最优
                    if cost < self.best_score:
                        self.best_score = cost
                        self.best_position = np.copy(self.X[i])
                        self.stall_count = 0  # 有提升，清零停滞

            # 如果本代全局最优没有提升，增加停滞计数
            if np.min(self.pbest_F) >= self.best_score:
                self.stall_count += 1

            # 5. 记录当前迭代的最优值
            self.curve[t] = self.best_score
            #print(f"HEOM-PSO - Iteration {t + 1}: Best Cost = {self.best_score:.4f}")

        return self.best_position, self.best_score, self.curve