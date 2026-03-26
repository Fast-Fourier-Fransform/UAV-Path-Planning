import numpy as np
import random
from .base_algorithm import BaseOptimizer


class SHADE(BaseOptimizer):
    def __init__(self, nPop, MaxIt, VarMin, VarMax, dim, CostFunc):
        super().__init__(nPop, MaxIt, VarMin, VarMax, dim, CostFunc)

        # SHADE 专属核心参数
        self.H = 100  # 历史记忆库大小 (Historical memory size)
        self.p = 0.11  # p-best 变异比例

        # 初始化历史记忆库 (CR 和 F 初始均设为 0.5)
        self.M_CR = np.ones(self.H) * 0.5
        self.M_F = np.ones(self.H) * 0.5

        self.archive = []  # 外部劣解存档 (Archive)
        self.k = 0  # 记忆库更新索引

    def optimize(self):
        # 1. 种群初始化
        self.X = np.random.uniform(self.VarMin, self.VarMax, (self.nPop, self.dim))
        self.scores = np.zeros(self.nPop)
        for i in range(self.nPop):
            self.scores[i] = self.CostFunc(self.X[i])

        best_idx = np.argmin(self.scores)
        self.best_score = self.scores[best_idx]
        self.best_position = np.copy(self.X[best_idx])

        # 2. 核心迭代
        for t in range(self.MaxIt):
            S_CR = []  # 记录本代成功的 CR
            S_F = []  # 记录本代成功的 F
            delta_f = []  # 记录适应度提升量

            # 为每个个体生成 CR 和 F
            r_idx = np.random.randint(0, self.H, self.nPop)
            CR = np.random.normal(self.M_CR[r_idx], 0.1)
            CR = np.clip(CR, 0, 1)

            F = np.zeros(self.nPop)
            for i in range(self.nPop):
                # F 的生成采用柯西分布，若 <=0 则重新生成，若 >1 则截断为 1
                while F[i] <= 0:
                    F[i] = self.M_F[r_idx[i]] + 0.1 * np.random.standard_cauchy()
                if F[i] > 1:
                    F[i] = 1

            # 找出当前种群的 Top p% 个体作为 p-best 候选池
            sort_idxs = np.argsort(self.scores)
            p_num = max(int(self.nPop * self.p), 1)

            trial_X = np.zeros_like(self.X)

            # DE/current-to-pbest/1 变异与交叉
            for i in range(self.nPop):
                # 随机选择一个 p-best
                pbest_idx = sort_idxs[np.random.randint(0, p_num)]
                x_pbest = self.X[pbest_idx]

                # 随机选择 r1 (不等于 i)
                idxs = list(range(self.nPop))
                idxs.remove(i)
                r1 = random.choice(idxs)
                idxs.remove(r1)

                # 随机选择 r2 (从 种群 U 存档 中选择)
                P_union_A = self.X.tolist() + self.archive
                r2_idx = random.randint(0, len(P_union_A) - 1)

                # 防止 r2 与 i 或 r1 重复
                while r2_idx == i or r2_idx == r1:
                    r2_idx = random.choice(range(len(P_union_A)))

                x_r2 = np.array(P_union_A[r2_idx])
                x_r1 = self.X[r1]

                # 变异算子
                v = self.X[i] + F[i] * (x_pbest - self.X[i]) + F[i] * (x_r1 - x_r2)
                v = np.clip(v, self.VarMin, self.VarMax)

                # 二项式交叉
                j_rand = random.randint(0, self.dim - 1)
                mask = np.random.rand(self.dim) < CR[i]
                mask[j_rand] = True
                trial_X[i] = np.where(mask, v, self.X[i])

            # 3. 严格选择与更新存档
            for i in range(self.nPop):
                trial_score = self.CostFunc(trial_X[i])

                # 如果试探个体优于原个体
                if trial_score < self.scores[i]:
                    # 将被淘汰的原个体加入外部存档
                    self.archive.append(self.X[i].copy())

                    # 记录成功参数和提升量
                    delta_f.append(self.scores[i] - trial_score)
                    S_CR.append(CR[i])
                    S_F.append(F[i])

                    # 替换原个体
                    self.X[i] = trial_X[i]
                    self.scores[i] = trial_score

                    # 更新全局最优
                    if trial_score < self.best_score:
                        self.best_score = trial_score
                        self.best_position = np.copy(trial_X[i])

            # 维护存档大小不超过 nPop
            if len(self.archive) > self.nPop:
                self.archive = random.sample(self.archive, self.nPop)

            # 4. 更新历史记忆库
            if len(S_CR) > 0:
                weights = np.array(delta_f) / np.sum(delta_f)

                # 加权算术平均更新 M_CR
                self.M_CR[self.k] = np.sum(weights * np.array(S_CR))

                # 加权 Lehmer 平均更新 M_F
                self.M_F[self.k] = np.sum(weights * (np.array(S_F) ** 2)) / np.sum(weights * np.array(S_F))

                self.k = (self.k + 1) % self.H

            self.curve[t] = self.best_score

        return self.best_position, self.best_score, self.curve