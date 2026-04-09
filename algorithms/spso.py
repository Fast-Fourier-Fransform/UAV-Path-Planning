import numpy as np
from .base_algorithm import BaseOptimizer
import config


class SPSO(BaseOptimizer):
    """
    SPSO: Spherical Vector-based Particle Swarm Optimization (2021)
    论文: Safety-enhanced UAV path planning with spherical vector-based particle swarm optimization
    """

    def __init__(self, nPop, MaxIt, VarMin, VarMax, dim, CostFunc):
        self.nodes_num = config.NODES_NUMBER
        # 🌟 核心：SPSO 原生搜索 3N 维的球面空间 (rho: 长度, psi: 极角, phi: 方位角)
        self.spherical_dim = 3 * self.nodes_num
        super().__init__(nPop, MaxIt, VarMin, VarMax, self.spherical_dim, CostFunc)

        # 设定球面向量的边界限制
        step_max = (config.END_POINT[0] - config.START_POINT[0]) / self.nodes_num * 2.0
        self.bounds_min = np.array([0, 0, -np.pi] * self.nodes_num)
        self.bounds_max = np.array([step_max, np.pi, np.pi] * self.nodes_num)

    def decode_spherical(self, spherical_pos):
        """将 3N 的球面向量序列，平滑映射到 2N 的降维笛卡尔坐标系中 (Y和Z)"""
        rhos = spherical_pos[0::3]
        psis = spherical_pos[1::3]
        phis = spherical_pos[2::3]

        Y = np.zeros(self.nodes_num)
        Z = np.zeros(self.nodes_num)

        curr_y = config.START_POINT[1]
        curr_z = config.START_POINT[2]

        for i in range(self.nodes_num):
            # 球面坐标系转换公式
            curr_y += rhos[i] * np.sin(psis[i]) * np.sin(phis[i])
            curr_z += rhos[i] * np.cos(psis[i])
            Y[i] = curr_y
            Z[i] = curr_z

        # 截断以确保不出界
        Y = np.clip(Y, 0, config.MAP_SIZE_Y)
        Z = np.clip(Z, 0, config.MAP_SIZE_Z)
        return np.concatenate([Y, Z])

    def optimize(self):
        w_max, w_min = 0.9, 0.4
        c1, c2 = 2.0, 2.0
        VelMax = 0.2 * (self.bounds_max - self.bounds_min)
        VelMin = -VelMax

        particles = []
        for i in range(self.nPop):
            # 粒子在 3N 维球面空间随机初始化
            pos = np.random.uniform(self.bounds_min, self.bounds_max, self.spherical_dim)
            # 转成笛卡尔坐标去适应你的 2N CostFunc
            cartesian_pos = self.decode_spherical(pos)
            cost = self.CostFunc(cartesian_pos)

            particles.append({
                'Position': pos,
                'Velocity': np.zeros(self.spherical_dim),
                'Cost': cost,
                'Best_Position': pos.copy(),
                'Best_Cost': cost
            })
            if cost < self.best_score:
                self.best_score = cost
                self.best_position = pos.copy()

        for it in range(self.MaxIt):
            w = w_max - it * ((w_max - w_min) / self.MaxIt)
            for i in range(self.nPop):
                r1, r2 = np.random.rand(self.spherical_dim), np.random.rand(self.spherical_dim)
                particles[i]['Velocity'] = (w * particles[i]['Velocity'] +
                                            c1 * r1 * (particles[i]['Best_Position'] - particles[i]['Position']) +
                                            c2 * r2 * (self.best_position - particles[i]['Position']))

                particles[i]['Velocity'] = np.clip(particles[i]['Velocity'], VelMin, VelMax)
                particles[i]['Position'] += particles[i]['Velocity']

                # 越界反弹惩罚
                is_out = (particles[i]['Position'] < self.bounds_min) | (particles[i]['Position'] > self.bounds_max)
                particles[i]['Velocity'][is_out] *= -0.5
                particles[i]['Position'] = np.clip(particles[i]['Position'], self.bounds_min, self.bounds_max)

                # 计算适应度
                cartesian_pos = self.decode_spherical(particles[i]['Position'])
                cost = self.CostFunc(cartesian_pos)
                particles[i]['Cost'] = cost

                if cost < particles[i]['Best_Cost']:
                    particles[i]['Best_Position'] = particles[i]['Position'].copy()
                    particles[i]['Best_Cost'] = cost
                    if cost < self.best_score:
                        self.best_score = cost
                        self.best_position = particles[i]['Position'].copy()

            self.curve[it] = self.best_score

        # 🌟 绝杀：输出前，转换为 2N 的笛卡尔坐标，保证 main.py 解码画图不崩溃！
        return self.decode_spherical(self.best_position), self.best_score, self.curve