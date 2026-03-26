import numpy as np
import matplotlib.pyplot as plt


# --- 1. 定义经典基准测试函数 (全部支持多维自适应) ---
def sphere(x):
    return np.sum(x ** 2, axis=-1)


def ackley(x):
    dim = x.shape[-1]
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2, axis=-1) / dim))
    term2 = -np.exp(np.sum(np.cos(2.0 * np.pi * x), axis=-1) / dim)
    return term1 + term2 + 20.0 + np.e


def rosenbrock(x):
    return np.sum(100.0 * (x[..., 1:] - x[..., :-1] ** 2) ** 2 + (x[..., :-1] - 1.0) ** 2, axis=-1)


def rastrigin(x):
    A = 10
    dim = x.shape[-1]
    return A * dim + np.sum(x ** 2 - A * np.cos(2 * np.pi * x), axis=-1)


# --- 2. 核心算法类：HEOM-PSO vs Standard PSO ---
class HEOM_PSO:
    def __init__(self, dim, pop_size, max_iter, bounds, init_X, init_V, func, use_heom=True):
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.bounds = bounds
        self.func = func

        # 机制开关：True 为 HEOM改进版，False 为 标准PSO
        self.use_heom = use_heom

        # 经典 PSO 参数
        self.w_max, self.w_min = 0.9, 0.4
        self.c1, self.c2 = 2.0, 2.0

        # 统一初始化，保证公平对比
        self.X = np.copy(init_X)
        self.V = np.copy(init_V)
        self.pbest_X = np.copy(self.X)
        self.pbest_F = self.func(self.X)

        best_idx = np.argmin(self.pbest_F)
        self.gbest_X = np.copy(self.pbest_X[best_idx])
        self.gbest_F = self.pbest_F[best_idx]
        self.history = []

        # 停滞计数器
        self.stall_count = 0
        self.stall_limit = 15  # 全局最优连续 15 代不更新则判定为局部死锁

    def optimize(self):
        v_limit = (self.bounds[1] - self.bounds[0]) * 0.2

        for t in range(self.max_iter):
            # 线性递减惯性权重
            w = self.w_max - (self.w_max - self.w_min) * (t / self.max_iter)
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)

            # 1. 基础速度与位置更新 (Standard PSO)
            self.V = w * self.V + self.c1 * r1 * (self.pbest_X - self.X) + self.c2 * r2 * (self.gbest_X - self.X)
            self.V = np.clip(self.V, -v_limit, v_limit)
            self.X = np.clip(self.X + self.V, self.bounds[0], self.bounds[1])

            # ==========================================================
            # 2. HEOM 核心机制插入 (消融实验模块)
            # 终端保护期：最后 15% 的迭代关闭一切扰动，进行纯数学极致收敛
            if self.use_heom and t < int(self.max_iter * 0.85):
                if self.stall_count > self.stall_limit:

                    # 策略 A：Gbest 自适应高斯变异 (引自 HCLDMS-PSO)
                    # 变异方差随迭代次数非线性收缩，实现前期大步长跳跃，后期小步长微调
                    decay = (1.0 - t / self.max_iter) ** 2
                    sigma = (self.bounds[1] - self.bounds[0]) * 0.1 * decay
                    mut_gbest = self.gbest_X + np.random.normal(0, sigma, self.dim)
                    mut_gbest = np.clip(mut_gbest, self.bounds[0], self.bounds[1])
                    mut_F = self.func(np.array([mut_gbest]))[0]

                    # 贪婪替换
                    if mut_F < self.gbest_F:
                        self.gbest_X = np.copy(mut_gbest)
                        self.gbest_F = mut_F

                    # 策略 B：劣质粒子动态边界对立学习 (引自 LCPSO 改进)
                    # 取出当前适应度最差的 20% 粒子
                    sort_idx = np.argsort(self.pbest_F)
                    worst_idx = sort_idx[-int(self.pop_size * 0.2):]

                    # 计算当前种群的动态包围盒 (Dynamic Bounding Box)
                    c_min = np.min(self.X, axis=0)
                    c_max = np.max(self.X, axis=0)

                    for idx in worst_idx:
                        # 在动态包围盒内生成对立解，瞬间填补搜索盲区
                        self.X[idx] = c_min + c_max - self.X[idx]
                        self.X[idx] = np.clip(self.X[idx], self.bounds[0], self.bounds[1])
                        # 赋予随机初速度，打破死锁惯性
                        self.V[idx] = np.random.uniform(-v_limit, v_limit, self.dim)

                    self.stall_count = 0  # 机制触发完毕，重置停滞计数器
            # ==========================================================

            # 3. 评估与更新 Pbest, Gbest
            current_F = self.func(self.X)
            improve_mask = current_F < self.pbest_F
            self.pbest_X[improve_mask] = self.X[improve_mask]
            self.pbest_F[improve_mask] = current_F[improve_mask]

            best_idx = np.argmin(self.pbest_F)
            if self.pbest_F[best_idx] < self.gbest_F:
                self.gbest_F = self.pbest_F[best_idx]
                self.gbest_X = np.copy(self.pbest_X[best_idx])
                self.stall_count = 0  # 找到更优解，清零停滞
            else:
                self.stall_count += 1

            self.history.append(self.gbest_F)

        return self.history


# --- 3. 运行高维消融实验测试 ---
if __name__ == "__main__":
    dim = 300  # 极限高维测试 (可改500试试威力)
    pop_size = 50  # 种群数量
    max_iter = 1500  # 最大迭代次数

    tasks = [
        ("Sphere", sphere, [-100, 100]),
        ("Ackley", ackley, [-32, 32]),
        ("Rastrigin", rastrigin, [-5.12, 5.12]),
        ("Rosenbrock", rosenbrock, [-30, 30])
    ]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()

    for idx, (name, func, bounds) in enumerate(tasks):
        print(f"正在挑战 {name} 函数 (Dim={dim})...")

        # 绝对公平：生成一模一样的初始种群和初速度
        init_X = np.random.uniform(bounds[0], bounds[1], (pop_size, dim))
        init_V = np.random.uniform(-1, 1, (pop_size, dim))

        # 运行基准 PSO (use_heom=False)
        std_pso = HEOM_PSO(dim, pop_size, max_iter, bounds, init_X, init_V, func, use_heom=False)
        pso_history = std_pso.optimize()

        # 运行改进的 HEOM-PSO (use_heom=True)
        heom_pso = HEOM_PSO(dim, pop_size, max_iter, bounds, init_X, init_V, func, use_heom=True)
        heom_history = heom_pso.optimize()

        # 绘图
        axs[idx].plot(pso_history, label='Standard PSO', linewidth=2, alpha=0.8)
        axs[idx].plot(heom_history, label='HEOM-PSO (Proposed)', linewidth=2, alpha=0.8)
        axs[idx].set_title(f'{name} Function (Dim={dim})', fontsize=14)
        axs[idx].set_xlabel('Iteration', fontsize=12)
        axs[idx].set_ylabel('Fitness Value (Log Scale)', fontsize=12)
        axs[idx].set_yscale('log')
        axs[idx].grid(True, linestyle='--', alpha=0.6)
        axs[idx].legend(fontsize=12)

    plt.tight_layout()
    plt.show()
    print("测试跑完啦！快看看效果吧。")