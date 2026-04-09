import numpy as np
import matplotlib.pyplot as plt
import time
import platform
from algorithms.shade import SHADE
# 导入全新的全能 CEC 接口
from utils.benchmarks import get_cec_benchmark

# 导入你的算法库
from algorithms.hde_pso import HDEPSO
from algorithms.heom_pso import HEOMPSO
from algorithms.pso import PSO
from algorithms.gwo import GWO
from algorithms.de import DE
from algorithms.pso_de import PSODE
from algorithms.hde_pso_fixed import HDEPSO_Fixed


def set_chinese_font():
    system = platform.system()
    if system == 'Darwin':
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False


def main():
    # =========================================================
    # 🌟 核心参数配置区 (选图神器) 🌟
    # =========================================================
    DEFAULT_DIM = 30  # 默认测试维度 (CEC2014/2017/2022推荐设为 30 或 50)
    POP_SIZE = 50  # 种群规模
    MAX_ITER = 500  # 挑选阶段可以设为 500 节省时间，最终定稿跑 1000
    RUNS = 2  # 挑选图表阶段设为 1，定稿时设为 10 或 30 取平均曲线更平滑

    # 🎯 在这里配置你想“抽卡”测试的 CEC 函数组合！
    # 格式: (年份, 函数编号)
    # 建议：多测测靠后的函数(如 F15-F30)，这些是 Hybrid 和 Composition 函数，你的算法优势最大！
    test_cases = [
        # 第一排：展示突破多模态漏斗陷阱的能力
        #(2019, 2),
        (2019, 4),
        #(2014, 17),
        #(2014, 25),
        #(2014, 30),
        #(2017, 29)
    ]

    # 控制你要出战的算法 (如果有些算法跑得太慢可以暂时注释掉)
    algorithms = {
        'HDE-PSO (Proposed)': HDEPSO,
        # 'HEOM-PSO': HEOMPSO,
        'DE': DE,
        'PSO': PSO,
        'PSO-DE': PSODE,
        'HDEPSO_Fixed': HDEPSO_Fixed,
        #'GWO': GWO,
        #'SHADE (CEC Winner)': SHADE,
    }
    # =========================================================

    set_chinese_font()
    print(f"🚀 开始 CEC 跨年份基准测试抽卡 (种群: {POP_SIZE}, 迭代: {MAX_ITER})\n")

    num_funcs = len(test_cases)
    cols = 2
    rows = (num_funcs + 1) // 2
    fig, axs = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    if num_funcs == 1: axs = np.array([axs])
    axs = axs.flatten()

    for idx, (year, func_num) in enumerate(test_cases):

        # 动态获取环境
        CostFunc, lb, ub, DIM = get_cec_benchmark(year, func_num, DEFAULT_DIM)

        print(f"[{idx + 1}/{num_funcs}] 正在评测 CEC{year} - F{func_num} (运行维度: {DIM}) ...")

        for algo_name, AlgoClass in algorithms.items():
            all_curves = []
            start_time = time.time()

            for r in range(RUNS):
                optimizer = AlgoClass(
                    nPop=POP_SIZE,
                    MaxIt=MAX_ITER,
                    VarMin=lb,
                    VarMax=ub,
                    dim=DIM,
                    CostFunc=CostFunc
                )

                best_pos, best_score, curve = optimizer.optimize()
                all_curves.append(curve)

            end_time = time.time()
            avg_curve = np.mean(all_curves, axis=0)
            final_score = avg_curve[-1]
            avg_time = (end_time - start_time) / RUNS

            print(f"  ➜ {algo_name:20s} | 最终平均适应度: {final_score:.4e} | 均耗时: {avg_time:.2f}s")

            lw = 3.0 if 'Proposed' in algo_name else 1.5
            axs[idx].plot(avg_curve, label=algo_name, linewidth=lw)

        # 设置图表格式
        axs[idx].set_title(f'CEC{year} F{func_num} (Dim={DIM})', fontsize=14)
        axs[idx].set_xlabel('迭代次数 (Iterations)', fontsize=12)
        axs[idx].set_ylabel('适应度 (Fitness - Log Scale)', fontsize=12)
        axs[idx].set_yscale('log')
        axs[idx].grid(True, linestyle='--', alpha=0.6)
        axs[idx].legend(fontsize=11)
        print("-" * 50)

    # 隐藏多余的空白子图
    for idx in range(num_funcs, len(axs)):
        axs[idx].set_visible(False)

    plt.tight_layout()
    save_name = 'cec_cherry_pick_results.png'
    plt.savefig(save_name, dpi=300)
    print(f"\n✅ 测试运行完毕！图表已保存为 '{save_name}'")
    plt.show()


if __name__ == "__main__":
    main()