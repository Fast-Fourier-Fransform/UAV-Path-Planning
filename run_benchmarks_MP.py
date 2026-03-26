import numpy as np
import matplotlib.pyplot as plt
import time
import platform
import concurrent.futures
import multiprocessing
from algorithms.shade import SHADE
# 导入全新的全能 CEC 接口
from algorithms.pso_de import PSODE
from algorithms.hde_pso_fixed import HDEPSO_Fixed
from utils.benchmarks import get_cec_benchmark

# 导入你的算法库
from algorithms.hde_pso import HDEPSO
from algorithms.pso import PSO
from algorithms.gwo import GWO
from algorithms.de import DE


def set_chinese_font():
    system = platform.system()
    if system == 'Darwin':
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False


# =========================================================
# 🌟 独立的工作进程函数 (核心黑科技) 🌟
# 注意：多进程无法传递复杂的嵌套闭包函数，所以必须在子进程中重新获取 CostFunc
# =========================================================
def run_single_task(algo_name, year, func_num, target_dim, pop_size, max_iter):
    # 1. 在独立的 CPU 核心中重新加载该地形函数
    CostFunc, lb, ub, actual_dim = get_cec_benchmark(year, func_num, target_dim)

    # 2. 映射需要实例化的算法
    algorithms = {
        'HDE-PSO (Proposed)': HDEPSO,
        # 'HEOM-PSO': HEOMPSO,
        'DE': DE,
        'PSO': PSO,
        'PSO-DE': PSODE,
        'HDEPSO_Fixed': HDEPSO_Fixed,
        # 'GWO': GWO,
        # 'SHADE (CEC Winner)': SHADE,
    }
    AlgoClass = algorithms[algo_name]

    # 3. 执行寻优
    optimizer = AlgoClass(
        nPop=pop_size,
        MaxIt=max_iter,
        VarMin=lb,
        VarMax=ub,
        dim=actual_dim,
        CostFunc=CostFunc
    )

    _, _, curve = optimizer.optimize()

    # 返回算法名称和收敛曲线，以便主进程收集
    return algo_name, curve


def main():
    # =========================================================
    # 🌟 核心参数配置区 🌟
    # =========================================================
    DEFAULT_DIM = 30
    POP_SIZE = 50
    MAX_ITER = 500
    RUNS = 20  # 🚀 现在有了多进程，你可以大胆地设为 10 甚至 30！

    test_cases = [
        (2019, 2),
        (2019, 4),
        (2014, 17),
        (2014, 25),
        (2014, 30),
        (2017, 29)
    ]

    algorithms_to_test = ['HDE-PSO (Proposed)', 'DE', 'PSO', 'HDEPSO_Fixed', 'PSO-DE']

    # 获取电脑的 CPU 逻辑核心数，预留1个核心保持系统流畅
    max_workers = max(1, multiprocessing.cpu_count() - 1)
    # =========================================================

    set_chinese_font()
    print(f"🚀 开始 CEC 多进程狂飙测试 (维度: {DEFAULT_DIM}, 种群: {POP_SIZE}, 迭代: {MAX_ITER})")
    print(f"🔥 已开启多进程并行加速，调用核心数: {max_workers}，每个算法重复运行: {RUNS} 次\n")

    num_funcs = len(test_cases)
    cols = 2
    rows = (num_funcs + 1) // 2
    fig, axs = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    if num_funcs == 1: axs = np.array([axs])
    axs = axs.flatten()

    for idx, (year, func_num) in enumerate(test_cases):

        # 仅为了主进程打印提示获取一次实际维度
        _, _, _, actual_DIM = get_cec_benchmark(year, func_num, DEFAULT_DIM)
        print(f"[{idx + 1}/{num_funcs}] 正在并行评测 CEC{year} - F{func_num} (运行维度: {actual_DIM}) ...")

        # 用于收集该函数下所有算法的多次运行结果
        results_collection = {algo: [] for algo in algorithms_to_test}
        start_time = time.time()

        # 启动进程池
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_algo = {}

            # 把所有任务（算法数量 * 运行次数）全部丢进池子里排队
            for algo_name in algorithms_to_test:
                for r in range(RUNS):
                    future = executor.submit(
                        run_single_task,
                        algo_name, year, func_num, DEFAULT_DIM, POP_SIZE, MAX_ITER
                    )
                    future_to_algo[future] = algo_name

            # 监听任务完成进度 (哪个核心跑完就收集哪个)
            completed_tasks = 0
            total_tasks = len(algorithms_to_test) * RUNS

            for future in concurrent.futures.as_completed(future_to_algo):
                algo_name = future_to_algo[future]
                completed_tasks += 1
                try:
                    _, curve = future.result()
                    results_collection[algo_name].append(curve)
                    # 进度条提示
                    print(f"\r  └─ 并行计算进度: {completed_tasks}/{total_tasks} 任务已完成", end="", flush=True)
                except Exception as exc:
                    print(f"\n[错误] {algo_name} 运行产生异常: {exc}")

        end_time = time.time()
        print(f"\n  ✅ 评测完成！总耗时: {(end_time - start_time):.2f} 秒")

        # 绘制该函数子图的曲线
        for algo_name in algorithms_to_test:
            curves = results_collection[algo_name]
            if curves:
                avg_curve = np.mean(curves, axis=0)
                final_score = avg_curve[-1]
                print(f"     ➜ {algo_name:20s} | 平均精度: {final_score:.4e}")

                lw = 3.0 if 'Proposed' in algo_name else 1.5
                axs[idx].plot(avg_curve, label=algo_name, linewidth=lw)

        # 设置图表格式
        axs[idx].set_title(f'CEC{year} F{func_num} (Dim={actual_DIM})', fontsize=14)
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
    save_name = 'cec_parallel_results.png'
    plt.savefig(save_name, dpi=300)
    print(f"\n🎉 所有多进程并行测试完毕！图表已保存为 '{save_name}'")
    plt.show()


if __name__ == "__main__":
    main()