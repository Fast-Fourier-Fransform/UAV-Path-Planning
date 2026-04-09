import os
import time
import random
import numpy as np
import concurrent.futures
import csv
import config

# ==========================================
# 🚀 论文环境与基础配置
# ==========================================
TARGET_ENV = 'mountain_radar'  # 测试的极难地形
NUM_RUNS = 5  # 论文要求的独立运行次数

config.set_env(TARGET_ENV)
from environment import UAVEnvironment

# 导入所有算法
from algorithms.pso import PSO
from algorithms.dcw_pso import DCWPSO
from algorithms.hspso import HSPSO
from algorithms.mpsogoa import MPSOGOA
from algorithms.vn_ipso import VNIPSO
from algorithms.pso_de import PSODE
from algorithms.hde_pso_fixed import HDEPSO_Fixed

try:
    from algorithms.spso import SPSO
except ImportError:
    SPSO = None
from algorithms.hde_pso import HDEPSO


# ==========================================
# 🧠 核心：子进程单次独立评估任务
# ==========================================
def run_single_trial(args):
    algo_name, run_id, target_env = args

    # 1. 先初始化环境
    config.set_env(target_env)
    local_env = UAVEnvironment()

    # 2. 🌟 核心修复：在环境初始化【之后】强制注入绝对随机种子！
    local_seed = (int(time.time() * 1000) + os.getpid() + hash(algo_name) + run_id) % 2000000000
    np.random.seed(local_seed)
    random.seed(local_seed)

    # 注册表中的键名必须与下面列表中的名字一模一样
    algorithm_registry = {
        '1. Standard PSO': PSO,
        '2. DCWPSO (2024)': DCWPSO,
        '3. HSPSO (2024)': HSPSO,
        '4. MPSOGOA (2024)': MPSOGOA,
        '5. VN-IPSO (2024)': VNIPSO,
       # '6. SPSO (2021)': SPSO,
        '7. PSODE': PSODE,  # 消融实验1
        '8. HDEPSO-Fixed': HDEPSO_Fixed,  # 消融实验2
        '9. HDE-PSO (Proposed)': HDEPSO  # 所提算法
    }

    AlgoClass = algorithm_registry.get(algo_name)
    if AlgoClass is None:
        return algo_name, run_id, None, 0

    start_time = time.time()
    optimizer = AlgoClass(
        nPop=config.SEARCH_AGENTS_NO,
        MaxIt=config.MAX_ITERATION,
        VarMin=config.LB,
        VarMax=config.UB,
        dim=config.DIM,
        CostFunc=local_env.cost_function
    )

    _, best_score, _ = optimizer.optimize()
    cost_time = time.time() - start_time

    print(f"  ⚡️ [核心计算完成] | 算法: {algo_name:<22} | 独立运行: 第 {run_id:02d} 次 | 最优代价: {best_score:.4f}")

    return algo_name, run_id, best_score, cost_time


# ==========================================
# 👑 主控程序：拉起 M4 并发与统计分析
# ==========================================
def main():
    # 🚨 修复：补充了缺失的逗号，并将消融实验算法放在 Proposed 前面，符合论文逻辑顺序
    algorithms_to_run = [
        '1. Standard PSO',
        '2. DCWPSO (2024)',
        '3. HSPSO (2024)',
        '4. MPSOGOA (2024)',
        '5. VN-IPSO (2024)',
        '9. HDE-PSO (Proposed)'
       # '7. PSODE',
        #'8. HDEPSO-Fixed',

    ]

    if SPSO is not None:
        algorithms_to_run.insert(5, '6. SPSO (2021)')

    # 获取 M4 总核心数，强制减 1，留一个核心给系统和后台软件
    total_cores = os.cpu_count() or 4
    workers = max(1, total_cores - 1)

    print("=" * 85)
    print(f"🚀 启动 UAV 论文级盲测实验 | 地形: {TARGET_ENV} | 独立运行: {NUM_RUNS} 次")
    print(f"💻 正在调用多核并发引擎 (已分配工作核心: {workers}/{total_cores}，保留1核待机)...")
    print("=" * 85)

    tasks = [(algo, i + 1, TARGET_ENV) for algo in algorithms_to_run for i in range(NUM_RUNS)]
    results_raw = {algo: {'scores': [], 'times': []} for algo in algorithms_to_run}

    total_start = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for res in executor.map(run_single_trial, tasks):
            algo_name, run_id, best_score, cost_time = res
            if best_score is not None:
                results_raw[algo_name]['scores'].append(best_score)
                results_raw[algo_name]['times'].append(cost_time)

    total_end = time.time()

    # 计算统计指标
    stats_list = []
    for algo, data in results_raw.items():
        scores = np.array(data['scores'])
        times = np.array(data['times'])

        if len(scores) == 0:
            continue

        stats_list.append({
            'Algorithm': algo,
            'Best (Min)': np.min(scores),
            'Worst (Max)': np.max(scores),
            'Mean': np.mean(scores),
            'Std Dev': np.std(scores),
            'Variance': np.var(scores),
            'Avg Time (s)': np.mean(times)
        })

    # 按均值从小到大排序
    stats_list.sort(key=lambda x: x['Mean'])

    print("\n\n" + "=" * 115)
    print("🏆 UAV 路径规划学术统计表 (Statistical Results of 10 Independent Runs)")
    print("=" * 115)
    header = f"{'Algorithm':<25} | {'Best(Min)':<12} | {'Worst(Max)':<12} | {'Mean':<12} | {'Std Dev':<12} | {'Variance':<12} | {'Avg Time(s)':<12}"
    print(header)
    print("-" * 115)

    for row in stats_list:
        print(
            f"{row['Algorithm']:<25} | {row['Best (Min)']:<12.4f} | {row['Worst (Max)']:<12.4f} | {row['Mean']:<12.4f} | {row['Std Dev']:<12.4f} | {row['Variance']:<12.1f} | {row['Avg Time (s)']:<12.2f}")
    print("=" * 115)
    print(f"✨ 实验全部完成！(总耗时仅: {(total_end - total_start) / 60:.2f} 分钟)\n")

    csv_filename = f"paper_statistical_results_{TARGET_ENV}.csv"
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Algorithm', 'Best (Min)', 'Worst (Max)', 'Mean', 'Std Dev', 'Variance',
                                               'Avg Time (s)'])
        writer.writeheader()
        writer.writerows(stats_list)
    print(f"💾 纯净数据已自动导出至: {csv_filename} ，可直接使用 Excel 打开！")


if __name__ == "__main__":
    main()