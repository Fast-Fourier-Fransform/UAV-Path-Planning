import concurrent.futures
import time
import config
from algorithms.spso import SPSO
# ==========================================
# 🚀 环境切换：
# 'mountain' -> 复杂山地地形
# 'cylinder' -> 城市雷达圆柱体
# mountain_radar
# ==========================================
TARGET_ENV = 'mountain_radar'
# TARGET_ENV = 'cylinder'
config.set_env(TARGET_ENV)
from environment import UAVEnvironment
# 1. 算法库
from algorithms.pso import PSO
from algorithms.hde_pso import HDEPSO
from algorithms.pso_de import PSODE
from algorithms.hde_pso_fixed import HDEPSO_Fixed


# 🌟 新增：2024 前沿对比算法
from algorithms.dcw_pso import DCWPSO
from algorithms.hspso import HSPSO
from algorithms.mpsogoa import MPSOGOA
from algorithms.vn_ipso import VNIPSO  # 加上最后一个算法

# 2. 可视化工具
from utils.visualizer import plot_multiple_3d_paths, plot_convergence_curves


def run_single_algorithm(algo_name):
    """
    子进程函数，用于并行执行单个算法
    """
    import os
    import random
    import numpy as np

    # 1. 必须【先】初始化环境！让它内部把 np.random.seed(42) 执行完
    config.set_env(TARGET_ENV)
    local_env = UAVEnvironment()

    # 2. 🌟 核心修复：在环境初始化【之后】强制注入绝对随机种子！
    # 彻底覆盖掉地形生成遗留的 42 种子，让粒子群真正的随机撒点
    local_seed = (int(time.time() * 1000) + os.getpid() + hash(algo_name)) % 2000000000
    np.random.seed(local_seed)
    random.seed(local_seed) # 同步重置 Python 原生 random，双重保险

    # 🌟 算法注册表（键名将直接作为图例显示在图中，保持英文）
    algorithm_registry = {
        '1. Standard PSO':PSO,
        #'2. PSO-DE':PSODE,             # 注释掉消融变体，保持图表不至于过于拥挤
        #'3. HDE-PSO-Fixed':HDEPSO_Fixed,
        '2. DCWPSO':DCWPSO,
        '3. HSPSO':HSPSO,
        '4. MPSOGOA':MPSOGOA,
        '5. VN-IPSO':VNIPSO,
        '6. HDE-PSO (Proposed)':HDEPSO,
        #'9. SPSO': SPSO  # 🌟 新增 SPSO
    }

    optimizer_class = algorithm_registry.get(algo_name)
    if optimizer_class is None:
        raise ValueError(f"算法 '{algo_name}' 未在注册表中找到，请检查拼写！")

    optimizer = optimizer_class(
        nPop=config.SEARCH_AGENTS_NO,
        MaxIt=config.MAX_ITERATION,
        VarMin=config.LB,
        VarMax=config.UB,
        dim=config.DIM,
        CostFunc=local_env.cost_function
    )

    print(f"[进程启动] >>> 正在运行: {algo_name} ...")
    start_time = time.time()

    best_pos, best_score, curve = optimizer.optimize()

    cost_time = time.time() - start_time
    print(f"[进程完成] <<< [{algo_name}] 最优适应度: {best_score:.4f}, 耗时: {cost_time:.2f}s")

    X_seq, Y_seq, Z_seq, _, _, _ = local_env.get_path_line(best_pos)

    return {
        'algo_name': algo_name,
        'best_score': best_score,
        'curve_data': (curve, algo_name),
        'path_data': (X_seq, Y_seq, Z_seq, algo_name),
        'cost_time': cost_time
    }

def main():
    print("1. 初始化多进程环境...")
    main_env = UAVEnvironment()

    # 🌟 待运行的算法队列（注意：这里的字符串必须与上面注册表里的完全一致，每一行末尾必须有逗号！）
    algorithms_to_run = [
        '1. Standard PSO',
        #'2. PSO-DE',            # 注释掉消融变体，保持图表不至于过于拥挤
        #'3. HDE-PSO-Fixed',
        '2. DCWPSO',
        '3. HSPSO',
        '4. MPSOGOA',
        '5. VN-IPSO',
        #'9. SPSO',
        '6. HDE-PSO (Proposed)'  # 故意放到最后，确保渲染时它画在最上层
    ]

    print(f"\n2. 💥 在 M4 芯片上启动多进程，共 {len(algorithms_to_run)} 个算法并行运行...\n")

    results_list = []
    all_paths_data = []
    all_curves_data = []

    total_start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=len(algorithms_to_run)) as executor:
        results = executor.map(run_single_algorithm, algorithms_to_run)

        for res in results:
            results_list.append({
                'Algorithm': res['algo_name'],
                'Best Cost': res['best_score'],
                'Time (s)': round(res['cost_time'], 2)
            })
            all_curves_data.append(res['curve_data'])
            all_paths_data.append(res['path_data'])

    total_cost_time = time.time() - total_start_time
    print(f"\n✨ 优化完成！总耗时: {total_cost_time:.2f} s")

    # ==========================================
    # 📊 在控制台直接输出定量实验数据对比
    # ==========================================
    print("\n" + "=" * 50)
    print("🏆 SOTA 算法对比实验定量结果")
    print("=" * 50)

    sorted_results = sorted(results_list, key=lambda x: x['Best Cost'])

    for rank, res in enumerate(sorted_results, 1):
        print(
            f"排名 {rank} | 最优适应度: {res['Best Cost']:<10.4f} | 耗时: {res['Time (s)']:<5.2f}s | 算法: {res['Algorithm']}")
    print("=" * 50)

    # ==========================================
    # 📈 渲染学术图表（图表内文本均为英文）
    # ==========================================
    if config.ENV_TYPE == 'mountain':
        env_name = "Mountainous"
    elif config.ENV_TYPE == 'mountain_radar':
        env_name = "Hybrid Mountain-Radar"
    else:
        env_name = "Urban"

    print("\n🎨 正在渲染 3D 轨迹图与收敛曲线图...")
    plot_convergence_curves(all_curves_data, title=f"Convergence Curves in {env_name} Terrain")
    plot_multiple_3d_paths(main_env.map_data, all_paths_data, title=f"3D Path Comparison in {env_name} Terrain")


if __name__ == "__main__":
    main()