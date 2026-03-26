import concurrent.futures
import time
import config

# ==========================================
# 🚀 环境切换：
# 'mountain' -> 复杂山地地形
# 'cylinder' -> 城市雷达圆柱体
# ==========================================
TARGET_ENV = 'mountain'
#TARGET_ENV = 'cylinder'
config.set_env(TARGET_ENV)

from environment import UAVEnvironment

# 1. 算法库
from algorithms.pso import PSO
from algorithms.hde_pso import HDEPSO
from algorithms.pso_de import PSODE
from algorithms.hde_pso_fixed import HDEPSO_Fixed

# 2. 可视化工具
from utils.visualizer import plot_multiple_3d_paths, plot_convergence_curves


def run_single_algorithm(algo_name):
    """
    子进程函数，用于并行执行单个算法
    """
    config.set_env(TARGET_ENV)
    local_env = UAVEnvironment()

    # 🌟 算法注册表（键名将直接作为图例显示在图中，保持英文）
    algorithm_registry = {
        '1. Standard PSO': PSO,
        '2. PSO-DE': PSODE,
        '3. HDE-PSO-Fixed': HDEPSO_Fixed,
        '4. HDE-PSO (Proposed)': HDEPSO
    }

    optimizer_class = algorithm_registry.get(algo_name)
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

    # 🌟 待运行的算法队列（键名英文，用于图例）
    algorithms_to_run = [
        '1. Standard PSO',
        '2. PSO-DE',
        '3. HDE-PSO-Fixed',
        '4. HDE-PSO (Proposed)'
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
    print("🏆 消融实验定量结果")
    print("=" * 50)

    sorted_results = sorted(results_list, key=lambda x: x['Best Cost'])

    for rank, res in enumerate(sorted_results, 1):
        print(
            f"排名 {rank} | 最优适应度: {res['Best Cost']:<10.4f} | 耗时: {res['Time (s)']:<5.2f}s | 算法: {res['Algorithm']}")
    print("=" * 50)

    # ==========================================
    # 📈 渲染学术图表（图表内文本均为英文）
    # ==========================================
    env_name = "Mountainous" if config.ENV_TYPE == 'mountain' else "Urban"

    plot_convergence_curves(all_curves_data, title=f"Convergence Curves in {env_name} Terrain")
    plot_multiple_3d_paths(main_env.map_data, all_paths_data, title=f"3D Path Comparison in {env_name} Terrain")


if __name__ == "__main__":
    main()