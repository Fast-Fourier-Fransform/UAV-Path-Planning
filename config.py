import numpy as np

# ==========================================
# ⚙️ 算法通用核心参数 (不同地图共享)
# ==========================================
SEARCH_AGENTS_NO = 50
MAX_ITERATION = 150
# 将控制点减少，恢复航线的“物理刚性”，彻底消灭麻花弯！
NODES_NUMBER = 6       # 建议设为 5 到 8 之间
DIM = 2 * NODES_NUMBER
LB = 0

# ==========================================
# 🗺️ 动态环境变量 (默认初始化为山地，将被 set_env 动态覆盖)
# ==========================================
MAP_SIZE_X = 200
MAP_SIZE_Y = 200
MAP_SIZE_Z = 50
UB = 200
START_POINT = [10.0, 10.0, 15.0]
END_POINT = [190.0, 190.0, 15.0]
W1, W2, W3 = 0.5, 0.1, 0.4
OBSTACLES = []
ENV_TYPE = 'mountain'


def set_env(env_type):
    """一键切换地图环境的全局配置函数"""
    global MAP_SIZE_X, MAP_SIZE_Y, MAP_SIZE_Z, UB, START_POINT, END_POINT, W1, W2, W3, OBSTACLES, ENV_TYPE
    ENV_TYPE = env_type

    if env_type == 'mountain':
        print("🌍 全局配置已切换为：[复杂山地地形]")
        MAP_SIZE_X, MAP_SIZE_Y, MAP_SIZE_Z = 200, 200, 50
        UB = 200
        START_POINT = [10.0, 10.0, 15.0]
        END_POINT = [190.0, 190.0, 15.0]
        # 山地倾向于拉直并紧贴地形
        W1, W2, W3 = 0.8, 0.1, 0.1
        OBSTACLES = []

    elif env_type == 'cylinder':
        print("🏙️ 全局配置已切换为：[城市/雷达圆柱阵列]")
        MAP_SIZE_X, MAP_SIZE_Y, MAP_SIZE_Z = 100, 100, 50
        UB = 100
        START_POINT = [5.0, 5.0, 10.0]
        END_POINT = [95.0, 95.0, 10.0]
        # 城市倾向于暴力拉直，避免绕路
        W1, W2, W3 = 0.85, 0.1, 0.05
        OBSTACLES = np.array([
            [30, 30, 10, 40], [60, 60, 15, 25], [70, 20, 10, 35],
            [30, 80, 12, 30], [50, 45, 8, 45], [85, 50, 10, 20], [15, 55, 8, 25]
        ])