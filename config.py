import numpy as np

# ==========================================
# ⚙️ 算法通用核心参数 (不同地图共享)
# ==========================================
SEARCH_AGENTS_NO = 50
MAX_ITERATION = 300
# 🌟 修复 1: 将控制点从 5 增加到 7。因为雷达缝隙狭窄，控制点太少会导致曲线过于僵硬扫到柱子，算法只能被迫绕大圈。
NODES_NUMBER = 5
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
        W1, W2, W3 = 0.7, 0.1, 0.2
        OBSTACLES = []

    elif env_type == 'cylinder':
        print("🏙️ 全局配置已切换为：[城市/雷达圆柱阵列]")
        MAP_SIZE_X, MAP_SIZE_Y, MAP_SIZE_Z = 100, 100, 50
        UB = 100
        START_POINT = [5.0, 5.0, 10.0]
        END_POINT = [95.0, 95.0, 10.0]
        W1, W2, W3 = 0.85, 0.1, 0.05
        OBSTACLES = np.array([
            [30, 30, 10, 40], [60, 60, 15, 25], [70, 20, 10, 35],
            [30, 80, 12, 30], [50, 45, 8, 45], [85, 50, 10, 20], [15, 55, 8, 25]
        ])

    elif env_type == 'mountain_radar':
        print("⛰️+📡 全局配置已切换为：[复杂山地与雷达禁飞区混合地形]")
        MAP_SIZE_X, MAP_SIZE_Y, MAP_SIZE_Z = 200, 200, 50
        UB = 200
        START_POINT = [10.0, 10.0, 15.0]
        END_POINT = [190.0, 190.0, 15.0]

        # 🌟 修复 2: 极端拔高路径长度权重(W1=0.8)，压低高度权重(W2=0.1)，严厉惩罚算法的绕路行为！
        W1, W2, W3 = 0.8, 0.1, 0.1

        # 格式: [X, Y, R, H]
        OBSTACLES = np.array([
            [60, 50, 15, 120],
            [120, 80, 18, 120],
            [70, 140, 20, 120],
            [150, 130, 15, 120],
            [100, 100, 12, 120]
        ])