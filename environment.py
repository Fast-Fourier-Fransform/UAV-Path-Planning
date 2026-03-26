import numpy as np
from scipy.interpolate import CubicSpline
import config


class UAVEnvironment:
    def __init__(self):
        # 根据全局配置，智能生成对应的底层地形数据
        if config.ENV_TYPE == 'mountain':
            self.map_data = self._generate_mountain_map()
        else:
            self.map_data = np.zeros((config.MAP_SIZE_X, config.MAP_SIZE_Y))

    # ==================== 山地生成核心函数 ====================
    def map_value_function(self, x, y):
        a, b, c, d, e, f_const, g = 1, 1, 1, 1, 1, 1, 1
        z = (np.sin(y + a) + b * np.sin(x) + c * np.cos(d * (y ** 2 + x ** 2)) +
             e * np.cos(y) + f_const * np.sin(f_const * (y ** 2 + x ** 2)) + g * np.cos(y))
        A = np.array(
            [[55, 45], [70, 100], [160, 45], [160, 150], [90, 160], [45, 135], [124, 73], [108, 117], [160, 110],
             [30, 180], [25, 75], [115, 28], [23, 49], [27, 103], [182, 86], [186, 129], [120, 181], [23, 148]])
        hi = np.array([50, 75, 95, 80, 85, 65, 75, 70, 65, 55, 50, 55, 40, 45, 75, 60, 65, 72.5])
        ai = np.array([20, 25, 18, 23, 17, 13, 12, 13, 15, 18, 10, 17, 13, 12, 13, 13.6, 13.7, 12.3])
        bi = np.array([24, 22, 32, 24, 28, 13, 12, 14, 13, 18, 10, 18, 13, 14, 15.6, 17, 14.1, 14])
        h = np.sum(hi * np.exp(-((x - A[:, 0]) ** 2 / ai ** 2) - ((y - A[:, 1]) ** 2 / bi ** 2)))
        return max(z, h)

    def _generate_mountain_map(self):
        map_data = np.zeros((config.MAP_SIZE_X, config.MAP_SIZE_Y))
        for i in range(config.MAP_SIZE_X):
            for j in range(config.MAP_SIZE_Y):
                # 修复：移除偏移，严格对应 X=i, Y=j
                map_data[i, j] = self.map_value_function(i, j)
        return map_data

    # ==================== 统一的路径插值 ====================
    def get_path_line(self, position):
        import config  # 确保导入了 config
        # ====== 这里需要导入 B样条工具 ======
        from scipy.interpolate import make_interp_spline
        import numpy as np

        postionP = np.zeros((config.NODES_NUMBER, 3))
        for i in range(config.NODES_NUMBER):
            postionP[i, 0] = config.START_POINT[0] + (config.END_POINT[0] - config.START_POINT[0]) * (i + 1) / (
                        config.NODES_NUMBER + 1)
        postionP[:, 1] = position[0:config.NODES_NUMBER]
        postionP[:, 2] = position[config.NODES_NUMBER:2 * config.NODES_NUMBER]
        sort_index = np.argsort(postionP[:, 0])
        postionP = postionP[sort_index]

        # 针对不同地形修正 Z 轴高度基线
        if config.ENV_TYPE == 'mountain':
            for i in range(config.NODES_NUMBER):
                # 修复：严格传入 (X坐标, Y坐标)
                postionP[i, 2] += self.map_value_function(postionP[i, 0], postionP[i, 1])

        PALL = np.vstack([config.START_POINT, postionP, config.END_POINT])
        x_seq, y_seq, z_seq = PALL[:, 0], PALL[:, 1], PALL[:, 2]

        i_seq = np.linspace(0, 1, len(PALL))
        I_seq = np.linspace(0, 1, 200)

        # ========================================================
        # 🔥 核心修正：摒弃 CubicSpline，换装 B-Spline (B样条) 🔥
        # k=3 表示三次 B 样条，它能在保持平滑的同时，彻底消除 U型过冲
        # ========================================================
        try:
            bs_x = make_interp_spline(i_seq, x_seq, k=3)
            bs_y = make_interp_spline(i_seq, y_seq, k=3)
            bs_z = make_interp_spline(i_seq, z_seq, k=3)

            return bs_x(I_seq), bs_y(I_seq), bs_z(I_seq), x_seq, y_seq, z_seq
        except ValueError:
            # 极小概率下的降级保护
            from scipy.interpolate import interp1d
            f_x = interp1d(i_seq, x_seq, kind='linear')
            f_y = interp1d(i_seq, y_seq, kind='linear')
            f_z = interp1d(i_seq, z_seq, kind='linear')
            return f_x(I_seq), f_y(I_seq), f_z(I_seq), x_seq, y_seq, z_seq

    # ==================== 智能分发的代价函数 ====================
    def cost_function(self, position):
        if config.ENV_TYPE == 'mountain':
            return self._cost_mountain(position)
        else:
            return self._cost_cylinder(position)

    def _cost_mountain(self, position):
        X_seq, Y_seq, Z_seq, x_seq, y_seq, z_seq = self.get_path_line(position)
        dx, dy, dz = np.diff(X_seq), np.diff(Y_seq), np.diff(Z_seq)
        PathLength = np.sum(np.sqrt(dx ** 2 + dy ** 2 + dz ** 2))

        # 修复：严格使用 (X, Y) 顺序评估地形
        z_interp_seq = np.array([self.map_value_function(p[0], p[1]) for p in np.column_stack((X_seq, Y_seq))])
        Height = np.sum(np.abs(Z_seq - z_interp_seq))

        Dx, Dy, Dz = np.diff(x_seq), np.diff(y_seq), np.diff(z_seq)
        Curve = 0
        for i in range(len(Dx) - 1):
            num = Dx[i] * Dx[i + 1] + Dy[i] * Dy[i + 1] + Dz[i] * Dz[i + 1]
            den = np.sqrt(Dx[i] ** 2 + Dy[i] ** 2 + Dz[i] ** 2) * np.sqrt(
                Dx[i + 1] ** 2 + Dy[i + 1] ** 2 + Dz[i + 1] ** 2)
            if den != 0: Curve += (np.cos(np.pi / 2) - num / den)

        penalty = 0
        # 🌟 引入物理安全余量：要求无人机必须离山体表面至少 3.0 米
        SAFE_MARGIN = 3.0

        for i in range(len(Z_seq)):
            x, y, z = X_seq[i], Y_seq[i], Z_seq[i]
            # 修复：严格使用 (X, Y) 获取该点真正的地形高度
            z_interp = self.map_value_function(x, y)

            # 撞山判定：高度一旦低于 (山体真实高度 + 安全隔离带)，立刻遭受惩罚
            if z <= z_interp + SAFE_MARGIN:
                penalty += 10000 + (z_interp + SAFE_MARGIN - z) * 10000

        return config.W1 * PathLength + config.W2 * Height + config.W3 * Curve + penalty

    def _cost_cylinder(self, position):
        X_seq, Y_seq, Z_seq, x_seq, y_seq, z_seq = self.get_path_line(position)
        dx, dy, dz = np.diff(X_seq), np.diff(Y_seq), np.diff(Z_seq)
        PathLength = np.sum(np.sqrt(dx ** 2 + dy ** 2 + dz ** 2))

        Height = np.sum(np.abs(Z_seq - config.START_POINT[2]))

        Dx, Dy, Dz = np.diff(x_seq), np.diff(y_seq), np.diff(z_seq)
        Curve = 0
        for i in range(len(Dx) - 1):
            num = Dx[i] * Dx[i + 1] + Dy[i] * Dy[i + 1] + Dz[i] * Dz[i + 1]
            den = np.sqrt(Dx[i] ** 2 + Dy[i] ** 2 + Dz[i] ** 2) * np.sqrt(
                Dx[i + 1] ** 2 + Dy[i + 1] ** 2 + Dz[i + 1] ** 2)
            if den != 0: Curve += (np.cos(np.pi / 2) - num / den)

        penalty = 0
        for i in range(len(X_seq)):
            x, y, z = X_seq[i], Y_seq[i], Z_seq[i]

            # 越界绝对硬惩罚
            if x < 0 or x > config.MAP_SIZE_X or y < 0 or y > config.MAP_SIZE_Y or z < 0 or z > config.MAP_SIZE_Z:
                penalty += 1e8

            for obs in config.OBSTACLES:
                obs_x, obs_y, obs_r, obs_h = obs
                dist_2d = np.sqrt((x - obs_x) ** 2 + (y - obs_y) ** 2)

                # 🔥 阶跃式硬惩罚：绝不容忍切入柱体
                if dist_2d <= obs_r and z <= obs_h:
                    penalty += 10000 + ((obs_r - dist_2d) + (obs_h - z)) * 5000

        return config.W1 * PathLength + config.W2 * Height + config.W3 * Curve + penalty