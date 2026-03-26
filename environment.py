import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.ndimage import zoom  # 用于生成分形噪声
import config


class UAVEnvironment:
    def __init__(self):
        # 初始化时生成并缓存真实地形，后续碰撞检测全部查表，速度极快！
        if config.ENV_TYPE == 'mountain':
            self.map_data = self._generate_realistic_mountain()
        else:
            self.map_data = np.zeros((config.MAP_SIZE_X, config.MAP_SIZE_Y))

    # ==================== 1. 地形生成引擎 ====================
    def _base_gaussian_map(self, x, y):
        """原来的平滑高斯山峰骨架"""
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

    def _generate_realistic_mountain(self):
        """生成具备真实岩石质感的褶皱地形 (Fractal Noise)"""
        print("🌍 正在生成具备真实岩石纹理的分形山体...")
        shape = (config.MAP_SIZE_X, config.MAP_SIZE_Y)
        base_map = np.zeros(shape)

        # 1. 生成平滑骨架
        for i in range(shape[0]):
            for j in range(shape[1]):
                base_map[i, j] = self._base_gaussian_map(i, j)

        # 2. 生成分形噪声 (5 个八度叠加，模拟真实自然风化)
        np.random.seed(42)  # 固定种子，保证每次运行地形一致
        noise = np.zeros(shape)
        octaves = 5
        roughness = 0.55  # 岩石粗糙度

        for i in range(octaves):
            grid_size = (max(shape[0] // (2 ** (i + 1)), 1), max(shape[1] // (2 ** (i + 1)), 1))
            rand_grid = np.random.rand(*grid_size)
            zoom_x = shape[0] / grid_size[0]
            zoom_y = shape[1] / grid_size[1]
            octave_noise = zoom(rand_grid, (zoom_x, zoom_y), order=3)
            noise += octave_noise * (roughness ** i)

        # 将噪声居中并控制幅度 (约正负 15 米的褶皱起伏)
        noise = (noise - np.mean(noise)) * 30.0

        # 3. 骨架与岩石细节融合
        realistic_map = base_map + noise
        realistic_map = np.maximum(realistic_map, 0)  # 防止凹陷到地下
        return realistic_map

    def get_real_z(self, x, y):
        """
        高级查表法：使用双线性插值 (Bilinear Interpolation)。
        让数学底层的碰撞箱，与 Plotly 3D 渲染的表面达到 100% 完美的亚像素级贴合！
        """
        # 限制在安全索引范围内
        x = np.clip(x, 0, config.MAP_SIZE_X - 1.001)
        y = np.clip(y, 0, config.MAP_SIZE_Y - 1.001)

        # 找到包围该点的 4 个整数网格坐标
        x0, y0 = int(x), int(y)
        x1, y1 = x0 + 1, y0 + 1

        # 计算小数部分的偏移量
        dx, dy = x - x0, y - y0

        # 获取这 4 个角的真实岩石高度
        z00 = self.map_data[x0, y0]
        z10 = self.map_data[x1, y0]
        z01 = self.map_data[x0, y1]
        z11 = self.map_data[x1, y1]

        # 按照距离比例，精确混合出这个亚像素点的绝对高度
        z_interp = (1 - dx) * (1 - dy) * z00 + dx * (1 - dy) * z10 + \
                   (1 - dx) * dy * z01 + dx * dy * z11

        return z_interp

    # ==================== 2. 航迹插值 ====================
    def get_path_line(self, position):
        postionP = np.zeros((config.NODES_NUMBER, 3))
        for i in range(config.NODES_NUMBER):
            postionP[i, 0] = config.START_POINT[0] + (config.END_POINT[0] - config.START_POINT[0]) * (i + 1) / (
                        config.NODES_NUMBER + 1)
        postionP[:, 1] = position[0:config.NODES_NUMBER]
        postionP[:, 2] = position[config.NODES_NUMBER:2 * config.NODES_NUMBER]
        sort_index = np.argsort(postionP[:, 0])
        postionP = postionP[sort_index]

        if config.ENV_TYPE == 'mountain':
            for i in range(config.NODES_NUMBER):
                # 依据真实的岩石地形抬高基础控制点
                postionP[i, 2] += self.get_real_z(postionP[i, 0], postionP[i, 1])
        else:
            postionP[:, 2] = np.maximum(postionP[:, 2], 0)

        PALL = np.vstack([config.START_POINT, postionP, config.END_POINT])
        x_seq, y_seq, z_seq = PALL[:, 0], PALL[:, 1], PALL[:, 2]

        i_seq = np.linspace(0, 1, len(PALL))
        I_seq = np.linspace(0, 1, 200)

        try:
            bs_x = make_interp_spline(i_seq, x_seq, k=3)
            bs_y = make_interp_spline(i_seq, y_seq, k=3)
            bs_z = make_interp_spline(i_seq, z_seq, k=3)
            return bs_x(I_seq), bs_y(I_seq), bs_z(I_seq), x_seq, y_seq, z_seq
        except ValueError:
            from scipy.interpolate import interp1d
            f_x = interp1d(i_seq, x_seq, kind='linear')
            f_y = interp1d(i_seq, y_seq, kind='linear')
            f_z = interp1d(i_seq, z_seq, kind='linear')
            return f_x(I_seq), f_y(I_seq), f_z(I_seq), x_seq, y_seq, z_seq

    # ==================== 3. 适应度代阶 ====================
    def cost_function(self, position):
        if config.ENV_TYPE == 'mountain':
            return self._cost_mountain(position)
        else:
            return self._cost_cylinder(position)

    def _cost_mountain(self, position):
        X_seq, Y_seq, Z_seq, x_seq, y_seq, z_seq = self.get_path_line(position)
        dx, dy, dz = np.diff(X_seq), np.diff(Y_seq), np.diff(Z_seq)
        PathLength = np.sum(np.sqrt(dx ** 2 + dy ** 2 + dz ** 2))

        # 严密贴合岩石表面的高度代价
        z_interp_seq = np.array([self.get_real_z(p[0], p[1]) for p in np.column_stack((X_seq, Y_seq))])
        Height = np.sum(np.abs(Z_seq - z_interp_seq))

        Dx, Dy, Dz = np.diff(x_seq), np.diff(y_seq), np.diff(z_seq)
        Curve = 0
        for i in range(len(Dx) - 1):
            num = Dx[i] * Dx[i + 1] + Dy[i] * Dy[i + 1] + Dz[i] * Dz[i + 1]
            den = np.sqrt(Dx[i] ** 2 + Dy[i] ** 2 + Dz[i] ** 2) * np.sqrt(
                Dx[i + 1] ** 2 + Dy[i + 1] ** 2 + Dz[i + 1] ** 2)
            if den != 0: Curve += (np.cos(np.pi / 2) - num / den)

        penalty = 0
        SAFE_MARGIN = 3.0
        for i in range(len(Z_seq)):
            x, y, z = X_seq[i], Y_seq[i], Z_seq[i]
            z_interp = self.get_real_z(x, y)
            # 撞击真实岩石表面的严厉惩罚
            if z <= z_interp + SAFE_MARGIN:
                penalty += 1000000 + (z_interp + SAFE_MARGIN - z) * 10000

        return config.W1 * PathLength + config.W2 * Height + config.W3 * Curve + penalty

    def _cost_cylinder(self, position):
        # 城市环境代价计算保持不变
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
            if x < 0 or x > config.MAP_SIZE_X or y < 0 or y > config.MAP_SIZE_Y or z < 0 or z > config.MAP_SIZE_Z:
                penalty += 1e8
            for obs in config.OBSTACLES:
                obs_x, obs_y, obs_r, obs_h = obs
                dist_2d = np.sqrt((x - obs_x) ** 2 + (y - obs_y) ** 2)
                if dist_2d <= obs_r and z <= obs_h:
                    penalty += 1e8 + ((obs_r - dist_2d) + (obs_h - z)) * 5000
        return config.W1 * PathLength + config.W2 * Height + config.W3 * Curve + penalty