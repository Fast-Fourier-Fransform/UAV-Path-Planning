import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


def draw_fig_1_problem_definition():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1. 绘制示意性的复杂地形
    x = np.linspace(0, 500, 50)
    y = np.linspace(0, 500, 50)
    X, Y = np.meshgrid(x, y)
    Z = 100 * np.sin(X / 100) * np.cos(Y / 100) + 150  # 示意地形
    ax.plot_surface(X, Y, Z, cmap='terrain', alpha=0.6, edgecolor='none')

    # 2. 绘制半透明威胁圆柱体
    def draw_cylinder(center_x, center_y, radius, height, ax):
        z_cyl = np.linspace(0, height, 10)
        theta_cyl = np.linspace(0, 2 * np.pi, 20)
        theta_grid, z_grid = np.meshgrid(theta_cyl, z_cyl)
        x_grid = radius * np.cos(theta_grid) + center_x
        y_grid = radius * np.sin(theta_grid) + center_y
        ax.plot_surface(x_grid, y_grid, z_grid, color='red', alpha=0.3, edgecolor='none')

    threats = [(200, 200, 60, 400), (350, 100, 50, 400), (100, 400, 70, 400)]
    for t in threats:
        draw_cylinder(t[0], t[1], t[2], t[3], ax)

    # 3. 绘制平滑航迹
    path_points = np.array([[50, 50, 200], [250, 150, 300], [300, 350, 250], [450, 450, 350]])
    t = np.linspace(0, 1, len(path_points))
    t_smooth = np.linspace(0, 1, 100)
    spline_x = make_interp_spline(t, path_points[:, 0], k=3)
    spline_y = make_interp_spline(t, path_points[:, 1], k=3)
    spline_z = make_interp_spline(t, path_points[:, 2], k=3)

    ax.plot(spline_x(t_smooth), spline_y(t_smooth), spline_z(t_smooth), color='black', linewidth=3,
            label='Generated Path')

    # 4. 关键标注
    ax.scatter(50, 50, 200, c='black', s=100, marker='s', label='Start P_start')
    ax.scatter(450, 450, 350, c='black', s=100, marker='*', label='Goal P_goal')

    ax.set_title("Fig 1. Theoretical modeling of UAV path planning environment with terrain and cylindrical threats",
                 fontsize=14)
    ax.legend(loc='upper right')
    ax.set_xlabel("X (m)", fontsize=10)
    ax.set_ylabel("Y (m)", fontsize=10)
    ax.set_zlabel("Z (Altitude m)", fontsize=10)
    ax.view_init(elev=30, azim=40)
    plt.tight_layout()
    plt.show()

# 取消下方注释运行代码
    draw_fig_1_problem_definition()