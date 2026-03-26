import matplotlib.pyplot as plt
import numpy as np
import platform
import config


def set_academic_font():
    """配置学术论文标准英文排版字体"""
    plt.rcParams['font.family'] = 'serif'
    # 优先使用 Times New Roman，如果没有则回退到系统默认的 serif 字体
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['axes.unicode_minus'] = False


def plot_convergence_curves(all_curves_data, title="Convergence Curves"):
    """
    绘制收敛曲线图（所有文本均为英文，用于学术论文）
    """
    set_academic_font()

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    for curve, algo_name in all_curves_data:
        iterations = range(len(curve))

        # 根据算法名称设定线型、颜色和粗细
        if 'HDE-PSO' in algo_name and 'Proposed' in algo_name:
            ax.plot(iterations, curve, label=f"★ {algo_name}", color='#FF4500', linewidth=3.5, linestyle='-', zorder=10)
        elif 'Fixed' in algo_name:
            ax.plot(iterations, curve, label=algo_name, color='#9370DB', linewidth=2.0, linestyle='--')
        elif 'PSO-DE' in algo_name:
            ax.plot(iterations, curve, label=algo_name, color='#1E90FF', linewidth=2.0, linestyle='-.')
        else:
            ax.plot(iterations, curve, label=algo_name, color='#3CB371', linewidth=2.0, linestyle=':')

    ax.set_yscale('log')

    # 🌟 纯英文坐标轴标签
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Iterations', fontsize=14)
    ax.set_ylabel('Fitness Cost [Log Scale]', fontsize=14)

    ax.grid(True, which="both", ls="-", alpha=0.2, color='gray')
    ax.grid(True, which="major", ls="-", alpha=0.5, color='gray')

    legend = ax.legend(loc='upper right', fontsize=12, framealpha=0.9, edgecolor='black')
    legend.get_frame().set_linewidth(1.0)

    plt.tight_layout()
    file_name = f'convergence_curves_{config.ENV_TYPE}.png'
    plt.savefig(file_name, bbox_inches='tight')
    print(f"\n📊 收敛曲线已保存至: {file_name}")


def plot_multiple_3d_paths(map_data, all_paths_data, title="3D Path Comparison"):
    """
    使用 Plotly OpenGL 引擎渲染 MATLAB 级别的全英文 3D 图像
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    fig = go.Figure()

    # 根据环境类型绘制地形或圆柱体障碍物
    if config.ENV_TYPE == 'mountain':
        x_grid = np.arange(map_data.shape[0])
        y_grid = np.arange(map_data.shape[1])

        fig.add_trace(go.Surface(
            z=map_data.T,
            x=x_grid,
            y=y_grid,
            colorscale='Greys',
            opacity=1.0,
            showscale=False,
            contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
        ))
    else:
        # 城市环境：绘制地面
        fig.add_trace(go.Surface(
            z=np.zeros((config.MAP_SIZE_X, config.MAP_SIZE_Y)),
            x=np.arange(config.MAP_SIZE_X), y=np.arange(config.MAP_SIZE_Y),
            colorscale=[[0, 'lightgray'], [1, 'lightgray']],
            opacity=0.3, showscale=False
        ))

        # 绘制圆柱体障碍物
        colors = ['#FF6347', '#4682B4', '#32CD32', '#DAA520', '#9370DB']
        for idx, obs in enumerate(config.OBSTACLES):
            xc, yc, r, h = obs
            theta = np.linspace(0, 2 * np.pi, 30)
            z_cyl = np.linspace(0, h, 10)
            theta_grid, z_grid = np.meshgrid(theta, z_cyl)
            x_cyl = xc + r * np.cos(theta_grid)
            y_cyl = yc + r * np.sin(theta_grid)
            c = colors[idx % len(colors)]

            fig.add_trace(go.Surface(
                x=x_cyl, y=y_cyl, z=z_grid,
                colorscale=[[0, c], [1, c]], opacity=0.85, showscale=False
            ))

            # 绘制圆柱体顶面
            fig.add_trace(go.Mesh3d(
                x=xc + r * np.cos(theta), y=yc + r * np.sin(theta), z=np.full_like(theta, h),
                color=c, opacity=0.85,
                i=[0] * (len(theta) - 2), j=list(range(1, len(theta) - 1)), k=list(range(2, len(theta)))
            ))

    # 绘制算法路径
    for path_info in all_paths_data:
        X_seq, Y_seq, Z_seq, algo_name = path_info
        if 'HDE-PSO' in algo_name and 'Proposed' in algo_name:
            color, lw, label_name = '#FFD700', 8, f"★ {algo_name}"
        elif 'Fixed' in algo_name:
            color, lw, label_name = '#BA55D3', 5, algo_name
        elif 'PSO-DE' in algo_name:
            color, lw, label_name = '#FF4500', 5, algo_name
        else:
            color, lw, label_name = '#00FFFF', 4, algo_name

        fig.add_trace(go.Scatter3d(
            x=X_seq, y=Y_seq, z=Z_seq,
            mode='lines',
            name=label_name,
            line=dict(color=color, width=lw)
        ))

    # 🌟 纯英文起点/终点标记
    start_x, start_y, start_z = config.START_POINT
    end_x, end_y, end_z = config.END_POINT
    fig.add_trace(go.Scatter3d(
        x=[start_x], y=[start_y], z=[start_z], mode='markers', name='Start Point',
        marker=dict(symbol='circle', size=8, color='lime', line=dict(color='black', width=1))
    ))
    fig.add_trace(go.Scatter3d(
        x=[end_x], y=[end_y], z=[end_z], mode='markers', name='Target Point',
        marker=dict(symbol='diamond', size=8, color='fuchsia', line=dict(color='black', width=1))
    ))

    # 🌟 纯英文 3D 坐标轴和全局样式（使用 Serif 衬线体）
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20, family="Times New Roman, serif")),
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z / Height (m)',
            xaxis=dict(gridcolor='lightgray', backgroundcolor='white'),
            yaxis=dict(gridcolor='lightgray', backgroundcolor='white'),
            zaxis=dict(gridcolor='lightgray', backgroundcolor='white'),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.4),
            camera=dict(eye=dict(x=-1.5, y=-1.5, z=0.8))
        ),
        legend=dict(x=0.8, y=0.9, bgcolor='rgba(255, 255, 255, 0.8)'),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    fig.show()