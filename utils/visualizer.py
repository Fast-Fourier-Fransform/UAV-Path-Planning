import matplotlib.pyplot as plt
import numpy as np
import config


def set_academic_font():
    """配置学术论文标准英文排版字体"""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['axes.unicode_minus'] = False


# 🌟 全局独立配色与线型字典 (去掉了数字，免疫编号变化)
ALGO_STYLES = {
    'Standard PSO': {'color': '#808080', 'ls': '--'},
    'PSO-DE': {'color': '#A0522D', 'ls': '-.'},
    'HDE-PSO-Fixed': {'color': '#20B2AA', 'ls': '--'},
    'HDE-PSO (Proposed)': {'color': '#FF0000', 'ls': '-'},
    'DCWPSO': {'color': '#2E8B57', 'ls': '-.'},
    'HSPSO': {'color': '#1E90FF', 'ls': ':'},
    'MPSOGOA': {'color': '#8A2BE2', 'ls': '--'},
    'VN-IPSO': {'color': '#FF1493', 'ls': '-.'},
    'SPSO': {'color': '#FF8C00', 'ls': '-'}
}


def plot_convergence_curves(all_curves_data, title="Convergence Curves"):
    """绘制高标准学术收敛曲线图"""
    set_academic_font()
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    for curve, algo_name in all_curves_data:
        iterations = range(len(curve))

        # 1. 优先捕获所提算法：绝对的主角光环（加粗红色实线）
        if 'Proposed' in algo_name:
            ax.plot(iterations, curve, label=f"★ {algo_name}", color='#FF0000', linewidth=3.5, linestyle='-', zorder=10)
            continue

        # 2. 遍历字典，只要算法名字里包含字典的键名，就采用对应颜色
        matched_style = None
        for key, style in ALGO_STYLES.items():
            if key in algo_name:
                matched_style = style
                break

        if matched_style:
            ax.plot(iterations, curve, label=algo_name, color=matched_style['color'], linewidth=2.0,
                    linestyle=matched_style['ls'])
        else:
            ax.plot(iterations, curve, label=algo_name, color='black', linewidth=2.0, linestyle=':')

    ax.set_yscale('log')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Iterations', fontsize=14)
    ax.set_ylabel('Fitness Cost [Log Scale]', fontsize=14)

    ax.grid(True, which="both", ls="-", alpha=0.2, color='gray')
    ax.grid(True, which="major", ls="-", alpha=0.5, color='gray')

    legend = ax.legend(loc='upper right', fontsize=11, framealpha=0.9, edgecolor='black', ncol=2)
    legend.get_frame().set_linewidth(1.0)
    plt.tight_layout()

    file_name = f'Fig_Convergence_Curves_{config.ENV_TYPE}.png'
    plt.savefig(file_name, bbox_inches='tight')
    print(f"\n📊 收敛曲线已生成并保存至: {file_name}")
    plt.show()


def plot_multiple_3d_paths(map_data, all_paths_data, title="3D Path Comparison"):
    """使用 Plotly 渲染 3D 轨迹图"""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    fig = go.Figure()

    # 1. 绘制绝对不透明地形
    if config.ENV_TYPE in ['mountain', 'mountain_radar']:
        x_grid = np.arange(map_data.shape[0])
        y_grid = np.arange(map_data.shape[1])
        fig.add_trace(go.Surface(
            z=map_data.T, x=x_grid, y=y_grid,
            colorscale='Earth', opacity=1.0, showscale=False,
            contours_z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True)
        ))
    else:
        fig.add_trace(go.Surface(
            z=np.zeros((config.MAP_SIZE_X, config.MAP_SIZE_Y)),
            x=np.arange(config.MAP_SIZE_X), y=np.arange(config.MAP_SIZE_Y),
            colorscale=[[0, 'lightgray'], [1, 'lightgray']],
            opacity=1.0, showscale=False
        ))

    # 2. 绘制不透明的警戒雷达禁飞区
    if len(config.OBSTACLES) > 0:
        for obs in config.OBSTACLES:
            xc, yc, r, h = obs
            theta = np.linspace(0, 2 * np.pi, 30)
            z_cyl = np.linspace(0, h, 10)
            theta_grid, z_grid = np.meshgrid(theta, z_cyl)
            x_cyl = xc + r * np.cos(theta_grid)
            y_cyl = yc + r * np.sin(theta_grid)

            c = 'firebrick'
            fig.add_trace(go.Surface(
                x=x_cyl, y=y_cyl, z=z_grid,
                colorscale=[[0, c], [1, c]], opacity=1.0, showscale=False
            ))
            fig.add_trace(go.Mesh3d(
                x=xc + r * np.cos(theta), y=yc + r * np.sin(theta), z=np.full_like(theta, h),
                color=c, opacity=1.0,
                i=[0] * (len(theta) - 2), j=list(range(1, len(theta) - 1)), k=list(range(2, len(theta)))
            ))

    # 3. 绘制算法 3D 路径
    for path_info in all_paths_data:
        X_seq, Y_seq, Z_seq, algo_name = path_info

        # 所提算法：粗红线
        if 'Proposed' in algo_name:
            color, lw, label_name = '#FF0000', 8, f"★ {algo_name}"
        else:
            # 智能模糊匹配颜色
            matched_color = '#000000'  # 兜底黑色
            for key, style in ALGO_STYLES.items():
                if key in algo_name:
                    matched_color = style['color']
                    break
            color, lw, label_name = matched_color, 4, algo_name

        fig.add_trace(go.Scatter3d(
            x=X_seq, y=Y_seq, z=Z_seq,
            mode='lines', name=label_name, line=dict(color=color, width=lw)
        ))

    # 4. 起止点标记
    fig.add_trace(go.Scatter3d(
        x=[config.START_POINT[0]], y=[config.START_POINT[1]], z=[config.START_POINT[2]],
        mode='markers', name='START', marker=dict(symbol='circle', size=8, color='lime')
    ))
    fig.add_trace(go.Scatter3d(
        x=[config.END_POINT[0]], y=[config.END_POINT[1]], z=[config.END_POINT[2]],
        mode='markers', name='TARGET', marker=dict(symbol='diamond', size=8, color='fuchsia')
    ))

    # 5. 全局样式设置
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20, family="Times New Roman, serif")),
        scene=dict(
            xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Altitude (m)',
            aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.5),
            camera=dict(eye=dict(x=-1.5, y=-1.5, z=0.8))
        ),
        legend=dict(x=0.85, y=0.9, bgcolor='rgba(255, 255, 255, 1.0)'),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    fig.show()