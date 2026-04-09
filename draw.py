import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 🔥 HDE-PSO 炼丹机理分析：参数动态演化绘图脚本
# ==========================================

# 1. 设定科研标准字体与全局样式
plt.rcParams['font.family'] = 'sans-serif' # 默认 sans-serif，Visio 和 academic plot 常用
plt.rcParams['font.sans-serif'] = ['SimHei'] # 如果 Linux/Mac 系统无该字体，会导致中文字符变方块。可注释
plt.rcParams['axes.unicode_minus'] = False # 正确显示负号
# plt.style.use('seaborn-v0_8-ticks') # 可选：启用更符合科研风格的 ticks 样式

# 2. 核心数学模型参数设定 (与论文第 4.3 节和图 4 图注严密对齐)
Tmax = 500          # 最大迭代次数
w_max = 0.9         # PSO 初始惯性权重
w_min = 0.4         # PSO 终止惯性权重
F_base = 0.5        # DE 基础缩放因子
alpha = 0.4         # 震荡最大振幅控制系数
k_periods = 4       # 频率系数：ARGUMENT Argument from 0 to 4\pi over t=0 to Tmax (2 periods).

# 3. 生成迭代次数 $t$ 序列
t = np.arange(0, Tmax + 1) # 从 0 到 500，共 501 个点

# 4. 数学公式实例化计算
# 曲线 1：惯性权重 $w(t)$ 的线性衰减
w = w_max - (w_max - w_min) * (t / Tmax)

# 曲线 2：阻尼余弦震荡缩放因子 $F(t)$ 的演化
# ARGUMENT Argument from 0 to 4\pi over t=0 to Tmax (2 periods).
A_t = alpha * (1 - t / Tmax) # 振幅衰减控制项
cos_term = np.cos(k_periods * np.pi * t / Tmax) # ARGUMENT ranges 0 to 4\pi
F = F_base + A_t * cos_term

# 5. ==========================================
# 📈 绘制极具学术风的高清 2D 坐标图
# ==========================================
plt.figure(figsize=(10, 6), dpi=300) # 设定高清图尺寸与分辨率

# 绘制惯性权重 w(t) - 线性递减
plt.plot(t, w, label=r'Inertia Weight: $w(t) = 0.9 - 0.5 \cdot (t / 500)$ (Linear Decay)',
         color='blue', # 指定学术蓝色
         linewidth=2.5,
         linestyle='--') # 虚线区别

# 绘制缩放因子 F(t) - 震荡阻尼
plt.plot(t, F, label=r'Scaling Factor: $F(t) = 0.5 + A(t) \cdot \cos(4\pi \cdot (t/500))$ (Damped Cosine Oscillation)',
         color='orange', # 指定学术橙色
         linewidth=3.0, # 冠军曲线加粗
         linestyle='-') # 实线高亮

# 6. 设置图表格式
# 设置坐标轴范围与标签
plt.ylim(0.0, 1.0) # 统一纵坐标
plt.xticks(np.arange(0, Tmax + 1, 50), fontsize=12) # 横轴每 50 代一个刻度
plt.yticks(np.arange(0.0, 1.1, 0.1), fontsize=12)
plt.xlabel(r'迭代次数 $t$ (Iterations)', fontsize=14, fontweight='bold')
plt.ylabel(r'参数值 (Parameter Value)', fontsize=14, fontweight='bold')

# 添加网格与图例
plt.grid(True, linestyle='--', alpha=0.6) # 启用网格，轻微透明
plt.legend(loc='lower left', fontsize=11, framealpha=0.8, edgecolor='black') # 优化图例，带边框

# 添加标题与图注描述
# 与图注示例严密对齐
plt.title(f'HDE-PSO 核心参数的阻尼与震荡非线性动态演化轨迹 (t 从 0 到 {Tmax})', fontsize=16, fontweight='bold')

# 保存为高清图片
save_name = f'grid_search_param_evolution_T{Tmax}.png' # 或保存为 .pdf 等矢量格式
plt.tight_layout()
plt.savefig(save_name, bbox_inches='tight') #bbox_inches='tight' 自动紧凑排版
print(f"\n🎉 炼丹机理图已高清生成！最佳参数演化图已保存为 '{save_name}'")
plt.show()