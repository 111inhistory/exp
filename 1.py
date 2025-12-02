# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "matplotlib",
#     "numpy",
# ]
#
# [[tool.uv.index]]
# url = "https://pypi.tuna.tsinghua.edu.cn/simple"
# default = true
# ///


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

input()

# --- 1. 物理参数设置 (参考题目数据) ---
f = 0.5            # 频率 (Hz)
v = 2.0            # 波速 (m/s)
A = 0.2            # 振幅 (m)
lam = v / f        # 波长 (4m)
k = 2 * np.pi / lam  # 波数
omega = 2 * np.pi * f # 角频率

# 波的传播方向设置
# 题目中波面夹角120度，假设两列波关于Y轴对称
# 为了让波看起来像图2那样向上方交汇，我们设置波矢量方向
theta = np.deg2rad(60) # 60度 (120度的一半)

# 波矢量 k1 和 k2
# k1 向右上方传播, k2 向左上方传播
k1_x, k1_y = k * np.sin(theta), k * np.cos(theta)
k2_x, k2_y = -k * np.sin(theta), k * np.cos(theta)

input()

# --- 2. 创建网格数据 ---
range_val = 10  # 模拟范围 (米)
resolution = 100
x = np.linspace(-range_val, range_val, resolution)
y = np.linspace(0, range_val*2, resolution)
X, Y = np.meshgrid(x, y)

# 寻找理论上的减弱点位置 (用于右下角的图)
# 当路径差 = (n + 0.5) * lambda 时为减弱点
# 简单计算：X轴方向的驻波部分 k_x * x = pi/2 -> x = pi / (2 * k_x)
x_node = np.pi / (2 * k * np.sin(theta)) 

# --- 3. 绘图初始化 ---
fig = plt.figure(figsize=(12, 7))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

# 子图1: 2D 平面俯视 (热力图)
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_title("两列波干涉俯视图 (类图1效果)")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")
ax1.set_aspect('equal')

input()

# 子图2: 振动加强线 (中间 O-C 连线)
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_title("振动加强线上的波形 (C点所在线)")
ax2.set_ylim(-2.5*A, 2.5*A) # 预留叠加后的振幅空间
ax2.set_xlim(0, range_val*2)
ax2.set_ylabel("振幅 Z (m)")
line_constructive, = ax2.plot([], [], 'r-', lw=2, label='加强 (2A)')
ax2.legend(loc='upper right')
ax2.grid(True, linestyle='--', alpha=0.6)

# 子图3: 振动减弱线 (旁边的线)
ax3 = fig.add_subplot(2, 2, 4)
ax3.set_title("振动减弱线上的波形")
ax3.set_ylim(-2.5*A, 2.5*A)
ax3.set_xlim(0, range_val*2)
ax3.set_ylabel("振幅 Z (m)")
ax3.set_xlabel("沿传播方向距离 y (m)")
line_destructive, = ax3.plot([], [], 'b--', lw=2, label='减弱 (~0)')
ax3.legend(loc='upper right')
ax3.grid(True, linestyle='--', alpha=0.6)

# 初始化图像对象
# 使用 pcolormesh 绘制平面波，vmin/vmax 设置为 2A 以便观察叠加
c_mesh = ax1.pcolormesh(X, Y, np.zeros_like(X), cmap='RdBu_r', vmin=-2*A, vmax=2*A, shading='auto')
fig.colorbar(c_mesh, ax=ax1, label='水位高度 (m)', orientation='horizontal', pad=0.1)

# 画出加强线和减弱线的辅助线
ax1.axvline(x=0, color='r', linestyle='-', alpha=0.5, label='加强线')
ax1.axvline(x=x_node, color='b', linestyle='--', alpha=0.5, label='减弱线')
ax1.legend(loc='upper right')

# --- 4. 动画更新函数 ---
def update(frame):
    t = frame * 0.05 # 时间步长
    
    # 计算两列波的波函数
    # Z = A * cos(k*r - w*t)
    wave1 = A * np.cos(k1_x * X + k1_y * Y - omega * t)
    wave2 = A * np.cos(k2_x * X + k2_y * Y - omega * t)
    
    # 叠加
    Z_total = wave1 + wave2
    
    # 更新平面图
    c_mesh.set_array(Z_total.ravel())
    
    # 更新加强线 (x=0 处) 的波形
    # 在 x=0 处，两列波相位相同，直接相加
    Z_strong = A * np.cos(k1_y * y - omega * t) + A * np.cos(k2_y * y - omega * t)
    line_constructive.set_data(y, Z_strong)
    
    # 更新减弱线 (x=x_node 处) 的波形
    # 在 x=x_node 处，两列波相位相差 pi，相互抵消
    Z_weak = A * np.cos(k1_x * x_node + k1_y * y - omega * t) + \
             A * np.cos(k2_x * x_node + k2_y * y - omega * t)
    line_destructive.set_data(y, Z_weak)
    
    return c_mesh, line_constructive, line_destructive

# --- 5. 运行动画 ---
ani = FuncAnimation(fig, update, frames=200, interval=50, blit=False)


plt.show()