import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
import matplotlib as mpl
import warnings
import numpy as np
from cycler import cycler


mpl.use("pgf")

# 配置 PGF + XeLaTeX 环境
pgf_config = {
    "pgf.texsystem": "xelatex",  # 指定使用 xelatex 编译
    "text.usetex": True,  # 启用 LaTeX 渲染
    "font.family": "serif",  # 基本字体族
    "font.size": 7,  # 字号
    "pgf.rcfonts": False,  # 不使用 matplotlib 默认字体设置
    "pgf.preamble": r"""
        \usepackage{xeCJK}                    % 支持中文
        \usepackage{unicode-math}            % 支持 unicode 数学符号，如 −
        \setmainfont{Times New Roman}        % 英文字体
        \setCJKmainfont{SimSun}              % 中文字体
        \xeCJKsetup{CJKmath=true}            % 中文参与数学公式
    """,
}

# 应用配置
mpl.rcParams.update(pgf_config)

# Suppress warnings from polyfit for clarity
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

plt.style.use(["science"])

# 设置全局默认颜色循环 (Nature Publishing Group 风格)
mpl.rcParams["axes.prop_cycle"] = cycler(
    color=[
        "#E64B35",
        "#4DBBD5",
        "#00A087",
        "#3C5488",
        "#F39B7F",
        "#8491B4",
        "#91D1C2",
        "#DC0000",
        "#7E6148",
        "#B09C85",
    ]
)

# don't modify the code above this line

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(4, 9))

ax1: plt.Axes
ax2: plt.Axes
ax3: plt.Axes

data1 = {
    "发射电流(10mA)": [0, 0.50, 0.60, 0.80, 1.00, 1.50, 2.00, 2.50, 3.00, 3.50],
    "正向偏置电压(V)": [0, 0.97, 0.98, 1.00, 1.01, 1.04, 1.07, 1.10, 1.13, 1.16],
    "光功率(mW)": [
        0.001,
        0.001,
        0.001,
        0.014,
        0.049,
        0.142,
        0.235,
        0.329,
        0.426,
        0.522,
    ],
}

ax1.set_xlabel("发射电流(mA)")
ax1_twin = ax1.twinx()
ax1.set_ylabel("正向偏置电压(V)")
ax1_twin.set_ylabel("光功率(mW)")

x = np.array(data1["发射电流(10mA)"]) * 10

ax1.plot(x, data1["正向偏置电压(V)"], marker="o", markersize=3, label="正向偏置电压")
ax1_twin.plot(
    x, data1["光功率(mW)"], marker="o", markersize=3, color="C1", label="光功率"
)

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower right")


data2 = [
    list(range(0, 5)),
    [-3] * 5,
    [105] * 5,
    [216] * 2 + [215] + [216] * 2,
]

tag2 = ["P=0mW", "P=0.1mW", "P=0.2mW"]

label = {"y": "光电流(\u03bcA)", "x": "反向偏置电压(V)"}

ax2.set_xlabel(label["x"])
ax2.set_ylabel(label["y"])
for i in range(len(data2) - 1):
    ax2.plot(data2[0], data2[i + 1], marker="o", markersize=3, label=tag2[i])
ax2.set_ylim((-15, 350))
ax2.legend()

data3 = [
    [0, 0.50, 1.00, 1.50, 2.00, 2.50, 3.00, 3.50, 4.00],
    [85.47, 83.33, 81.20, 78.65, 76.92, 74.63, 72.46, 70.42, 68.49],
]

x = np.array(data3[0])
y = np.array(data3[1])
y1 = y * 2 * np.pi
print(np.vstack([x, y1]))

print(y1)

# 线性拟合
k, b = np.polyfit(x, y1, 1)
print(k, b)
y_fit = k * x + b

# 计算 R^2
ss_res = np.sum((y1 - y_fit) ** 2)
ss_tot = np.sum((y1 - np.mean(y1)) ** 2)
r2 = 1 - (ss_res / ss_tot)

x_tag = "输入电压(V)"
y_tag = "输出角频率(kHz)"

ax3.set_xlabel(x_tag)
ax3.set_ylabel(y_tag)
ax3.plot(
    x,
    y_fit,
    color="C0",
    label=f"拟合: $R^2={r2:.4f}$\n$y={k:.2f}x+{b:.2f}$",
)
ax3.plot(x, y1, "o", markersize=3, label="实验数据")
ax3.legend()

ax1.set_title("(a) 激光二极管输出及伏安特性曲线")
ax2.set_title("(b) 光电二极管光电流与反向偏置电压关系曲线")
ax3.set_title("(c) V-f转换模块输入电压与输出角频率关系曲线")

fig.subplots_adjust(hspace=0.3)

fig.savefig("figure_template.png", dpi=300)
