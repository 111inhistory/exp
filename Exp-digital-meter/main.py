import os
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
import matplotlib as mpl
import warnings
import re
from cycler import cycler

mpl.use("pgf")

# 配置 PGF + XeLaTeX 环境
pgf_config = {
    "pgf.texsystem": "xelatex",  # 指定使用 xelatex 编译
    "text.usetex": True,  # 启用 LaTeX 渲染
    "font.family": "serif",  # 基本字体族
    "font.size": 8,  # 字号
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


fig, axs = plt.subplots(2, 2, figsize=(8, 6))
data: dict[str, pd.DataFrame] = {}

# level = [0.1, 0.2, 0.5, 1.0, 1.5]

file_list = sorted([f for f in os.listdir("./") if f.endswith(".csv")])
file_list[1], file_list[2] = file_list[2], file_list[1]
for i in file_list:
    data[i] = pd.read_csv(i)

regex = re.compile(r"data_(\d+)([a-zA-Z]+)\.csv")

for i, (filename, df) in enumerate(data.items()):
    df: pd.DataFrame
    ax: plt.Axes = axs.flatten()[i]
    match = regex.match(filename)
    if not match:
        continue

    range_str, unit_str = match.groups()

    (meter_mod, meter_std) = df.iloc[:, 0], df.iloc[:, 1]

    delta = meter_mod - meter_std
    data_name: str = df.columns[0][0]
    delta_name = f"$\\Delta {df.columns[0][0]}$ / {unit_str}"

    # 绘制 x-y 散点图
    ax.plot(meter_mod, delta, "x", linestyle="-", markersize=5, color="green")
    x_lim = ax.get_xlim()
    ax.hlines(
        0, xmin=x_lim[0], colors="gray", xmax=x_lim[1], linestyles="-."
    )
    ax.set_xlim(x_lim)

    ax.set_xlabel(f"${data_name}_改$ / {unit_str}")
    ax.set_ylabel(delta_name)
    ax.set_title(f"{range_str}{unit_str}量程数字电表校正曲线")
    # ax.legend()

plt.tight_layout()
plt.savefig("output.png", dpi=300)
plt.show()
