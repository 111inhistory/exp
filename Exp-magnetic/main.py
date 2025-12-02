import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
import matplotlib as mpl
import pandas as pd
import numpy as np
import scipy.interpolate as interp
import warnings

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

# don't modify the content above except for new imports

# y unit is mV, and x unit is "格"
# x: 5格=1V

# const definitions
N1 = N2 = 100  # 匝
S = 1.24e-4  # m^2
L = 0.130  # m
R1 = 11  # 欧
R2 = 2.1e4  # 欧
C = 1e-6  # F
f = 25  # Hz

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

# 处理data1.csv
df1 = pd.read_csv("data1.csv")
ax1: plt.Axes

x = df1.iloc[:, 0].to_numpy() / 5  # V
y = df1.iloc[:, 1].to_numpy()  # mV to V
y[1:-1] += 1  # 修正漂移，其中首尾为手动读出
y = y / 1000  # V

H = N1 * x / (L * R1)  # A/m
B = C * R2 * y / (N2 * S)

# Sort data based on H for interpolation
sort_indices = np.argsort(H)
H_sorted = H[sort_indices]
B_sorted = B[sort_indices]

# Create spline interpolation
spline = interp.CubicSpline(H_sorted, B_sorted)
H_new = np.linspace(H_sorted.min(), H_sorted.max(), 500)
B_new = spline(H_new)

mu = B_new[1:] / H_new[1:]

ax1_twin = ax1.twinx()  # 创建第二个 y 轴

# 绘制 B-H 曲线
(line1,) = ax1.plot(
    H, B, linestyle="none", marker="o", markersize=2, color="C0", label="实验数据点"
)
(line2,) = ax1.plot(H_new, B_new, label="三次样条拟合曲线", color="C0")
ax1.set_xlabel("磁场强度 H (A/m)")
ax1.set_ylabel("磁感应强度 B (T)")
ax1.tick_params(axis="y", which="both", direction="in")
ax1.tick_params(axis="x", which="both", direction="in")
ax1.minorticks_off()

# 绘制 mu-H 曲线
(line3,) = ax1_twin.plot(
    H_new[1:], mu, label=r"磁导率 $\mu = \frac{B}{H}$", color="green"
)
ax1_twin.set_ylabel(r"磁导率 $\mu$ (T·m/A)")
ax1_twin.tick_params(axis="y", which="both", direction="in")
ax1_twin.minorticks_off()
ax1.set_title("初始磁化曲线与拟合磁导率")

# 合并图例
handles, labels = ax1.get_legend_handles_labels()
handles_twin, labels_twin = ax1_twin.get_legend_handles_labels()
ax1.legend(handles + handles_twin, labels + labels_twin, loc="lower right")


# 处理data.csv
df = pd.read_csv("data.csv")
ax2: plt.Axes

x = df.iloc[:, 0].to_numpy() / 5  # V
y_upper = df.iloc[:, 1].to_numpy()  # mV
y_lower = df.iloc[:, 2].to_numpy()  # mV
y_upper[1:-1] += 1  # 修正漂移，其中首尾为手动读出
y_lower[1:-1] += 1  # 修正
y_upper = y_upper / 1000  # V
y_lower = y_lower / 1000  # V

# 矫顽力点
Hc = np.array([-2.25, 2.3]) / 5 * N1 / (L * R1)  # A/m
Bc = np.array([0, 0])  # T

# 剩磁点 (从 data.csv 中 x=0 的行读取)
br_y_values = df[df.iloc[:, 0] == 0].iloc[0, 1:3].to_numpy()  # mV
br_y_values[br_y_values > 0] += 1  # 修正漂移
br_y_values[br_y_values < 0] += 1  # 修正
Br = C * R2 * (br_y_values / 1000) / (N2 * S)  # T
Hr = np.array([0, 0])  # A/m

H = N1 * x / (L * R1)  # A/m
B_upper = C * R2 * y_upper / (N2 * S)  # T
B_lower = C * R2 * y_lower / (N2 * S)  # T

spline_upper = interp.interp1d(H, B_upper, kind="cubic")
spline_lower = interp.interp1d(H, B_lower, kind="cubic")
H_new = np.linspace(H.min(), H.max(), 500)
B_upper_new = spline_upper(H_new)
B_lower_new = spline_lower(H_new)

# 绘制磁滞回线，合并图例
ax2.plot(H_new, B_upper_new, color="C0")
ax2.plot(H_new, B_lower_new, color="C0", label="磁滞回线")
ax2.scatter(H, B_upper, s=5, color="C0")
ax2.scatter(H, B_lower, s=5, color="C0")
ax2.legend(loc="lower right")


# 标注剩磁点 Br
ax2.scatter(Hr, Br, color="red", zorder=5, s=8)
ax2.annotate(
    f"$B_r$ ≈ {Br[0]:.3f} T",
    xy=(Hr[0], Br[0]),
    xytext=(-60, 5),
    textcoords="offset points",
)
ax2.annotate(
    f"$B_r$ ≈ {Br[1]:.3f} T",
    xy=(Hr[1], Br[1]),
    xytext=(5, -15),
    textcoords="offset points",
)

# 标注矫顽力点 Hc
ax2.scatter(Hc, Bc, color="red", zorder=5, s=8)
ax2.annotate(
    f"$H_c$ ≈ {Hc[0]:.2f} A/m",
    xy=(Hc[0], Bc[0]),
    xytext=(-60, 5),
    textcoords="offset points",
)
ax2.annotate(
    f"$H_c$ ≈ {Hc[1]:.2f} A/m",
    xy=(Hc[1], Bc[1]),
    xytext=(10, -15),
    textcoords="offset points",
)

# 添加坐标轴
ax2.axhline(0, color="gray", linestyle="-.")
ax2.axvline(0, color="gray", linestyle="-.")

ax2.set_xlabel("磁场强度 H (A/m)")
ax2.set_ylabel("磁感应强度 B (T)")
ax2.minorticks_off()
ax2.legend(loc="best")

ax2.set_title("磁滞回线（样品1，$f=25$Hz）")

ax1.set_box_aspect(4 / 5)  # 设置第一个子图的显示框高宽比为 4:5
ax2.set_box_aspect(4 / 5)  # 设置第二个子图的显示框高宽比为 4:5

fig.tight_layout()
# fig.savefig("BH_curve.png", bbox_inches="tight", dpi=500)

print(B_upper)
print(B_lower)
print(H)


# --- 输出转换后的坐标到 Excel ---
# 创建一个 Excel writer
def export_to_excel():
    with pd.ExcelWriter("transformed_coordinates.xlsx") as writer:
        # --- 处理 data1.csv ---
        df1_raw = pd.read_csv("data1.csv")
        x_div1 = df1_raw.iloc[:, 0].to_numpy()
        y_mv1 = df1_raw.iloc[:, 1].to_numpy()
        y_mv1[1:-1] += 1  # 修正漂移
        y_div1 = y_mv1 / 50  # 100mV/格

        # 创建 DataFrame
        df_out1 = pd.DataFrame({"x_div": x_div1, "y_div": y_div1})
        df_out1.to_excel(writer, sheet_name="data1_divs", index=False)

        # --- 处理 data.csv ---
        df_raw = pd.read_csv("data.csv")
        x_div = df_raw.iloc[:, 0].to_numpy()
        y_upper_mv = df_raw.iloc[:, 1].to_numpy()
        y_lower_mv = df_raw.iloc[:, 2].to_numpy()

        # 修正漂移
        y_upper_mv[1:-1] += 1
        y_lower_mv[1:-1] += 1

        # 转换为 "格"
        y_upper_div = y_upper_mv / 50
        y_lower_div = y_lower_mv / 50

        # 创建 DataFrame
        df_out = pd.DataFrame({
            "x_div": x_div,
            "y_upper_div": y_upper_div,
            "y_lower_div": y_lower_div,
        })
        df_out.to_excel(writer, sheet_name="data_divs", index=False)


# export_to_excel()
