import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.optimize import fsolve
from utils import draw_hline

df_200 = pd.read_csv("./data_200.csv")
df_500 = pd.read_csv("./data_500_1.csv")

L = 0.1  # unit: H
C = 0.047e-6  # unit: F
R0 = 50  # unit: Ω
RL = 25  # unit: Ω
f0_theo = 1 / (2 * np.pi * np.sqrt(L * C))
print(f0_theo)
E = 1  # unit: V

f_200 = df_200["x"].values
U_200 = df_200["y"].values / 1000
R_200 = 200
I_200 = U_200 / R_200
f0_200 = 2318.5
UC_200 = 5.288  # unit: V
UL_200 = 5.325  # unit: V
ZL_200 = UL_200 / max(I_200)
RL_200_calc = math.sqrt(abs(ZL_200**2 - (2 * math.pi * f0_200 * L) ** 2))
RL_200_calc1 = R_200 / U_200.max() * E - R_200 - R0
RL_200_calc2 = math.sqrt((UL_200 / max(I_200)) ** 2 - (UC_200 / max(I_200)) ** 2)
print(
    f"""RL_200: 
使用阻抗计算电阻: {RL_200_calc:.4f}
使用电阻分压计算，假设ZC+ZL虚部为0: {RL_200_calc1:.4f}
使用电容分压计算: {RL_200_calc2:.4f}"""
)

f_500 = df_500["x"].values
U_500 = df_500["y"].values / 1000
R_500 = 500
I_500 = U_500 / R_500
f0_500 = 2321.5
UC_500 = 2.528  # unit: V
UL_500 = 2.565  # unit: V
ZL_500 = UL_500 / max(I_500)
RL_500_calc = math.sqrt(ZL_500**2 - (2 * math.pi * f0_500 * L) ** 2)
RL_500_calc1 = R_500 / U_500.max() * E - R_500 - R0
RL_500_calc2 = math.sqrt((UL_500 / max(I_500)) ** 2 - (UC_500 / max(I_500)) ** 2)
print(
    f"""RL_500: 
使用阻抗计算电阻: {RL_500_calc:.4f}
使用电阻分压计算，假设ZC+ZL虚部为0: {RL_500_calc1:.4f}
使用电容分压计算: {RL_500_calc2:.4f}"""
)



plt.figure(figsize=(10, 6))
plt.scatter(f_200, I_200, label="R = 200 Ω Data", color="blue", s=5)
plt.scatter(f_500, I_500, label="R = 500 Ω Data", color="orange", s=5)
plt.axvline(
    f0_theo, color="green", linestyle="--", label="Theoretical f0", linewidth=0.5
)
plt.axvline(
    f0_200, color="blue", linestyle=":", label="Experimental f0 (200 Ω)", linewidth=0.5
)
plt.axvline(
    f0_500,
    color="orange",
    linestyle=":",
    label="Experimental f0 (500 Ω)",
    linewidth=0.5,
)
# plt.plot(f_200, I_200, label='R = 200 Ω', color='blue')
# plt.plot(f_500, I_500, label='R = 500 Ω', color='orange')


# # Avoid Runge's phenomenon
# def smooth(
#     x: np.ndarray, y: np.ndarray, x_peak, window_size=3
# ) -> tuple[np.ndarray, np.ndarray]:
#     mask = (x >= x_peak - window_size) & (x <= x_peak + window_size)

#     x_masked = x[mask]
#     # y_masked = y[mask]
#     # print(x_masked, "\n", y_masked)

#     x_new, y_new = (
#         np.linspace(x_masked[0], x_masked[-1], 4),
#         np.empty(4),
#     )

#     left = interpolate.interp1d(
#         x[x <= x_peak + 0.00001], y[x <= x_peak + 0.00001], kind=1
#     )
#     right = interpolate.interp1d(
#         x[x >= x_peak - 0.00001], y[x >= x_peak - 0.00001], kind=1
#     )

#     y_new[:3] = left(x_new[:3])
#     y_new[3:] = right(x_new[3:])
#     # print(x_new, "\n", y_new)
#     x_res = np.concatenate((x[~mask], x_new))
#     y_res = np.concatenate((y[~mask], y_new))

#     plt.scatter(x_new, y_new, color="red", s=10, label="Smoothed Points", marker="x")

#     sorted_indices = np.argsort(x_res)
#     return x_res[sorted_indices], y_res[sorted_indices]


# f_200_smooth, I_200_smooth = smooth(f_200, I_200, f0_200, window_size=5)


x_linspace = np.linspace(1600, 3000, 100000)
# f_200_interp = interpolate.interp1d(f_200_smooth, I_200_smooth, kind=3)
f_200_interp = interpolate.interp1d(f_200, I_200, kind=3)
f_500_interp = interpolate.interp1d(f_500, I_500, kind=3)
I_200_smooth = f_200_interp(x_linspace)
I_500_smooth = f_500_interp(x_linspace)

plt.plot(x_linspace, I_200_smooth, label="R = 200 Ω", color="blue", linewidth=1)
plt.plot(x_linspace, I_500_smooth, label="R = 500 Ω", color="orange", linewidth=1)

# 找到最大值和最大值/sqrt(2)
I_200_max = np.max(I_200_smooth)
I_500_max = np.max(I_500_smooth)
I_200_half_power = I_200_max / np.sqrt(2)
I_500_half_power = I_500_max / np.sqrt(2)

print(f"R = 200 Ω: I_max = {I_200_max:.6f} A, I_max/√2 = {I_200_half_power:.6f} A")
print(f"R = 500 Ω: I_max = {I_500_max:.6f} A, I_max/√2 = {I_500_half_power:.6f} A")


# 为200Ω找交点
def f_200_minus_target(x):
    return f_200_interp(x) - I_200_half_power


# 为500Ω找交点
def f_500_minus_target(x):
    return f_500_interp(x) - I_500_half_power


# 估计初始值：在谐振频率左右找交点
f0_200_guess_left = f0_200 - 200  # 左侧交点初始猜测
f0_200_guess_right = f0_200 + 200  # 右侧交点初始猜测
f0_500_guess_left = f0_500 - 200
f0_500_guess_right = f0_500 + 200

# 求解200Ω的交点
try:
    intersection_200_left = fsolve(f_200_minus_target, f0_200_guess_left)[0]
    intersection_200_right = fsolve(f_200_minus_target, f0_200_guess_right)[0]
    intersections_200 = [intersection_200_left, intersection_200_right]
    print(f"R = 200 Ω 交点频率: {intersections_200}")

    # 画出交点
    plt.plot(
        intersections_200,
        [I_200_half_power] * 2,
        "bo",
        markersize=4,
        label="R = 200 Ω interceptions",
    )

    # 计算与f0的差值
    diff_200_left = abs(intersections_200[0] - f0_200)
    diff_200_right = abs(intersections_200[1] - f0_200)
    bandwidth_200 = intersections_200[1] - intersections_200[0]
    print(f"R = 200 Ω: 左交点与f0差值 = {diff_200_left:.2f} Hz")
    print(f"R = 200 Ω: 右交点与f0差值 = {diff_200_right:.2f} Hz")
    print(f"R = 200 Ω: 带宽 = {bandwidth_200:.2f} Hz")
    Q_200_exp_f = f0_200 / bandwidth_200
    print(f"R = 200 Ω: 基于频率的实验Q值 = {Q_200_exp_f:.4f}")
    Q_200_exp_UC = UC_200 / E
    print(f"R = 200 Ω: 基于电压的实验Q值 = {Q_200_exp_UC:.4f}")
    Q_200_theo = math.sqrt(L / ((R_200 + R0 + RL) ** 2 * C))
    print(f"R = 200 Ω: 理论Q值(RL Fixed) = {Q_200_theo:.4f}")
    Q_200_theo = math.sqrt(L / ((R_200 / U_200.max()) ** 2 * C))
    print(f"R = 200 Ω: 理论Q值(RL Variable) = {Q_200_theo:.4f}")

except Exception:
    print("R = 200 Ω: 无法找到交点")

# 求解500Ω的交点
try:
    intersection_500_left = fsolve(f_500_minus_target, f0_500_guess_left)[0]
    intersection_500_right = fsolve(f_500_minus_target, f0_500_guess_right)[0]
    intersections_500 = [intersection_500_left, intersection_500_right]
    print(f"R = 500 Ω 交点频率: {intersections_500}")

    # 画出交点
    plt.plot(
        intersections_500,
        [I_500_half_power] * 2,
        "o",
        color="darkorange",
        markersize=4,
        label="R = 500 Ω interceptions",
    )

    # 计算与f0的差值
    diff_500_left = abs(intersections_500[0] - f0_500)
    diff_500_right = abs(intersections_500[1] - f0_500)
    bandwidth_500 = intersections_500[1] - intersections_500[0]
    print(f"R = 500 Ω: 左交点与f0差值 = {diff_500_left:.2f} Hz")
    print(f"R = 500 Ω: 右交点与f0差值 = {diff_500_right:.2f} Hz")
    print(f"R = 500 Ω: 带宽 = {bandwidth_500:.2f} Hz")
    Q_500_exp = f0_500 / bandwidth_500
    print(f"R = 500 Ω: 基于频率的实验Q值 = {Q_500_exp:.4f}")
    Q_500_exp_UC = UC_500 / E
    print(f"R = 500 Ω: 基于电压的实验Q值 = {Q_500_exp_UC:.4f}")
    Q_500_theo = math.sqrt(L / ((R_500 + R0 + RL) ** 2 * C))
    print(f"R = 500 Ω: 理论Q值(RL Fixed) = {Q_500_theo:.4f}")
    Q_500_theo = math.sqrt(L / ((R_500 / U_500.max()) ** 2 * C))
    print(f"R = 500 Ω: 理论Q值(RL Variable) = {Q_500_theo:.4f}")

except Exception:
    print("R = 500 Ω: 无法找到交点")

# 画出最大值/sqrt(2)的水平线
draw_hline(
    I_200_half_power,
    *intersections_200,
    color="blue",
    linestyle="--",
    alpha=0.7,
    linewidth=0.8,
    label="R = 200 Ω: I_max/√2",
)
draw_hline(
    I_500_half_power,
    *intersections_500,
    color="orange",
    linestyle="--",
    alpha=0.7,
    linewidth=0.8,
    label="R = 500 Ω: I_max/√2",
)

plt.xlim(1500, 3100)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Current (A)")
plt.title("Current vs Frequency for Different Resistances")
# plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("RLC_frequency_response.svg")
plt.savefig("RLC_frequency_response.png", dpi=300)
plt.show()
