import pandas as pd
import numpy as np
from scipy import optimize as opt
import scipy.interpolate as interp
import matplotlib.pyplot as plt

n = 0.4


def interp1d(x: np.ndarray, y: np.ndarray, kind: str = "linear") -> interp.interp1d:
    return interp.interp1d(x, y, kind=kind)


def poly_fit(x: np.ndarray, y: np.ndarray, degree: int) -> tuple[np.poly1d, float]:
    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)
    mat = np.vstack((x, y))
    corr_mat = np.corrcoef(mat)
    corr_xy = corr_mat[0, 1]
    return poly, corr_xy


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(40, 12))
ax1: plt.Axes
ax2: plt.Axes

# Fig 1: ratio arm size
df = pd.read_csv("data/arm_ratio.csv")

ratio = df["arm_ratio"]
size = df["size"]
r0 = df["r0"]
r0_ = df["r0_"]
delta_r = (r0 - r0_).abs()
S = n / (delta_r / r0)
S_error = np.vstack((
    (n / (delta_r / 10 * 11 / r0) - S).abs(),
    (n / (delta_r / 10 * 9 / r0) - S).abs(),
))

# int_func1 = interp1d(size.values, S.values, kind="cubic")
# xnew1 = np.linspace(size.min(), size.max(), 100)
# ynew1 = int_func1(xnew1)

regr_func, corr1 = poly_fit(size.values, S.values, 1)
xnew1_ = np.linspace(size.min(), size.max(), 100)
ynew1_ = regr_func(xnew1_)
print(f"R of data set 1 is {corr1:.4f}")

ax1.scatter(size, S, label="size")
ax1.errorbar(size, S, yerr=S_error, fmt="o", label="error bar")
# ax1.plot(xnew1, ynew1, "r-", label="cubic interpolation")
ax1.plot(xnew1_, ynew1_, "r-", label="linear regression")

ax1.set_xlabel("arm size (m)")
ax1.set_ylabel("S")
ax1.set_title("Delta r vs Arm Size")
ax1.legend()
ax1.grid()

# # 填充r0为r0.avg
# r0 = np.array([r0.mean()] * len(df))
# delta_r = (r0 - r0_).abs()
# S = n / (delta_r / r0)
# r0_error = np.vstack((
#     (n / (delta_r / 10 * 11 / r0) - S).abs(),
#     (n / (delta_r / 10 * 9 / r0) - S).abs(),
# ))
# ax3.scatter(size, S, label="size (r0 fixed to avg)")
# ax3.errorbar(size, S, yerr=r0_error, fmt="o", label="error bar")
# regr_func, corr1 = poly_fit(size.values, S.values, 1)
# xnew1_ = np.linspace(size.min(), size.max(), 100)
# ynew1_ = regr_func(xnew1_)
# print(f"R of data set 1 is {corr1:.4f}")
# ax3.plot(xnew1_, ynew1_, "r-", label="linear regression (from fig 1)")
# ax3.set_xlabel("arm size (m)")
# ax3.set_ylabel("S")
# ax3.set_title("Delta r vs Arm Size (r0 fixed to avg)")
# ax3.legend()
# ax3.grid()

E = 2.5


def perform_curve_fit_and_plot(
    ax: plt.Axes, x_data: np.ndarray, y_data: np.ndarray, curve_func, p0: list
):
    """
    对数据进行非线性拟合，并绘制结果。
    """
    popt, pcov = opt.curve_fit(curve_func, x_data, y_data, p0=p0)

    print(f"Fitted params: {popt}")
    C_fit, r0_fit, rg_fit = popt
    C_fit_err, r0_fit_err, rg_fit_err = np.sqrt(np.diag(pcov))
    print(f"Fitted C: {C_fit:.4f} ± {C_fit_err:.4f}")
    print(f"Fitted r0: {r0_fit:.4f} ± {r0_fit_err:.4f}")
    print(f"Fitted rg: {rg_fit:.4f} ± {rg_fit_err:.4f}")
    print(f"Covariance Matrix:\n{pcov}")

    x_new = np.linspace(x_data.min(), x_data.max(), 100)
    y_new = curve_func(x_new, *popt)

    ax.plot(x_new, y_new, label="non-linear regression")
    ax.scatter(x_data, y_data, label="data points")
    ax.legend()
    ax.set_title("Delta R - R1 (with non-linear regression)")
    ax.grid()


def curve(R1, C, Rx, Rg):
    """
    计算电桥灵敏度 S 与 R1 关系的理论曲线函数。
    S = Δn / (ΔR0 / R0)

    参数:
    R1 (numpy.ndarray): R1 的一系列阻值 (自变量)。
    C (float): 待拟合的综合比例常数 (C = Vs / k)。
    Rx (float): 待拟合的待测电阻 Rx 的值。
    Rg (float): 待拟合的微安表内阻 Rg 的值。

    返回:
    numpy.ndarray: 对应每个R1值的理论灵敏度S。
    """
    # 为防止除以零等数学错误，增加一个极小值
    epsilon = 1e-12

    # 根据理论公式计算
    # 分子
    numerator = C * R1 * Rx

    # 分母
    term1 = R1 + Rx
    term2 = 11 * R1 * Rx + Rg * (R1 + Rx)
    denominator = term1 * term2

    return numerator / (denominator + epsilon)


r1 = size.values
delta_r = (r0 - r0_).abs().values
S = n / (delta_r / r0.values)

initial_guess = [1, r0.mean() / 10, 100]
perform_curve_fit_and_plot(ax2, r1, S, curve, p0=initial_guess)

# S1 = np.concatenate((S[0:2], S[2:4] - S_error[0, 2:4], [S[4]]))
# S2 = np.concatenate((S[0:2], S[2:4] + S_error[1, 2:4], [S[4]]))

# perform_curve_fit_and_plot(ax2, r1, S1, curve, p0=initial_guess)
# perform_curve_fit_and_plot(ax2, r1, S2, curve, p0=initial_guess)

plt.show()
fig.savefig("Delta_r_vs_ArmSize.png", dpi=300)
