import pandas as pd
import numpy as np
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


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(60, 12))
ax1: plt.Axes
ax2: plt.Axes
ax3: plt.Axes

# Fig 1: ratio arm size
df = pd.read_csv("data/arm_ratio.csv")

ratio = df["arm_ratio"]
size = df["size"]
r0 = df["r0"]
r0_ = df["r0_"]
delta_r = (r0 - r0_).abs()
S = n / (delta_r / r0)
r0_error = np.vstack((
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
ax1.errorbar(size, S, yerr=r0_error, fmt="o", label="error bar")
# ax1.plot(xnew1, ynew1, "r-", label="cubic interpolation")
ax1.plot(xnew1_, ynew1_, "r-", label="linear regression")

ax1.set_xlabel("arm size (m)")
ax1.set_ylabel("S")
ax1.set_title("Delta r vs Arm Size")
ax1.legend()
ax1.grid()

# 填充r0为r0.avg
r0 = np.array([r0.mean()] * (len(df) - 1))
r0_ = r0_[1:]
delta_r = (r0 - r0_).abs()
S = n / (delta_r / r0)
size = size[1:]
r0_error = np.vstack((
    (n / (delta_r / 10 * 11 / r0) - S).abs(),
    (n / (delta_r / 10 * 9 / r0) - S).abs(),
))
ax3.scatter(size, S, label="size (r0 fixed to avg)")
ax3.errorbar(size, S, yerr=r0_error, fmt="o", label="error bar")
regr_func, corr1 = poly_fit(size.values, S.values, 1)
xnew1_ = np.linspace(size.min(), size.max(), 100)
ynew1_ = regr_func(xnew1_)
print(f"R of data set 1 is {corr1:.4f}")
ax3.plot(xnew1_, ynew1_, "r-", label="linear regression (from fig 1)")
ax3.set_xlabel("arm size (m)")
ax3.set_ylabel("S")
ax3.set_title("Delta r vs Arm Size (r0 fixed to avg)")
ax3.legend()
ax3.grid()

# Fig 2: Power Voltage
df2 = pd.read_csv("data/power_voltage.csv")


def draw_lin_regr(ax2: plt.Axes, df2: pd.DataFrame, **kwargs):
    E = df2["E"]
    r0_1 = df2["r0"]
    r0_1_ = df2["r0_"]

    delta_r1 = (r0_1 - r0_1_).abs()
    S1 = n / (delta_r1 / r0_1)

    regr_func2, corr = poly_fit(E.values, S1.values, 1)
    print(f"R of data set 2 is {corr:.4f}")
    xnew = np.linspace(E.min(), E.max(), 100)
    ynew = regr_func2(xnew)
    ax2.plot(xnew, ynew, **kwargs)
    return E, S1, xnew, ynew


E, S1, xnew2_, ynew2_ = draw_lin_regr(ax2, df2, color="r")
ax2.scatter(E, S1, label="E")
# Drop point 2 (in desc order of E)
df2.drop(1, axis=0, inplace=True)
draw_lin_regr(ax2, df2, label="linear regression (drop point 2)", color="g")

ax2.set_xlabel("Power Voltage (V)")
ax2.set_ylabel("S")
ax2.set_title("Delta r vs Power Voltage")
ax2.legend()
ax2.grid()

# plt.show()
fig.tight_layout()
fig.savefig("Delta_r_vs_ArmSize_and_PowerVoltage.png", dpi=600)
