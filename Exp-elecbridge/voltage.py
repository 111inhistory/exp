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


fig, ax2 = plt.subplots(figsize=(20, 12))
ax2: plt.Axes

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
print(S1)
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
fig.savefig("PowerVoltage.png", dpi=100)