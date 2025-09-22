import pandas as pd
import numpy as np
import scipy.interpolate as interp
import scipy.optimize as opt

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))

# Fig 1: ratio arm size
df = pd.read_csv("data/arm_ratio.csv", sep=", ")
ratio = df["arm_ratio"]
size = df["size"]
r0 = df["r0"]
r0_ = df["r0_"]
delta_r = (r0 - r0_).abs()
S = 1 / (delta_r / r0)
ax1.scatter(size, S, label="size")
# int_func1 = interp.interp1d(size, S, kind="cubic")
# xnew1 = np.linspace(size.min(), size.max(), 100)
# ynew1 = int_func1(xnew1)
regr_func = np.polyfit(size, S, 1)
xnew1 = np.linspace(size.min(), size.max(), 100)
ynew1 = regr_func[0] * xnew1 + regr_func[1]
ax1.plot(xnew1, ynew1, 'r-', label="cubic interpolation")
ax1.set_xlabel("arm size (m)")
ax1.set_ylabel("S")
ax1.set_title("Delta r vs Arm Size")
ax1.legend()
ax1.grid()

# Fig 2: Power Voltage
df2 = pd.read_csv("data/power_voltage.csv", sep=", ")
E = df2["E"]
r0_1 = df2["r0"]
r0_1_ = df2["r0_"]
delta_r1 = (r0_1 - r0_1_).abs()
S1 = 1 / (delta_r1 / r0_1)
ax2.scatter(E, S1, label="E")
int_func2 = interp.interp1d(E, S1, kind="cubic")
xnew2 = np.linspace(E.min(), E.max(), 100)
ynew2 = int_func2(xnew2)
ax2.plot(xnew2, ynew2, 'r-', label="cubic interpolation")
ax2.set_xlabel("Power Voltage (V)")
ax2.set_ylabel("S")
ax2.set_title("Delta r vs Power Voltage")
ax2.legend()
ax2.grid()

plt.show()
fig.tight_layout()
# fig.savefig("Delta_r_vs_ArmSize_and_PowerVoltage.png", dpi=600)