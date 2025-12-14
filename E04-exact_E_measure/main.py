import numpy as np

t0 = 21.0  # degC
E20 = 1.01860  # 20 degC ref power voltage
En = E20 - 4.06e-5 * (t0 - 20.0)  # power voltage at t0 degC, arount 1.01856V
print(f"En={En:.6f} V")
En = 1.01856  # V
Rn = En * 1e3  # ohm
print(f"Rn={Rn:.2f} ohm")

R1 = 313.93  # ohm
R2 = 1486.09  # ohm
Rall = R1 + R2

workingI = 1e-3  # A
Ex = R2 * workingI
print(f"Ex={Ex:.5f} V")

# 温度的精度为1℃，那么En的精度为4.06e-5 V
# 电阻箱精度均为0.02%


Deltan = 0.4
R2_1 = 1487.00  # ohm
DeltaEx1 = abs(R2 - R2_1) * workingI

S = Deltan / DeltaEx1
print(f"S={S:.2f} V^-1")

U_Ex = np.sqrt(2e-4**2 + 2e-4**2 + (4.06e-5 / En) ** 2 + (Deltan / S) ** 2) * Ex
print(f"U_Ex={U_Ex:.5f} V")
print(f"Relative uncertainty of Ex: {U_Ex/Ex*100:.5f} %")

Rs = 1000.0  # ohm
DeltaRs = 0.1  # ohm
R2_2 = 1480.39  # ohm
Ux = R2_2 * workingI

# 这个是测试内阻，并联了一个1000欧姆的电阻
rx = (Ex - Ux) / Ux * Rs
print(f"rx={rx:.5f} ohm")

U_rx = (
    np.sqrt(
        (4.06e-5 / En) ** 2
        + 2e-4**2
        + 2e-4**2
        + (DeltaRs / Rs) ** 2
        + (Deltan / S) ** 2
    )
    * rx
)
print(f"U_rx={U_rx:.5f} ohm")
print(f"Relative uncertainty of rx: {U_rx/rx*100:.5f} %")