import math

def calculate_box_uncertainty(resistance, error_map):
    """
    计算电阻箱的标准不确定度。
    根据电阻值拆分成不同档位，并按其误差比例计算不确定度，
    最后用不确定度传播公式（平方和开方）求和。
    """
    uncertainty_squared_sum = 0
    temp_resistance = resistance
    
    # 确定电阻值的最高位（最高幂次）
    if resistance > 0:
        power = int(math.log10(resistance))
    else:
        return 0

    # 从最高位开始，逐位计算不确定度
    for i in range(power, -2, -1):
        decade_value = 10**i
        if decade_value in error_map:
            # 提取该档位的数值
            digit = int(temp_resistance // decade_value)
            if digit > 0:
                value_at_decade = digit * decade_value
                error_percentage = error_map[decade_value]
                # 计算该档位的不确定度
                uncertainty_term = value_at_decade * error_percentage
                uncertainty_squared_sum += uncertainty_term**2
                
            # 减去已处理的数值，以便处理下一位
            temp_resistance %= decade_value
    
    return math.sqrt(uncertainty_squared_sum)

# 1. 定义已知参数
R0 = 61199.0
R1 = 900.0
R2 = 9000.0
extra_R0_error = 12.2938

# 定义各档位的误差比例（用字典存储，更清晰）
# 注意：这里将百分比转换为小数
error_percentages = {
    10000: 0.001,  # 10^4
    1000: 0.001,   # 10^3
    100: 0.005,    # 10^2
    10: 0.01,      # 10^1
    1: 0.02,       # 10^0
    0.1: 0.05      # 10^-1
}

# 2. 计算各电阻的标准不确定度
# R0的不确定度（来自电阻箱）
u_R0_box = calculate_box_uncertainty(R0, error_percentages)
# R0的总不确定度（电阻箱不确定度与附加误差的平方和再开方）
u_R0 = math.sqrt(u_R0_box**2 + extra_R0_error**2)

# R1的不确定度
u_R1 = calculate_box_uncertainty(R1, error_percentages)

# R2的不确定度
u_R2 = calculate_box_uncertainty(R2, error_percentages)

print("--- 各电阻不确定度计算 ---")
print(f"R0 的电阻箱不确定度: {u_R0_box:.4f} Ω")
print(f"R0 的总不确定度: {u_R0:.4f} Ω")
print(f"R1 的不确定度: {u_R1:.4f} Ω")
print(f"R2 的不确定度: {u_R2:.4f} Ω\n")

# 3. 计算 Rx 的值
Rx = (R0 * R1) / R2
print(f"计算得到的 Rx 值为: {Rx:.4f} Ω\n")

# 4. 根据不确定度传播公式计算 Rx 的总不确定度
# 公式为: (u_Rx/Rx)^2 = (u_R0/R0)^2 + (u_R1/R1)^2 + (u_R2/R2)^2
relative_uncertainty_squared = (u_R0/R0)**2 + (u_R1/R1)**2 + (u_R2/R2)**2
relative_uncertainty_Rx = math.sqrt(relative_uncertainty_squared)
u_Rx = Rx * relative_uncertainty_Rx

print("--- Rx 总不确定度计算 ---")
print(f"Rx 的相对不确定度: {relative_uncertainty_Rx * 100:.4f} %")
print(f"Rx 的总不确定度 (u_Rx): {u_Rx:.4f} Ω")