import pandas as pd
import numpy as np
import scienceplots
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
import warnings
import re

mpl.use("pgf")

# 配置 PGF + XeLaTeX 环境
pgf_config = {
    "pgf.texsystem": "xelatex",               # 指定使用 xelatex 编译
    "text.usetex": True,                      # 启用 LaTeX 渲染
    "font.family": "serif",                   # 基本字体族
    "font.size": 8,                          # 字号
    "pgf.rcfonts": False,                     # 不使用 matplotlib 默认字体设置
    "pgf.preamble": r"""
        \usepackage{xeCJK}                    % 支持中文
        \usepackage{unicode-math}            % 支持 unicode 数学符号，如 −
        \setmainfont{Times New Roman}        % 英文字体
        \setCJKmainfont{SimSun}              % 中文字体
        \xeCJKsetup{CJKmath=true}            % 中文参与数学公式
    """
}

# 应用配置
mpl.rcParams.update(pgf_config)

# Suppress warnings from polyfit for clarity
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

plt.style.use(['science', 'no-latex', 'grid'])


def generate_report(exponential_results, linear_results, report_path):
    """Generates a markdown report of the fitting results."""
    k_B = 1.380649e-23  # Boltzmann constant in J/K

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# PN结特性实验拟合报告\n\n")

        f.write("## 1. I-V 特性指数拟合\n\n")
        f.write("根据物理模型 $I_F = I_s \\cdot e^{eU_F/(n k_B T)}$，我们对不同温度下的数据进行线性化拟合。\n\n")
        f.write(f"使用的玻尔兹曼常数标准值为: $k_B = {k_B:.6e}$ J/K。\n\n")
        f.write("拟合得到的参数 `k_fit` 实际上对应于 `n * k_B`，其中 `n` 是理想因子。因此，我们计算 `n = k_fit / k_B` 来评估PN结的理想程度。\n\n")

        for result in exponential_results:
            temp_c = result['temp_c']
            data = result['data']
            I_s_fit = result['I_s_fit']
            k_fit = result['k_fit']
            r_squared = result['r_squared']
            
            # Calculate ideality factor n
            n_factor = k_fit / k_B

            f.write(f"### 1.{result['index']}. 温度: {temp_c}\u2103\n\n")
            f.write("**拟合结果:**\n\n")
            f.write("| 参数 | 拟合值 | 单位 |\n")
            f.write("|:---|:---|:---|\n")
            f.write(f"| 反向饱和电流 $I_s$ | {I_s_fit:.3e} | µA |\n")
            f.write(f"| 拟合参数 $k_{{fit}}$ | {k_fit:.3e} | J/K |\n")
            f.write(f"| 拟合优度 $R^2$ | {r_squared:.5f} | - |\n")
            f.write(f"| **相对误差 $E(k)$** | **{abs(n_factor-1)*100:.2f}%** | - |\n\n")
            
            f.write("**原始数据:**\n\n")
            f.write("| 正向电压 U_F (V) | 正向电流 I_F (µA) |\n")
            f.write("|:---|:---|\n")
            for _, row in data.iterrows():
                f.write(f"| {row['U_F']:.4f} | {row['I_F']:.4e} |\n")
            f.write("\n---\n\n")

        f.write("## 2. U-T 特性线性拟合\n\n")
        if linear_results:
            U_g_fit = linear_results['U_g_fit']
            S_fit = linear_results['S_fit']
            r_squared_linear = linear_results['r_squared_linear']
            data_linear = linear_results['data']

            f.write("根据物理模型 $U_F = U_g + S \\cdot T$，我们对正向电压和绝对温度的关系进行线性拟合。\n\n")
            f.write("**拟合结果:**\n\n")
            f.write("| 参数 | 拟合值 | 单位 |\n")
            f.write("|:---|:---|:---|\n")
            f.write(f"| 禁带电压 $U_g$ | {U_g_fit:.4f} | V |\n")
            f.write(f"| 温度系数 $S$ | {S_fit:.4f} | V/K |\n")
            f.write(f"| 拟合优度 $R^2$ | {r_squared_linear:.5f} | - |\n\n")

            f.write("**原始数据:**\n\n")
            f.write("| 绝对温度 T (K) | 正向电压 U_F (V) |\n")
            f.write("|:---|:---|\n")
            for _, row in data_linear.iterrows():
                f.write(f"| {row['T']:.2f} | {row['U_F']:.4f} |\n")
            f.write("\n")
        else:
            f.write("未找到线性拟合的数据。\n")
            
    print(f"成功生成报告: {report_path}")


def format_scientific_latex(num, precision=2):
    """Formats a number into scientific notation for LaTeX with no leading zeros in the exponent."""
    s = f'{num:.{precision}e}'
    mantissa, exponent = s.split('e')
    # Remove leading zero from exponent
    exponent_val = int(exponent)
    return f'{mantissa} \\times 10^{{{exponent_val}}}'


def main():
    """
    Main function to read data, perform fits, plot results, and generate a report.
    """
    excel_file_path = '在特定温度下pn结正向电压和正向电流的关系.xlsx'
    e_charge = 1.60217663e-19  # Elementary charge in Coulombs

    # # --- Setup Matplotlib ---
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['SimSun', 'Noto Serif SC'],
        'font.size': 8, # Smaller font for a more compact plot
    #     'font.weight': 'normal',
    #     'font.variant': 'normal',
    #     'font.stretch': 'normal',
    })

    try:
        xls = pd.ExcelFile(excel_file_path)
    except FileNotFoundError:
        print(f"错误：找不到文件 '{excel_file_path}'。")
        return

    sheet_names = xls.sheet_names
    num_exp_sheets = len(sheet_names) - 1
    print(f"找到 {len(sheet_names)} 个工作表: {sheet_names}")

    exponential_results = []
    
    # --- Part 1: Exponential Fits (Data Processing) ---
    # This part is the same for all figures, so we process it once.
    for index, sheet_name in enumerate(sheet_names[:-1]):
        df = pd.read_excel(xls, sheet_name=sheet_name, header=None)

        temp_val_str = None
        for _, row in df.iterrows():
            for cell in row:
                if isinstance(cell, str) and 't=' in cell:
                    temp_val_str = cell
                    break
            if temp_val_str:
                break
        
        if not temp_val_str:
            print(f"警告: 在工作表 '{sheet_name}' 中找不到温度值。跳过此表。")
            continue

        try:
            match = re.search(r't\s*=\s*(-?\d+\.?\d*)', temp_val_str)
            if match:
                temp_c = float(match.group(1))
                temp_k = temp_c + 273.15
            else:
                raise ValueError("Could not find temperature value with regex")
        except (ValueError, IndexError):
            print(f"警告: 无法从 '{temp_val_str}' 解析温度。跳过工作表 '{sheet_name}'。")
            continue

        first_numeric_row = -1
        for i, row in df.iterrows():
            try:
                pd.to_numeric(row)
                first_numeric_row = i
                break
            except (ValueError, TypeError):
                continue
        
        if first_numeric_row == -1:
            print(f"警告: 在工作表 '{sheet_name}' 中找不到数值数据。")
            continue

        data = df.iloc[first_numeric_row:].dropna(how='all', axis=1).dropna(how='any', axis=0)
        data = data.iloc[:, :2].apply(pd.to_numeric, errors='coerce').dropna()
        data.columns = ['U_F', 'I_F']
        
        U_F = data['U_F'].values
        I_F = data['I_F'].values

        if len(U_F) < 2:
            print(f"警告: 工作表 '{sheet_name}' 中的数据不足以进行拟合。")
            continue

        try:
            positive_current_mask = I_F > 0
            if np.sum(positive_current_mask) < 2:
                print(f"警告: 工作表 '{sheet_name}' 中没有足够的正电流数据点进行对数拟合。")
                continue

            U_F_lin = U_F[positive_current_mask]
            I_F_lin = I_F[positive_current_mask]
            log_I_F = np.log(I_F_lin)

            slope, intercept, r_value, p_value, std_err = linregress(U_F_lin, log_I_F)
            
            k_fit = e_charge / (slope * temp_k)
            I_s_fit = np.exp(intercept)
            r_squared = r_value**2

            exponential_results.append({
                'index': index + 1,
                'temp_c': temp_c,
                'temp_k': temp_k,
                'data': data.copy(),
                'I_s_fit': I_s_fit,
                'k_fit': k_fit,
                'slope': slope,
                'r_squared': r_squared
            })
            print(f"已处理工作表 '{sheet_name}' (T={temp_c}\u2103) 的指数数据。")
            print(exponential_results)

        except Exception as e:
            print(f"错误: 无法对工作表 '{sheet_name}' 进行指数拟合: {e}")

    # --- Part 2: Linear Fit Data Reading ---
    df_linear_original = None
    if sheet_names:
        last_sheet_name = sheet_names[-1]
        try:
            df_linear_original = pd.read_excel(xls, sheet_name=last_sheet_name, header=0)
            df_linear_original = df_linear_original.iloc[:, 1:3].apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
            df_linear_original.columns = ['T', 'U_F']
        except Exception as e:
            print(f"错误: 无法读取或处理最后一个工作表 '{last_sheet_name}': {e}")
            df_linear_original = None

    # --- Part 3: Generate 3 Versions of Plots and Reports ---
    if df_linear_original is None or df_linear_original.empty:
        print("错误：无法为 U-T 特性加载数据，已中止。")
        return

    for i in range(3):
        # --- Create a new figure for each version ---
        fig = plt.figure(figsize=(8, 6.5))
        # Main GridSpec: 2 rows, 1 column. Top row for exp plot + legend, bottom row for the other two.
        gs_main = fig.add_gridspec(2, 1, height_ratios=[1, 1])

        # Nested GridSpec for the top row: 1 row, 2 columns (plot on left, legend on right)
        gs_top = gs_main[0].subgridspec(1, 2, width_ratios=[7, 1])
        # gs_top = gs_main[0].subgridspec(1, 2)
        ax_exp = fig.add_subplot(gs_top[0])
        ax_legend_container = fig.add_subplot(gs_top[1])
        ax_legend_container.axis('off') # This axis is just for placing the legend

        # Nested GridSpec for the bottom row: 1 row, 2 columns for the two plots
        gs_bottom = gs_main[1].subgridspec(1, 2)
        ax_lin = fig.add_subplot(gs_bottom[0])
        ax_last = fig.add_subplot(gs_bottom[1])

        # --- Plot Exponential Fits (same for all versions) ---
        for result in exponential_results:
            data = result['data']
            U_F = data['U_F'].values
            I_F = data['I_F'].values
            temp_c = result['temp_c']
            temp_k = result['temp_k']
            I_s_fit = result['I_s_fit']
            k_fit = result['k_fit']
            slope = result['slope']
            r_squared = result['r_squared']

            def exp_model_func(u, i_s, k_val):
                return i_s * np.exp(e_charge * u / (k_val * temp_k))

            U_F_curve = np.linspace(U_F.min(), U_F.max(), 200)
            I_F_curve = exp_model_func(U_F_curve, I_s_fit, k_fit)
            
            label_line1 = f'{temp_c}\u2103, $R^2={r_squared:.4f}$'
            label_line2 = f'$I_F = {format_scientific_latex(I_s_fit)} \\cdot e^{{{slope:.2f} U_F}}$'
            
            a = ax_exp.semilogy(U_F, I_F, 'o', markersize=3)
            color = a[0].get_color()
            ax_exp.semilogy(U_F_curve, I_F_curve, '-', color=color, linewidth=1.5, label=f'{label_line1}\n{label_line2}')

            ax_lin.plot(U_F, I_F, 'o', color=color, markersize=3)
            ax_lin.plot(U_F_curve, I_F_curve, '-', color=color, linewidth=1.5, label=f'{temp_c}\u2103 ($R^2={r_squared:.4f}$)')

        # --- Process and Plot Linear Fit for the current group ---
        linear_results = {}
        if len(df_linear_original) > 2:
            middle_data = df_linear_original.iloc[1:-1]
            group_data = middle_data[middle_data.reset_index(drop=True).index % 3 == i]
            current_data = pd.concat([df_linear_original.iloc[[0]], group_data, df_linear_original.iloc[[-1]]]).drop_duplicates()
        else:
            current_data = df_linear_original
        
        T_linear = current_data['T'].values
        U_F_linear = current_data['U_F'].values

        if len(T_linear) >= 2:
            slope, intercept, r_value, _, _ = linregress(T_linear, U_F_linear)
            U_g_fit = intercept
            S_fit = slope
            r_squared_linear = r_value**2

            linear_results = {
                'data': current_data.copy(),
                'U_g_fit': U_g_fit,
                'S_fit': S_fit,
                'r_squared_linear': r_squared_linear
            }
            print(linear_results)
            
            ax_last.scatter(T_linear, U_F_linear, s=15, label='实验数据')
            
            T_fit_line = np.array([T_linear.min(), T_linear.max()])
            U_F_fit_line = S_fit * T_fit_line + U_g_fit
            
            label_line1_linear = f'拟合, $R^2 = {r_squared_linear:.4f}$'
            label_line2_linear = f'$U_F = {S_fit:.4f}T + {U_g_fit:.4f}$'
            ax_last.plot(T_fit_line, U_F_fit_line, '-', color=plt.cm.get_cmap("viridis")(0.85), label=f'{label_line1_linear}\n{label_line2_linear}')
        else:
            print(f"警告: 第 {i+1} 组的数据不足以进行线性拟合。")

        # --- Finalize and Save the Figure for the current group ---
        ax_exp.set_xlabel('正向电压 $U_F$ (V)')
        ax_exp.set_ylabel('正向电流 $I_F$ (µA)')
        ax_exp.set_title('(a) PN结伏安特性曲线 (指数坐标)', loc='left')
        
        # Get handles and labels from ax_exp and place them in the dedicated legend axis
        handles, labels = ax_exp.get_legend_handles_labels()
        ax_legend_container.legend(handles, labels, fontsize=7, loc='center right', frameon=True)

        ax_lin.set_xlabel('正向电压 $U_F$ (V)')
        ax_lin.set_ylabel('正向电流 $I_F$ (µA)')
        ax_lin.set_title('(b) PN结伏安特性曲线 (线性坐标)', loc='left')
        ax_lin.legend(fontsize=7)

        ax_last.set_xlabel('绝对温度 $T$ (K)')
        ax_last.set_ylabel('正向电压 $U_F$ (V)')
        ax_last.set_title(f'(c) U-T关系 (第 {i+1} 组)', loc='left')
        ax_last.legend(fontsize=8, loc='best')
        ymin, ymax = ax_last.get_ylim()
        if ymin > ymax:
            ax_last.set_ylim(ymax, ymin)

        # Adjust layout to make room for the legend outside the plot
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.3, hspace=0.4)
        
        combined_filename = f'pn_knot_combined_plots_group_{i+1}.pdf'
        fig.savefig(combined_filename, format='pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"成功生成合并图表: {combined_filename}")

        # --- Generate Report for the current group ---
        report_filename = f'results_report_group_{i+1}.md'
        generate_report(exponential_results, linear_results, report_filename)


if __name__ == '__main__':
    main()