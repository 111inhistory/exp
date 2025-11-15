import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
import matplotlib as mpl
import warnings


def init_style():
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
