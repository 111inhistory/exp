import matplotlib.pyplot as plt
import numpy as np

def draw_hline(y: float, x_min: float, x_max: float, **kwargs):
    x = np.linspace(x_min, x_max, 2)
    y = np.array([y,y])
    plt.plot(x, y, **kwargs)
