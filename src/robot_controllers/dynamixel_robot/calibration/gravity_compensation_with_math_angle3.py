import matplotlib.pyplot as plt
from numpy import cos
import numpy as np
from scipy.optimize import curve_fit

servo_3_dynamic_offsets = {'0.0': 0, '0.1': 4, '0.2': 8, '0.3': 10, '0.4': 13, '0.5': 15, '0.6': 13, '0.7': 10, '0.8': 8, '0.9': 4, '1.0': 0, '1.1': -5, '1.2': -7}

x = []
y = []

for angle3_raw in servo_3_dynamic_offsets:
    angle3 = float(angle3_raw) - 0.5
    x.append(angle3)
    offset = servo_3_dynamic_offsets[angle3_raw]
    y.append(offset)

plot_x = np.array(x)
plot_y = np.array(y)


def func(variables, a, b):
    a3 = variables
    return a * np.cos(b * a3)


fit, _ = curve_fit(func, plot_x, plot_y)
print(fit)  # [13.50604571  3.15347015]

y_cos = fit[0] * cos(fit[1] * plot_x)

plt.plot(plot_x, plot_y, 'g')
plt.plot(plot_x, y_cos, 'y')
plt.show()
