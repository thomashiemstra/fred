import json

import matplotlib.pyplot as plt
from numpy import cos
import numpy as np
from scipy.optimize import curve_fit

servo_3_dynamic_offsets_raw = {"0.9": [2], "0.85": [3, 3], "0.8": [5, 4], "0.75": [6, 5, 6], "0.7": [7, 7, 7], "0.65": [7, 8, 7, 8], "0.6": [9, 7, 9, 10], "0.55": [9, 9, 10, 9, 9], "0.5": [10, 10, 9, 9, 10, 11], "0.45": [8, 10, 9, 10, 9, 11, 11], "0.4": [8, 8, 9, 8, 9, 9, 10], "0.35": [9, 8, 9, 9, 9, 9, 1], "0.3": [8, 8, 7, 8, 7, 4], "0.25": [6, 6, 7, 6, 6], "0.2": [5, 5, 6, 5], "0.15": [4, 4, 5, 5], "0.1": [0, 3, 3, 3], "0.05": [0, 2, 1], "0.0": [0]}


servo_3_dynamic_offsets = {}
sorted_tuple_list = sorted(servo_3_dynamic_offsets_raw.items())
for key_value in sorted_tuple_list:
    servo_3_dynamic_offsets[key_value[0]] = int(np.max(key_value[1]))

print(json.dumps(servo_3_dynamic_offsets))

x = []
y = []

for angle3_raw in servo_3_dynamic_offsets:
    angle3 = round(float(angle3_raw) - 0.5, 1) * np.pi
    x.append(angle3)
    offset = servo_3_dynamic_offsets[angle3_raw]
    y.append(offset)

plot_x = np.array(x)
plot_y = np.array(y)


def func(variables, a):
    a3 = variables
    return a * np.cos(a3)


fit, _ = curve_fit(func, plot_x, plot_y)
print(fit)

y_cos = fit[0] * cos(plot_x)

plt.plot(plot_x, plot_y, 'g')
plt.plot(plot_x, y_cos, 'y')
plt.show()
