import json

import matplotlib.pyplot as plt
from numpy import cos
import numpy as np
from scipy.optimize import curve_fit

servo_3_dynamic_offsets_raw = {"0.9": 2, "0.85": 3, "0.8": 3, "0.75": 3, "0.7": 4, "0.65": 5, "0.6": 5, "0.55": 5, "0.5": 6, "0.45": 6, "0.4": 5, "0.35": 5, "0.3": 5, "0.25": 4, "0.2": 3, "0.15": 2, "0.1": 1, "0.05": 1, "0.0": 0}

servo_3_dynamic_offsets = {}
sorted_tuple_list = sorted(servo_3_dynamic_offsets_raw.items())
for key_value in sorted_tuple_list:
    servo_3_dynamic_offsets[key_value[0]] = key_value[1]

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


def func(variables, a, b):
    a3 = variables
    return a * np.cos(b * a3)


fit, _ = curve_fit(func, plot_x, plot_y)
print(fit)  # [13.50604571  3.15347015]

y_cos = fit[0] * cos(fit[1] * plot_x)

plt.plot(plot_x, plot_y, 'g')
plt.plot(plot_x, y_cos, 'y')
plt.show()
