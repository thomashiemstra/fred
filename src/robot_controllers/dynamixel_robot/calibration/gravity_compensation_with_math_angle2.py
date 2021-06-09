import matplotlib.pyplot as plt
from numpy import cos, sin
import numpy as np
from scipy.optimize import curve_fit

servo_2_dynamic_offsets = {"0.0": {"0.4": -10, "0.5": -10}, "0.1": {"0.2": -10, "0.3": -10, "0.4": -10, "0.5": -9}, "0.2": {"-0.1": -8, "0.0": -8, "0.1": -8, "0.2": -8, "0.3": -9, "0.4": -8, "0.5": -8}, "0.3": {"-0.3": -6, "-0.2": -6, "-0.1": -6, "0.0": -7, "0.1": -7, "0.2": -6, "0.3": -6, "0.4": -6, "0.5": -5}, "0.4": {"-0.3": -3, "-0.2": -3, "-0.1": -3, "-0.0": -3, "0.1": -3, "0.2": -3, "0.3": -3, "0.4": -3, "0.5": -3}, "0.5": {"-0.3": -2, "-0.2": -2, "-0.1": -2, "-0.0": -2, "0.1": -2, "0.2": -2, "0.3": -2, "0.4": -1, "0.5": 1}, "0.6": {"-0.3": -1, "-0.2": -2, "-0.1": -2, "-0.0": -1, "0.1": -2, "0.2": -1, "0.3": 1, "0.4": 1, "0.5": 2}, "0.7": {"-0.3": 2, "-0.2": 2, "-0.1": 1, "-0.0": 1, "0.1": 2, "0.2": 2, "0.3": 2, "0.4": 3, "0.5": 3}}

x = []
y = []
z = []

for angle2_raw in servo_2_dynamic_offsets:
    angle2 = round(float(angle2_raw), 1)
    for angle3_raw in servo_2_dynamic_offsets[angle2_raw]:
        angle3 = round(float(angle3_raw) - 0.5, 1)
        offset = servo_2_dynamic_offsets[angle2_raw][angle3_raw]
        x.append(angle2)
        y.append(angle3)
        z.append(offset)
        print(angle2, angle3, servo_2_dynamic_offsets[angle2_raw][angle3_raw])

x_axis = np.array(x)
y_axis = np.array(y)
z_axis = np.array(z)


def func(variables, a, b, c, d):
    a2, a3 = variables
    return a * np.cos(b * a2) + c * np.cos(d * (a2 + a3))


fit, _ = curve_fit(func, (x_axis, y_axis), z_axis)
print(fit)  # [-5.45690379 -4.13802897 -4.63319359  1.63811251]

z_fitted = fit[0] * cos(fit[1] * x_axis) + fit[2] * cos(fit[3] * (x_axis + y_axis))

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(x_axis, y_axis, z_axis, 'g')
ax.plot3D(x_axis, y_axis, z_fitted, 'b')
ax.set_xlabel('angle_2')
ax.set_ylabel('angle_3')
ax.set_zlabel('offset')

plt.show()

# plot_x = np.array(x)
# plot_y = np.array(y)
#
# a = 1
# b = 1
# c = 1
# d = 1
# a2 = a3 = 1
#
#
# y_cos = a * cos(b * a2) + d * cos(c * (a2 + a3))
#
# plt.plot(plot_x, plot_y, 'g')
# # plt.plot(plot_x, y_cos, 'y')
# plt.show()
