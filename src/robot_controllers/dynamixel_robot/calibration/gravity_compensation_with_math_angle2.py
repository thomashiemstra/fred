import matplotlib.pyplot as plt
from numpy import cos, sin
import numpy as np
from numpy import pi
from scipy.optimize import curve_fit

servo_2_dynamic_offsets = {"0.45": {"0.45": -2, "0.4": -2, "0.35": -2, "0.3": -2, "0.25": -2, "0.2": -2, "0.15": -2, "0.1": -2, "0.05": -2, "0.0": -2, "-0.05": -2, "-0.1": -2, "-0.15": -2, "-0.2": -2, "-0.25": -2, "-0.3": -2, "-0.35": -2}, "0.4": {"0.45": -2, "0.4": -2, "0.35": -2, "0.3": -2, "0.25": -2, "0.2": -2, "0.15": -2, "0.1": -2, "0.05": -2, "0.0": -2, "-0.05": -2, "-0.1": -2, "-0.15": -2, "-0.2": -2, "-0.25": -2, "-0.3": -2, "-0.35": -2}, "0.3": {"0.45": -3, "0.4": -3, "0.35": -3, "0.3": -3, "0.25": -3, "0.2": -3, "0.15": -3, "0.1": -3, "0.05": -3, "0.0": -3, "-0.05": -3, "-0.1": -3, "-0.15": -3, "-0.2": -3, "-0.25": -3, "-0.3": -3}, "0.2": {"0.45": -3, "0.4": -3, "0.35": -3, "0.3": -3, "0.25": -3, "0.2": -3, "0.15": -3, "0.1": -3, "0.05": -3, "0.0": -3, "-0.05": -3, "-0.1": -3, "-0.15": -3}, "0.05": {"0.45": -3, "0.4": -3, "0.35": -3, "0.3": -3, "0.25": -3}, "0.0": {"0.45": -3, "0.4": -3, "0.35": -3}}

x = []
y = []
z = []

for angle2_raw in servo_2_dynamic_offsets:
    angle2 = round(float(angle2_raw), 1) * pi
    for angle3_raw in servo_2_dynamic_offsets[angle2_raw]:
        angle3 = round(float(angle3_raw) - 0.5, 1) * pi
        offset = servo_2_dynamic_offsets[angle2_raw][angle3_raw]
        x.append(angle2)
        y.append(angle2 + angle3)
        z.append(offset)
        print(angle2, angle2 + angle3, servo_2_dynamic_offsets[angle2_raw][angle3_raw])

x_axis = np.array(x)
y_axis = np.array(y)
z_axis = np.array(z)


def func(variables, a, b, c, d):
    a2, a23 = variables
    return a * np.cos(b * a2) + c * np.cos(d * a23)


fit, _ = curve_fit(func, (x_axis, y_axis), z_axis)
np.set_printoptions(precision=5, suppress=True)
print(fit)  # [-5.45690379 -4.13802897 -4.63319359  1.63811251]

z_fitted = fit[0] * cos(fit[1] * x_axis) + fit[2] * cos(fit[3] * y_axis)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(x_axis, y_axis, z_axis, 'g')
ax.plot3D(x_axis, y_axis, z_fitted, 'b')
ax.set_xlabel('angle_2')
ax.set_ylabel('angle_2 + angle_3')
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
