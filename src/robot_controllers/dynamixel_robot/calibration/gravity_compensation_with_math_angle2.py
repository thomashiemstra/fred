import matplotlib.pyplot as plt
from numpy import cos, sin
import numpy as np
from numpy import pi
from scipy.optimize import curve_fit

servo_2_dynamic_offsets = {"0.45": {"0.45": 4, "0.4": 3, "0.35": 3, "0.3": 2, "0.25": 2, "0.2": 1, "0.15": 1, "0.1": 1, "0.05": 1, "0.0": 1, "-0.05": 1, "-0.1": 1, "-0.15": 1, "-0.2": 1, "-0.25": 1, "-0.3": 2, "-0.35": 1}, "0.4": {"0.45": 3, "0.4": 3, "0.35": 2, "0.3": 1, "0.25": 1, "0.2": 1, "0.15": 0, "0.1": -2, "0.05": -2, "0.0": -3, "-0.05": -2, "-0.1": -2, "-0.15": -2, "-0.2": -2, "-0.25": -2, "-0.3": -2, "-0.35": -2}, "0.3": {"0.45": 2, "0.4": -2, "0.35": -1, "0.3": -2, "0.25": -2, "0.2": -2, "0.15": -2, "0.1": -2, "0.05": -2, "0.0": -2, "-0.05": -2, "-0.1": -2, "-0.15": -1, "-0.2": -1, "-0.25": -1, "-0.3": -2}, "0.2": {"0.45": -1, "0.4": -2, "0.35": -2, "0.3": -2, "0.25": -2, "0.2": -2, "0.15": -2, "0.1": -2, "0.05": -2, "0.0": -2, "-0.05": -2, "-0.1": -2, "-0.15": -2}, "0.1": {"0.45": -2, "0.4": -2, "0.35": -2, "0.3": -2, "0.25": -2, "0.2": -2, "0.15": -3}, "0.05": {"0.45": -2, "0.4": -2, "0.35": -2, "0.3": -2, "0.25": -2}, "0.0": {"0.45": -2, "0.4": -3, "0.35": -2}}


x = []
y = []
z = []

for angle2_raw in servo_2_dynamic_offsets:
    angle2 = round(float(angle2_raw), 1) * pi
    for angle3_raw in servo_2_dynamic_offsets[angle2_raw]:
        angle3 = round(float(angle3_raw) - 0.5, 1) * pi
        offset = servo_2_dynamic_offsets[angle2_raw][angle3_raw]
        x.append(angle2)
        y.append(angle3)
        z.append(-offset)
        # print(angle2, angle2 + angle3, servo_2_dynamic_offsets[angle2_raw][angle3_raw])

x_axis = np.array(x)
y_axis = np.array(y)
z_axis = np.array(z)


def func(variables, a, b):
    a2, a3 = variables
    return a * np.cos(a2) + b * np.cos(a2 + a3)


fit, _ = curve_fit(func, (x_axis, y_axis), z_axis)
np.set_printoptions(precision=5, suppress=True)
print(fit)  # [-5.45690379 -4.13802897 -4.63319359  1.63811251]

z_fitted = fit[0] * cos(x_axis) + fit[1] * cos((x_axis + y_axis))

x_test = np.arange(0, 0.5 * np.pi)
y_test = np.arange(0, 0.5 * np.pi)
z_test = fit[0] * cos(x_test) + fit[1] * cos((x_test + y_test))


def f(x, y):
    return fit[0] * cos(x) + fit[1] * cos((x + y))


X, Y = np.meshgrid(x_axis, y_axis)
Z = f(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('angle_2')
ax.set_ylabel('angle_3')
ax.set_zlabel('offset')


# fig = plt.figure()
# ax = plt.axes(projection='3d')
ax.plot3D(x_axis, y_axis, z_axis, 'g')
# ax.plot3D(x_axis, y_axis, z_fitted, 'b')
# ax.set_xlabel('angle_2')
# ax.set_ylabel('angle_3')
# ax.set_zlabel('offset')

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
