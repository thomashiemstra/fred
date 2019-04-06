from __future__ import division
from math import ceil
from time import sleep
import numpy as np

from src import global_objects
from src.kinematics.kinematics import inverse_kinematics
from src.kinematics.kinematics_utils import Pose
import logging as log
from scipy.interpolate import splev, splrep, CubicSpline, splprep, interp1d, InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def line(start_pose, stop_pose, servo_controller, time):
    """go from start to stop pose in time amount of seconds"""
    flip = stop_pose.flip
    dx = stop_pose.x - start_pose.x
    dy = stop_pose.y - start_pose.y
    dz = stop_pose.z - start_pose.z
    d_alpha = stop_pose.alpha - start_pose.alpha
    d_beta = stop_pose.beta - start_pose.beta
    d_gamma = stop_pose.gamma - start_pose.gamma

    steps_per_second = 10
    total_steps = ceil(time * steps_per_second)  # 50 steps per second
    dt = 1.0 / steps_per_second

    for i in range(total_steps):
        t = i / total_steps

        curve_value = (6 * np.power(t, 5) - 15 * np.power(t, 4) + 10 * np.power(t, 3))
        x = start_pose.x + dx * curve_value
        y = start_pose.y + dy * curve_value
        z = start_pose.z + dz * curve_value
        r = np.sqrt(np.power(x, 2) + np.power(y, 2))

        alpha = start_pose.alpha + d_alpha * curve_value
        beta = start_pose.beta + d_beta * curve_value
        gamma = start_pose.gamma + d_gamma * curve_value

        z_adjust = r * 0.008 if i > 3 else 0
        temp_pose = Pose(x, y, z + z_adjust, flip, alpha, beta, gamma)

        servo_controller.move_to_pose(temp_pose)

        sleep(dt)

    return stop_pose


def pose_to_pose(start_pose, stop_pose, robot_config, servo_controller, time=None):
    start_angles = inverse_kinematics(start_pose, robot_config)
    stop_angles = inverse_kinematics(stop_pose, robot_config)
    if time is None:
        time = stop_pose.time

    angles_to_angles(start_angles, stop_angles, time, servo_controller)
    return stop_pose


def angles_to_angles(start_angles, stop_angles, time, servo_controller):
    """go from start to stop angles in time amount of seconds"""
    delta_angle = np.zeros(7, dtype=np.float64)

    for i in range(1, 7):
        delta_angle[i] = stop_angles[i] - start_angles[i]

    current_angles = start_angles.copy()

    steps = 10
    total_steps = ceil(time * steps)  # 50 steps per second
    dt = 1.0 / steps

    for i in range(total_steps + 1):
        t = i / total_steps

        curve_value = (6 * np.power(t, 5) - 15 * np.power(t, 4) + 10 * np.power(t, 3))
        for j in range(1, 7):
            current_angles[j] = start_angles[j] + delta_angle[j] * curve_value

        sleep(dt)
        servo_controller.move_servos(current_angles)


def get_curve_val(t):
    return 6 * np.power(t, 5) - 15 * np.power(t, 4) + 10 * np.power(t, 3)


def plot_curve(x, y, z):
    fig2 = plt.figure(2)
    ax3d = fig2.add_subplot(111, projection='3d')

    ax3d.plot(x, y, z, 'g*')
    fig2.show()
    plt.show()


# todo test this function!
def b_spline_curve(poses, time, servo_controller, workspace_limits=None, center=None, plot_first=False):
    """
    Move along a B-spline defined by the poses provided
    :param poses: array of Pose, knot points for the B-spline
    :param time: total time for the movement
    :param servo_controller:
    :param workspace_limits:
    :param plot_first:
    :return: final pose
    """
    if len(poses) < 2:
        log.warning("not enough poses")
        return

    k_val = min(len(poses) - 1, 3)

    x = [pose.x for pose in poses]
    y = [pose.y for pose in poses]
    z = [pose.z for pose in poses]

    # noinspection PyTupleAssignmentBalance
    tck, u = splprep([x, y, z], k=k_val, s=0)

    total_steps = ceil(time * global_objects.steps_per_second)
    dt = 1.0 / global_objects.steps_per_second
    lin = np.linspace(0, 1, total_steps)
    path_parameter = [get_curve_val(t) for t in lin]

    # todo check workspace limits if they are given
    x_steps, y_steps, z_steps = splev(path_parameter, tck)

    start_pose = poses[0]
    stop_pose = poses[-1]

    # todo change orientation to always face the center if it's not None
    d_alpha = stop_pose.alpha - start_pose.alpha
    d_beta = stop_pose.beta - start_pose.beta
    d_gamma = stop_pose.gamma - start_pose.gamma

    flip = stop_pose.flip

    if plot_first:
        plot_curve(x_steps, y_steps, z_steps)

    # todo what if the curve does not exactly starts at start_pose because of fitting?
    for i in range(total_steps):
        x = x_steps[i]
        y = y_steps[i]
        z = z_steps[i]

        alpha = start_pose.alpha + d_alpha * path_parameter[i]
        beta = start_pose.beta + d_beta * path_parameter[i]
        gamma = start_pose.gamma + d_gamma * path_parameter[i]

        temp_pose = Pose(x, y, z, flip, alpha, beta, gamma)

        servo_controller.move_to_pose(temp_pose)

        sleep(dt)

    return stop_pose

