from __future__ import division
from math import ceil
from time import sleep
import numpy as np

import src.global_constants
from src.kinematics.kinematics import inverse_kinematics
from src.kinematics.kinematics_utils import Pose
import logging as log
from scipy.interpolate import splev, splprep
from copy import copy

from src.utils.movement_exception import MovementException

def pose_to_pose(start_pose, stop_pose, servo_controller, time=None):
    robot_config = servo_controller.robot_config
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

    steps = src.global_constants.steps_per_second
    total_steps = ceil(time * steps)  # 50 steps per second
    dt = 1.0 / steps

    for i in range(total_steps + 1):
        t = i / total_steps

        curve_value = (6 * np.power(t, 5) - 15 * np.power(t, 4) + 10 * np.power(t, 3))
        for j in range(1, 7):
            current_angles[j] = start_angles[j] + delta_angle[j] * curve_value

        sleep(dt)
        servo_controller.move_servos(current_angles)


def from_current_angles_to_pose(pose, servo_controller, time):
    current_angles = servo_controller.get_current_angles()
    target_angles = servo_controller.pose_to_angles(pose)
    angles_to_angles(current_angles, target_angles, time, servo_controller)


def get_curve_val(t):
    return 6 * np.power(t, 5) - 15 * np.power(t, 4) + 10 * np.power(t, 3)


def plot_curve(x, y, z, poses):
    import matplotlib.pyplot as plt

    fig2 = plt.figure(2)
    ax3d = fig2.add_subplot(111, projection='3d')

    pose_x = [pose.x for pose in poses]
    pose_y = [pose.y for pose in poses]
    pose_z = [pose.z for pose in poses]

    ax3d.plot(pose_x, pose_y, pose_z, 'r*')
    ax3d.plot(x, y, z, 'g')
    fig2.show()
    plt.show()


def get_adjustments_and_stop_pose(start_pose, stop_pose, x_steps, y_steps, z_steps):
    x_start, y_start, z_start = start_pose.x, start_pose.y, start_pose.z

    dx = x_steps[0] - x_start
    dy = y_steps[0] - y_start
    dz = z_steps[0] - z_start

    actual_stop_pose = copy(stop_pose)
    actual_stop_pose.x = x_steps[-1] - dx
    actual_stop_pose.y = y_steps[-1] - dy
    actual_stop_pose.z = z_steps[-1] - dz

    return dx, dy, dz, actual_stop_pose


def b_spline_curve(poses, time, servo_controller, workspace_limits=None, center=None):
    """
    Move along a B-spline defined by the poses provided
    :param poses: array of Pose, knot points for the B-spline
    :param time: total time for the movement
    :param servo_controller:
    :param workspace_limits:
    :param center: [x, y, z] the end effector will always be oriented towards this center point
    :return: final pose
    """
    x_steps, y_steps, z_steps, total_steps, path_parameter = get_spline_step_arrays(poses, time)

    start_pose = poses[0]
    stop_pose = poses[-1]
    # If the curve does not exactly go through the start and stop pose we make sure it starts at start_pose
    # by shifting the spline and calculate where it actually ends
    dx, dy, dz, actual_stop_pose = get_adjustments_and_stop_pose(start_pose, stop_pose, x_steps, y_steps, z_steps)

    d_alpha, d_beta, d_gamma = get_delta_angles(start_pose, stop_pose)

    flip = stop_pose.flip

    if workspace_limits is not None and not check_workspace_limits(x_steps, y_steps, z_steps, total_steps, workspace_limits):
        raise MovementException('curve goes outside of workspace limits!')

    alpha, beta, gamma = start_pose.alpha, start_pose.beta, start_pose.gamma
    if center is not None:
        fix_initial_orientation(alpha, beta, center, gamma, servo_controller, start_pose)

    dt = 1.0 / src.global_constants.steps_per_second
    for i in range(total_steps):
        x = x_steps[i] - dx
        y = y_steps[i] - dy
        z = z_steps[i] - dz

        if center is not None:
            alpha, beta, gamma = get_angles_center(x, y, z, center)
        else:
            alpha, beta, gamma = get_angles_no_center(start_pose, d_alpha, d_beta, d_gamma, path_parameter[i])

        temp_pose = Pose(x, y, z, flip, alpha, beta, gamma)

        rec_time, time_taken = servo_controller.move_to_pose(temp_pose)

        print("rec time: {}, dt: {}".format(rec_time, dt))
        sleep_time = np.maximum(dt, rec_time)
        sleep(sleep_time)

    if center is not None:
        actual_stop_pose.alpha = alpha
        actual_stop_pose.beta = beta
        actual_stop_pose.gamma = gamma

    return actual_stop_pose


def b_spline_plot(poses):
    x_steps, y_steps, z_steps, total_steps, path_parameter = get_spline_step_arrays(poses, 1)
    plot_curve(x_steps, y_steps, z_steps, poses)


def b_spline_curve_calculate_only(poses, time, workspace_limits):
    x_steps, y_steps, z_steps, total_steps, path_parameter = get_spline_step_arrays(poses, time)
    if not check_workspace_limits(x_steps, y_steps, z_steps, total_steps, workspace_limits):
        raise MovementException('curve goes outside of workspace limits!')

    start_pose = poses[0]
    stop_pose = poses[-1]
    _, _, _, actual_stop_pose = get_adjustments_and_stop_pose(start_pose, stop_pose, x_steps, y_steps, z_steps)
    return actual_stop_pose


def get_spline_step_arrays(poses, time):
    if len(poses) < 2:
        raise ValueError("Not enough poses, need at least 2 for a b spline movement")

    k_val = min(len(poses) - 1, 3)

    x_poses = [pose.x for pose in poses]
    y_poses = [pose.y for pose in poses]
    z_poses = [pose.z for pose in poses]

    # noinspection PyTupleAssignmentBalance
    tck, u = splprep([x_poses, y_poses, z_poses], k=k_val, s=2)

    total_steps = ceil(time * src.global_constants.steps_per_second)
    lin = np.linspace(0, 1, total_steps)
    path_parameter = [get_curve_val(t) for t in lin]

    x_steps, y_steps, z_steps = splev(path_parameter, tck)

    return x_steps, y_steps, z_steps, total_steps, path_parameter


def check_workspace_limits(x_steps, y_steps, z_steps, total_steps, workspace_limits):
    if workspace_limits is None:
        return True
    for i in range(total_steps):
        r = np.sqrt(x_steps[i]*x_steps[i] + y_steps[i]*y_steps[i] + z_steps[i]*z_steps[i])
        if r < workspace_limits.radius_min or r > workspace_limits.radius_max:
            return False
        if z_steps[i] < workspace_limits.z_min:
            return False
    return True


def get_delta_angles(start_pose, stop_pose):
    d_alpha = stop_pose.alpha - start_pose.alpha
    d_beta = stop_pose.beta - start_pose.beta
    d_gamma = stop_pose.gamma - start_pose.gamma

    return d_alpha, d_beta, d_gamma


def get_angles_no_center(start_pose, d_alpha, d_beta, d_gamma, path_param):
    alpha = start_pose.alpha + d_alpha * path_param
    beta = start_pose.beta + d_beta * path_param
    gamma = start_pose.gamma + d_gamma * path_param

    return alpha, beta, gamma


def get_angles_center(x, y, z, center):
    dx = center[0] - x
    dy = center[1] - y
    dz = center[2] - z
    dr = np.sqrt(np.power(dx, 2) + np.power(dy, 2))
    alpha = -np.arctan2(dx, dy)
    gamma = np.arctan2(dz, dr)
    return alpha, 0, gamma


# If the start orientation is not the same as the orientation of the start pose
# we first move to the correct start orientation
def fix_initial_orientation(alpha, beta, center, gamma, servo_controller, start_pose):
    start_alpha, start_beta, start_gamma = get_angles_center(start_pose.x, start_pose.y, start_pose.z, center)
    if not (np.isclose(alpha, start_alpha) and np.isclose(beta, start_beta) and np.isclose(gamma, start_gamma)):
        adjusted_start_pose = copy(start_pose)
        adjusted_start_pose.alpha = start_alpha
        adjusted_start_pose.beta = start_beta
        adjusted_start_pose.gamma = start_gamma
        pose_to_pose(start_pose, adjusted_start_pose, servo_controller, 0.5)

