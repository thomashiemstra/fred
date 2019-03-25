from __future__ import division
from math import ceil
from time import sleep
import numpy as np
from numpy import pi
from src.kinematics.kinematics import inverse_kinematics
from src.kinematics.kinematics_utils import Pose
import logging as log


def line(start_pose, stop_pose, robot_config, servo_controller):
    """go from start to stop pose in time amount of seconds"""
    flip = stop_pose.flip
    dx = stop_pose.x - start_pose.x
    dy = stop_pose.y - start_pose.y
    dz = stop_pose.z - start_pose.z
    d_alpha = stop_pose.alpha - start_pose.alpha
    d_beta = stop_pose.beta - start_pose.beta
    d_gamma = stop_pose.gamma - start_pose.gamma
    time = stop_pose.time

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

        current_angles = inverse_kinematics(temp_pose, robot_config)
        servo_controller.move_servos(current_angles)

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


# Given 2 poses both on the same level and with orientations only in the x-y plane
# find the intersection of the 2 poses
def get_centre(pose1, pose2):
    tolerance = 0.1
    if not np.isclose(pose1.z, pose2.z, tolerance):
        log.warning("z's do not match")
        return None

    if pose1.gamma > tolerance or pose1.beta > tolerance:
        log.warning("pose1 is not orientated in the x-y plane")
        return None

    if pose2.gamma > tolerance or pose2.beta > tolerance:
        log.warning("pose2 is not orientated in the x-y plane")
        return None

    alpha1 = pose1.alpha
    alpha2 = pose2.alpha

    a1 = 1/np.tan(alpha1)
    b1 = pose1.y - a1*pose1.x

    a2 = 1/np.tan(alpha2)
    b2 = pose2.y - a2 * pose2.x

    if np.isclose(a1, a2, 0.01):
        log.warning("The poses are parallel")
        return None

    if np.isclose(alpha1, 0, 0.01):
        x_center = pose1.x
        y_center = b2
    elif np.isclose(alpha2, 0, 0.01):
        x_center = pose2.x
        y_center = b1
    else:
        x_center = (b2 - b1) / (a1 - a2)
        y_center = a1*x_center + b1
    z_center = pose1.z

    return np.array([x_center, y_center, z_center])


# Move along an ellipse
def arc(start_pose, stop_pose, center, robot_config, servo_controller):
    pass


def find_ellipse_radii(first_point, second_point, centre):
    """
    :param first_point: array of (x_0, y_0)
    :param second_point: array of (x_1, y_1)
    :param centre: array of (x_c, y_c)
    :return: the 2 radii a and b or None if you gave nonsense input
    """
    x_0, y_0 = first_point[0], first_point[1]
    x_1, y_1 = second_point[0], second_point[1]
    x_c, y_c = centre[0], centre[1]

    if np.isclose(x_0, x_1) and np.isclose(y_1, y_1):
        log.warning("the same point is supplied twice")
        return 0, 0

    # both points lined up in the y-direction
    # since the centre should be in front of the robot, this make no sense
    if np.isclose(x_0, x_1):
        log.warning("Both points have the same x coordinate, go in a line instead")
        return 0, 0

    # subtract centres from coordinates
    x_0_p = x_0 - x_c
    x_1_p = x_1 - x_c
    y_0_p = y_0 - y_c
    y_1_p = y_1 - y_c

    # both points lined up in the x direction
    # it should be a circle
    if np.isclose(y_0, y_1):
        log.warning("Both points have the same y, "
                    "assuming this should be a circle, taking the first point for the radius")
        r = np.sqrt(np.power(x_0_p, 2) + np.power(y_0_p, 2))
        return r, r

    # Solve a_matrix.X = b_matrix where X = [A, B] with A = 1/a^2 and B = 1/b^2
    a_matrix = np.array([[x_0_p ** 2, y_0_p ** 2],
                        [x_1_p ** 2, y_1_p ** 2]])
    b_matrix = np.array([1, 1])

    # singular probably means we want a circle
    if is_singular(a_matrix):
        log.warning("singular matrix found, impossible combination supplied: ", first_point, second_point, centre)
        return 0, 0

    res = np.linalg.solve(a_matrix, b_matrix)
    a = 1 / np.sqrt(res[0])
    b = 1 / np.sqrt(res[1])

    return a, b


def is_singular(a):
    return not (a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0])
