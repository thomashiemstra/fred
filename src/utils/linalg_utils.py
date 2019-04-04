import numpy as np
import logging as log
from numpy import pi


# Unused code, but it will probably come in handy later
def get_rotation_matrix_params(start_pose, stop_pose, center):
    """
    We want the coordinate system for a plane spanned by (x,y,z) of the 2 poses and the center
    x will point from start_pose to stop_pose
    y will point in the direction of center
    z will follow from the right hand rule
    :param start_pose:
    :param stop_pose:
    :param center: array of (x,y,z) of the center
    :return:
    """
    # x is easy
    p_0 = np.array([start_pose.x, start_pose.y, start_pose.z])
    p_1 = np.array([stop_pose.x, stop_pose.y, stop_pose.z])

    p_0_p_1 = p_1 - p_0
    norm = np.linalg.norm(p_0_p_1)
    x = p_0_p_1 / norm

    # next we determine y and z with some cross product magic
    p_c = np.array(center)
    # the line pointing to the center, with this one and x we have the xy-plane defined
    p_0_p_c = p_c - p_0

    z = np.cross(x, p_0_p_c)
    norm = np.linalg.norm(z)
    if np.isclose(norm, 0.0):
        log.warning("all point are aligned, invalid input")
        return None
    z = z / norm

    y = np.cross(z, x)
    norm = np.linalg.norm(y)
    y = y / norm

    # and now we have an orthonormal basis from which we construct the rotation matrix
    # also return the vectors of p_0, p_1 and p_c expressed in this new coordinate frame
    return np.column_stack((x, y, z)), [0, 0, 0], p_0_p_1, p_0_p_c


def get_center(pose1, pose2):
    """
    Use two poses both at the same z-height to determine a point in space
    Used to have the camera always look at the same point while moving
    Both poses should point towards the positive y-axis, so -pi/2 <= alpha <= pi/2

    :param pose1: the first pose
    :param pose2: the second pose
    :return: array of x,y,z of the centre
    """
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

    a1, b1 = get_line_coefficients(alpha1, pose1.x, pose1.y)
    a2, b2 = get_line_coefficients(alpha2, pose2.x, pose2.y)

    if not (a1 is None or a2 is None):
        if np.isclose(a1, a2, 0.01):
            log.warning("The poses are parallel")
            return None

    if a1 is None and a2 is None:
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


def get_line_coefficients(alpha, x, y):
    """
    calculates y = a*x + b
    find the coefficients of the line passing through x,y with gradient determined by the angle alpha
    alpha should be between -pi/2 and pi/2 because of the constraint from get_centre()
    :param alpha: angle of the line
    :param x: x-coordinate
    :param y: 7-coordinate
    :return: a and b, returns none of the line is along the y-axis
    """
    temp = np.tan(alpha)
    if alpha >= pi / 2 or alpha <= -pi / 2:
        a = 0
    elif temp != 0:
        a = 1 / temp
    else:
        a = None  # the line points along the y axis
    b = y - a * x if a is not None else None

    return a, b
