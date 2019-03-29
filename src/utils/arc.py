import numpy as np
import logging as log
from numpy import pi


def arc(start_pose, stop_pose, center, robot_config, servo_controller):
    """
    move along an arc passing through the 2 poses with the given center
    :param start_pose:
    :param stop_pose:
    :param center:
    :param robot_config:
    :param servo_controller:
    :return:
    """






    # first get a and b
    x_c, y_c = center[0], center[1]
    x_0, y_0 = start_pose.x, start_pose.y
    x_1, y_1 = stop_pose.x, start_pose.y

    a, b = find_ellipse_radii([x_0, y_0], [x_1, y_1], [x_c, y_c])

    pass


def get_rotation_matrix_params(start_pose, stop_pose, center):
    """
    We want the coordinate system for a plane spanned by the 2 poses and the center
    x will point from start_pose to stop_pose
    y will point in the direction of center
    z will follow from the right hand rule
    :param start_pose:
    :param stop_pose:
    :param center:
    :return:
    """
    p_0 = np.array([start_pose.x, start_pose.y, start_pose.z])
    p_1 = np.array([stop_pose.x, stop_pose.y, stop_pose.z])
    p_c = np.array(center)

    x = p_0 - p_1
    x /= np.linalg.norm(x)


def get_parametric_parameter(a, b, x_c, y_c, x, y):
    """
    given an ellipse with x = a*cos(t) + x_c and y = b*sin(t) + y_c
    find the value for t for the point (x,y) on the ellipse
    """

    # first check if the point (x,y) actually lies on the ellipse
    res = (((x - x_c) ** 2) / a ** 2) + (((y - y_c) ** 2) / b ** 2)
    if not np.isclose(res, 1):
        log.warning("The point x,y does not lie on the ellipse")
        return None

    return np.arctan2((a * (y - y_c)), (b * (x - x_c)))


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
