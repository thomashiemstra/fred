"""
The check whether a pose of the arm is inside its workspace limits, see global_constants.WorkSpaceLimits. Robots
expose it as AbstractRobotController.check_workspace(angles); whoever moves the robot runs it before executing a
move, move_servos itself doesn't.
"""
import numpy as np

from src.kinematics.kinematics import forward_position_kinematics
from src.utils.movement_exception import MovementException

# a pose on a limit comes out of the kinematics a rounding error past it, that is still inside
TOLERANCE = 1e-6  # cm


def workspace_violation(angles, robot_config, limits):
    """
    why the pose of the angles is outside the limits, or None when it is inside
    :param angles: the joint angles, indexed by servo id, in radians
    :param robot_config: the link lengths
    :param limits: a WorkSpaceLimits (class or object with its attributes), None checks nothing
    :return: the reason as text, or None
    """
    if limits is None:
        return None
    _, _, _, wrist, end_effector = forward_position_kinematics(angles, robot_config)

    radius = np.linalg.norm(end_effector)
    if radius < limits.radius_min - TOLERANCE or radius > limits.radius_max + TOLERANCE:
        return 'the end effector would move to {}, {:.1f}cm from the base, outside of [{}, {}]'.format(
            np.round(end_effector, 1), radius, limits.radius_min, limits.radius_max)
    if end_effector[2] < limits.z_min - TOLERANCE:
        return 'the end effector would move to {}, below the minimum height of {}'.format(
            np.round(end_effector, 1), limits.z_min)
    if end_effector[1] < limits.y_min - TOLERANCE:
        return 'the end effector would move to {}, closer than y={} to the base'.format(
            np.round(end_effector, 1), limits.y_min)

    wrist_z_min = getattr(limits, 'wrist_z_min', limits.z_min)
    if wrist[2] < wrist_z_min - TOLERANCE:
        return 'the wrist would move to {}, below the minimum height of {}'.format(np.round(wrist, 1), wrist_z_min)
    return None


def check_workspace(angles, robot_config, limits):
    """
    :raises MovementException: when the pose of the angles is outside the limits
    """
    violation = workspace_violation(angles, robot_config, limits)
    if violation is not None:
        raise MovementException(violation)
