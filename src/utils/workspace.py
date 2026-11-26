"""
The check whether a pose of the arm is inside its workspace limits, see global_constants.WorkSpaceLimits. Robots
expose it as AbstractRobotController.check_workspace(angles); whoever moves the robot runs it before executing a
move, move_servos itself doesn't. The checks are the reach of the wrist from the shoulder, the distance of the wrist
from the base axis (wrist_radius_min), and the radius, heights and y of the gripper tip and the wrist.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.kinematics.kinematics import forward_position_kinematics
from src.utils.movement_exception import MovementException

if TYPE_CHECKING:
    from src.global_constants import WorkSpaceLimits
    from src.kinematics.kinematics_utils import RobotConfig

# a pose on a limit comes out of the kinematics a rounding error past it, that is still inside
TOLERANCE = 1e-6  # cm


def workspace_violation(angles: np.ndarray, robot_config: RobotConfig,
                        limits: type[WorkSpaceLimits] | WorkSpaceLimits | None) -> str | None:
    """
    why the pose of the angles is outside the limits, or None when it is inside
    :param angles: the joint angles, indexed by servo id, in radians
    :param robot_config: the link lengths
    :param limits: a WorkSpaceLimits (class or object with its attributes), None checks nothing
    :return: the reason as text, or None
    """
    if limits is None:
        return None
    shoulder, _, _, wrist, end_effector = forward_position_kinematics(angles, robot_config)

    reach = np.linalg.norm(wrist - shoulder)
    if reach > limits.wrist_reach_max + TOLERANCE:
        return 'the wrist would move to {}, {:.1f}cm from the shoulder, farther than {}'.format(  # type: ignore[str-format]  # numpy float is fine for :.1f
            np.round(wrist, 1), reach, limits.wrist_reach_max)
    # limits objects of tests and scripts may not have the minimum of the wrist radius, they check nothing for it
    wrist_radius_min = getattr(limits, 'wrist_radius_min', None)
    if wrist_radius_min is not None:
        wrist_radius = np.hypot(wrist[0], wrist[1])
        if wrist_radius < wrist_radius_min - TOLERANCE:
            return 'the wrist would move to {}, {:.1f}cm from the base, closer than {}'.format(
                np.round(wrist, 1), wrist_radius, wrist_radius_min)
    radius = np.linalg.norm(end_effector)
    if radius < limits.radius_min - TOLERANCE:
        return 'the end effector would move to {}, {:.1f}cm from the base, closer than {}'.format(  # type: ignore[str-format]  # numpy float is fine for :.1f
            np.round(end_effector, 1), radius, limits.radius_min)
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


def check_workspace(angles: np.ndarray, robot_config: RobotConfig,
                    limits: type[WorkSpaceLimits] | WorkSpaceLimits | None) -> None:
    """
    :raises MovementException: when the pose of the angles is outside the limits
    """
    violation = workspace_violation(angles, robot_config, limits)
    if violation is not None:
        raise MovementException(violation)
