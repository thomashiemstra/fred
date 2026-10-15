"""
Movements in cartesian space: the end effector moves in a straight line while its orientation turns smoothly.

A movement is planned completely before anything moves, every step is checked against the workspace limits, the
angle limits of the servos and the joint speeds, so a movement either is executed completely or not at all.
"""
from copy import copy
from math import ceil
from time import perf_counter, sleep

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

import src.global_constants
from src.kinematics.kinematics import inverse_kinematics, forward_kinematics, forward_position_kinematics
from src.kinematics.kinematics_utils import Pose
from src.utils.movement_exception import MovementException
from src.utils.movement_utils import get_curve_val

# how far the end effector is allowed to be from the planned straight line, larger means it can't be reached
MAX_POSITION_ERROR = 0.1  # cm
# how far the first step of a movement is allowed to be from the current angles of the robot
MAX_START_ANGLE_ERROR = 0.05  # rad
# a joint at the limit of its range comes out of the inverse kinematics just past it because of rounding errors
ANGLE_LIMIT_TOLERANCE = 1e-6  # rad


class PlannedMove:
    """the angles of every step of a movement, see execute_move"""

    def __init__(self, description, angles, duration):
        self.description = description
        self.angles = angles  # array of shape (steps, 7), the angles of each step start at index 1
        self.duration = duration

    def __str__(self):
        return '{} in {:.1f}s ({} steps)'.format(self.description, self.duration, len(self.angles))


def plan_move_to_pose(current_angles, target_pose, robot_config, workspace_limits=src.global_constants.WorkSpaceLimits,
                      angle_limits=None, linear_speed=5.0, angular_speed=0.5):
    """
    Plan a straight line from the current pose of the robot to the target pose.

    The wrist keeps its current flip, both flips reach the same pose. Changing the flip turns the wrist through the
    point where it is straight, which sweeps the end effector around the wrist, too far to do it safely in place.
    :param current_angles: the current angles of the robot, starting at index 1
    :param target_pose: the pose to move to, its flip is ignored
    :param robot_config: link lengths
    :param workspace_limits: the end effector and the wrist have to stay within these limits
    :param angle_limits: mapping of joint number to (min angle, max angle) in radians
    :param linear_speed: average speed of the end effector in cm/s
    :param angular_speed: average speed of the rotation of the end effector in rad/s
    :return: PlannedMove
    :raises MovementException: if any step of the movement is not possible
    """
    current_pose = forward_kinematics(current_angles, robot_config)
    target_pose = copy(target_pose)
    target_pose.flip = current_pose.flip
    return plan_straight_line(current_pose, target_pose, robot_config, workspace_limits, angle_limits,
                              linear_speed, angular_speed, current_angles)


def plan_straight_line(start_pose, stop_pose, robot_config, workspace_limits=src.global_constants.WorkSpaceLimits,
                       angle_limits=None, linear_speed=5.0, angular_speed=0.5, current_angles=None):
    """
    Plan a straight line of the end effector from the start to the stop pose, the orientation is interpolated
    with a slerp. The poses have to be flipped the same way, otherwise the wrist would turn around halfway.
    :param current_angles: if given, the first step has to match these angles
    :raises MovementException: if any step of the movement is not possible
    """
    if start_pose.flip != stop_pose.flip:
        raise MovementException('the start and stop pose are flipped differently')

    start_position = np.array([start_pose.x, start_pose.y, start_pose.z])
    stop_position = np.array([stop_pose.x, stop_pose.y, stop_pose.z])
    rotations = Rotation.from_matrix([start_pose.get_euler_matrix(), stop_pose.get_euler_matrix()])
    slerp = Slerp([0.0, 1.0], rotations)

    distance = np.linalg.norm(stop_position - start_position)
    rotation = (rotations[0].inv() * rotations[1]).magnitude()
    duration = max(distance / linear_speed, rotation / angular_speed, 1.0)

    curve_values = _get_curve_values(duration)
    positions = start_position + np.outer(curve_values, stop_position - start_position)
    orientations = slerp(curve_values).as_matrix()

    angles = np.zeros((len(curve_values), 7), dtype=np.float64)
    for i in range(len(curve_values)):
        pose = Pose(*positions[i], flip=stop_pose.flip, euler_matrix=orientations[i])
        angles[i] = inverse_kinematics(pose, robot_config)
        reached = forward_position_kinematics(angles[i], robot_config)[-1]
        if np.linalg.norm(reached - positions[i]) > MAX_POSITION_ERROR:
            raise MovementException('position {} of the straight line can not be reached'
                                    .format(np.round(positions[i], 1)))

    description = 'move in a straight line from {} to {} ({:.1f}cm, {:.2f}rad rotation)'.format(
        np.round(start_position, 1), np.round(stop_position, 1), distance, rotation)
    move = PlannedMove(description, angles, duration)
    check_move(move, robot_config, workspace_limits, angle_limits, current_angles)
    return move


def check_move(move, robot_config, workspace_limits=None, angle_limits=None, current_angles=None):
    """
    check every step of a planned movement
    :raises MovementException: if any step is outside the limits
    """
    if current_angles is not None:
        error = np.abs(move.angles[0][1:] - np.asarray(current_angles)[1:]).max()
        if error > MAX_START_ANGLE_ERROR:
            raise MovementException('the movement starts {:.2f}rad away from the current angles'.format(error))

    max_joint_speed = src.global_constants.recommended_max_servo_speed
    joint_speeds = np.abs(np.diff(move.angles[:, 1:], axis=0)) * src.global_constants.steps_per_second
    if len(joint_speeds) > 0 and joint_speeds.max() > max_joint_speed:
        step, joint = np.unravel_index(joint_speeds.argmax(), joint_speeds.shape)
        raise MovementException('joint {} would move at {:.1f}rad/s at step {}, the maximum is {}rad/s'
                                .format(joint + 1, joint_speeds.max(), step, max_joint_speed))

    if angle_limits is not None:
        for joint, (min_angle, max_angle) in angle_limits.items():
            joint_angles = move.angles[:, joint]
            below = min_angle - joint_angles.min()
            above = joint_angles.max() - max_angle
            if max(below, above) > ANGLE_LIMIT_TOLERANCE:
                raise MovementException('joint {} would move {:.3g}rad {} its range [{:.2f}, {:.2f}]'.format(
                    joint, max(below, above), 'below' if below > above else 'above', min_angle, max_angle))

    if workspace_limits is not None:
        for angles in move.angles:
            _, _, _, wrist, end_effector = forward_position_kinematics(angles, robot_config)
            _check_workspace_limits(end_effector, workspace_limits, 'the end effector')
            if wrist[2] < workspace_limits.z_min:
                raise MovementException('the wrist would move to {}, below the minimum height of {}'
                                        .format(np.round(wrist, 1), workspace_limits.z_min))


def execute_move(robot, move):
    """send every step of a planned movement to the robot, at src.global_constants.steps_per_second"""
    dt = 1.0 / src.global_constants.steps_per_second
    next_step = perf_counter()
    for angles in move.angles:
        robot.move_servos(angles)
        next_step += dt
        sleep(max(0.0, next_step - perf_counter()))


def _get_curve_values(duration):
    """the progress (0 to 1) along a minimum jerk curve for every step of a movement"""
    total_steps = max(ceil(duration * src.global_constants.steps_per_second), 1)
    return np.array([get_curve_val(i / total_steps) for i in range(total_steps + 1)])


def _check_workspace_limits(position, workspace_limits, name):
    radius = np.linalg.norm(position)
    if radius < workspace_limits.radius_min or radius > workspace_limits.radius_max:
        raise MovementException('{} would move to {}, {:.1f}cm from the base, outside of [{}, {}]'.format(
            name, np.round(position, 1), radius, workspace_limits.radius_min, workspace_limits.radius_max))
    if position[2] < workspace_limits.z_min:
        raise MovementException('{} would move to {}, below the minimum height of {}'.format(
            name, np.round(position, 1), workspace_limits.z_min))
    if position[1] < workspace_limits.y_min:
        raise MovementException('{} would move to {}, closer than y={} to the base'.format(
            name, np.round(position, 1), workspace_limits.y_min))
