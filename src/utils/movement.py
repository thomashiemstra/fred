"""
Moving the arm: a move is planned completely, checked against the robot and only then executed, or not at all.

A PlannedMove is the angles of every step of a move at global_constants.steps_per_second. Three planners make one:
    plan_move_to_pose     a straight line of the end effector to a pose, its orientation turns with a slerp
    plan_move_to_angles   a minimum jerk interpolation in joint space
    plan_spline           a smooth curve through recorded poses (the xbox controller's recorded moves)
check_move checks every step against the robot: its workspace limits (robot.check_workspace), the ranges of its
servos (robot.get_angle_limits) and the joint speeds. execute_move sends the steps on an absolute schedule. The
robot's own move_servos checks nothing, so whoever moves the robot plans and checks first.

move_between_poses and move_from_current_angles plan a joint space move and execute it right away. The Movement
classes at the end are the recorded moves of the xbox controller.
"""
import json
import logging
from abc import ABC, abstractmethod
from copy import copy
from math import ceil
from time import perf_counter, sleep

import numpy as np
from scipy.interpolate import splev, splprep
from scipy.spatial.transform import Rotation, Slerp

import src.global_constants
from src.kinematics.kinematics import inverse_kinematics, forward_kinematics, forward_position_kinematics
from src.kinematics.kinematics_utils import Pose
from src.utils.movement_exception import MovementException

logger = logging.getLogger(__name__)

# how far the end effector is allowed to be from the planned position of a step, larger means it can't be reached
MAX_POSITION_ERROR = 0.1  # cm
# how far the first step of a movement is allowed to be from the current angles of the robot
MAX_START_ANGLE_ERROR = 0.05  # rad
# a joint at the limit of its range comes out of the inverse kinematics just past it because of rounding errors
ANGLE_LIMIT_TOLERANCE = 1e-6  # rad
# a move with a given duration that would turn a joint faster than recommended_max_servo_speed is slowed down,
# up to this factor; beyond it check_move refuses the move (a joint jumping in a single step, for example)
MAX_STRETCH = 4.0
# how far the robot may be from the target angles to count as already there
ALREADY_THERE = 0.2  # rad


def get_curve_val(t):
    """the progress (0 to 1) along a minimum jerk curve at the time fraction t (0 to 1)"""
    return 6 * np.power(t, 5) - 15 * np.power(t, 4) + 10 * np.power(t, 3)


def _get_curve_values(duration):
    """the progress (0 to 1) along a minimum jerk curve for every step of a movement, both ends included"""
    total_steps = max(ceil(duration * src.global_constants.steps_per_second), 1)
    return np.array([get_curve_val(i / total_steps) for i in range(total_steps + 1)])


class PlannedMove:
    """the angles of every step of a movement, see execute_move"""

    def __init__(self, description, angles, duration):
        self.description = description
        self.angles = angles  # array of shape (steps, 7), the angles of each step start at index 1
        self.duration = duration

    def __str__(self):
        return '{} in {:.1f}s ({} steps)'.format(self.description, self.duration, len(self.angles))


# checking and executing


def check_move(move, robot, current_angles=None):
    """
    check every step of a planned movement against the robot: the joint speeds, the ranges of the servos
    (robot.get_angle_limits(), nothing is checked when it returns None) and the workspace limits
    (robot.check_workspace)
    :param current_angles: if given, the first step has to match these angles
    :raises MovementException: if any step is outside the limits
    """
    if current_angles is not None:
        error = np.abs(move.angles[0][1:] - np.asarray(current_angles)[1:]).max()
        if error > MAX_START_ANGLE_ERROR:
            raise MovementException('the movement starts {:.2f}rad away from the current angles'.format(error))

    max_joint_speed = src.global_constants.recommended_max_servo_speed
    joint_speeds = _joint_speeds(move.angles)
    if len(joint_speeds) > 0 and joint_speeds.max() > max_joint_speed:
        step, joint = np.unravel_index(joint_speeds.argmax(), joint_speeds.shape)
        raise MovementException('joint {} would move at {:.1f}rad/s at step {}, the maximum is {}rad/s'
                                .format(joint + 1, joint_speeds.max(), step, max_joint_speed))

    angle_limits = robot.get_angle_limits()
    if angle_limits is not None:
        for joint, (min_angle, max_angle) in angle_limits.items():
            joint_angles = move.angles[:, joint]
            below = min_angle - joint_angles.min()
            above = joint_angles.max() - max_angle
            if max(below, above) > ANGLE_LIMIT_TOLERANCE:
                raise MovementException('joint {} would move {:.3g}rad {} its range [{:.2f}, {:.2f}]'.format(
                    joint, max(below, above), 'below' if below > above else 'above', min_angle, max_angle))

    for angles in move.angles:
        robot.check_workspace(angles)


def execute_move(robot, move: PlannedMove, should_stop=None):
    """
    send every step of a planned movement to the robot, at src.global_constants.steps_per_second
    :param move: the PlannedMove to be executed
    :param should_stop: optional function that is checked before every step, the movement stops when it returns True
    :return: True if the whole movement was executed, False if it was stopped
    """
    dt = 1.0 / src.global_constants.steps_per_second
    next_step = perf_counter()
    for angles in move.angles:
        if should_stop is not None and should_stop():
            return False
        robot.move_servos(angles)
        next_step += dt
        sleep(max(0.0, next_step - perf_counter()))
    return True


def _joint_speeds(angles):
    """the speed of every joint between consecutive steps, in rad/s"""
    return np.abs(np.diff(angles[:, 1:], axis=0)) * src.global_constants.steps_per_second


def _within_speed_limit(plan, duration):
    """
    the angles of plan(duration), with the duration stretched (up to MAX_STRETCH times) until no joint turns
    faster than recommended_max_servo_speed; beyond that the angles of the longest duration are returned and
    check_move refuses them
    :param plan: function of the duration returning the angles of every step
    :return: (angles, duration)
    """
    limit = src.global_constants.recommended_max_servo_speed
    requested = duration
    for _ in range(4):
        angles = plan(duration)
        speeds = _joint_speeds(angles)
        fastest = speeds.max() if len(speeds) else 0.0
        if fastest <= limit or duration >= requested * MAX_STRETCH:
            break
        duration = min(duration * fastest / limit * 1.02, requested * MAX_STRETCH)
    if duration > requested:
        logger.info('slowed a move of %.1fs down to %.1fs to keep the joints below %srad/s', requested, duration, limit)
    return angles, duration


# straight lines


def plan_move_to_pose(robot, current_angles, target_pose, linear_speed=5.0, angular_speed=0.5):
    """
    Plan a straight line from the current pose of the robot to the target pose.

    The wrist keeps its current flip, both flips reach the same pose. Changing the flip turns the wrist through the
    point where it is straight, which sweeps the end effector around the wrist, too far to do it safely in place.
    :param robot: the robot the move is for: its link lengths (robot_config), workspace limits (check_workspace)
        and the ranges of its servos (get_angle_limits, None when they are not known)
    :param current_angles: the current angles of the robot, starting at index 1
    :param target_pose: the pose to move to, its flip is ignored
    :param linear_speed: average speed of the end effector in cm/s
    :param angular_speed: average speed of the rotation of the end effector in rad/s
    :return: PlannedMove
    :raises MovementException: if any step of the movement is not possible
    """
    current_pose = forward_kinematics(current_angles, robot.robot_config)
    target_pose = copy(target_pose)
    target_pose.flip = current_pose.flip
    return plan_straight_line(robot, current_pose, target_pose, linear_speed, angular_speed, current_angles)


def plan_straight_line(robot, start_pose, stop_pose, linear_speed=5.0, angular_speed=0.5, current_angles=None):
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
    poses = [Pose(*positions[i], flip=stop_pose.flip, euler_matrix=orientations[i]) for i in range(len(curve_values))]

    description = 'move in a straight line from {} to {} ({:.1f}cm, {:.2f}rad rotation)'.format(
        np.round(start_position, 1), np.round(stop_position, 1), distance, rotation)
    move = PlannedMove(description, _angles_of_poses(robot, poses, 'the straight line'), duration)
    check_move(move, robot, current_angles)
    return move


def _angles_of_poses(robot, poses, what):
    """the inverse kinematics of every pose, checked to reach the position of the pose"""
    angles = np.zeros((len(poses), 7), dtype=np.float64)
    for i, pose in enumerate(poses):
        angles[i] = inverse_kinematics(pose, robot.robot_config)
        reached = forward_position_kinematics(angles[i], robot.robot_config)[-1]
        if np.linalg.norm(reached - [pose.x, pose.y, pose.z]) > MAX_POSITION_ERROR:
            raise MovementException('position {} of {} can not be reached'.format(
                np.round([pose.x, pose.y, pose.z], 1), what))
    return angles


# joint space


def plan_move_to_angles(robot, current_angles, target_angles, angular_speed=0.5, duration=None):
    """
    Plan a movement in joint space from the current angles to the target angles: every joint follows a minimum
    jerk profile, so the joints start and stop together. The end effector doesn't move in a straight line, but
    every step is checked like one (workspace, joint ranges, joint speed).
    :param robot: see plan_move_to_pose
    :param current_angles: the current angles of the robot, starting at index 1
    :param target_angles: the angles to move to, starting at index 1
    :param angular_speed: average speed of the joint that turns the most, in rad/s, when no duration is given;
        a move at a speed that is too fast is refused
    :param duration: the time of the move in seconds; it is stretched (see MAX_STRETCH) when a joint would turn
        too fast
    :return: PlannedMove
    :raises MovementException: if any step of the movement is not possible
    """
    current = np.asarray(current_angles, dtype=np.float64)
    target = np.asarray(target_angles, dtype=np.float64)
    largest = np.abs((target - current)[1:]).max()
    if duration is None:
        duration = max(largest / angular_speed, 1.0)
        angles = _joint_steps(current, target, duration)
    else:
        angles, duration = _within_speed_limit(lambda seconds: _joint_steps(current, target, seconds), duration)

    description = 'move the joints to {} degrees ({:.2f}rad at most)'.format(np.round(np.degrees(target[1:]), 1),
                                                                             largest)
    move = PlannedMove(description, angles, duration)
    check_move(move, robot, current_angles)
    return move


def _joint_steps(current, target, duration):
    """the angles of every step of a minimum jerk interpolation between the angle arrays, joint 0 stays"""
    delta = target - current
    delta[0] = 0.0
    return current + np.outer(_get_curve_values(duration), delta)


def move_between_poses(robot, start_pose, stop_pose, duration=None):
    """
    move in joint space from the start pose to the stop pose and return the stop pose, the robot has to be at the
    start pose
    :param duration: seconds, by default the time of the stop pose
    :raises MovementException: if the move is not possible
    """
    if duration is None:
        duration = stop_pose.time
    start_angles = inverse_kinematics(start_pose, robot.robot_config)
    stop_angles = inverse_kinematics(stop_pose, robot.robot_config)
    execute_move(robot, plan_move_to_angles(robot, start_angles, stop_angles, duration=duration))
    return stop_pose


def move_from_current_angles(robot, pose, duration):
    """
    move in joint space from the current angles of the robot to the pose, nothing happens when the robot is
    already within ALREADY_THERE of it
    :raises MovementException: if the move is not possible, or the current angles can't be read
    """
    current_angles = robot.get_current_angles()
    if current_angles is None:
        raise MovementException('failed to read the current angles')
    target_angles = robot.pose_to_angles(pose)
    if np.linalg.norm(current_angles - target_angles) < ALREADY_THERE:
        return
    execute_move(robot, plan_move_to_angles(robot, current_angles, target_angles, duration=duration))


# splines through recorded poses


def plan_spline(robot, poses, duration, center=None, s=None):
    """
    Plan a smooth curve of the end effector through the poses: a B-spline, timed with a minimum jerk profile,
    shifted to start exactly at the first pose. The orientation follows a spline through the orientations of the
    poses, or points at the center: then the robot first turns to point at it, within the same plan.
    :param duration: seconds for the curve, stretched (see MAX_STRETCH) when a joint would turn too fast
    :param center: [x, y, z] the end effector is kept pointed towards this point
    :param s: the smoothing factor of the spline, by default the number of poses plus the square root of twice that
    :return: (PlannedMove, the pose the curve actually ends at)
    :raises MovementException: if any step of the movement is not possible
    """
    angles, duration = _within_speed_limit(
        lambda seconds: _angles_of_poses(robot, spline_poses(poses, seconds, center, s)[0], 'the curve'), duration)
    _, actual_stop_pose = spline_poses(poses, duration, center, s)

    turn_duration = 0.0
    if center is not None:
        turn = _turn_towards_center(robot, poses[0], center)
        if turn is not None:
            turn_angles, turn_duration = turn
            angles = np.vstack([turn_angles[:-1], angles])

    description = 'move along a curve through {} poses from {} to {}'.format(
        len(poses), np.round([poses[0].x, poses[0].y, poses[0].z], 1),
        np.round([actual_stop_pose.x, actual_stop_pose.y, actual_stop_pose.z], 1))
    move = PlannedMove(description, angles, duration + turn_duration)
    check_move(move, robot)
    return move, actual_stop_pose


TURN_TOWARDS_CENTER = 0.5  # seconds, stretched when the wrist would turn too fast


def _turn_towards_center(robot, start_pose, center):
    """
    a joint space turn of the start pose towards the center, None when it points at it already
    :return: (the angles of every step, the duration in seconds)
    """
    alpha, beta, gamma = get_angles_center(start_pose.x, start_pose.y, start_pose.z, center)
    if np.allclose([start_pose.alpha, start_pose.beta, start_pose.gamma], [alpha, beta, gamma]):
        return None
    pointed = copy(start_pose)
    pointed.alpha, pointed.beta, pointed.gamma = alpha, beta, gamma
    start_angles = inverse_kinematics(start_pose, robot.robot_config)
    pointed_angles = inverse_kinematics(pointed, robot.robot_config)
    return _within_speed_limit(lambda seconds: _joint_steps(start_angles, pointed_angles, seconds),
                               TURN_TOWARDS_CENTER)


def spline_poses(poses, time, center=None, s=None):
    """
    The pose of every step along the B-spline through the poses, shifted so the curve starts at the first pose, and
    the pose the curve actually ends at.
    :param center: [x, y, z] the end effector is kept pointed towards this point, otherwise the orientation follows
        the spline through the orientations of the poses
    :param s: desired value for the smoothing factor s
    :return: (list of Pose per step, the actual stop pose)
    """
    x_steps, y_steps, z_steps, total_steps, alpha_steps, gamma_steps, _ = get_spline_step_arrays(poses, time, s)

    start_pose = poses[0]
    stop_pose = poses[-1]
    # If the curve does not exactly go through the start and stop pose we make sure it starts at start_pose
    # by shifting the spline and calculate where it actually ends
    dx, dy, dz, actual_stop_pose = get_adjustments_and_stop_pose(start_pose, stop_pose, x_steps, y_steps, z_steps)
    flip = stop_pose.flip

    steps = []
    for i in range(total_steps):
        x = x_steps[i] - dx
        y = y_steps[i] - dy
        z = z_steps[i] - dz
        if center is not None:
            alpha, beta, gamma = get_angles_center(x, y, z, center)
        else:
            alpha, beta, gamma = alpha_steps[i], 0, gamma_steps[i]
        steps.append(Pose(x, y, z, flip, alpha, beta, gamma))

    if center is not None and steps:
        actual_stop_pose.alpha = steps[-1].alpha
        actual_stop_pose.beta = steps[-1].beta
        actual_stop_pose.gamma = steps[-1].gamma
    return steps, actual_stop_pose


def get_spline_step_arrays(poses, time, s=None):
    data_points = len(poses)
    if data_points < 2:
        raise ValueError("Not enough poses, need at least 2 for a b spline movement")

    k_val = min(data_points - 1, 5)

    x_poses = [pose.x for pose in poses]
    y_poses = [pose.y for pose in poses]
    z_poses = [pose.z for pose in poses]
    alphas = [pose.alpha for pose in poses]
    gammas = [pose.gamma for pose in poses]

    if s is None:
        s = data_points + np.sqrt(2 * data_points)

    # noinspection PyTupleAssignmentBalance
    tck, u = splprep([x_poses, y_poses, z_poses, alphas, gammas], k=k_val, s=s)

    total_steps = ceil(time * src.global_constants.steps_per_second)
    lin = np.linspace(0, 1, total_steps)
    path_parameter = [get_curve_val(t) for t in lin]

    x_steps, y_steps, z_steps, alpha_steps, gamma_steps = splev(path_parameter, tck)

    return x_steps, y_steps, z_steps, total_steps, alpha_steps, gamma_steps, path_parameter


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


def get_angles_center(x, y, z, center):
    """the orientation (alpha, beta, gamma) that points the end effector at (x, y, z) towards the center"""
    dx = center[0] - x
    dy = center[1] - y
    dz = center[2] - z
    dr = np.sqrt(np.power(dx, 2) + np.power(dy, 2))
    alpha = -np.arctan2(dx, dy)
    gamma = np.arctan2(dz, dr)
    return alpha, 0, gamma


def b_spline_plot(poses, s=None):
    import matplotlib.pyplot as plt

    x_steps, y_steps, z_steps, _, _, _, _ = get_spline_step_arrays(poses, 1, s)
    fig2 = plt.figure(2)
    ax3d = fig2.add_subplot(111, projection='3d')
    ax3d.set_xlim3d(-40, 40)
    ax3d.set_ylim3d(0, 40)
    ax3d.set_zlim3d(0, 40)
    ax3d.plot([pose.x for pose in poses], [pose.y for pose in poses], [pose.z for pose in poses], 'r*')
    ax3d.plot(x_steps, y_steps, z_steps, 'g')
    fig2.show()
    plt.show()


# the recorded moves of the xbox controller


def convert_center_to_float(center):
    if center is None:
        return None
    return [float(x) for x in center]


class Movement(ABC):
    """a recorded move through poses, planned and checked against the robot when it is executed"""

    def __init__(self, poses, time, center=None) -> None:
        if len(poses) < 2:
            raise ValueError("at least 2 poses should be given in order to make a movement")
        self.poses = poses
        self.time = float(time)
        self.center = convert_center_to_float(center)

    def go_to_start_of_move(self, robot, time=None):
        if time is None:
            time = 4
        move_from_current_angles(robot, self.poses[0], time)

    def move(self, robot):
        if not self.is_robot_at_start_pose(self.poses[0], robot):
            raise MovementException("robot is not at the start pose, not executing move")
        return self._move_internal(self.poses, robot)

    def move_reversed(self, robot):
        if not self.is_robot_at_start_pose(self.poses[-1], robot):
            raise MovementException("robot is not at the start pose, not executing move")
        return self._move_internal(self.poses[::-1], robot)

    @abstractmethod
    def _move_internal(self, poses, robot):
        pass

    @abstractmethod
    def is_possible(self, robot):
        """whether the robot can execute the move: it can be planned and passes the checks of check_move"""
        pass

    def to_json(self):
        json_poses = [pose.to_json() for pose in self.poses]
        dump_dict = {'poses': json_poses}
        return json.dumps(dump_dict)

    @staticmethod
    def is_robot_at_start_pose(start_pose, robot):
        current_angles = robot.get_current_angles()
        start_pose_angles = robot.pose_to_angles(start_pose)
        return np.allclose(current_angles, start_pose_angles, atol=0.4)


class SplineMovement(Movement):

    def __init__(self, poses, time, center=None, s=None) -> None:
        super().__init__(poses, time, center)
        self._s = s

    def _move_internal(self, poses, robot):
        move, actual_stop_pose = plan_spline(robot, poses, self.time, self.center, self._s)
        execute_move(robot, move)
        return actual_stop_pose

    def is_possible(self, robot):
        try:
            plan_spline(robot, self.poses, self.time, self.center, self._s)
        except MovementException:
            return False
        return True


class PoseToPoseMovement(Movement):

    def __init__(self, poses, time, center=None) -> None:
        super().__init__(poses, time, center)
        if len(self.poses) > 2:
            logger.warning("more than 2 poses provided for a pose to pose movement, "
                           "using the first and the last pose given")

    def _move_internal(self, poses, robot):
        return move_between_poses(robot, poses[0], poses[-1], duration=self.time)

    def is_possible(self, robot):
        try:
            plan_move_to_angles(robot, robot.pose_to_angles(self.poses[0]), robot.pose_to_angles(self.poses[-1]),
                                duration=self.time)
        except MovementException:
            return False
        return True
