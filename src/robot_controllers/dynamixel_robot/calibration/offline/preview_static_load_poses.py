"""
Lists the poses record_static_load.py will visit and how close the arm comes to the table at each of them, without
the robot: nothing here opens a serial port, whatever global_constants.use_simulation is. Run it from the project
root with the options of the recorder, the recorder's command line can be pasted as it is:
    python -m src.robot_controllers.dynamixel_robot.calibration.offline.preview_static_load_poses --poses 30
    --joints 2 3 4 5 --range 2=45:115 --seed 1 [--warn 3.0]
For the same options and the same --seed these are exactly the poses the recorder visits, in the same order (the code
from the options to the poses is the recorder's own, record_static_load.select_poses). Without --seed the poses of a
later recording differ from these. The options that only matter while recording (--settle, --readings, --speed,
--output, --yes) are accepted and ignored.

One line per pose: the angles of joints 1 to 6 in degrees, and the height in cm above the table of the wrist centre
and of the gripper tip at the pose, at the overshoot above it (+delta, where the recorder approaches the pose from,
going down) and at the overshoot below it (-delta). A pose where the tip or the wrist comes below --warn cm at any of
the three is marked LOW. The summary has the lowest tip, wrist and elbow of all the poses and which pose and
overshoot they belong to. The planner (and so the recorder's choice of poses) checks the wrist and the tip against
the floor, not the elbow or the gripper fingers, so these heights are what to look at before a recording.

The planner's answers come from an OfflineRobot, an AbstractRobotController without hardware: its angle limits are
those of servo_config.yaml, its link lengths and workspace limits those of the real robot, so a pose is accepted here
exactly when it is accepted on the arm. The return code is 0 when poses were listed, 1 when no pose is accepted or
the options are wrong.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.global_constants import dynamixel_robot_config
from src.kinematics.kinematics import forward_position_kinematics, inverse_kinematics
from src.kinematics.kinematics_utils import Pose, RobotConfig
from src.robot_controllers.abstract_robot_controller import AbstractRobotController
from src.robot_controllers.dynamixel_robot.calibration import record_static_load
from src.robot_controllers.dynamixel_robot.calibration.load_directions import DOWN
from src.robot_controllers.dynamixel_robot.dynamixel_robot_controller import ARM_IDS
from src.robot_controllers.dynamixel_robot.servo_config import ServoConfig, load_servo_config

# cm: a tip or wrist lower than this at any of a pose's three stops is marked. 1 cm above the limit of the tip
# (WorkSpaceLimits.z_min, 1 cm), so the marked poses are the ones that come within 1 cm of what the planner refuses
DEFAULT_WARN = 2.0
# the three stops at every pose, in the order of the columns
STOPS = ('at the pose', 'at the overshoot above (+delta)', 'at the overshoot below (-delta)')
# the indices into the points forward_position_kinematics returns: shoulder, elbow, forearm middle, wrist, tip
ELBOW, WRIST, TIP = 1, 3, 4


@dataclass
class PoseHeights:
    """a pose and the heights in cm above the table of the elbow, the wrist centre and the gripper tip at its three
    stops (see STOPS)"""
    number: int  # from 1, as the recorder numbers its poses
    angles: np.ndarray  # 7 elements, in radians, the first is not used
    elbow: tuple[float, float, float]
    wrist: tuple[float, float, float]
    tip: tuple[float, float, float]

    @property
    def lowest(self) -> float:
        """the lowest of the wrist and tip heights, the ones the planner checks"""
        return min(*self.wrist, *self.tip)


class NoHardwareError(RuntimeError):
    """raised by the OfflineRobot when something asks it to move or read the arm"""

    def __init__(self, what: str) -> None:
        super().__init__('{} needs the robot, this one has no hardware'.format(what))


class OfflineRobot(AbstractRobotController):
    """
    what the planner asks of the robot, without the robot: check_move uses get_angle_limits, robot_config and
    workspace_limits (the checks of AbstractRobotController.check_workspace are the same code as on the arm), and none
    of these touches a servo. It is not a DynamixelRobotController: that one writes to the servos when it is built.
    Every method that would move or read the arm raises, so a use of this class that needs the hardware fails
    instead of doing something.
    """

    def __init__(self, robot_config: RobotConfig = dynamixel_robot_config,
                 servo_config: ServoConfig | None = None) -> None:
        """
        :param robot_config: the link lengths, by default those of the real robot
        :param servo_config: where the joint ranges come from, by default servo_config.yaml as the real robot
        :raises ServoConfigError: when servo_config.yaml is not valid
        """
        self.robot_config = robot_config
        self.servo_config = servo_config if servo_config is not None else load_servo_config()

    def get_angle_limits(self) -> dict[int, tuple[float, float]]:
        """the (min angle, max angle) of the arm servos, as DynamixelRobotController.get_angle_limits"""
        limits = {}
        for servo_id in ARM_IDS:
            servo = self.servo_config.servos[servo_id]
            limits[servo_id] = (min(servo.min_angle, servo.max_angle), max(servo.min_angle, servo.max_angle))
        return limits

    def pose_to_angles(self, pose: Pose) -> np.ndarray:
        return inverse_kinematics(pose, self.robot_config)

    def enable_servos(self) -> bool | None:
        raise NoHardwareError('enable_servos')

    def disable_servos(self) -> bool | None:
        raise NoHardwareError('disable_servos')

    def move_to_pose(self, pose: Pose) -> float | None:
        raise NoHardwareError('move_to_pose')

    def move_to_pose_and_give_new_angles(self, pose: Pose) -> np.ndarray:
        raise NoHardwareError('move_to_pose_and_give_new_angles')

    def move_servos(self, angles: np.ndarray, velocities: np.ndarray | None = None) -> float | None:
        raise NoHardwareError('move_servos')

    def get_current_angles(self) -> np.ndarray | None:
        raise NoHardwareError('get_current_angles')

    def set_gripper(self, new_gripper_state: float) -> bool | None:
        raise NoHardwareError('set_gripper')

    def reset_to_pose(self, pose: Pose) -> None:
        raise NoHardwareError('reset_to_pose')


def pose_heights(number: int, pose: np.ndarray, delta: float, joints: Sequence[int],
                 angle_limits: dict[int, tuple[float, float]], robot_config: RobotConfig = dynamixel_robot_config
                 ) -> PoseHeights:
    """
    the heights at a pose and at the overshoots the recorder approaches it from, which are the recorder's own
    (record_static_load.approach_sequence, clipped to the joint ranges)
    :param number: the number of the pose, from 1
    :param pose: the 7 element angle array, in radians
    :param delta: the overshoot in radians
    :param joints: the varied joints
    :param angle_limits: mapping of joint to (min angle, max angle) in radians, the joint ranges of the robot
    """
    stops = {'pose': pose}
    for overshoot, _, direction in record_static_load.approach_sequence(pose, delta, joints, angle_limits):
        stops['above' if direction == DOWN else 'below'] = overshoot
    points = [forward_position_kinematics(stops[name], robot_config) for name in ('pose', 'above', 'below')]

    def heights(index: int) -> tuple[float, float, float]:
        return float(points[0][index][2]), float(points[1][index][2]), float(points[2][index][2])
    return PoseHeights(number, pose, heights(ELBOW), heights(WRIST), heights(TIP))


def lowest(poses: Sequence[PoseHeights], part: str) -> tuple[float, int, str]:
    """the lowest height of 'elbow', 'wrist' or 'tip' of all the poses and stops: (cm, pose number, what stop)"""
    best = (float('inf'), 0, '')
    for pose in poses:
        for height, stop in zip(getattr(pose, part), STOPS):
            if height < best[0]:
                best = (height, pose.number, stop)
    return best


def format_pose(pose: PoseHeights, warn: float) -> str:
    """the line of a pose: its number, the angles of the joints 1 to 6 in degrees, the wrist and tip heights"""
    marked = pose.lowest < warn
    return '{:4d}  {}  | {} | {} {}'.format(
        pose.number, ' '.join('{:6.1f}'.format(angle) for angle in np.degrees(pose.angles[1:7])),
        ' '.join('{:5.1f}'.format(height) for height in pose.wrist),
        ' '.join('{:5.1f}'.format(height) for height in pose.tip), ' <-- LOW' if marked else '').rstrip()


def print_report(args: argparse.Namespace, robot: OfflineRobot, poses: Sequence[np.ndarray],
                 ranges: dict[int, tuple[float, float]]) -> None:
    """print the options in force, the limits, the poses with their heights and the summary"""
    limits = robot.workspace_limits
    delta = np.radians(args.delta)
    angle_limits = robot.get_angle_limits()
    print('preview of the poses of record_static_load, nothing is connected: joints {}, ranges {}, base {} degrees, '
          'overshoot {} degrees, seed {}'.format(args.joints, ' '.join(args.range) or 'default', args.base,
                                                 args.delta, args.seed if args.seed is not None else 'none'))
    print('limits in force: floor {} cm (wrist {} cm){}'.format(
        limits.z_min, limits.wrist_z_min, ' set by --floor' if args.floor is not None else ''))
    for joint, (low, high) in ranges.items():
        print('  joint {}: {:.0f} to {:.0f} degrees'.format(joint, np.degrees(low), np.degrees(high)))
    if args.seed is None:
        print('no --seed: the poses of a later recording will differ from these, give the recording the same --seed '
              'to get exactly them')

    reports = [pose_heights(number, pose, delta, args.joints, angle_limits, robot.robot_config)
               for number, pose in enumerate(poses, 1)]
    print()
    print('angles in degrees, heights in cm above the table: wrist centre and gripper tip at the pose, at the '
          'overshoot above (+{0:g} degrees) and at the overshoot below (-{0:g} degrees)'.format(args.delta))
    print('pose  {}  | {} | {}'.format(' '.join('{:>6s}'.format('j' + str(joint)) for joint in range(1, 7)),
                                       'wrist: pose    +     -', 'tip: pose    +     -'))
    for report in reports:
        print(format_pose(report, args.warn))

    tip, tip_pose, tip_stop = lowest(reports, 'tip')
    wrist, wrist_pose, wrist_stop = lowest(reports, 'wrist')
    elbow, elbow_pose, elbow_stop = lowest(reports, 'elbow')
    low = sum(1 for report in reports if report.lowest < args.warn)
    print()
    print('{} poses of {} asked'.format(len(reports), args.poses))
    print('lowest gripper tip: {:.1f} cm, pose {}, {}'.format(tip, tip_pose, tip_stop))
    print('lowest wrist centre: {:.1f} cm, pose {}, {}'.format(wrist, wrist_pose, wrist_stop))
    print('lowest elbow: {:.1f} cm, pose {}, {}'.format(elbow, elbow_pose, elbow_stop))
    print('{} poses with the tip or the wrist under {:g} cm (marked LOW)'.format(low, args.warn))
    # the numbers are those of the limits in force (WorkSpaceLimits, or --floor), its docstring says what they mean
    print('Heights are cm above the table: the planner lets the gripper tip down to {} cm and the wrist centre to {} '
          'cm (the tip of the model is the end of the closed fingertips, so with the gripper pointing down the '
          'fingertips are at the height of the tip; the wrist servo hangs below the wrist centre when the gripper '
          'points up, see WorkSpaceLimits in src/global_constants.py), and it does not check the gripper fingers or '
          'the elbow.'.format(limits.z_min, limits.wrist_z_min))


def build_parser() -> argparse.ArgumentParser:
    """the options of the recorder and --warn"""
    parser = record_static_load.build_parser()
    parser.description = ('list the poses record_static_load will visit and their heights above the table, without '
                          'the robot; the options of the recording that do not matter here are ignored')
    parser.add_argument('--warn', type=float, default=DEFAULT_WARN,
                        help='cm: mark the poses where the wrist or the tip comes lower than this (default {})'
                        .format(DEFAULT_WARN))
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as e:  # argparse has printed the usage or the help
        return 0 if e.code in (0, None) else 1

    problem = record_static_load.invalid_joint_message(args.joints)
    if problem is not None:
        print(problem)
        return 1
    try:
        robot = OfflineRobot()
        poses, ranges = record_static_load.select_poses(args, robot)
    except (ValueError, argparse.ArgumentTypeError) as e:  # a bad --range, a joint without a range, a bad yaml
        print(e)
        return 1
    if not poses:
        print('no pose the planner accepts, widen the ranges or lower the floor')
        return 1
    print_report(args, robot, poses, ranges)
    return 0


if __name__ == '__main__':
    sys.exit(main())
