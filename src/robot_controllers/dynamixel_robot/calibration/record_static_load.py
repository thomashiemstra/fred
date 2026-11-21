"""
Records the static load of the arm servos: the steady state position error at random poses, approached from both
directions, as the data for calibration/fit_static_load.py. Run it from the project root with the arm powered, its
torque off and nothing in its way:
    python -m src.robot_controllers.dynamixel_robot.calibration.record_static_load --poses 40 --output static_load.csv
    [--yes]
It asks to type yes before it enables the servos; --yes skips the question, for an unattended run (without --yes and
without any input, stdin from /dev/null, it prints that and stops with the servos off). The moves between the poses
run at --speed, by default REST_SPEED (0.3 rad/s): joint 2 of this arm must not be moved fast.

It enables the servos, moves to every pose plus an overshoot of --delta degrees on the varied joints, then to the
pose (approached from above, direction down), waits --settle seconds and logs every arm servo; then the same from
below (direction up). The poses are random in joint space; a pose is only used when the planner accepts it and
its two overshoot poses: every move is planned in joint space and checked like the straight line moves of the xbox
controller (workspace limits, joint ranges, joint speed); --floor replaces WorkSpaceLimits.z_min (cm above the
table for the wrist and the gripper tip) for the recording. The planner checks the wrist and the gripper tip, not
the elbow or the gripper fingers, so keep the default ranges of joints 4 and 5 (plus or minus 90 degrees,
--range 5=-45:45 to narrow or widen one) until you have watched the arm go through them. The robot stops before
every approach when a servo is warmer than the warning temperature or has a hardware error. At the end the arm is
lowered to the rest pose just above the table and the torque goes off there. Ctrl+C stops the movement and lowers
the arm the same way; Ctrl+C during the lowering stops it with the servos holding and asks you to confirm the arm is
supported before the torque goes off (the same prompt comes when the arm can't be lowered, for instance after a
hardware error). Without any input to answer that prompt the servos keep holding their position, and
calibration/lower_arm.py lowers the arm later.

Logged per servo and pose direction: the commanded angle, the goal position of the angle without any compensation
and without any velocity lead (Servo.trajectory_position), the compensation that was added to it (0 unless the yaml has
one), the velocity lead that was added to it (0 at rest, and the arm is at rest here; the column is there because the
sweep recordings, which move, fill it), the present position and the error (present minus the goal of the angle itself,
in positions) as the mean of --readings readings 0.05 s apart (samples is their number), the gains p, i and d of the
servo, the measured present pwm (mean, 885 is full drive), the present current register (mA for the XM and XL servos,
0.1% load for the XC430), the temperature and the input voltage. With a
compensation active the error is the sag that remains: the whole static load is error minus compensation, which
fit_static_load.py uses, together with p to convert it to the holding drive in pwm units. The mode column is
'static' for these rows, speed is blank (a recording of sweeps fills both).
Joints 2, 3, 4 and 5 are varied by default (--joints), joint 1 is kept at --base degrees (90 is straight ahead)
and joint 6 at 0.
"""
import argparse
import csv
import logging
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from typing import Any, TypeVar

import numpy as np

from src.global_constants import WorkSpaceLimits
from src.global_objects import get_robot
from src.robot_controllers.abstract_robot_controller import AbstractRobotController
from src.robot_controllers.combined_robot.combined_robot import CombinedRobot
from src.robot_controllers.dynamixel_robot.dynamixel_robot_controller import ARM_IDS, DynamixelRobotController
from src.robot_controllers.dynamixel_robot.dynamixel_utils import DynamixelError
from src.robot_controllers.dynamixel_robot.calibration.load_directions import UP, DOWN
from src.kinematics.kinematics import forward_position_kinematics, inverse_kinematics
from src.kinematics.kinematics_utils import Pose
from src.utils.movement import MAX_POSITION_ERROR, plan_move_to_angles, execute_move, check_move, PlannedMove
from src.utils.movement_exception import MovementException

# the unit of the present current register per servo model, (name, value of one unit)
CURRENT_UNITS = {
    'XM430-W350': ('mA', 2.69),
    'XM540-W270': ('mA', 2.69),
    'XL330-M288': ('mA', 1.0),
    'XC330-T288-T': ('mA', 1.0),
    'XC430-W240': ('load_0.1%', 1.0),  # no current sensor, the register is the present load
    'XC430-W150': ('load_0.1%', 1.0),
}
FIELDS = ('time', 'pose', 'direction', 'servo', 'angle', 'goal_position', 'compensation', 'lead', 'present_position',
          'error', 'current_raw', 'current', 'current_unit', 'temperature', 'voltage', 'mode', 'speed', 'samples', 'p',
          'i', 'd', 'pwm')
MODE = 'static'  # the value of the mode column for the rows of this script
DEFAULT_JOINTS = (2, 3, 4, 5)
# readings of the servos per row, and the pause between them in seconds: the position of a servo that holds a load
# jitters by a position or two, the mean of a few readings is steadier
DEFAULT_READINGS = 5
READING_PAUSE = 0.05
# the wrist joints are only sampled in this range by default: the planner doesn't know the gripper fingers or the
# elbow, and a wrist bent further than this can fold the gripper onto the upper arm
DEFAULT_RANGES = {4: (np.radians(-90), np.radians(90)), 5: (np.radians(-90), np.radians(90))}
# how many random poses are tried for every pose the planner accepts, before giving up
SAMPLING_ATTEMPTS = 50
# reads of the servo status before a failure stops the run: a corrupt status packet now and then happens on the bus
READ_ATTEMPTS = 3
# where the arm lies on the table with the torque off, as read on 13 November 2026 (gripper tip x=-0.9 y=39.6 z=3.7
# cm, alpha=-0.007 beta=-0.143 gamma=0.099), rounded to straight ahead with the gripper level and a bit over 1 cm
# higher so the arm doesn't hit the table
REST_POSE = Pose(0.0, 40, 5)
REST_SPEED = 0.3  # rad/s of the joint that turns the most, for the move to the rest pose

T = TypeVar('T')


class StopRecording(Exception):
    """raised to stop the recording, with the reason"""


def limits_with_floor(floor: float, limits: type[WorkSpaceLimits] = WorkSpaceLimits) -> type[WorkSpaceLimits]:
    """the workspace limits with the floor of the gripper tip (z_min, cm above the table) replaced; the wrist keeps
    its own floor unless this one is higher"""
    wrist_z_min = max(float(floor), getattr(limits, 'wrist_z_min', float(floor)))
    return type('RecordingLimits', (limits,), {'z_min': float(floor), 'wrist_z_min': wrist_z_min})


def sampling_ranges(angle_limits: dict[int, tuple[float, float]], joints: Iterable[int], margin: float,
                    overrides: dict[int, tuple[float, float]] | None = None) -> dict[int, tuple[float, float]]:
    """
    the range of every varied joint: the servo range less the margin, narrowed by DEFAULT_RANGES and the overrides
    :param angle_limits: mapping of joint to (min angle, max angle) of the servos in radians
    :param margin: the varied joints stay this far from the ends of the servo range, so the overshoot fits
    :param overrides: mapping of joint to (min angle, max angle) in radians
    :return: mapping of joint to (min angle, max angle) in radians
    """
    ranges = {}
    for joint in joints:
        low, high = angle_limits[joint]
        low, high = low + margin, high - margin
        for narrower in (DEFAULT_RANGES.get(joint), (overrides or {}).get(joint)):
            if narrower is not None:
                low, high = max(low, narrower[0]), min(high, narrower[1])
        if low >= high:
            raise ValueError('joint {} has no range left to sample'.format(joint))
        ranges[joint] = (low, high)
    return ranges


def angle_limits_of(robot: AbstractRobotController) -> dict[int, tuple[float, float]]:
    """the joint ranges of the robot in radians, the recording needs them"""
    limits = robot.get_angle_limits()
    if limits is None:
        raise ValueError('this robot does not know the ranges of its joints')
    return limits


def pose_is_accepted(pose: np.ndarray, robot: AbstractRobotController, delta: float, joints: Iterable[int]) -> bool:
    """whether the planner accepts the pose and the two overshoot poses it is approached from: every move is
    checked against the workspace limits of the robot (the wrist and the gripper tip) and its joint ranges"""
    for overshoot, target, _ in approach_sequence(pose, delta, joints, angle_limits_of(robot)):
        for angles in (overshoot, target):
            try:
                check_move(PlannedMove('pose', angles[np.newaxis], 0.0), robot)
            except MovementException:
                return False
    return True


def sample_poses(count: int, ranges: dict[int, tuple[float, float]], base_angle: float, rng: np.random.Generator,
                 accept: Callable[[np.ndarray], bool] | None = None,
                 attempts: int = SAMPLING_ATTEMPTS) -> list[np.ndarray]:
    """
    random poses in joint space
    :param count: number of poses
    :param ranges: mapping of joint to (min angle, max angle) in radians, the joints to vary; the others stay at 0
    :param base_angle: the angle of joint 1 in radians
    :param rng: numpy random generator
    :param accept: optional function of a pose that says whether to keep it; after attempts * count rejected poses
        the list is returned short
    :return: list of 7 element angle arrays
    """
    poses: list[np.ndarray] = []
    tries = 0
    while len(poses) < count and tries < attempts * count:
        tries += 1
        angles = np.zeros(7, dtype=np.float64)
        angles[1] = base_angle
        for joint, (low, high) in ranges.items():
            angles[joint] = rng.uniform(low, high)
        if accept is None or accept(angles):
            poses.append(angles)
    return poses


def approach_sequence(pose: np.ndarray, delta: float, joints: Iterable[int],
                      angle_limits: dict[int, tuple[float, float]]) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """
    the moves that approach a pose from both directions: [(overshoot pose, pose, direction), ...]
    :param delta: how far beyond the pose the varied joints go first, in radians
    :return: the pose approached from above (direction DOWN) and from below (direction UP)
    """
    sequence = []
    for sign, direction in ((1, DOWN), (-1, UP)):
        overshoot = pose.copy()
        for joint in joints:
            low, high = angle_limits[joint]
            overshoot[joint] = float(np.clip(pose[joint] + sign * delta, low, high))
        sequence.append((overshoot, pose, direction))
    return sequence


def current_in_units(raw: float, model: str | None) -> tuple[float | None, str]:
    """returns (current, unit name) for the raw register value of a servo model, (None, '?') for an unknown model"""
    if model not in CURRENT_UNITS:
        return None, '?'
    unit, per_unit = CURRENT_UNITS[model]
    return raw * per_unit, unit


def readings(robot: DynamixelRobotController | CombinedRobot, models: dict[int, str], pose_index: int, direction: str,
             angles: np.ndarray, now: float, count: int = DEFAULT_READINGS, pause: float = READING_PAUSE,
             sleep: Callable[[float], None] = time.sleep) -> list[dict[str, Any]] | None:
    """
    read every arm servo count times and return a csv row per servo, or None when a read fails
    :param models: mapping of servo id to model name
    :param count: the number of readings, at least 1; the row has the mean of the present position, the error and
        the present pwm, and the current, the temperature and the voltage of the last reading
    :param pause: seconds between the readings
    """
    if count < 1:
        raise ValueError('at least 1 reading is needed, got {}'.format(count))
    positions: dict[int, list[int]] = {servo_id: [] for servo_id in ARM_IDS}
    pwms: dict[int, list[int]] = {servo_id: [] for servo_id in ARM_IDS}
    currents: dict[int, int] = {}
    status: dict[int, tuple[int, float]] = {}
    for reading in range(count):
        if reading:
            sleep(pause)
        read_positions = robot.get_current_positions()
        read_currents = robot.get_present_currents()
        read_pwms = robot.get_present_pwms()
        read_status = robot.get_temperatures_and_voltages()
        if read_positions is None or read_currents is None or read_pwms is None or read_status is None:
            return None
        currents, status = read_currents, read_status
        for servo_id in ARM_IDS:
            positions[servo_id].append(int(read_positions[servo_id]))
            pwms[servo_id].append(int(read_pwms[servo_id]))
    rows = []
    for servo_id in ARM_IDS:
        servo = robot.servo_handler.get_servo(servo_id)
        # the error is measured against the goal of the angle itself, whatever compensation and velocity lead were
        # added to it (the lead is 0 at rest)
        goal = servo.trajectory_position
        position = float(np.mean(positions[servo_id]))
        current, unit = current_in_units(currents[servo_id], models.get(servo_id))
        temperature, voltage = status[servo_id]
        rows.append({'time': now, 'pose': pose_index, 'direction': direction, 'servo': servo_id,
                     'angle': '{:.6f}'.format(angles[servo_id]), 'goal_position': goal,
                     'compensation': servo.compensation_positions, 'lead': servo.lead_positions,
                     'present_position': '{:.2f}'.format(position), 'error': '{:.2f}'.format(position - goal),
                     'current_raw': currents[servo_id], 'current': '' if current is None else '{:.2f}'.format(current),
                     'current_unit': unit, 'temperature': temperature, 'voltage': voltage, 'mode': MODE,
                     'speed': '', 'samples': count, 'p': servo.p, 'i': servo.i, 'd': servo.d,
                     'pwm': '{:.1f}'.format(float(np.mean(pwms[servo_id])))})
    return rows


def read_status(read: Callable[[], T | None], what: str) -> T:
    """the result of the read, tried up to READ_ATTEMPTS times; raises StopRecording when every attempt failed"""
    for _ in range(READ_ATTEMPTS):
        value = read()
        if value is not None:
            return value
    raise StopRecording('failed to read {} {} times in a row'.format(what, READ_ATTEMPTS))


def check_servos(robot: DynamixelRobotController | CombinedRobot, temperature_warning: float) -> None:
    """raises StopRecording when a servo has a hardware error or is warmer than the warning temperature"""
    errors = read_status(robot.get_hardware_errors, 'the hardware errors')
    if errors:
        raise StopRecording('servos {} have a hardware error'.format(sorted(errors)))
    status = read_status(robot.get_temperatures_and_voltages, 'the temperatures')
    warm = {servo_id: temperature for servo_id, (temperature, _) in status.items()
            if temperature > temperature_warning}
    if warm:
        raise StopRecording('servos {} are warmer than {}C, let them cool down'.format(warm, temperature_warning))


def move_to(robot: DynamixelRobotController | CombinedRobot, target: np.ndarray, angular_speed: float,
            should_stop: Callable[[], bool] | None = None) -> bool:
    """plan and execute a joint space move from the current angles, returns False when the planner refuses it"""
    current = robot.get_current_angles()
    if current is None:
        raise StopRecording('failed to read the current angles')
    try:
        move = plan_move_to_angles(robot, current, target, angular_speed=angular_speed)
    except MovementException as e:
        print('  skipped: {}'.format(e))
        return False
    if not execute_move(robot, move, should_stop):
        raise StopRecording('stopped')
    return True


def record(robot: DynamixelRobotController | CombinedRobot, poses: Sequence[np.ndarray], writer: csv.DictWriter[str],
           models: dict[int, str], temperature_warning: float, joints: Sequence[int] = DEFAULT_JOINTS,
           delta: float = np.radians(5.0), settle: float = 3.0, angular_speed: float = 0.5,
           sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.time,
           reading_count: int = DEFAULT_READINGS, reading_pause: float = READING_PAUSE) -> int:
    """
    visit the poses from both directions and write a row per servo and direction, returns the number of poses done
    :param robot: the robot of global_objects.get_robot with its servos enabled; every move is checked against its workspace
        limits (robot.workspace_limits, see limits_with_floor) and joint ranges
    :param poses: list of 7 element angle arrays
    :param writer: csv.DictWriter with FIELDS
    :param models: mapping of servo id to model name, for the unit of the current
    :param temperature_warning: stop when a servo gets warmer than this
    :param reading_count: the readings behind every row, see readings
    :param reading_pause: seconds between the readings
    """
    angle_limits = robot.get_angle_limits()
    done = 0
    for pose_index, pose in enumerate(poses):
        print('pose {} of {}: {} degrees'.format(pose_index + 1, len(poses), np.round(np.degrees(pose[1:]), 1)))
        rows_of_pose = []
        for overshoot, target, direction in approach_sequence(pose, delta, joints, angle_limits):
            check_servos(robot, temperature_warning)
            if not move_to(robot, overshoot, angular_speed):
                break
            if not move_to(robot, target, angular_speed):
                break
            sleep(settle)
            rows = readings(robot, models, pose_index, direction, target, clock(), reading_count, reading_pause, sleep)
            if rows is None:
                raise StopRecording('failed to read the servos')
            rows_of_pose.extend(rows)
            print('  {:4s} errors: {}'.format(direction, ' '.join(
                '{}={:+.1f}'.format(row['servo'], float(row['error'])) for row in rows if row['servo'] in joints)))
        else:
            writer.writerows(rows_of_pose)
            done += 1
    return done


def disable_torque_on_request(robot: DynamixelRobotController | CombinedRobot) -> None:
    """
    ask to switch the torque off with the arm supported, and do it on enter; Ctrl+C, or no input at all (stdin is
    closed in an unattended run), leaves the servos holding their position
    """
    try:
        input('\npress enter to disable the torque, SUPPORT THE ARM, it will drop. '
              'Ctrl+C to exit and keep the torque enabled\n')
    except KeyboardInterrupt:
        print('\nexiting, the servos keep holding their position')
        return
    except EOFError:
        # nobody can support the arm, so the torque stays on
        print('there is no input to answer the question, the servos keep holding their position; '
              'python -m src.robot_controllers.dynamixel_robot.calibration.lower_arm lowers the arm')
        return
    robot.disable_servos()


def lower_to_rest(robot: DynamixelRobotController | CombinedRobot, speed: float = REST_SPEED, settle: float = 0.5,
                  sleep: Callable[[float], None] = time.sleep) -> str | None:
    """
    lower the arm to REST_POSE just above the table and switch the torque off there, so it doesn't drop from height;
    never asks anything: the torque is only switched off directly after a completed move, and when the arm can't be
    lowered the servos are left as they are (see lower_and_disable_torque for the fallback to a prompt)
    :param speed: speed of the joint that turns the most, in rad/s
    :param settle: seconds to wait at the pose before the torque goes off
    :return: None if the arm was lowered and the torque is off, else why not: a servo has shut down or can't be read,
        the planner refuses the pose, the angles can't be read, or Ctrl+C came during the move
    """
    why: str | None = None
    try:
        errors = robot.get_hardware_errors()
        if errors is None:
            why = 'the hardware errors could not be read'
        elif errors:
            why = 'servos {} have a hardware error and are limp, the others are not moved'.format(sorted(errors))
        else:
            target = inverse_kinematics(REST_POSE, robot.robot_config)
            reached = forward_position_kinematics(target, robot.robot_config)[-1]
            if not np.all(np.isfinite(target)) or \
                    np.linalg.norm(reached - [REST_POSE.x, REST_POSE.y, REST_POSE.z]) > MAX_POSITION_ERROR:
                why = 'the rest pose can not be reached with these link lengths'
            else:
                print('lowering the arm to the table, the torque goes off after it')
                if move_to(robot, target, speed):
                    sleep(settle)
                    robot.disable_servos()
                    print('the torque is off')
                    return None
                why = 'the planner refuses the move to the rest pose'
    except MovementException as e:
        why = 'the move to the rest pose is refused: {}'.format(e)
    except StopRecording as e:
        why = str(e)
    except KeyboardInterrupt:
        why = 'interrupted, the servos hold where they are'
    return why


def lower_and_disable_torque(robot: DynamixelRobotController | CombinedRobot, speed: float = REST_SPEED,
                             settle: float = 0.5, sleep: Callable[[float], None] = time.sleep) -> bool:
    """
    lower the arm to REST_POSE just above the table and switch the torque off there, so it doesn't drop from height;
    falls back to disable_torque_on_request (the arm is not moved, or stays where it is) and returns False when a
    servo has shut down or can't be read, the planner refuses the pose, the angles can't be read, or Ctrl+C comes
    during the move: the torque is only switched off directly after a completed move
    :param speed: speed of the joint that turns the most, in rad/s
    :param settle: seconds to wait at the pose before the torque goes off
    :return: True if the arm was lowered and the torque is off
    """
    why = lower_to_rest(robot, speed, settle, sleep)
    if why is None:
        return True
    print('the arm is not lowered: {}'.format(why))
    disable_torque_on_request(robot)
    return False


def parse_ranges(values: Iterable[str]) -> dict[int, tuple[float, float]]:
    """--range entries like 5=-45:45 (degrees) into a mapping of joint to (min, max) in radians"""
    ranges = {}
    for value in values:
        try:
            joint, span = value.split('=')
            low, high = (float(bound) for bound in span.split(':'))
        except ValueError:
            raise argparse.ArgumentTypeError('range entries look like 5=-45:45, got {!r}'.format(value))
        ranges[int(joint)] = (np.radians(low), np.radians(high))
    return ranges


def positive_int(value: str) -> int:
    """argparse type of a whole number of at least 1"""
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError('should be a whole number, got {!r}'.format(value))
    if number < 1:
        raise argparse.ArgumentTypeError('should be at least 1, got {}'.format(number))
    return number


def build_parser() -> argparse.ArgumentParser:
    """the options of the recorder; offline/preview_static_load_poses.py takes the same ones, so a command line of
    the recorder can be pasted there as it is"""
    parser = argparse.ArgumentParser(description='record the static load of the arm servos at random poses')
    parser.add_argument('--poses', type=int, default=40, help='number of poses')
    parser.add_argument('--joints', type=int, nargs='+', default=list(DEFAULT_JOINTS), help='the joints to vary')
    parser.add_argument('--range', nargs='+', default=[], help='joint=min:max in degrees, narrows a joint')
    parser.add_argument('--base', type=float, default=90.0, help='angle of joint 1 in degrees')
    parser.add_argument('--delta', type=float, default=5.0, help='overshoot before every pose in degrees')
    parser.add_argument('--floor', type=float, default=None,
                        help='cm above the table the wrist and gripper tip stay, default WorkSpaceLimits.z_min ({})'
                        .format(WorkSpaceLimits.z_min))
    parser.add_argument('--settle', type=float, default=3.0, help='seconds to wait before reading')
    parser.add_argument('--readings', type=positive_int, default=DEFAULT_READINGS,
                        help='readings of the servos per row, the row has their mean ({} s apart)'
                        .format(READING_PAUSE))
    # the speed of the sweep recorder between its poses: joint 2 of this arm must not be moved fast
    parser.add_argument('--speed', type=float, default=REST_SPEED,
                        help='joint speed in rad/s (default {}, joint 2 must not be moved fast)'.format(REST_SPEED))
    parser.add_argument('--seed', type=int, default=None, help='seed of the random poses')
    parser.add_argument('--output', default='static_load_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M')))
    parser.add_argument('--yes', action='store_true', help="don't ask before enabling the servos")
    return parser


def invalid_joint_message(joints: Iterable[int]) -> str | None:
    """why a joint of --joints can not be varied, or None when all of them can"""
    for joint in joints:
        if joint not in (2, 3, 4, 5, 6):
            return 'only joints 2-6 can be varied, not {}'.format(joint)
    return None


def select_poses(args: argparse.Namespace, robot: AbstractRobotController,
                 overrides: dict[int, tuple[float, float]] | None = None
                 ) -> tuple[list[np.ndarray], dict[int, tuple[float, float]]]:
    """
    the poses a recording with these options visits, in this order: the steps from the options to the poses, shared
    with offline/preview_static_load_poses.py so that the preview lists exactly what the recorder will do. With
    --floor the workspace limits of the robot are replaced (the robot object is changed).
    :param args: the parsed options of build_parser
    :param robot: where the angle limits, the link lengths and the workspace limits come from
    :param overrides: the parsed --range entries, by default parse_ranges(args.range)
    :return: (the accepted poses, which can be fewer than asked or none, and the sampling range of every varied joint)
    :raises ValueError: when a varied joint has no range left to sample
    :raises argparse.ArgumentTypeError: when a --range entry is malformed
    """
    if overrides is None:
        overrides = parse_ranges(args.range)
    delta = np.radians(args.delta)
    if args.floor is not None:
        robot.workspace_limits = limits_with_floor(args.floor, robot.workspace_limits)
    ranges = sampling_ranges(angle_limits_of(robot), args.joints, delta, overrides)
    rng = np.random.default_rng(args.seed)
    poses = sample_poses(args.poses, ranges, np.radians(args.base), rng,
                         accept=lambda pose: pose_is_accepted(pose, robot, delta, tuple(args.joints)))
    return poses, ranges


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

    problem = invalid_joint_message(args.joints)
    if problem is not None:
        print(problem)
        return 1
    overrides = parse_ranges(args.range)
    delta = np.radians(args.delta)

    try:
        robot = get_robot()
    except DynamixelError as e:
        print(e)
        return 1
    if not isinstance(robot, CombinedRobot):
        print('the static load can only be recorded on the real robot, use_simulation is set')
        return 1
    config = robot.servo_config
    poses, ranges = select_poses(args, robot, overrides)
    if not poses:
        print('no pose the planner accepts, widen the ranges or lower the floor')
        return 1

    print('{} poses (of {} asked), joints {}, overshoot {} degrees, floor {} cm (wrist {} cm), settle {} s, '
          'output {}'.format(len(poses), args.poses, args.joints, args.delta, robot.workspace_limits.z_min,
                             robot.workspace_limits.wrist_z_min, args.settle, args.output))
    for joint, (low, high) in ranges.items():
        print('  joint {}: {:.0f} to {:.0f} degrees'.format(joint, np.degrees(low), np.degrees(high)))
    print('the arm will move through these ranges, nothing may be in its way')
    if not args.yes:
        try:
            answer = input('type yes to enable the servos and start: ')
        except EOFError:
            print('there is no input to answer the question, use --yes for an unattended run')
            return 1
        if answer.strip().lower() != 'yes':
            return 0
    if not robot.enable_servos():
        print('failed to enable the servos')
        return 1

    done = 0
    try:
        with open(args.output, 'w', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=FIELDS)
            writer.writeheader()
            done = record(robot, poses, writer, config.models, config.temperature_warning, tuple(args.joints),
                          delta, args.settle, args.speed, reading_count=args.readings)
    except StopRecording as e:
        print('\nstopped: {}'.format(e))
    except KeyboardInterrupt:
        print('\nstopped, the servos are holding their last position')
    print('{} poses recorded in {}'.format(done, args.output))
    lower_and_disable_torque(robot)  # falls back to the prompt by itself when the arm can't be lowered
    return 0


if __name__ == '__main__':
    sys.exit(main())
