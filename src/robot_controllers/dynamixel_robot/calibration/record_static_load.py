"""
Records the static load of the arm servos: the steady state position error at random poses, approached from both
directions, as the data for calibration/fit_static_load.py. Run it from the project root with the arm powered, its
torque off and nothing in its way:
    python -m src.robot_controllers.dynamixel_robot.calibration.record_static_load --poses 40 --output static_load.csv

It enables the servos, moves to every pose plus an overshoot of --delta degrees on the varied joints, then to the
pose (approached from above, direction down), waits --settle seconds and logs every arm servo; then the same from
below (direction up). The poses are random in joint space; a pose is only used when the planner accepts it and
its two overshoot poses: every move is planned in joint space and checked like the straight line moves of the xbox
controller (workspace limits, joint ranges, joint speed); --floor replaces WorkSpaceLimits.z_min (cm above the
table for the wrist and the gripper tip) for the recording. The planner checks the wrist and the gripper tip, not
the elbow or the gripper fingers, so keep the default ranges of joints 4 and 5 (plus or minus 90 degrees,
--range 5=-45:45 to narrow or widen one) until you have watched the arm go through them. The robot stops before
every approach when a servo is warmer than the warning temperature or has a hardware error. Ctrl+C stops the
movement, the servos keep holding until you confirm the arm is supported.

Logged per servo and reading: the commanded angle, the goal position of the angle without any compensation, the
compensation that was added to it (0 unless the yaml has one), the present position, the error (present minus the
uncompensated goal, in positions, so the whole static load even while a compensation is active), the present
current register (mA for the XM and XL servos, 0.1% load for the XC430), the temperature and the input voltage.
Joints 2, 3, 4 and 5 are varied by default (--joints), joint 1 is kept at --base degrees (90 is straight ahead)
and joint 6 at 0.
"""
import argparse
import csv
import logging
import sys
import time
from datetime import datetime

import numpy as np

from src import global_constants
from src.global_constants import WorkSpaceLimits
from src.robot_controllers.dynamixel_robot.dynamixel_robot_controller import DynamixelRobotController, ARM_IDS
from src.robot_controllers.dynamixel_robot.dynamixel_utils import DynamixelError
from src.robot_controllers.dynamixel_robot.static_compensation import UP, DOWN
from src.utils.movement import plan_move_to_angles, execute_move, check_move, PlannedMove
from src.utils.movement_exception import MovementException

# the unit of the present current register per servo model, (name, value of one unit)
CURRENT_UNITS = {
    'XM430-W350': ('mA', 2.69),
    'XM540-W270': ('mA', 2.69),
    'XL330-M288': ('mA', 1.0),
    'XC430-W240': ('load_0.1%', 1.0),  # no current sensor, the register is the present load
    'XC430-W150': ('load_0.1%', 1.0),
}
FIELDS = ('time', 'pose', 'direction', 'servo', 'angle', 'goal_position', 'compensation', 'present_position', 'error',
          'current_raw', 'current', 'current_unit', 'temperature', 'voltage')
DEFAULT_JOINTS = (2, 3, 4, 5)
# the wrist joints are only sampled in this range by default: the planner doesn't know the gripper fingers or the
# elbow, and a wrist bent further than this can fold the gripper onto the upper arm
DEFAULT_RANGES = {4: (np.radians(-90), np.radians(90)), 5: (np.radians(-90), np.radians(90))}
# how many random poses are tried for every pose the planner accepts, before giving up
SAMPLING_ATTEMPTS = 50


class StopRecording(Exception):
    """raised to stop the recording, with the reason"""


def limits_with_floor(floor, limits=WorkSpaceLimits):
    """the workspace limits with the floor of the gripper tip (z_min, cm above the table) replaced; the wrist keeps
    its own floor unless this one is higher"""
    wrist_z_min = max(float(floor), getattr(limits, 'wrist_z_min', float(floor)))
    return type('RecordingLimits', (limits,), {'z_min': float(floor), 'wrist_z_min': wrist_z_min})


def sampling_ranges(angle_limits, joints, margin, overrides=None):
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


def pose_is_accepted(pose, robot, delta, joints):
    """whether the planner accepts the pose and the two overshoot poses it is approached from: every move is
    checked against the workspace limits of the robot (the wrist and the gripper tip) and its joint ranges"""
    for overshoot, target, _ in approach_sequence(pose, delta, joints, robot.get_angle_limits()):
        for angles in (overshoot, target):
            try:
                check_move(PlannedMove('pose', angles[np.newaxis], 0.0), robot)
            except MovementException:
                return False
    return True


def sample_poses(count, ranges, base_angle, rng, accept=None, attempts=SAMPLING_ATTEMPTS):
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
    poses = []
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


def approach_sequence(pose, delta, joints, angle_limits):
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


def current_in_units(raw, model):
    """returns (current, unit name) for the raw register value of a servo model, (None, '?') for an unknown model"""
    if model not in CURRENT_UNITS:
        return None, '?'
    unit, per_unit = CURRENT_UNITS[model]
    return raw * per_unit, unit


def readings(robot, models, pose_index, direction, angles, now):
    """
    read every arm servo once and return a csv row per servo, or None when a read fails
    :param models: mapping of servo id to model name
    """
    positions = robot.get_current_positions()
    currents = robot.get_present_currents()
    status = robot.get_temperatures_and_voltages()
    if positions is None or currents is None or status is None:
        return None
    rows = []
    for servo_id in ARM_IDS:
        servo = robot.servo_handler.get_servo(servo_id)
        # the error is measured against the goal of the angle itself, whatever compensation was added to it
        goal = servo.target_position - servo.compensation_positions
        current, unit = current_in_units(currents[servo_id], models.get(servo_id))
        temperature, voltage = status[servo_id]
        rows.append({'time': now, 'pose': pose_index, 'direction': direction, 'servo': servo_id,
                     'angle': '{:.6f}'.format(angles[servo_id]), 'goal_position': goal,
                     'compensation': servo.compensation_positions,
                     'present_position': int(positions[servo_id]), 'error': int(positions[servo_id]) - goal,
                     'current_raw': currents[servo_id], 'current': '' if current is None else '{:.2f}'.format(current),
                     'current_unit': unit, 'temperature': temperature, 'voltage': voltage})
    return rows


def check_servos(robot, temperature_warning):
    """raises StopRecording when a servo has a hardware error or is warmer than the warning temperature"""
    errors = robot.get_hardware_errors()
    if errors is None:
        raise StopRecording('failed to read the hardware errors')
    if errors:
        raise StopRecording('servos {} have a hardware error'.format(sorted(errors)))
    status = robot.get_temperatures_and_voltages()
    if status is None:
        raise StopRecording('failed to read the temperatures')
    warm = {servo_id: temperature for servo_id, (temperature, _) in status.items()
            if temperature > temperature_warning}
    if warm:
        raise StopRecording('servos {} are warmer than {}C, let them cool down'.format(warm, temperature_warning))


def move_to(robot, target, angular_speed, should_stop=None):
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


def record(robot, poses, writer, models, temperature_warning, joints=DEFAULT_JOINTS, delta=np.radians(5.0),
           settle=3.0, angular_speed=0.5, sleep=time.sleep, clock=time.time):
    """
    visit the poses from both directions and write a row per servo and reading, returns the number of poses done
    :param robot: a DynamixelRobotController with its servos enabled; every move is checked against its workspace
        limits (robot.workspace_limits, see limits_with_floor) and joint ranges
    :param poses: list of 7 element angle arrays
    :param writer: csv.DictWriter with FIELDS
    :param models: mapping of servo id to model name, for the unit of the current
    :param temperature_warning: stop when a servo gets warmer than this
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
            rows = readings(robot, models, pose_index, direction, target, clock())
            if rows is None:
                raise StopRecording('failed to read the servos')
            rows_of_pose.extend(rows)
            print('  {:4s} errors: {}'.format(direction, ' '.join(
                '{}={:+d}'.format(row['servo'], row['error']) for row in rows if row['servo'] in joints)))
        else:
            writer.writerows(rows_of_pose)
            done += 1
    return done


def disable_torque_on_request(robot):
    try:
        input('\npress enter to disable the torque, SUPPORT THE ARM, it will drop. '
              'Ctrl+C to exit and keep the torque enabled\n')
    except KeyboardInterrupt:
        print('\nexiting, the servos keep holding their position')
        return
    robot.disable_servos()


def parse_ranges(values):
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


def main(argv=None):
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
    parser.add_argument('--speed', type=float, default=0.5, help='joint speed in rad/s')
    parser.add_argument('--seed', type=int, default=None, help='seed of the random poses')
    parser.add_argument('--output', default='static_load_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M')))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

    for joint in args.joints:
        if joint not in (2, 3, 4, 5, 6):
            print('only joints 2-6 can be varied, not {}'.format(joint))
            return 1
    overrides = parse_ranges(args.range)
    delta = np.radians(args.delta)

    try:
        robot = DynamixelRobotController(None, global_constants.dynamixel_robot_config)
    except DynamixelError as e:
        print(e)
        return 1
    if args.floor is not None:
        robot.workspace_limits = limits_with_floor(args.floor, robot.workspace_limits)
    config = robot.servo_config
    ranges = sampling_ranges(robot.get_angle_limits(), args.joints, delta, overrides)
    rng = np.random.default_rng(args.seed)
    poses = sample_poses(args.poses, ranges, np.radians(args.base), rng,
                         accept=lambda pose: pose_is_accepted(pose, robot, delta, tuple(args.joints)))
    if not poses:
        print('no pose the planner accepts, widen the ranges or lower the floor')
        return 1

    print('{} poses (of {} asked), joints {}, overshoot {} degrees, floor {} cm (wrist {} cm), settle {} s, '
          'output {}'.format(len(poses), args.poses, args.joints, args.delta, robot.workspace_limits.z_min,
                             robot.workspace_limits.wrist_z_min, args.settle, args.output))
    for joint, (low, high) in ranges.items():
        print('  joint {}: {:.0f} to {:.0f} degrees'.format(joint, np.degrees(low), np.degrees(high)))
    print('the arm will move through these ranges, nothing may be in its way')
    if input('type yes to enable the servos and start: ').strip().lower() != 'yes':
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
                          delta, args.settle, args.speed)
    except StopRecording as e:
        print('\nstopped: {}'.format(e))
    except KeyboardInterrupt:
        print('\nstopped, the servos are holding their last position')
    print('{} poses recorded in {}'.format(done, args.output))
    disable_torque_on_request(robot)
    return 0


if __name__ == '__main__':
    sys.exit(main())
