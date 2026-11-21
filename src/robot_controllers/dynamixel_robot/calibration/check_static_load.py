"""
Reports how good the static load compensation of the arm joints is, from recordings of record_static_load.py (poses at
rest) or record_load_sweep.py (slow sweeps of one joint). It moves nothing and needs no robot and no yaml:
    python -m src.robot_controllers.dynamixel_robot.calibration.check_static_load recording.csv [more.csv ...]
        [--joints 2 3] [--limit 1.0] [--worst 10] [--table]
fit_static_load.py fits a new model, this script measures the compensation that was active while recording.

The servos run a P-only position loop, so a joint that holds a load rests a little off its goal. The csv has per servo,
pose and direction the error, present minus the UNCOMPENSATED goal in positions (4096 per turn, one position is
360 / 4096 degrees), and the compensation in positions that was added to the goal. With a compensation active the
error is what is left of the sag. Friction rests a joint off its goal with the sign of the direction it arrived from
(up: from below, down: from above), so at every pose the mean of the two directions is the sag, the gravity that is
not compensated, and half the difference is the friction band. A joint is reported from the poses where it has an
error in both directions; the recordings are pooled, a pose is identified by its file and its index in the file.

How to read the report, per joint:
- the sag that is left: the sag (mean of up and down) of every pose, with the compensation as it was recorded. This is
  the number to judge the compensation by: it should be close to 0 and without a trend over the range of the joint.
- the sag without the compensation: the sag less the compensation, what the joint would have had without it, and how
  much smaller the rms has become. Left out when no compensation was active in the recording.
- the error of each direction, the largest absolute error of either, and the friction band. The friction is not
  compensated, it is the part of the error that stays when the sag is gone.
- the poses where the absolute sag is over --limit (positions), largest first, at most --worst of them, with the
  angles of joints 1 to 6 in whole degrees, and one verdict line.
- with --table every pose of the joint, sorted by the angle of the joint and then by the other angles, to read a trend
  along the range of the joint off the sag column.
In a sweep the joint moves while it is measured, so the errors of the two directions include the lag of the speed (it
has opposite signs and cancels in the mean): the sag is right, the friction band is larger than at rest. A joint with
rows of both modes (static and sweep) is warned about, since the two are mixed in its statistics.
"""
import argparse
import csv
import math
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from src.robot_controllers.dynamixel_robot.calibration.fit_static_load import DEGREES_PER_POSITION, read_rows
from src.robot_controllers.dynamixel_robot.calibration.load_directions import DOWN, UP

ARM_JOINTS = (1, 2, 3, 4, 5, 6)  # the gripper (servo 7) has no compensation and is left out
DEFAULT_LIMIT = 1.0  # positions, the sag above which a pose is listed
DEFAULT_WORST = 10  # the number of poses over the limit that are listed


@dataclass
class PoseJoint:
    """one joint at one pose: the errors of the two directions, in positions, and what the pose was recorded with"""
    joint: int
    file: str
    pose: int
    angles: np.ndarray  # 7 element array, radians, nan for a joint that has no row at the pose
    error_up: float  # present minus the uncompensated goal, arriving from below
    error_down: float  # the same arriving from above
    compensation_up: float  # positions added to the goal, 0 for a joint without compensation
    compensation_down: float
    p: float | None  # the P gain of the servo, None when the csv has none
    mode: str  # 'static' or 'sweep'
    speed: float | None = None  # degrees per second of a sweep, None at rest

    @property
    def compensation(self) -> float:
        """the compensation added at the pose, the mean of the two directions (the same in both for a smooth one)"""
        return (self.compensation_up + self.compensation_down) / 2.0

    @property
    def sag(self) -> float:
        """the sag that is left: the mean of the two directions, friction cancels in it"""
        return (self.error_up + self.error_down) / 2.0

    @property
    def uncompensated(self) -> float:
        """the sag the joint would have without the compensation"""
        return self.sag - self.compensation

    @property
    def friction(self) -> float:
        """half the difference between the directions: the friction band, which is not compensated"""
        return abs(self.error_up - self.error_down) / 2.0


@dataclass
class Stats:
    """the numbers of a list of values, in the unit of the values"""
    mean: float
    rms: float
    low: float
    high: float
    largest: float  # the largest absolute value


@dataclass
class JointSummary:
    """the numbers of the report of one joint"""
    joint: int
    count: int  # the poses with both directions
    gains: tuple[float, ...]  # the P gains seen, ascending
    modes: tuple[str, ...]  # 'static' and 'sweep', as seen
    speeds: tuple[float, ...]  # the sweep speeds seen, degrees per second
    sag: Stats  # (up + down) / 2 per pose
    uncompensated: Stats  # the sag less the compensation
    compensation: Stats | None  # the compensation added per pose, None when it was 0 in every row
    up: Stats  # the errors arriving from below
    down: Stats  # the errors arriving from above
    largest_error: float  # the largest absolute error of either direction
    friction_mean: float
    friction_max: float
    rms_reduction: float | None  # percent the rms is smaller than without the compensation, None without one
    limit: float
    over: list[PoseJoint]  # the poses whose absolute sag is over the limit, the largest first


def _blank(value: str | None) -> bool:
    return value is None or value.strip() == ''


def _number(row: dict[str, str], name: str) -> float:
    """
    the float in a column of a row
    :raises ValueError: naming the file and the pose when the column is missing or is not a finite number
    """
    where = '{}: pose {} servo {}'.format(row.get('file', 'the csv'), row.get('pose'), row.get('servo'))
    value = row.get(name)
    if value is None:
        raise ValueError('{} has no column {}'.format(where, name))
    try:
        number = float(value)
    except ValueError:
        raise ValueError('{} has {} {!r}, which is not a number'.format(where, name, value)) from None
    if not math.isfinite(number):
        raise ValueError('{} has {} {!r}, which is not finite'.format(where, name, value))
    return number


def _optional(row: dict[str, str], name: str) -> float | None:
    """the float in a column of a row, None when it is blank or the column is missing"""
    return None if _blank(row.get(name)) else _number(row, name)


def poses_from_rows(rows: Iterable[dict[str, str]]) -> dict[int, list[PoseJoint]]:
    """
    group the rows per pose and joint: the angles of the pose and the errors of both directions of every arm joint.
    A joint has a PoseJoint only at the poses where it has an error in both directions. A row without an error (a
    servo that was not measured) only gives the angle of its joint to the pose, and a row of the gripper is ignored.
    The pose index only counts within a file, so rows need a 'file' field when they come from more than one file.
    :return: mapping of joint to its PoseJoints, in the order the poses appear in the rows
    :raises ValueError: for a row with a number that can't be read, naming the file and the pose
    """
    # per (file, pose) the angles, and per direction and joint the row that holds the error
    angles_of_pose: dict[tuple[str, int], np.ndarray] = {}
    measured: dict[tuple[str, int], dict[str, dict[int, dict[str, str]]]] = {}
    for row in rows:
        servo = int(_number(row, 'servo'))
        if servo not in ARM_JOINTS:
            continue
        key = (row.get('file', ''), int(_number(row, 'pose')))
        angles_of_pose.setdefault(key, np.full(7, np.nan))[servo] = _number(row, 'angle')
        if _blank(row.get('error')) or row.get('direction') not in (UP, DOWN):
            continue
        _number(row, 'error')  # a wrong number is reported now, with the file and the pose
        measured.setdefault(key, {}).setdefault(row['direction'], {})[servo] = row

    poses: dict[int, list[PoseJoint]] = {}
    for key, directions in measured.items():
        if UP not in directions or DOWN not in directions:
            continue
        for joint in sorted(set(directions[UP]) & set(directions[DOWN])):
            up, down = directions[UP][joint], directions[DOWN][joint]
            poses.setdefault(joint, []).append(PoseJoint(
                joint=joint, file=key[0], pose=key[1], angles=angles_of_pose[key],
                error_up=_number(up, 'error'), error_down=_number(down, 'error'),
                compensation_up=_optional(up, 'compensation') or 0.0,
                compensation_down=_optional(down, 'compensation') or 0.0,
                p=_optional(up, 'p'), mode=(up.get('mode') or '').strip() or 'static', speed=_optional(up, 'speed')))
    return poses


def stats_of(values: Sequence[float]) -> Stats:
    """the statistics of a list of values, which must not be empty"""
    array = np.array(values, dtype=np.float64)
    # + 0.0 turns a negative zero into a zero, which would print as -0.00
    return Stats(mean=float(np.mean(array)) + 0.0, rms=float(np.sqrt(np.mean(np.square(array)))),
                 low=float(np.min(array)) + 0.0, high=float(np.max(array)) + 0.0, largest=float(np.max(np.abs(array))))


def summarise(joint: int, poses: Sequence[PoseJoint], limit: float = DEFAULT_LIMIT) -> JointSummary:
    """
    the numbers of the report of a joint
    :param poses: the PoseJoints of the joint, at least one
    :param limit: the absolute sag in positions above which a pose is over the limit
    """
    if not poses:
        raise ValueError('joint {} has no poses to summarise'.format(joint))
    sag = stats_of([pose.sag for pose in poses])
    uncompensated = stats_of([pose.uncompensated for pose in poses])
    active = any(pose.compensation_up != 0 or pose.compensation_down != 0 for pose in poses)
    reduction = 100.0 * (1.0 - sag.rms / uncompensated.rms) if active and uncompensated.rms > 0 else None
    frictions = [pose.friction for pose in poses]
    over = sorted((pose for pose in poses if abs(pose.sag) > limit), key=lambda pose: -abs(pose.sag))
    up = stats_of([pose.error_up for pose in poses])
    down = stats_of([pose.error_down for pose in poses])
    return JointSummary(
        joint=joint, count=len(poses), gains=tuple(sorted({pose.p for pose in poses if pose.p is not None})),
        modes=tuple(sorted({pose.mode for pose in poses})),
        speeds=tuple(sorted({pose.speed for pose in poses if pose.speed is not None})),
        sag=sag, uncompensated=uncompensated, compensation=stats_of([pose.compensation for pose in poses])
        if active else None, up=up, down=down, largest_error=max(up.largest, down.largest),
        friction_mean=float(np.mean(frictions)), friction_max=max(frictions), rms_reduction=reduction, limit=limit,
        over=over)


def _degrees(positions: float) -> float:
    return positions * DEGREES_PER_POSITION


def _joined(values: Iterable[float]) -> str:
    return ', '.join('{:g}'.format(value) for value in values)


def _limit_text(limit: float) -> str:
    """the limit with a decimal point, e.g. '1.0' and '0.5'"""
    text = '{:g}'.format(limit)
    return text if any(character in text for character in '.en') else text + '.0'


def _whole_degrees(angles: np.ndarray) -> str:
    """the angles of joints 1 to 6 in whole degrees, '-' for a joint that has no row at the pose"""
    return ' '.join('{:>4s}'.format('-' if math.isnan(angle) else str(int(round(math.degrees(angle)))))
                    for angle in angles[1:7])


def _stats_line(label: str, stats: Stats) -> list[str]:
    return ['  {}:'.format(label),
            '    mean {:+.2f}, rms {:.2f}, lowest {:+.2f}, highest {:+.2f} positions'.format(
                stats.mean, stats.rms, stats.low, stats.high),
            '    rms {:.3f} degrees, largest absolute value {:.2f} positions ({:.3f} degrees)'.format(
                _degrees(stats.rms), stats.largest, _degrees(stats.largest))]


def describe(summary: JointSummary, worst: int = DEFAULT_WORST) -> str:
    """
    the report of one joint as text
    :param worst: the most poses over the limit that are listed, the others are counted
    """
    lines = ['joint {}: {} poses with an error in both directions, P {}, mode {}'.format(
        summary.joint, summary.count, _joined(summary.gains) if summary.gains else 'unknown',
        ' and '.join(summary.modes))]
    if 'sweep' in summary.modes:
        speeds = ' ({} degrees/s)'.format(_joined(summary.speeds)) if summary.speeds else ''
        lines.append('  sweep{}: the errors of the two directions include the lag of the speed, their mean is the '
                     'sag'.format(speeds))
    if len(summary.modes) > 1:
        lines.append('  warning: the rows are of both modes ({}), the statistics mix them'.format(
            ' and '.join(summary.modes)))
    lines += _stats_line('sag that is left (mean of up and down, with the compensation as recorded)', summary.sag)
    if summary.compensation is None:
        lines.append('  no compensation was active in these rows: the sag above is the sag of the joint')
    else:
        lines += _stats_line('sag without the compensation (sag less compensation)', summary.uncompensated)
        if summary.rms_reduction is not None:
            lines.append('  rms {:.2f} without, {:.2f} with: {:.0f}% {}'.format(
                summary.uncompensated.rms, summary.sag.rms, abs(summary.rms_reduction),
                'less' if summary.rms_reduction >= 0 else 'more'))
        lines.append('  compensation added: from {:+.1f} to {:+.1f} positions, mean {:+.1f}'.format(
            summary.compensation.low, summary.compensation.high, summary.compensation.mean))
    lines.append('  arriving from below (up): mean {:+.2f}, from {:+.2f} to {:+.2f} positions'.format(
        summary.up.mean, summary.up.low, summary.up.high))
    lines.append('  arriving from above (down): mean {:+.2f}, from {:+.2f} to {:+.2f} positions'.format(
        summary.down.mean, summary.down.low, summary.down.high))
    lines.append('  largest absolute error of either direction: {:.2f} positions ({:.3f} degrees)'.format(
        summary.largest_error, _degrees(summary.largest_error)))
    lines.append('  friction band (half the up/down difference, not compensated): mean {:.2f}, largest {:.2f} '
                 'positions'.format(summary.friction_mean, summary.friction_max))
    lines += _over_lines(summary, worst)
    return '\n'.join(lines)


def _over_lines(summary: JointSummary, worst: int) -> list[str]:
    """the poses over the limit, at most worst of them, and the verdict line"""
    lines = []
    if summary.over:
        lines.append('  poses over {} positions, largest sag first (angles of joints 1 to 6 in degrees):'.format(
            _limit_text(summary.limit)))
        lines.append('    {:>{width}s}  {:>6s}  {:>7s}  {:>7s}  {:>7s}'.format(
            'q1 .. q6', 'comp', 'up', 'down', 'sag', width=_angles_width()))
        for pose in summary.over[:max(worst, 0)]:
            lines.append('    {}  {:+6.1f}  {:+7.2f}  {:+7.2f}  {:+7.2f}'.format(
                _whole_degrees(pose.angles), pose.compensation, pose.error_up, pose.error_down, pose.sag))
        if len(summary.over) > max(worst, 0):
            lines.append('    and {} more'.format(len(summary.over) - max(worst, 0)))
        lines.append('  {} of {} poses are over {} positions'.format(
            len(summary.over), summary.count, _limit_text(summary.limit)))
    else:
        lines.append('  the sag stays within {} positions at all {} poses'.format(
            _limit_text(summary.limit), summary.count))
    return lines


def _angles_width() -> int:
    return len(_whole_degrees(np.zeros(7)))


def table(joint: int, poses: Sequence[PoseJoint]) -> str:
    """
    every pose of a joint in one aligned table, sorted by the angle of the joint and then by the other angles so a
    trend along the range of the joint can be read off the sag column
    """
    def key(pose: PoseJoint) -> tuple[float, ...]:
        # a joint without a row at the pose has nan, which has no order: it goes last
        angles = [math.inf if math.isnan(angle) else float(angle) for angle in pose.angles[1:7]]
        return (angles[joint - 1], *angles)

    with_speed = any(pose.speed is not None for pose in poses)
    header = '  ' + ' '.join('{:>7s}'.format('q{}'.format(joint_id)) for joint_id in ARM_JOINTS)
    if with_speed:
        header += '  {:>6s}'.format('deg/s')
    header += '  {:>6s}  {:>7s}  {:>7s}  {:>7s}'.format('comp', 'up', 'down', 'sag')
    lines = ['joint {}: every pose, by the angle of the joint (angles in degrees, errors in positions)'.format(joint),
             header]
    for pose in sorted(poses, key=key):
        line = '  ' + ' '.join('{:>7s}'.format('-' if math.isnan(angle) else '{:.1f}'.format(math.degrees(angle)))
                               for angle in pose.angles[1:7])
        if with_speed:
            line += '  {:>6s}'.format('' if pose.speed is None else '{:g}'.format(pose.speed))
        line += '  {:+6.1f}  {:+7.2f}  {:+7.2f}  {:+7.2f}'.format(
            pose.compensation, pose.error_up, pose.error_down, pose.sag)
        lines.append(line)
    return '\n'.join(lines)


def report(poses: dict[int, list[PoseJoint]], joints: Sequence[int], limit: float = DEFAULT_LIMIT,
           worst: int = DEFAULT_WORST, with_table: bool = False) -> str:
    """
    the report of the joints as text, a paragraph per joint, with a line for a joint that has no data
    :param poses: mapping of joint to its PoseJoints, see poses_from_rows
    :param joints: the joints to report, in this order
    """
    paragraphs = []
    for joint in joints:
        if not poses.get(joint):
            paragraphs.append('joint {}: no poses with an error in both directions'.format(joint))
            continue
        paragraph = describe(summarise(joint, poses[joint], limit), worst)
        if with_table:
            paragraph += '\n\n' + table(joint, poses[joint])
        paragraphs.append(paragraph)
    return '\n\n'.join(paragraphs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='report how good the static load compensation was in recordings')
    parser.add_argument('files', nargs='+', help='csv files of record_static_load.py or record_load_sweep.py')
    parser.add_argument('--joints', type=int, nargs='+', help='the joints to report, default every arm joint with data')
    parser.add_argument('--limit', type=float, default=DEFAULT_LIMIT,
                        help='the absolute sag in positions above which a pose is listed')
    parser.add_argument('--worst', type=int, default=DEFAULT_WORST, help='the most poses over the limit to list')
    parser.add_argument('--table', action='store_true', help='also print every pose of every joint')
    args = parser.parse_args(argv)

    try:
        poses = poses_from_rows(read_rows(args.files))
    except (OSError, ValueError, csv.Error) as e:  # a missing file, a csv that is not one, a number that can't be read
        print('can not read the recordings: {}'.format(e))
        return 1
    joints = list(dict.fromkeys(args.joints)) if args.joints else sorted(poses)
    if not any(poses.get(joint) for joint in joints):
        print('no poses with an error in both directions for {} in {}'.format(
            'joints {}'.format(' '.join(str(joint) for joint in joints)) if joints else 'any arm joint', args.files))
        return 1
    print(report(poses, joints, args.limit, args.worst, args.table))
    return 0


if __name__ == '__main__':
    sys.exit(main())
