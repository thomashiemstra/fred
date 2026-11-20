"""
Fits the static load compensation of the arm servos to the data of record_static_load.py and prints the
`compensation` blocks for servo_config.yaml:
    python -m src.robot_controllers.dynamixel_robot.calibration.fit_static_load static_load.csv [more.csv ...]
        [--joints 2 3 5] [--holdout 0.2] [--seed 0] [--max-positions 60]

The fit is in pwm units, the holding drive of the joint, so it stays valid when the P gain is retuned (a fit in
positions would only hold for the P it was recorded at). The csv has per servo, pose and direction the error, present
minus the UNCOMPENSATED goal in positions, and the compensation in positions that was added to the goal. With a
compensation active the error is only the sag that remains: the whole load is error - compensation, and that is what
is fitted. The servos run a P-only position loop, so they drive with pwm = P / 128 * (goal sent - present), where
goal sent - present = -(error - compensation). Every row is converted with its own P (column p of the csv, the gain the
servo ran with), so recordings at different gains can be fitted together:
    drive = -(error - compensation) * p / 128
The drive is positive towards higher positions. A row with an error needs a p, a csv recorded before the gains were
logged can't be fitted. A row without an error (a servo that was not measured) only gives the angle of its joint to
the pose, and a joint has a sample at a pose only when it has an error in both directions.

For every joint the mean of the drives of the two directions at a pose is its static load, which is gravity: friction
rests the joint off its goal with the sign of the direction it arrived from, so it cancels in the mean. Half the
difference is the friction, reported to show how large it is; it is not compensated (the compensation is for gravity
only). The static load is fitted with linear least squares on the basis of static_compensation.py. A share of the
poses is held out to report how well the fit generalises. The report gives the errors in pwm units and, at the P of
the joint in the csv (the mean when there are several), in positions and degrees. When the csv has the measured
present pwm, the report compares it to the drive computed from the position error, which checks the factor 128.
"""
import argparse
import csv
import math
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from src.robot_controllers.dynamixel_robot.dynamixel_x_config import PWM_PER_POSITION_PER_P
from src.robot_controllers.dynamixel_robot.calibration.load_directions import DOWN, UP
from src.robot_controllers.dynamixel_robot.static_compensation import regressor_row, term_names

DEGREES_PER_POSITION = 360.0 / 4096
DEFAULT_JOINTS = (2, 3, 5)


class Sample:
    """one pose with the drive of every servo: the static load (mean of both directions) and the friction"""

    def __init__(self, angles: np.ndarray, static: dict[int, float], hysteresis: dict[int, float],
                 pwm: dict[int, float] | None = None) -> None:
        self.angles = angles  # 7 element array, radians
        self.static = static  # mapping of servo id to the mean drive in pwm units
        # mapping of servo id to half the difference between the directions in pwm units: the friction, which is
        # not compensated
        self.hysteresis = hysteresis
        self.pwm = pwm if pwm is not None else {}  # mapping of servo id to the mean measured pwm of both directions


MAX_CONDITION = 1000.0  # above this the fit is warned about: some terms can't be told apart in the data


class Fit:
    """the result of fitting one joint, in pwm units"""

    def __init__(self, joint: int, names: Sequence[str], coefficients: np.ndarray, standard_errors: np.ndarray,
                 condition: float, rms_train: float, rms_holdout: float | None, train_count: int,
                 holdout_count: int) -> None:
        self.joint = joint
        self.names = names
        self.coefficients = dict(zip(names, coefficients))
        self.standard_errors = dict(zip(names, standard_errors))
        self.condition = condition
        self.rms_train = rms_train
        self.rms_holdout = rms_holdout  # None without hold-out poses
        self.train_count = train_count
        self.holdout_count = holdout_count

    def undetermined(self) -> list[str]:
        """the terms whose coefficient is smaller than twice its standard error: the data doesn't pin them down"""
        return [name for name in self.names
                if abs(self.coefficients[name]) < 2 * self.standard_errors[name]]

    def warnings(self) -> list[str]:
        lines = []
        if self.condition > MAX_CONDITION:
            lines.append('condition number {:.0f}: some terms can not be told apart in this data, vary the joints '
                         'more independently or fit fewer terms'.format(self.condition))
        if self.undetermined():
            lines.append('not determined by the data (smaller than twice their standard error): {}'.format(
                ', '.join(self.undetermined())))
        return lines


def read_rows(paths: Iterable[str]) -> list[dict[str, str]]:
    """the rows of the csv files of record_static_load.py, with the file they came from under 'file'"""
    rows = []
    for path in paths:
        with open(path, newline='') as csv_file:
            for row in csv.DictReader(csv_file):
                row['file'] = path
                rows.append(row)
    return rows


def _blank(value: str | None) -> bool:
    return value is None or value.strip() == ''


def _gain(row: dict[str, str]) -> float:
    """the P gain of the servo of a row that has an error; raises a ValueError naming the file when it has none"""
    try:
        p = float(row['p']) if not _blank(row.get('p')) else 0.0
    except ValueError:
        p = 0.0
    if not math.isfinite(p) or p <= 0:
        raise ValueError('{}: pose {} servo {} has an error but no usable p, the csv was recorded before the gains '
                         'were logged: record it again with the current record_static_load.py'.format(
                             row.get('file', 'the csv'), row.get('pose'), row.get('servo')))
    return p


def _drive(row: dict[str, str], p: float) -> float:
    """the holding drive in pwm units of a row: the servo drives with P / 128 * (goal sent - present), and that
    difference is minus the error once the compensation, which was added to the goal, is taken out of it"""
    compensation = 0.0 if _blank(row.get('compensation')) else float(row['compensation'])
    return -(float(row['error']) - compensation) * p * PWM_PER_POSITION_PER_P


def gains_of_joints(rows: Iterable[dict[str, str]]) -> dict[int, list[float]]:
    """mapping of servo id to the P gain of every row of that servo that has an error"""
    gains: defaultdict[int, list[float]] = defaultdict(list)
    for row in rows:
        if not _blank(row.get('error')):
            gains[int(row['servo'])].append(_gain(row))
    return dict(gains)


def samples_from_rows(rows: Iterable[dict[str, str]]) -> list[Sample]:
    """
    group the rows per pose: the angles of the pose and the holding drive per servo for both directions. Poses
    without both directions are left out, and so are poses where no servo has an error in both. A row without an
    error only gives the angle of its servo. The pose index only counts within a file, so rows need a 'file' field
    when they come from more than one file.
    :raises ValueError: for a row with an error and without a P gain
    """
    # (file, pose) -> direction -> servo -> drive in pwm units, and the same for the measured pwm
    drives: defaultdict[tuple[str, int], defaultdict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    pwms: defaultdict[tuple[str, int], defaultdict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    angles_of_pose: dict[tuple[str, int], np.ndarray] = {}
    for row in rows:
        key = (row.get('file', ''), int(row['pose']))
        servo = int(row['servo'])
        angles_of_pose.setdefault(key, np.zeros(7))[servo] = float(row['angle'])
        if _blank(row.get('error')):
            continue
        drives[key][row['direction']][servo] = _drive(row, _gain(row))
        if not _blank(row.get('pwm')):
            pwms[key][row['direction']][servo] = float(row['pwm'])

    samples = []
    for key, directions in drives.items():
        if UP not in directions or DOWN not in directions:
            continue
        servos = set(directions[UP]) & set(directions[DOWN])
        if not servos:
            continue
        static = {servo: (directions[UP][servo] + directions[DOWN][servo]) / 2.0 for servo in servos}
        hysteresis = {servo: (directions[UP][servo] - directions[DOWN][servo]) / 2.0 for servo in servos}
        pwm = {servo: (pwms[key][UP][servo] + pwms[key][DOWN][servo]) / 2.0 for servo in servos
               if servo in pwms[key][UP] and servo in pwms[key][DOWN]} if key in pwms else {}
        samples.append(Sample(angles_of_pose[key], static, hysteresis, pwm))
    return samples


def fit_joint(joint: int, samples: Sequence[Sample], holdout: float = 0.2, rng: np.random.Generator | None = None,
              names: Sequence[str] | None = None) -> Fit:
    """
    fit the terms of a joint to the samples
    :param holdout: share of the poses kept out of the fit to measure the error on
    :param names: the terms to fit, by default all terms of the joint
    :return: Fit, in pwm units like the samples
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    if names is None:
        names = term_names(joint)
    rows: Any = []
    targets: Any = []
    for sample in samples:
        if joint not in sample.static:
            continue
        rows.append(regressor_row(joint, sample.angles, names))
        targets.append(sample.static[joint])
    if len(rows) < len(names):
        raise ValueError('joint {} has {} samples, at least {} are needed for {} terms'.format(
            joint, len(rows), len(names), len(names)))
    rows, targets = np.array(rows), np.array(targets)

    holdout_count = int(round(holdout * len(rows))) if len(rows) > len(names) + 1 else 0
    indices = rng.permutation(len(rows))
    holdout_indices, train_indices = indices[:holdout_count], indices[holdout_count:]

    train_rows, train_targets = rows[train_indices], targets[train_indices]
    coefficients, _, _, _ = np.linalg.lstsq(train_rows, train_targets, rcond=None)
    condition = float(np.linalg.cond(train_rows))
    residuals = train_rows @ coefficients - train_targets
    rms_train = _rms(residuals)
    rms_holdout = _rms(rows[holdout_indices] @ coefficients - targets[holdout_indices]) if holdout_count else None
    # the standard errors of the coefficients from the residual variance
    degrees_of_freedom = max(len(train_rows) - len(names), 1)
    variance = float(np.sum(np.square(residuals))) / degrees_of_freedom
    standard_errors = np.sqrt(np.clip(np.diag(np.linalg.pinv(train_rows.T @ train_rows)) * variance, 0, None))
    return Fit(joint, names, coefficients, standard_errors, condition, rms_train, rms_holdout, len(train_indices),
               holdout_count)


def _rms(residuals: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(residuals)))) if len(residuals) else 0.0


def pwm_ratio(joint: int, samples: Iterable[Sample]) -> float | None:
    """
    the least squares ratio of the measured pwm (mean of both directions) to the drive computed from the position
    error, 1.0 when the factor P / 128 is right; None without measured pwm or without any drive
    """
    pairs = [(sample.pwm[joint], sample.static[joint]) for sample in samples if joint in sample.pwm]
    if not pairs:
        return None
    measured, computed = np.array(pairs).T
    denominator = float(np.sum(np.square(computed)))
    return float(np.sum(measured * computed)) / denominator if denominator > 0 else None


def yaml_block(fits: dict[int, Fit], max_positions: int) -> str:
    """
    the compensation blocks for servo_config.yaml. The fits are of the holding drive in pwm units (positive towards
    higher positions, the sign is in the conversion of the rows), which is what the compensation takes: its
    coefficients are the fitted ones as they are.
    :param fits: mapping of joint to Fit of the static load
    :param max_positions: the clamp of the compensation, in positions
    """
    lines = []
    for joint in sorted(fits):
        lines.append('  {}:'.format(joint))
        lines.append('    compensation:')
        lines.append('      max_positions: {}'.format(max_positions))
        lines.append('      terms: {}'.format(_terms_line(fits[joint].coefficients)))
    return '\n'.join(lines) + '\n'


def _terms_line(coefficients: dict[str, float]) -> str:
    """the coefficients of the compensation in pwm units, as fitted"""
    return '{' + ', '.join('{}: {:.3f}'.format(name, value) for name, value in coefficients.items()) + '}'


def in_positions(drive: float, p: float | None) -> str:
    """a drive in pwm units as positions and degrees at the P gain, e.g. ' (1.2 positions, 0.11 degrees at P 1000)';
    empty without a P"""
    if p is None:
        return ''
    positions = drive / (p * PWM_PER_POSITION_PER_P)
    return ' ({:.1f} positions, {:.2f} degrees at P {:g})'.format(positions, positions * DEGREES_PER_POSITION, p)


def describe(fit: Fit, p: float | None = None) -> str:
    """
    one paragraph about a fit
    :param p: the P gain of the joint in the recording; with it the errors are also given in positions and degrees
    """
    lines = ['joint {}: {} poses fitted, {} held out, condition {:.1f}'.format(
        fit.joint, fit.train_count, fit.holdout_count, fit.condition)]
    for name, value in fit.coefficients.items():
        lines.append('  {:12s} {:+9.3f} pwm (standard error {:.3f}){}'.format(
            name, value, fit.standard_errors[name], '  <- not determined' if name in fit.undetermined() else ''))
    lines.append('  rms error on the fitted poses {:.1f} pwm{}'.format(fit.rms_train, in_positions(fit.rms_train, p)))
    if fit.rms_holdout is not None:
        lines.append('  rms error on the held out poses {:.1f} pwm{}'.format(
            fit.rms_holdout, in_positions(fit.rms_holdout, p)))
    for warning in fit.warnings():
        lines.append('  warning: ' + warning)
    return '\n'.join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='fit the static load compensation to recorded data')
    parser.add_argument('files', nargs='+', help='csv files of record_static_load.py')
    parser.add_argument('--joints', type=int, nargs='+', default=list(DEFAULT_JOINTS))
    parser.add_argument('--holdout', type=float, default=0.2, help='share of the poses held out of the fit')
    parser.add_argument('--seed', type=int, default=0, help='seed of the hold-out selection')
    parser.add_argument('--max-positions', type=int, default=60, help='clamp of the compensation in the yaml')
    args = parser.parse_args(argv)

    rows = read_rows(args.files)
    try:
        samples = samples_from_rows(rows)
        gains = gains_of_joints(rows)
    except ValueError as e:
        print(e)
        return 1
    if not samples:
        print('no poses with both directions in {}'.format(args.files))
        return 1
    print('{} poses with both directions'.format(len(samples)))

    fits = {}
    for joint in args.joints:
        hysteresis = [abs(sample.hysteresis[joint]) for sample in samples if joint in sample.hysteresis]
        if not hysteresis:
            print('\njoint {}: no data'.format(joint))
            continue
        # the fit converts every row with its own P, the report with the mean of them
        p_seen = sorted(set(gains.get(joint, [])))
        p = float(np.mean(gains[joint])) if joint in gains else None
        fits[joint] = fit_joint(joint, samples, args.holdout, np.random.default_rng(args.seed))
        print()
        print(describe(fits[joint], p))
        if p is not None and len(p_seen) > 1:
            print('  P values seen: {}, the errors in positions use their mean {:.0f}'.format(
                ', '.join('{:g}'.format(value) for value in p_seen), p))
        mean_hysteresis, max_hysteresis = float(np.mean(hysteresis)), float(np.max(hysteresis))
        print('  friction (half the up/down difference, not compensated): mean {:.1f} pwm{}, max {:.0f} pwm{}'.format(
            mean_hysteresis, in_positions(mean_hysteresis, p), max_hysteresis, in_positions(max_hysteresis, p)))
        ratio = pwm_ratio(joint, samples)
        if ratio is not None:
            print('  measured pwm is {:.2f} times the drive computed from the position error (1.0 expected)'.format(
                ratio))

    print('\nfor servo_config.yaml (the holding drive in pwm units, positive towards higher positions; the '
          'compensation converts it to positions with the P gain of the servo):')
    print(yaml_block(fits, args.max_positions))
    return 0


if __name__ == '__main__':
    sys.exit(main())
