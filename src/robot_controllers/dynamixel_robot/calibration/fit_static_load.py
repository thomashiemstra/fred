"""
Fits the static load compensation of the arm servos to the data of record_static_load.py and prints the
`compensation` blocks for servo_config.yaml:
    python -m src.robot_controllers.dynamixel_robot.calibration.fit_static_load static_load.csv [more.csv ...]
        [--joints 2 3 5] [--holdout 0.2] [--seed 0] [--split] [--max-positions 60]

For every joint the mean of the errors of the two directions at a pose is its static load, half the difference is
the friction hysteresis. The static load is fitted with linear least squares on the basis of static_compensation.py
(gravity; the spring terms only when servo_config.yaml has a spring section, which it hasn't since the spring was
removed from the robot on 23 October 2026). A share of the poses is held out to report how
well the fit generalises, in positions and degrees. --split fits the two directions separately as well and prints
them as the up and down blocks, which compensates the hysteresis too.
"""
import argparse
import csv
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from src.robot_controllers.dynamixel_robot.servo_config import load_servo_config
from src.robot_controllers.dynamixel_robot.static_compensation import regressor_row, term_names, \
    UP, DOWN, SpringGeometry

DEGREES_PER_POSITION = 360.0 / 4096
DEFAULT_JOINTS = (2, 3, 5)


class Sample:
    """one pose with the error of every servo: the static load (mean of both directions) and the hysteresis"""

    def __init__(self, angles: np.ndarray, static: dict[int, float], hysteresis: dict[int, float],
                 by_direction: dict[str, dict[int, float]]) -> None:
        self.angles = angles  # 7 element array, radians
        self.static = static  # mapping of servo id to the mean error in positions
        self.hysteresis = hysteresis  # mapping of servo id to half the difference between the directions
        self.by_direction = by_direction  # mapping of direction to mapping of servo id to error


MAX_CONDITION = 1000.0  # above this the fit is warned about: some terms can't be told apart in the data


class Fit:
    """the result of fitting one joint"""

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


def samples_from_rows(rows: Iterable[dict[str, str]]) -> list[Sample]:
    """
    group the rows per pose: the angles of the pose and the error per servo for both directions. Poses without
    both directions are left out. The pose index only counts within a file, so rows need a 'file' field when they
    come from more than one file.
    """
    by_pose: defaultdict[tuple[str, int], defaultdict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))  # (file, pose) -> direction -> servo -> error
    angles_of_pose: dict[tuple[str, int], np.ndarray] = {}
    for row in rows:
        key = (row.get('file', ''), int(row['pose']))
        servo = int(row['servo'])
        by_pose[key][row['direction']][servo] = float(row['error'])
        angles_of_pose.setdefault(key, np.zeros(7))[servo] = float(row['angle'])

    samples = []
    for key, directions in by_pose.items():
        if UP not in directions or DOWN not in directions:
            continue
        servos = set(directions[UP]) & set(directions[DOWN])
        static = {servo: (directions[UP][servo] + directions[DOWN][servo]) / 2.0 for servo in servos}
        hysteresis = {servo: (directions[UP][servo] - directions[DOWN][servo]) / 2.0 for servo in servos}
        samples.append(Sample(angles_of_pose[key], static, hysteresis, directions))
    return samples


def fit_joint(joint: int, samples: Sequence[Sample], spring: SpringGeometry | None = None, holdout: float = 0.2,
              rng: np.random.Generator | None = None, direction: str | None = None,
              names: Sequence[str] | None = None) -> Fit:
    """
    fit the terms of a joint to the samples
    :param spring: SpringGeometry, with it the spring terms are fitted too
    :param holdout: share of the poses kept out of the fit to measure the error on
    :param direction: fit the errors of one direction instead of the static load
    :param names: the terms to fit, by default all terms of the joint
    :return: Fit
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    if names is None:
        names = term_names(joint, spring is not None)
    rows: Any = []
    targets: Any = []
    for sample in samples:
        errors = sample.static if direction is None else sample.by_direction[direction]
        if joint not in errors:
            continue
        rows.append(regressor_row(joint, sample.angles, spring, names))
        targets.append(errors[joint])
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


def yaml_block(fits: dict[int, Fit], max_positions: int, directional: dict[int, dict[str, Fit]] | None = None) -> str:
    """
    the compensation blocks for servo_config.yaml. The fits describe the error (present minus goal) and the
    compensation is added to the goal to cancel it, so the coefficients in the yaml are the negated fits.
    :param fits: mapping of joint to Fit of the static load
    :param directional: optional mapping of joint to {UP: Fit, DOWN: Fit}
    """
    lines = []
    for joint in sorted(fits):
        lines.append('  {}:'.format(joint))
        lines.append('    compensation:')
        lines.append('      max_positions: {}'.format(max_positions))
        lines.append('      terms: {}'.format(_terms_line(fits[joint].coefficients)))
        if directional and joint in directional:
            for direction in (UP, DOWN):
                lines.append('      {}: {}'.format(direction, _terms_line(directional[joint][direction].coefficients)))
    return '\n'.join(lines) + '\n'


def _terms_line(error_coefficients: dict[str, float]) -> str:
    """the coefficients of the compensation: minus the coefficients of the error"""
    return '{' + ', '.join('{}: {:.3f}'.format(name, -value) for name, value in error_coefficients.items()) + '}'


def describe(fit: Fit) -> str:
    """one paragraph about a fit"""
    lines = ['joint {}: {} poses fitted, {} held out, condition {:.1f}'.format(
        fit.joint, fit.train_count, fit.holdout_count, fit.condition)]
    for name, value in fit.coefficients.items():
        lines.append('  {:12s} {:+9.3f} positions (standard error {:.3f}){}'.format(
            name, value, fit.standard_errors[name], '  <- not determined' if name in fit.undetermined() else ''))
    lines.append('  rms error on the fitted poses {:.1f} positions ({:.2f} degrees)'.format(
        fit.rms_train, fit.rms_train * DEGREES_PER_POSITION))
    if fit.rms_holdout is not None:
        lines.append('  rms error on the held out poses {:.1f} positions ({:.2f} degrees)'.format(
            fit.rms_holdout, fit.rms_holdout * DEGREES_PER_POSITION))
    for warning in fit.warnings():
        lines.append('  warning: ' + warning)
    return '\n'.join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='fit the static load compensation to recorded data')
    parser.add_argument('files', nargs='+', help='csv files of record_static_load.py')
    parser.add_argument('--joints', type=int, nargs='+', default=list(DEFAULT_JOINTS))
    parser.add_argument('--holdout', type=float, default=0.2, help='share of the poses held out of the fit')
    parser.add_argument('--seed', type=int, default=0, help='seed of the hold-out selection')
    parser.add_argument('--split', action='store_true', help='also fit the two directions separately')
    parser.add_argument('--max-positions', type=int, default=60, help='clamp of the compensation in the yaml')
    parser.add_argument('--no-spring', action='store_true',
                        help='fit without the spring terms even when the yaml has a spring section')
    args = parser.parse_args(argv)

    samples = samples_from_rows(read_rows(args.files))
    if not samples:
        print('no poses with both directions in {}'.format(args.files))
        return 1
    spring = None if args.no_spring else load_servo_config().spring
    print('{} poses with both directions, spring {}'.format(len(samples), 'from servo_config.yaml' if spring else 'not modelled'))

    fits, directional = {}, {}
    for joint in args.joints:
        hysteresis = [abs(sample.hysteresis[joint]) for sample in samples if joint in sample.hysteresis]
        if not hysteresis:
            print('\njoint {}: no data'.format(joint))
            continue
        fits[joint] = fit_joint(joint, samples, spring, args.holdout, np.random.default_rng(args.seed))
        print()
        print(describe(fits[joint]))
        print('  hysteresis (half the up/down difference): mean {:.1f}, max {:.0f} positions'.format(
            np.mean(hysteresis), np.max(hysteresis)))  # type: ignore[str-format]  # numpy floating is not a float for mypy
        if args.split:
            directional[joint] = {direction: fit_joint(joint, samples, spring, args.holdout,
                                                       np.random.default_rng(args.seed), direction)
                                  for direction in (UP, DOWN)}
            for direction in (UP, DOWN):
                fit = directional[joint][direction]
                print('  {}: rms {:.1f} positions on the fitted poses{}'.format(
                    direction, fit.rms_train,
                    '' if fit.rms_holdout is None else ', {:.1f} held out'.format(fit.rms_holdout)))

    print('\nfor servo_config.yaml (the fits are of the error, present minus goal; the compensation is added to the '
          'goal to cancel it, so these are the negated fits):')
    print(yaml_block(fits, args.max_positions, directional if args.split else None))
    return 0


if __name__ == '__main__':
    sys.exit(main())
