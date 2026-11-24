"""
Fits the coefficient of the velocity lead and the friction drive of the arm servos to the feedforward recordings of
tune_servo_gains.py (tune_joint<N>_<date>_<time>_ff.csv) and prints the `velocity_feedforward` blocks for
servo_config.yaml:
    python -m src.robot_controllers.dynamixel_robot.calibration.fit_velocity_lead FILE.csv [FILE.csv ...]
        [--max-positions 30]

The servos run a P-only position loop, pwm = P / 128 * (goal - present). A turning motor needs a drive in proportion
to its speed (its back voltage), the only source of that drive is P times the error, so a moving joint trails its
trajectory by an error that grows with the speed. The PC shifts every goal by exactly that error:
    lead in positions = coefficient * commanded velocity * 128 / P
with the coefficient in pwm units per degree/s of the joint. It is fitted in pwm units so that it stays valid when P is
retuned (in positions it would only hold for the P it was recorded at).

The lead has a second term, friction: the motor needs a fixed extra drive to keep the joint sliding, in the direction
of the motion and whatever the speed (pwm_for_friction in pwm units, see velocity_feedforward.py). It is the intercept
of the lag fit below: friction drive = friction_positions * P / 128, not below 0, the third line of the yaml block.

In a recording one joint sweeps to and fro at a few speeds, every leg ramping up, cruising at constant speed and
ramping down. Only the cruise rows are used. Per speed and
direction the script takes the mean drive (pwm * direction) and the mean error behind the servo's trajectory
((position_trajectory - position) * direction, positive when the servo trails; the trajectory includes the lead the PC
added, so this is the whole error the P term works on, with or without a lead). The two directions are averaged, so
the pull of gravity, which pushes the same way in both while the direction factor flips it, cancels. Per group of rows
(one file, joint, lead gain and friction drive: a recording with and without a lead is two groups that follow each
other) two least squares lines against the speed in degrees/s are fitted:
    pwm = friction_pwm + k_pwm * speed          measured directly, the present pwm of the servo
    error = friction_positions + slope * speed  from the lag; k_lag = slope * P / 128 needs the P of the servo
The two coefficients are the same quantity found in two ways, so k_pwm / k_lag is 1.0 when the factor 128 and the
reasoning hold. The P is the p column of the csv. The value for the yaml is k_lag when the P is known: the lead works
through P times the error, so the coefficient fitted from the error is the one that cancels the slope of the error by
construction; a group whose p cells are blank has no P and gives k_pwm.
The friction part is what does not depend on the speed: the speed term of the lead does not touch it. The error the
lines work on includes the lead and the friction drive that were in force, so the fit gives the same coefficient and
the same friction drive whatever was switched on.

How to read the output: the table has a row per speed with the cruise samples, the mean pwm and error, the mean lead
that was added to the goal (0 without a lead), the lag behind the INTENDED trajectory (error - lead: what the servo
trails the path the planner meant) and the std of the error and of the pwm, which show whether a lead made the joint
rougher. Without a lead the lag behind the intended trajectory is the error itself. With the right lead it is flat
over the speeds, only the friction part is left, and the report says whether it changes by more than a position. With
the friction drive in the lead as well, nothing is left: the next line gives the mean of the lag behind the intended
trajectory over the speeds and says whether every speed is within a position of zero (the friction part is taken out).
A ratio far from 1 is a finding about the pwm reading or the factor 128, not necessarily a bug of the script. A
recording without the columns p, lead and lead_gain (those made before 21 November 2026 lack them) is refused: record
the joint again with tune_servo_gains.py --ff-only. The column lead_friction, the pwm_for_friction in force during the
sweep, is optional: a recording without it, or a blank cell, has no friction drive.
"""
import argparse
import csv
import math
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from src.robot_controllers.dynamixel_robot.calibration.gain_tuning import FF_LAG_TOLERANCE, MIN_CRUISE_SAMPLES, fit_line
from src.robot_controllers.dynamixel_robot.dynamixel_x_config import PWM_PER_POSITION_PER_P

DEGREES_PER_POSITION = 360.0 / 4096
# positions the lag behind the intended trajectory may change over the speeds to count as flat; the feedforward stage
# of tune_servo_gains.py corrects its coefficient above the same value
LAG_TOLERANCE = FF_LAG_TOLERANCE
# positions the lag behind the intended trajectory may be from zero at every speed for the friction part to count as
# taken out (with the lead and the friction drive in force the lag should be 0, not only flat)
INTENDED_TOLERANCE = 1.0
RATIO_TOLERANCE = 0.15  # the ratio of the two coefficients may be this far from 1 before it is a warning
LEAD_SPEED = 10.0  # degrees per second at which the report says how large the lead is
REQUIRED_COLUMNS = ('joint', 'speed', 'direction', 'cruise', 'position', 'position_trajectory', 'pwm')
# the columns the stage logs since 21 November 2026: the P of the servo, the lead and its coefficient
LOGGED_COLUMNS = ('p', 'lead', 'lead_gain')


class GroupKey(NamedTuple):
    """what the rows of one fit have in common"""
    file: str
    joint: int
    lead_gain: float  # pwm per degree/s in force during the sweep, 0 for no lead
    lead_friction: float = 0.0  # pwm_for_friction in force during the sweep in pwm units, 0 for none


class SpeedStatistics(NamedTuple):
    """the cruise rows of one commanded speed, per direction and then averaged over the two directions, so that the
    pull of gravity cancels"""
    speed: float  # degrees per second
    samples: int  # the cruise samples of both directions
    pwm: float  # pwm units, in the direction of the move
    error: float  # positions the servo trails its trajectory (which includes the lead), in the direction of the move
    lead: float  # positions the lead added to the goal, in the direction of the move
    pwm_std: float
    error_std: float  # positions

    @property
    def intended(self) -> float:
        """positions the servo trails the path that was meant, without the lead"""
        return self.error - self.lead


@dataclass(frozen=True)
class VelocityFit:
    """the two lines fitted to the statistics of the speeds of a group"""
    joint: int
    lead_gain: float
    p: float | None  # None when the P of the servo is not known
    friction_pwm: float  # pwm units at no speed, in the direction of the move
    k_pwm: float  # pwm per degree/s, from the measured pwm
    friction_positions: float  # positions at no speed
    slope: float  # positions per degree/s
    slowest: SpeedStatistics
    fastest: SpeedStatistics
    intended_slope: float = 0.0  # positions per degree/s the lag behind the intended trajectory still grows by
    lead_friction: float = 0.0  # pwm_for_friction in force during the sweep in pwm units, 0 for none
    # the lag behind the intended trajectory over the speeds: what is left of the lag with the lead and the friction
    # drive that were in force
    intended_mean: float = 0.0  # positions, the mean of the speeds
    intended_largest: float = 0.0  # positions, the largest distance from zero of a speed

    @property
    def friction_drive(self) -> float | None:
        """
        pwm units the motor needs at no speed to keep the joint sliding, for pwm_for_friction in the yaml: the lag at no
        speed times P / 128, not below 0. None without a P.
        """
        return None if self.p is None else max(0.0, self.friction_positions * self.p * PWM_PER_POSITION_PER_P)

    @property
    def friction_taken_out(self) -> bool:
        """whether the lag behind the intended trajectory is within INTENDED_TOLERANCE of zero at every speed"""
        return self.intended_largest <= INTENDED_TOLERANCE

    @property
    def leftover_coefficient(self) -> float | None:
        """
        pwm per degree/s the lead of this group still lacks (negative when it is too large): the slope of the lag
        behind the intended trajectory times P / 128. The coefficient that would have flattened the lag is the one in
        force (lead_gain of the group) plus this. None without a P.
        """
        return None if self.p is None else self.intended_slope * self.p * PWM_PER_POSITION_PER_P

    @property
    def k_lag(self) -> float | None:
        """pwm per degree/s from the lag: slope * P / 128, None without a P"""
        return None if self.p is None else self.slope * self.p * PWM_PER_POSITION_PER_P

    @property
    def ratio(self) -> float | None:
        """k_pwm / k_lag, 1.0 when the factor 128 holds; None without a P or without a lag coefficient"""
        k_lag = self.k_lag
        return None if k_lag is None or k_lag == 0 else self.k_pwm / k_lag

    @property
    def coefficient(self) -> float:
        """the value for the yaml: k_lag when the P is known, else k_pwm"""
        k_lag = self.k_lag
        return self.k_pwm if k_lag is None else k_lag

    @property
    def intended_change(self) -> float:
        """positions the lag behind the intended trajectory changes between the slowest and the fastest speed"""
        return self.fastest.intended - self.slowest.intended


@dataclass(frozen=True)
class GroupResult:
    """one group of rows with what was made of it"""
    key: GroupKey
    p: float | None
    statistics: list[SpeedStatistics]
    notes: list[str]  # speeds that were left out and why
    fit: VelocityFit | None
    problem: str  # why there is no fit, empty with one


def read_rows(paths: Iterable[str]) -> list[dict[str, str]]:
    """
    the rows of the csv files of the feedforward stage, with the file they came from under 'file'
    :raises ValueError: for a file without the columns the fit needs
    :raises OSError: for a file that can't be read
    """
    rows = []
    for path in paths:
        with open(path, newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            missing = [name for name in REQUIRED_COLUMNS if name not in (reader.fieldnames or [])]
            if missing:
                raise ValueError('{}: no column {}, it is not a recording of the feedforward stage'.format(
                    path, ', '.join(missing)))
            not_logged = [name for name in LOGGED_COLUMNS if name not in (reader.fieldnames or [])]
            if not_logged:
                raise ValueError('{}: no column {}, the recording was made before the stage logged the gain, the lead '
                                 'and its coefficient (21 November 2026): record the joint again with '
                                 'tune_servo_gains.py --ff-only'.format(path, ', '.join(not_logged)))
            for row in reader:
                row['file'] = path
                rows.append(row)
    return rows


def _number(row: dict[str, str], name: str, default: float | None = None) -> float:
    """a column of a row as a float; a blank or absent column gives the default when there is one
    :raises ValueError: naming the file and the column when there is no default and no usable number"""
    value = row.get(name)
    if value is None or value.strip() == '':
        if default is None:
            raise ValueError('{}: a row has no {}'.format(row.get('file', 'the csv'), name))
        return default
    try:
        number = float(value)
    except ValueError:
        raise ValueError('{}: {} is {!r}, not a number'.format(row.get('file', 'the csv'), name, value)) from None
    if not math.isfinite(number):
        raise ValueError('{}: {} is {!r}, not a number'.format(row.get('file', 'the csv'), name, value))
    return number


def group_rows(rows: Iterable[dict[str, str]]) -> dict[GroupKey, list[dict[str, str]]]:
    """
    the rows per file, joint, lead gain and friction drive (a missing lead_friction column or a blank cell is 0), in
    the order of the files, then by joint and with the rows without a lead before those with one and those without a
    friction drive before those with one, so that the groups of one file that can be compared follow each other
    """
    groups: dict[GroupKey, list[dict[str, str]]] = defaultdict(list)
    order: dict[str, int] = {}
    for row in rows:
        order.setdefault(row['file'], len(order))
        key = GroupKey(row['file'], int(_number(row, 'joint')), _number(row, 'lead_gain', 0.0),
                       _number(row, 'lead_friction', 0.0))
        groups[key].append(row)
    return {key: groups[key] for key in sorted(
        groups, key=lambda k: (order[k.file], k.joint, k.lead_gain, k.lead_friction))}


def group_p(rows: Sequence[dict[str, str]]) -> tuple[float | None, str]:
    """
    the P gain the servo ran with: the mean of the p column of the rows (a blank cell counts as 0 and is left out)
    :return: the P (None when no row has one), a remark about it (may be empty)
    """
    values = [_number(row, 'p', 0.0) for row in rows]
    values = [value for value in values if value > 0]
    if values:
        distinct = sorted(set(values))
        remark = 'the P changes within the group: {}, the mean is used'.format(
            ', '.join('{:g}'.format(value) for value in distinct)) if len(distinct) > 1 else ''
        return float(np.mean(values)), remark
    return None, ''


def speed_statistics(rows: Sequence[dict[str, str]],
                     minimum: int = MIN_CRUISE_SAMPLES) -> tuple[list[SpeedStatistics], list[str]]:
    """
    the statistics of every commanded speed of the rows, the slowest first: the cruise rows of each direction of the
    position (1 when it rises) give the mean of pwm * direction, of (position_trajectory - position) * direction and of
    lead * direction (a blank lead counts as 0), the result is the mean of the two directions; the std is that of the
    rows of a direction, averaged the same way. A speed that lacks a direction or has fewer cruise rows than the
    minimum in one is left out.
    :return: the statistics, and a line for every speed that was left out
    """
    speeds = sorted({_number(row, 'speed') for row in rows})
    notes = []
    statistics = []
    for speed in speeds:
        means = []  # per direction: pwm, error, lead, pwm std, error std
        counts = {}
        for direction, name in ((1, 'up'), (-1, 'down')):
            chosen = [row for row in rows if _number(row, 'speed') == speed and _number(row, 'cruise') > 0
                      and _number(row, 'direction') == direction]
            counts[name] = len(chosen)
            pwm = np.array([_number(row, 'pwm') * direction for row in chosen])
            error = np.array([(_number(row, 'position_trajectory') - _number(row, 'position')) * direction
                              for row in chosen])
            lead = np.array([_number(row, 'lead', 0.0) * direction for row in chosen])
            if len(chosen) >= minimum:
                means.append((pwm.mean(), error.mean(), lead.mean(), pwm.std(), error.std()))
        if len(means) < 2:
            notes.append('speed {:g} degrees/s left out: {} cruise samples going up and {} going down, at least {} '
                         'are needed in both'.format(speed, counts['up'], counts['down'], minimum))
            continue
        pwm_mean, error_mean, lead_mean, pwm_std, error_std = (float(np.mean(column)) for column in zip(*means))
        statistics.append(SpeedStatistics(speed, counts['up'] + counts['down'], pwm_mean, error_mean, lead_mean,
                                          pwm_std, error_std))
    return statistics, notes


def fit_group(joint: int, lead_gain: float, statistics: Sequence[SpeedStatistics], p: float | None,
              lead_friction: float = 0.0) -> VelocityFit:
    """
    fit pwm = friction_pwm + k_pwm * speed and error = friction_positions + slope * speed over the speeds, and take
    the mean and the largest of the lag behind the intended trajectory over the speeds
    :param p: the P gain of the servo, None when it is not known (there is no k_lag then)
    :param lead_friction: the pwm_for_friction of the group, only noted in the fit
    :raises ValueError: with fewer than two different speeds
    """
    speeds = [s.speed for s in statistics]
    if len(set(speeds)) < 2:
        raise ValueError('{} different speed(s) with cruise samples in both directions, at least two are needed'.format(
            len(set(speeds))))
    pwm_line = fit_line(speeds, [s.pwm for s in statistics])
    error_line = fit_line(speeds, [s.error for s in statistics])
    intended = [s.intended for s in statistics]
    intended_fit = fit_line(speeds, intended)
    ordered = sorted(statistics, key=lambda s: s.speed)
    return VelocityFit(joint=joint, lead_gain=lead_gain, p=p, friction_pwm=pwm_line.friction, k_pwm=pwm_line.slope,
                       friction_positions=error_line.friction, slope=error_line.slope, slowest=ordered[0],
                       fastest=ordered[-1], intended_slope=intended_fit.slope, lead_friction=lead_friction,
                       intended_mean=float(np.mean(intended)), intended_largest=float(np.max(np.abs(intended))))


def analyse_group(key: GroupKey, rows: Sequence[dict[str, str]], minimum: int = MIN_CRUISE_SAMPLES) -> GroupResult:
    """the statistics and the fit of one group of rows; a group that can't be fitted has the reason in `problem`"""
    p, remark = group_p(rows)
    statistics, notes = speed_statistics(rows, minimum)
    if remark:
        notes.append(remark)
    try:
        fit, problem = fit_group(key.joint, key.lead_gain, statistics, p, key.lead_friction), ''
    except ValueError as e:
        fit, problem = None, str(e)
    return GroupResult(key, p, statistics, notes, fit, problem)


def lead_meaning(coefficient: float, p: float) -> str:
    """what a coefficient does at the P of the joint: the lead in positions at LEAD_SPEED degrees/s and the equivalent
    time, the milliseconds of travel the lead is ahead of the commanded position"""
    slope = coefficient * 128.0 / p  # positions per degree/s
    milliseconds = slope * DEGREES_PER_POSITION * 1000.0
    return 'at P {:g} the lead is {:.1f} positions at {:g} degrees/s, the position of {:.0f} ms of travel'.format(
        p, slope * LEAD_SPEED, LEAD_SPEED, milliseconds)


def yaml_block(joint: int, coefficient: float, max_positions: int, pwm_for_friction: float | None = None) -> str:
    """the velocity_feedforward block of a servo for servo_config.yaml; the coefficient is in pwm units per degree/s of
    the joint, the friction drive in pwm units (left out of the block when it is None)"""
    lines = ['  {}:'.format(joint),
             '    velocity_feedforward:',
             '      max_positions: {}'.format(max_positions),
             '      pwm_per_degree_per_second: {:.2f}'.format(coefficient)]
    if pwm_for_friction is not None:
        lines.append('      pwm_for_friction: {:.1f}'.format(pwm_for_friction))
    return '\n'.join(lines) + '\n'


def table(statistics: Sequence[SpeedStatistics]) -> str:
    """a row per speed: the samples, pwm, error, lead and lag behind the intended trajectory, and the stds"""
    lines = ['  {:>8s} {:>8s} {:>9s} {:>9s} {:>9s} {:>9s} {:>9s} {:>9s}'.format(
        'deg/s', 'samples', 'pwm', 'error', 'lead', 'intended', 'err std', 'pwm std'),
        '  {:>8s} {:>8s} {:>9s} {:>9s} {:>9s} {:>9s} {:>9s} {:>9s}'.format(
            '', '', '(pwm)', '(pos)', '(pos)', '(pos)', '(pos)', '(pwm)')]
    for s in statistics:
        lines.append('  {:8g} {:8d} {:9.1f} {:9.2f} {:9.2f} {:9.2f} {:9.2f} {:9.1f}'.format(
            s.speed, s.samples, s.pwm, s.error, s.lead, s.intended, s.error_std, s.pwm_std))
    return '\n'.join(lines)


def is_flat(fit: VelocityFit) -> bool:
    """whether the lag behind the intended trajectory changes by at most LAG_TOLERANCE between the slowest and the
    fastest speed"""
    return abs(fit.intended_change) <= LAG_TOLERANCE


def intended_line(fit: VelocityFit) -> str:
    """the line about the change of the lag behind the intended trajectory over the speeds, flat or not"""
    return '  lag behind the intended trajectory changes by {:+.2f} positions from {:g} to {:g} degrees/s: {}'.format(
        fit.intended_change, fit.slowest.speed, fit.fastest.speed,
        'flat (within {:g} position)'.format(LAG_TOLERANCE) if is_flat(fit) else
        'more than {:g} position, the speed dependent part is not taken out'.format(LAG_TOLERANCE))


def friction_line(fit: VelocityFit) -> str:
    """the line about the lag behind the intended trajectory itself: its mean over the speeds and whether every speed
    is within INTENDED_TOLERANCE of zero, which is the friction part taken out"""
    return '  lag behind the intended trajectory: {:+.2f} positions on average over the speeds, {}'.format(
        fit.intended_mean,
        'every speed is within {:g} position of zero: the friction part is taken out'.format(INTENDED_TOLERANCE)
        if fit.friction_taken_out else
        'a speed is {:.2f} positions from zero, more than {:g}: the friction part is not taken out'.format(
            fit.intended_largest, INTENDED_TOLERANCE))


def fit_lines(fit: VelocityFit) -> list[str]:
    """the two fits, the check of the factor 128 by their ratio and the friction part, without the yaml block"""
    lines = ['  from the measured pwm: pwm = {:.1f} + {:.3f} * speed: {:.2f} pwm per degree/s'.format(
        fit.friction_pwm, fit.k_pwm, fit.k_pwm),
        '  from the lag: error = {:.2f} + {:.4f} * speed positions, {:.1f} ms of travel per speed'.format(
            fit.friction_positions, fit.slope, fit.slope * DEGREES_PER_POSITION * 1000.0)]
    k_lag, ratio = fit.k_lag, fit.ratio
    if fit.p is None:
        lines.append('  the cross-check needs the P of the servo (the p column of the csv has no value): only the pwm '
                     'result is given')
    else:
        lines.append('  at P {:g} the lag gives {:.2f} pwm per degree/s'.format(fit.p, k_lag or 0.0))
        if ratio is not None:
            lines.append('  ratio k_pwm / k_lag = {:.2f} (1.0 when the factor 128 and the reasoning hold)'.format(
                ratio))
            if abs(ratio - 1.0) > RATIO_TOLERANCE:
                lines.append('  warning: the measured pwm and the lag disagree by more than {:.0f} %'.format(
                    RATIO_TOLERANCE * 100))
        else:
            lines.append('  the lag does not grow with the speed, there is no ratio')
    lines.append('  friction part, what does not depend on the speed: {:.1f} pwm, {:.2f} positions'.format(
        fit.friction_pwm, fit.friction_positions))
    return lines


def describe_fit(fit: VelocityFit, max_positions: int) -> str:
    """the two fits, the ratio, the friction, the lag behind the intended trajectory and the yaml block, which has the
    friction drive of the lag fit when the P is known"""
    lines = fit_lines(fit)
    lines.append(intended_line(fit))
    lines.append(friction_line(fit))
    source = 'the lag (the lead works through P times the error)' if fit.p is not None else \
        'the measured pwm, the P is not known'
    lines.append('  for servo_config.yaml, from {}:'.format(source))
    lines.append(yaml_block(fit.joint, fit.coefficient, max_positions, fit.friction_drive).rstrip('\n'))
    if fit.coefficient <= 0:
        lines.append('  warning: the coefficient is not positive, no drive that grows with the speed was measured')
    if fit.p is not None:
        lines.append('  ' + lead_meaning(fit.coefficient, fit.p))
    return '\n'.join(lines)


def describe(result: GroupResult, max_positions: int) -> str:
    """the report of one group: its header, the table, the notes and the fit or the reason there is none"""
    key = result.key
    lead = 'lead gain {:g} pwm per degree/s'.format(key.lead_gain) if key.lead_gain else 'no lead'
    if key.lead_friction:
        lead += ', friction drive {:g} pwm'.format(key.lead_friction)
    p = 'P {:g}'.format(result.p) if result.p is not None else 'P not known'
    lines = ['{}: joint {}, {}, {}'.format(key.file, key.joint, lead, p)]
    if result.statistics:
        lines.append(table(result.statistics))
    lines.extend('  note: ' + note for note in result.notes)
    if result.fit is None:
        lines.append('  no fit: {}'.format(result.problem))
    else:
        lines.append(describe_fit(result.fit, max_positions))
    return '\n'.join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='fit the coefficient of the velocity lead to recorded sweeps')
    parser.add_argument('files', nargs='+', help='csv files of the feedforward stage of tune_servo_gains.py')
    parser.add_argument('--max-positions', type=int, default=30, help='clamp of the lead in the yaml, in positions')
    args = parser.parse_args(argv)

    try:
        rows = read_rows(args.files)
        results = [analyse_group(key, members) for key, members in group_rows(rows).items()]
    except (OSError, ValueError) as e:
        print(e)
        return 1
    for path in args.files:
        print('{}: {} rows'.format(path, sum(1 for row in rows if row['file'] == path)))

    for result in results:
        print()
        print(describe(result, args.max_positions))
    if not any(result.fit is not None for result in results):
        print('\nnothing could be fitted')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
