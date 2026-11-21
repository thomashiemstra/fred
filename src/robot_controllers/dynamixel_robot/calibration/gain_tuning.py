"""
The arithmetic of calibration/tune_servo_gains.py: the size of a test step, the metrics of a step response and of a
rest period, the acceptance criteria that mark the knee of a gain sweep, the live abort rules, and the choice of the
gains from the trials, and the feedforward stage (the velocity lead of the PC): the sweep of the joint, the statistics
of its cruise parts and the line fit of the lag against the speed (the fits of the stage itself are in
calibration/fit_velocity_lead.py), and the binning of a slow sweep by the angle of the joint that
calibration/record_load_sweep.py uses, and the swing stage: the size and the frequency of the oscillation of the lag
during the cruise of a sweep per direction of the angle, the spread between two runs and the choice of P from them.
Pure functions on lists of samples, nothing here talks to the robot.

A Sample is one read of the state block of the servo under test (ServoHandler.read_state) with the host time and the
goal position the host last wrote. The quantities are in the units of the servo registers: positions (4096 per turn),
pwm units (885 is 100 % duty) and the 0.229 rpm velocity unit.
"""
from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Sequence
from itertools import accumulate
from operator import sub
from typing import NamedTuple

import numpy as np

from src.robot_controllers.dynamixel_robot import dynamixel_x_config as cfg

# the pwm of the position controller per position of error is P / 128 according to the manual, in units that it
# doesn't give; the script measures it (pwm_per_position) and uses this to size the steps before it has
PWM_PER_POSITION_PER_P = cfg.PWM_PER_POSITION_PER_P


class Sample(NamedTuple):
    time: float  # host time in seconds since the start of the trial
    tick: int  # the realtime tick of the servo in ms, wraps at 32767
    goal: int  # the goal position the host last wrote
    position: int
    position_trajectory: int  # the setpoint of the position loop, made by the profile of the servo
    velocity: int
    velocity_trajectory: int
    pwm: int
    current: int  # present current (XM, XL servos) or present load (XC430)
    moving_status: int
    voltage: float
    temperature: int


class RestMetrics(NamedTuple):
    """what the servo does while it holds a position"""
    duration: float
    samples: int
    position_pp: int  # peak to peak of the position, positions
    position_std: float
    pwm_mean: float  # the holding pwm, the static load in pwm units
    pwm_std: float
    temperature_rise: float
    voltage_min: float
    period: float  # mean time between samples in seconds
    error_mean: float = 0.0  # mean of |position trajectory - position|, the sag at these gains
    # the std of the pwm that the position doesn't explain: a position flickering between two adjacent counts makes
    # the pwm step by P / 128 per count, which is the loop doing its job, not noise; this is the std of
    # pwm - slope * (position trajectory - position), see rest_metrics
    pwm_residual_std: float = 0.0
    pwm_slope: float | None = None  # the slope used for the residual, pwm units per position of error


class StepMetrics(NamedTuple):
    """the response of the servo to a step of the goal position"""
    step: int  # goal minus the previous goal, positions
    start: int  # the position before the step
    final: float  # the mean position of the last part of the hold
    moved: bool  # whether the servo covered at least half the step, false means it stuck
    overshoot: float  # how far the position went beyond the final value, in positions, 0 without overshoot
    ring_count: int  # how often the position crossed the final value after the first crossing
    settle_time: float | None  # seconds until the position stayed within the settle band, None when it didn't
    rise_time: float | None  # seconds from 10 to 90 % of the travel, None when it didn't travel
    error: float  # final minus goal, the steady state error in this direction
    pwm_final: float
    saturated_fraction: float  # fraction of the samples with the pwm at the clamp


class TrackingMetrics(NamedTuple):
    """the tracking of the profile during a streamed move"""
    peak_trajectory_error: int  # max of |position trajectory - position|
    rms_trajectory_error: float
    peak_goal_error: int  # max of |host goal - position|
    rms_goal_error: float
    final_error: float  # mean of goal - position over the last part of the hold after the move
    following_error: bool  # whether the servo raised its following error bit


class Criteria(NamedTuple):
    """the acceptance criteria of a trial, a trial that violates one is the knee of the sweep (except those
    marked as warnings, which are only reported)"""
    max_overshoot_fraction: float = 0.10  # of the step
    min_overshoot_positions: float = 2.0  # an overshoot below this is quantization, never flagged
    max_ring_count: int = 1
    max_settle_time: float = 0.5  # seconds
    max_rest_pp: float = 2.0  # positions peak to peak at rest, or
    rest_pp_factor: float = 2.0  # this times the peak to peak of the baseline, whichever is larger
    # pwm_std_factor and pwm_std_floor give a warning, not a flag: the pwm std at rest follows where the joint parks
    # relative to an encoder edge rather than the gain (joint 3, 13 November 2026); choose_d still uses them
    pwm_std_factor: float = 1.5  # times the pwm std of the baseline
    pwm_std_floor: float = 5.0  # pwm units, a std below this is noise however it compares to the baseline
    max_saturated_fraction: float = 0.3  # of the samples of a step, a warning
    max_temperature_rise: float = 3.0  # degrees during one trial, a warning


class AbortLimits(NamedTuple):
    """the live limits, a sample beyond one aborts the trial and restores the gains at once"""
    # degrees, the winding is warmer than the sensor; tune_servo_gains sets it to the temperature_limit of
    # servo_config.yaml, this default is for the tests
    max_temperature: float = 50.0
    max_voltage_drop: float = 1.5  # volts below the voltage at the start of the trial
    max_jump: int = 50  # positions between two samples, an encoder glitch or a slipped belt
    divergence_factor: float = 3.0  # times the step, or
    divergence_minimum: int = 40  # positions of error against the trajectory, or
    divergence_sag_factor: float = 2.5  # times the sag expected at these gains, whichever is the largest
    stream_divergence: int = 120  # positions behind the trajectory during a streamed move (plus the sag term)
    oscillation_factor: float = 3.0  # times the baseline peak to peak, the rise and the fall have to exceed it, or
    oscillation_minimum: float = 6.0  # positions the position has to go both up and down, whichever is larger
    # seconds, how far back the rule looks at most, never further back than the end of the settle grace (so slow
    # swings are seen as well as fast ones, and the step response itself is never inside the window)
    oscillation_window: float = 1.0
    saturation_fraction: float = 0.9  # of the pwm clamp
    saturation_window: float = 0.1  # seconds the pwm has to sit at the clamp
    settle_grace: float = 0.5  # seconds after a goal change before the divergence and oscillation rules apply


# the abort reasons that say nothing about the gains: the run has to stop and the trial is not a knee
ENVIRONMENT_ABORTS = ('temperature', 'voltage', 'position jumped')


def environmental_abort(reason: str) -> bool:
    """whether an abort reason of abort_reason is about the servo or the supply rather than the gains"""
    return reason.startswith(ENVIRONMENT_ABORTS)


def pwm_clamp(holding_pwm: float, factor: float = 3.0, floor_fraction: float = 0.5, limit: int = cfg.MAX_PWM) -> int:
    """
    the goal pwm clamp for the trials at a pose: a multiple of the pwm the servo needs to hold the pose, at least a
    fraction of the limit (the holding pwm is measured on one side of the friction band, the other side needs more,
    and a step needs room on top), never above the limit
    """
    return int(round(min(limit, max(floor_fraction * limit, factor * abs(holding_pwm)))))


def step_size(p: int, holding_pwm: float, clamp: int, fraction: float = 0.8, minimum: int = 8, maximum: int = 60,
              pwm_per_position: float | None = None) -> tuple[int, bool]:
    """
    the positions of a test step that keeps the pwm below the fraction of the clamp, on top of the holding pwm
    :param pwm_per_position: the measured pwm per position of error, by default P / 128
    :return: (positions, limited): limited is True when the step had to be raised to the minimum, the pwm then
        saturates during the step and the P gain is too high to test with a step at this pose
    """
    if p <= 0:
        raise ValueError('the P gain has to be positive')
    per_position = pwm_per_position if pwm_per_position is not None else p * PWM_PER_POSITION_PER_P
    headroom = fraction * clamp - abs(holding_pwm)
    size = int(headroom / per_position) if per_position > 0 else maximum
    return max(minimum, min(maximum, size)), size < minimum


def rest_metrics(samples: Sequence[Sample], p: int | None = None) -> RestMetrics:
    """
    :param p: the P gain during the rest, the fallback slope of P / 128 pwm per position of error for the pwm
        residual when the position doesn't vary enough to fit the slope (it doesn't change the residual then)
    """
    if len(samples) < 2:
        raise ValueError('a rest period needs at least two samples')
    positions = np.array([s.position for s in samples], dtype=float)
    pwm = np.array([s.pwm for s in samples], dtype=float)
    times = np.array([s.time for s in samples])
    signed_errors = np.array([s.position_trajectory - s.position for s in samples], dtype=float)
    # the pwm the position explains: fitted when the position moved, P / 128 per count otherwise
    slope: float | None
    if signed_errors.max() - signed_errors.min() >= 1:
        slope = float(np.polyfit(signed_errors, pwm, 1)[0])
    elif p is not None:
        slope = p * PWM_PER_POSITION_PER_P
    else:
        slope = None
    residual = pwm - (slope if slope is not None else 0.0) * signed_errors
    return RestMetrics(duration=float(times[-1] - times[0]), samples=len(samples),
                       position_pp=int(positions.max() - positions.min()), position_std=float(positions.std()),
                       pwm_mean=float(pwm.mean()), pwm_std=float(pwm.std()),
                       temperature_rise=float(samples[-1].temperature - samples[0].temperature),
                       voltage_min=float(min(s.voltage for s in samples)),
                       period=float(np.diff(times).mean()), error_mean=float(np.abs(signed_errors).mean()),
                       pwm_residual_std=float(residual.std()), pwm_slope=slope)


def step_metrics(samples: Sequence[Sample], step_time: float, previous_goal: int, goal: int, clamp: int,
                 band_fraction: float = 0.05, band_minimum: float = 2.0, ring_band: float = 1.5,
                 tail_fraction: float = 0.25) -> StepMetrics:
    """
    the response to the step of the goal from previous_goal to goal at step_time, measured against the final value
    (the mean of the last tail_fraction of the samples), not against the goal: with I = 0 the position settles at
    the goal minus the sag
    :param samples: the samples around the step, those before step_time give the start position
    :param clamp: the goal pwm clamp during the step, for the saturated fraction
    :param band_fraction: the settle band is this fraction of the step, at least band_minimum positions
    :param ring_band: crossings of the final value closer than this are quantization, not rings
    """
    after = [s for s in samples if s.time >= step_time]
    if len(after) < 4:
        raise ValueError('too few samples after the step')
    before = [s for s in samples if s.time < step_time]
    start = before[-1].position if before else after[0].position
    step = goal - previous_goal
    direction = 1 if step >= 0 else -1
    positions = np.array([s.position for s in after], dtype=float)
    times = np.array([s.time for s in after]) - step_time
    pwm = np.array([s.pwm for s in after], dtype=float)
    tail = max(2, int(len(after) * tail_fraction))
    final = float(positions[-tail:].mean())

    # positive beyond the final value in the direction of the step
    deviation = direction * (positions - final)
    overshoot = max(float(deviation.max()), 0.0)
    signs = np.sign(np.where(np.abs(deviation) <= ring_band, 0.0, deviation))
    nonzero = signs[signs != 0]
    crossings = int(np.count_nonzero(np.diff(nonzero))) if len(nonzero) > 1 else 0
    ring_count = max(crossings - 1, 0)

    band = max(band_minimum, band_fraction * abs(step))
    outside = np.flatnonzero(np.abs(positions - final) > band)
    settle_time: float | None
    if len(outside) == 0:
        settle_time = 0.0
    elif outside[-1] == len(positions) - 1:
        settle_time = None
    else:
        settle_time = float(times[outside[-1] + 1])

    travel = final - start
    rise_time: float | None = None
    if abs(travel) >= 2:
        progress = (positions - start) / travel
        past_10 = np.flatnonzero(progress >= 0.1)
        past_90 = np.flatnonzero(progress >= 0.9)
        if len(past_10) and len(past_90):
            rise_time = float(times[past_90[0]] - times[past_10[0]])
    moved = step != 0 and abs(travel) >= 0.5 * abs(step)
    return StepMetrics(step=step, start=start, final=final, moved=moved, overshoot=overshoot, ring_count=ring_count,
                       settle_time=settle_time, rise_time=rise_time, error=final - goal,
                       pwm_final=float(pwm[-tail:].mean()),
                       saturated_fraction=float(np.mean(np.abs(pwm) >= 0.9 * clamp)) if clamp > 0 else 0.0)


def tracking_metrics(samples: Sequence[Sample], tail_fraction: float = 0.25) -> TrackingMetrics:
    """the tracking during a streamed move followed by a hold, the final error over the last tail_fraction"""
    if len(samples) < 4:
        raise ValueError('too few samples of the move')
    trajectory_error = np.array([s.position_trajectory - s.position for s in samples], dtype=float)
    goal_error = np.array([s.goal - s.position for s in samples], dtype=float)
    tail = max(2, int(len(samples) * tail_fraction))
    return TrackingMetrics(peak_trajectory_error=int(np.abs(trajectory_error).max()),
                           rms_trajectory_error=float(np.sqrt(np.mean(trajectory_error ** 2))),
                           peak_goal_error=int(np.abs(goal_error).max()),
                           rms_goal_error=float(np.sqrt(np.mean(goal_error ** 2))),
                           final_error=float(goal_error[-tail:].mean()),
                           following_error=any(s.moving_status & cfg.MOVING_STATUS_FOLLOWING_ERROR for s in samples))


def pwm_per_position(points: Sequence[tuple[float, float]], minimum_spread: float = 4.0) -> float | None:
    """
    the slope of the pwm against the position error from the steady states of the steps and rests of a trial:
    the pwm per position of error the servo actually produces, to compare with P / 128
    :param points: (error = goal - position, pwm) pairs
    :return: the slope, or None when the errors don't spread enough for a fit
    """
    if len(points) < 3:
        return None
    errors = np.array([p[0] for p in points], dtype=float)
    pwm = np.array([p[1] for p in points], dtype=float)
    if errors.max() - errors.min() < minimum_spread:
        return None
    slope, _ = np.polyfit(errors, pwm, 1)
    return float(slope)


def evaluate(rest: RestMetrics, steps: Sequence[StepMetrics], baseline: RestMetrics,
             criteria: Criteria = Criteria(), p_ratio: float = 1.0,
             temperature_rise: float | None = None) -> tuple[list[str], list[str]]:
    """
    judge a trial: (flags, warnings). A flag means the trial fails and is the knee of the sweep, a warning is
    reported but doesn't fail the trial.
    :param baseline: the rest metrics at the gains of the yaml, at the same pose
    :param p_ratio: the P gain of the trial over that of the baseline: the pwm noise scales with P, so it is
        compared per unit of P
    :param temperature_rise: over the whole trial, by default the rise during the rest

    The pwm noise is the residual the position doesn't explain (RestMetrics.pwm_residual_std): a joint resting on
    an encoder edge flickers between two counts and its pwm steps by P / 128 with it, which is the loop working;
    the raw pwm std of that is half the step and would fail every stable joint at a high P. It is only a warning:
    on joint 3 on 13 November 2026 the residual equalled the raw std and followed where the sag parks the joint
    relative to an encoder edge, not the gain (the baseline at unchanged gains ranged from 1.3 to 5.3). Real
    oscillation at rest is the ripple flag and the live oscillation abort.
    """
    flags, warnings = set(), set()
    for m in steps:
        if m.overshoot > max(criteria.min_overshoot_positions, criteria.max_overshoot_fraction * abs(m.step)):
            flags.add('overshoot')
        if m.ring_count > criteria.max_ring_count:
            flags.add('ringing')
        if m.settle_time is None or m.settle_time > criteria.max_settle_time:
            flags.add('settle')
        if m.saturated_fraction > criteria.max_saturated_fraction:
            warnings.add('saturation')
        if not m.moved:
            warnings.add('stiction')
    if steps and not any(m.moved for m in steps):
        flags.add('stuck')  # nothing was excited, the trial says nothing about the loop
    if rest.position_pp > max(criteria.max_rest_pp, criteria.rest_pp_factor * baseline.position_pp):
        flags.add('ripple')
    if rest.pwm_residual_std / max(p_ratio, 1e-9) > max(criteria.pwm_std_floor,
                                                        criteria.pwm_std_factor * baseline.pwm_residual_std):
        warnings.add('pwm_noise')
    rise = rest.temperature_rise if temperature_rise is None else temperature_rise
    if rise > criteria.max_temperature_rise:
        warnings.add('heating')
    return sorted(flags), sorted(warnings)


def abort_reason(samples: Sequence[Sample], clamp: int, step: int, baseline_pp: float, start_voltage: float,
                 last_goal_time: float, limits: AbortLimits = AbortLimits(), sag: float = 0.0,
                 streaming: bool = False) -> str | None:
    """
    why the trial has to stop now, or None: checked on the latest samples while the servo is read
    :param clamp: the goal pwm the servo has during these samples
    :param step: the size of the current step, positions
    :param baseline_pp: the position peak to peak at rest at the baseline gains
    :param start_voltage: the input voltage at the start of the trial
    :param last_goal_time: the time of the last goal write, the divergence and oscillation rules wait after it
    :param sag: the position error expected at rest at these gains, the divergence rule allows a multiple of it
        (math.inf switches the rule off, for the baseline whose sag isn't known yet)
    :param streaming: the goal moves every step, so the position may lag the trajectory and only a large lag is
        divergence, and the oscillation rule doesn't apply
    """
    if not samples:
        return None
    s = samples[-1]
    if s.temperature >= limits.max_temperature:
        return 'temperature {}C'.format(s.temperature)
    if s.voltage < start_voltage - limits.max_voltage_drop:
        return 'voltage {:.1f}V, {:.1f}V at the start'.format(s.voltage, start_voltage)
    if len(samples) > 1 and abs(s.position - samples[-2].position) > limits.max_jump:
        return 'position jumped {} positions between two samples'.format(s.position - samples[-2].position)
    since_goal = s.time - last_goal_time
    error = s.position_trajectory - s.position
    if streaming:
        if abs(error) > max(limits.stream_divergence, limits.divergence_sag_factor * sag):
            return 'divergence, {} positions from the trajectory during the move'.format(error)
    elif since_goal > limits.settle_grace:
        if abs(error) > max(limits.divergence_factor * abs(step), limits.divergence_minimum,
                            limits.divergence_sag_factor * sag):
            return 'divergence, {} positions from the trajectory'.format(error)
        # oscillation is the position going both ways: a creep towards the goal in one direction, even a long
        # one, is the joint settling. The samples are the newest ones back to the bound, never from inside the
        # grace, so the step response itself is not in the window.
        bound = max(last_goal_time + limits.settle_grace, s.time - limits.oscillation_window)
        start = bisect_right(samples, bound, key=lambda x: x.time)  # the samples are in time order
        positions = [x.position for x in samples[start:]]  # never empty: the newest sample is after the bound
        rise = max(map(sub, positions, accumulate(positions, min)))  # the most the position went up
        fall = max(map(sub, accumulate(positions, max), positions))  # the most it went down
        if min(rise, fall) > max(limits.oscillation_factor * baseline_pp, limits.oscillation_minimum):
            return 'oscillation, {} positions back and forth at rest'.format(int(min(rise, fall)))
    if since_goal > limits.saturation_window:
        recent = samples[bisect_right(samples, s.time - limits.saturation_window, key=lambda x: x.time):]
        if recent and all(abs(x.pwm) >= limits.saturation_fraction * clamp for x in recent):
            return 'pwm at the clamp of {} for {} s'.format(clamp, limits.saturation_window)
    return None


def round_gain(value: float) -> int:
    return int(round(value / 10.0)) * 10


def p_values(start: int, factor: float, max_factor: float, cap: int = cfg.MAX_GAIN) -> list[int]:
    """the P gains of a sweep upwards: start, start * factor, ... up to start * max_factor and the cap"""
    if factor <= 1.0 or max_factor < 1.0:
        raise ValueError('the factor has to be above 1 and the maximum factor at least 1')
    values = []
    p = float(start)
    while p <= start * max_factor + 1e-9 and p <= cap:
        values.append(round_gain(p))
        p *= factor
    return values


def p_values_down(start: int, factor: float, steps: int = 3, minimum: int = 100) -> list[int]:
    """the P gains below start to try when start itself fails: start / factor, start / factor^2, ..."""
    values = []
    p = float(start)
    for _ in range(steps):
        p /= factor
        if p < minimum:
            break
        values.append(round_gain(p))
    return values


def d_ladder(start: int, factor: float, maximum: int = cfg.MAX_GAIN) -> list[int]:
    """the D gains of the grid: 0, then start, start * factor, ... up to the maximum. Near critical damping the D a
    loop needs grows with the square root of its P, so the steps are a factor apart rather than fractions of a P"""
    if start <= 0 or factor <= 1.0:
        raise ValueError('the D ladder needs a start above 0 and a factor above 1')
    values = [0]
    d = float(start)
    while d <= maximum + 1e-9:
        values.append(min(maximum, round_gain(d)))
        d *= factor
    return sorted(set(values))


def choose_p(results: Sequence[tuple[int, bool]], factor: float, margin: float = 0.65) -> tuple[int | None, int | None]:
    """
    the P gain to use from the trials of a sweep
    :param results: (p, passed) per trial
    :param factor: the factor of the sweep, the margin below the highest passed value when no knee was found
    :param margin: the recommended gain is at most this fraction of the knee, the first failing gain
    :return: (recommended P, knee P): the recommendation is None when nothing passed below the knee, the knee is
        None when no trial failed
    """
    passed = sorted(p for p, ok in results if ok)
    failed = sorted(p for p, ok in results if not ok)
    knee = failed[0] if failed else None
    if knee is not None:
        below = [p for p in passed if p < knee]
        if not below:
            return None, knee
        return round_gain(min(max(below), margin * knee)), knee
    if passed:
        return round_gain(max(passed) / factor), None
    return None, None


def choose_d(results: Sequence[tuple[int, bool, float, int, float, float]], pwm_std_factor: float = 1.5,
             pwm_std_floor: float = 5.0, min_improvement: float = 1.0) -> int | None:
    """
    the D gain to use from the trials at the chosen P
    :param results: (d, passed, overshoot in positions, ring count, pwm residual std at rest, mean of |final -
        goal| of the steps in positions) per trial
    :param min_improvement: positions of overshoot or of resting error a larger D has to gain over the reference
    :return: None when nothing passed. The candidates passed and keep their pwm noise below pwm_std_factor times
        that of the smallest D tried and the floor. The reference is the smallest candidate (D = 0 when it passed);
        a larger D replaces it only when it beats it by min_improvement in overshoot or in resting error, or removes
        a ring, without being worse by min_improvement in either. Of those the one with the least resting error,
        then overshoot, then rings, then the smallest D.

    The resting error is the signal where every D passes on overshoot: without damping the joint runs into the far
    side of its friction band, with damping it arrives at the setpoint (joint 6 on 23 October: D 0 rested at +4 +5
    +3 +5 / -6 -3 -6 -3 after the steps up and down, D 1100 at +0 +2 +0 +2 / -3 +0 -2 +0, both with an overshoot
    of 1 position or less).
    """
    if not results:
        return None
    noise_reference = min(results, key=lambda r: r[0])[4]
    limit = max(pwm_std_floor, pwm_std_factor * noise_reference)
    candidates = [r for r in results if r[1] and r[4] <= limit]
    if not candidates:
        return None
    reference = min(candidates, key=lambda r: r[0])
    margin = min_improvement - 1e-9  # an improvement of exactly min_improvement counts

    def qualifies(r: tuple[int, bool, float, int, float, float]) -> bool:
        overshoot_gain = reference[2] - r[2]
        error_gain = reference[5] - r[5]
        better = overshoot_gain >= margin or error_gain >= margin or r[3] < reference[3]
        worse = -overshoot_gain >= margin or -error_gain >= margin
        return better and not worse

    better = [r for r in candidates if r[0] > reference[0] and qualifies(r)]
    if not better:
        return reference[0]
    return min(better, key=lambda r: (round(r[5], 1), round(r[2], 1), r[3], r[0]))[0]


# the feedforward stage

FF_RAMP = 0.2  # seconds a leg of the sweep takes to ramp its speed up from rest, and to ramp it down again
FF_CRUISE_HOLD = 0.15  # seconds at the full speed before a step counts as cruise
MIN_CRUISE_SAMPLES = 3  # per direction, fewer than this and the statistics of a sweep are not made
FF_LAG_TOLERANCE = 1.0  # positions the lag behind the intended trajectory may change between the slowest and the
#                         fastest speed before the coefficient of the lead is corrected
FF_ROUGHNESS = 1.0  # positions the std of the lag may grow with the lead before it is a warning


class SweepStep(NamedTuple):
    """one step of the feedforward sweep: where the joint is commanded to and what the commanded motion is doing"""
    time: float  # seconds since the start of the sweep
    offset: float  # degrees from the pose
    direction: int  # of the leg the step is in: 1 up, -1 down
    cruise: bool  # the commanded speed has been at its full value for FF_CRUISE_HOLD seconds and still is
    leg: int = 0  # the number of the leg, the cruise parts of the legs going the same way are not one line


class _Leg(NamedTuple):
    """a rest to rest move of the sweep: the speed ramps up linearly, cruises, and ramps down again"""
    start: float  # seconds since the start of the sweep
    origin: float  # degrees
    direction: int
    distance: float  # degrees, positive
    peak: float  # degrees per second, below the speed of the sweep for a leg that is a triangle
    ramp_time: float
    cruise_time: float  # 0 for a triangle

    @property
    def duration(self) -> float:
        return 2 * self.ramp_time + self.cruise_time

    def offset_at(self, seconds: float) -> float:
        """degrees from the pose, seconds since the start of the leg"""
        tau = min(max(seconds, 0.0), self.duration)
        acceleration = self.peak / self.ramp_time
        if tau <= self.ramp_time:
            travelled = 0.5 * acceleration * tau ** 2
        elif tau <= self.ramp_time + self.cruise_time:
            travelled = 0.5 * acceleration * self.ramp_time ** 2 + self.peak * (tau - self.ramp_time)
        else:
            travelled = self.distance - 0.5 * acceleration * (self.duration - tau) ** 2
        return self.origin + self.direction * travelled

    def is_cruise(self, seconds: float, hold: float) -> bool:
        end = self.ramp_time + self.cruise_time
        return self.cruise_time > 0 and self.ramp_time + hold - 1e-9 <= seconds <= end + 1e-9


def ff_sweep(amplitude: float, speed: float, dt: float, ramp: float = FF_RAMP,
             cruise_hold: float = FF_CRUISE_HOLD, round_trips: int = 2) -> list[SweepStep]:
    """
    the steps of one sweep of the feedforward stage: the joint goes from the pose to +amplitude, to -amplitude, to
    +amplitude, to -amplitude and back to the pose (two round trips), every leg rest to rest with the speed ramping
    linearly up over `ramp` seconds, cruising at the speed and ramping down again (a triangle when the leg is too
    short to reach the speed, without cruise steps). The first step is at the pose at time 0, the last one at the
    pose at the end.
    :param amplitude: degrees either side of the pose
    :param speed: degrees per second of the cruise
    :param dt: seconds between the steps
    :param cruise_hold: seconds the speed has to have been at its full value before a step counts as cruise
    :param round_trips: how often the joint goes to +amplitude and -amplitude before it returns to the pose, at least 1
    """
    if amplitude <= 0 or speed <= 0 or dt <= 0 or ramp <= 0:
        raise ValueError('the amplitude, the speed, the step time and the ramp of a sweep have to be positive')
    if round_trips < 1:
        raise ValueError('a sweep needs at least 1 round trip, got {}'.format(round_trips))
    acceleration = speed / ramp
    legs: list[_Leg] = []
    start, origin = 0.0, 0.0
    for target in (amplitude, -amplitude) * round_trips + (0.0,):
        distance = abs(target - origin)
        full = distance >= speed * ramp
        peak = speed if full else math.sqrt(distance * acceleration)
        ramp_time = peak / acceleration
        cruise_time = (distance - speed * ramp) / speed if full else 0.0
        legs.append(_Leg(start, origin, 1 if target > origin else -1, distance, peak, ramp_time, cruise_time))
        start += legs[-1].duration
        origin = target
    total = start
    steps: list[SweepStep] = []
    index = 0
    for k in range(math.ceil(total / dt - 1e-9) + 1):
        seconds = min(k * dt, total)
        while index < len(legs) - 1 and seconds >= legs[index + 1].start - 1e-9:
            index += 1
        leg = legs[index]
        local = seconds - leg.start
        steps.append(SweepStep(seconds, leg.offset_at(local), leg.direction, leg.is_cruise(local, cruise_hold),
                               index))
    return steps


class SweepSample(NamedTuple):
    """a sample of the servo during a sweep with what the commanded motion was doing at the time"""
    sample: Sample
    direction: int  # in positions: 1 when the position rises, -1 when it falls (not the direction of the angle)
    cruise: bool
    leg: int = 0


class CruiseStatistics(NamedTuple):
    """the cruise parts of one sweep, per direction and then averaged over the two directions, so that the pull of
    gravity cancels"""
    speed: float  # positions per millisecond, from the goals the servo was sent
    lag: float  # positions the servo trails its trajectory, in the direction of the move
    pwm: float  # pwm units, in the direction of the move
    lag_std: float  # positions
    pwm_std: float
    samples: int  # the cruise samples of both directions


def cruise_statistics(samples: Sequence[SweepSample], minimum: int = MIN_CRUISE_SAMPLES) -> CruiseStatistics:
    """
    the statistics of the cruise samples of a sweep: for each direction (of the position: 1 when it rises, as the
    lag and the pwm are in positions) the mean of (position trajectory - position) times the direction (positive when
    the servo trails), the mean pwm times the direction, the std of both and the
    speed (the absolute slope of the goal against the time, one line per leg with the legs' own offsets); the result
    is the mean over the two directions
    :raises ValueError: when a direction has fewer cruise samples than the minimum
    """
    lags, pwms, lag_stds, pwm_stds, speeds = [], [], [], [], []
    for direction, name in ((1, 'up'), (-1, 'down')):
        legs: dict[int, list[Sample]] = {}
        for s in samples:
            if s.cruise and s.direction == direction:
                legs.setdefault(s.leg, []).append(s.sample)
        rows = [r for leg in legs.values() for r in leg]
        if len(rows) < minimum:
            raise ValueError('{} cruise samples going {}, at least {} are needed'.format(len(rows), name, minimum))
        lag = np.array([(r.position_trajectory - r.position) * direction for r in rows], dtype=float)
        pwm = np.array([r.pwm * direction for r in rows], dtype=float)
        lags.append(lag.mean())
        pwms.append(pwm.mean())
        lag_stds.append(lag.std())
        pwm_stds.append(pwm.std())
        # the slope of the goal against the time within the legs, pooled, in positions per millisecond
        covariance = variance = 0.0
        for leg in legs.values():
            times = np.array([r.time for r in leg])
            goals = np.array([r.goal for r in leg], dtype=float)
            covariance += float(((times - times.mean()) * (goals - goals.mean())).sum())
            variance += float(((times - times.mean()) ** 2).sum())
        if variance <= 0:
            raise ValueError('the cruise samples going {} are all at the same time'.format(name))
        speeds.append(abs(covariance / variance) / 1000.0)
    return CruiseStatistics(speed=float(np.mean(speeds)), lag=float(np.mean(lags)), pwm=float(np.mean(pwms)),
                            lag_std=float(np.mean(lag_stds)), pwm_std=float(np.mean(pwm_stds)),
                            samples=sum(1 for s in samples if s.cruise))


class LineFit(NamedTuple):
    """lag = friction + slope * speed"""
    friction: float  # positions, the lag at speed 0: what doesn't depend on the speed
    slope: float  # milliseconds (positions per position per millisecond)


def fit_line(speeds: Sequence[float], lags: Sequence[float]) -> LineFit:
    """the least squares line of the lag in positions against the speed in positions per millisecond"""
    if len(set(speeds)) < 2:
        raise ValueError('a line needs at least two different speeds')
    slope, friction = np.polyfit(np.asarray(speeds, dtype=float), np.asarray(lags, dtype=float), 1)
    return LineFit(friction=float(friction), slope=float(slope))


def lag_growth(fit: LineFit, statistics: Sequence[CruiseStatistics]) -> float:
    """the change of the lag in positions between the slowest and the fastest speed of the statistics, on the line"""
    speeds = [s.speed for s in statistics]
    return fit.slope * (max(speeds) - min(speeds))


def rougher_speeds(before: Sequence[float], after: Sequence[float], limit: float = FF_ROUGHNESS) -> list[int]:
    """
    the indexes of the speeds where a std grew by more than the limit
    :param before: the std at every speed without the velocity lead, in positions or in pwm units
    :param after: the std at the same speeds with it, in the same unit as `before`
    :param limit: in that unit; FF_ROUGHNESS is for positions
    """
    return [i for i, (b, a) in enumerate(zip(before, after)) if a - b > limit]


# the load sweep: the cruise samples of a slow sweep binned by the angle of the joint

MIN_BIN_SAMPLES = 30  # per direction, fewer than this and the mean of a bin is too uncertain to be paired


class BinSide(NamedTuple):
    """the samples of one direction in a bin"""
    error: float  # mean error in positions
    pwm: float  # mean measured pwm
    samples: int


class SweepBin(NamedTuple):
    """a bin of the angle of the joint that was passed in both directions"""
    angle: float  # degrees, the centre of the bin
    up: BinSide  # the angle was rising
    down: BinSide  # the angle was falling

    @property
    def sag(self) -> float:
        """the mean of the errors of the two directions, in positions: the sag under the load, with the lag of the
        speed and the friction, which both flip sign with the direction, taken out"""
        return (self.up.error + self.down.error) / 2.0

    @property
    def half_difference(self) -> float:
        """half the difference of the errors of the up and the down direction, in positions: minus the friction and
        the lag of the speed for a servo whose positions rise with its angle (a joint that trails its trajectory has
        an error against the direction it moves in), plus them for a servo mounted reversed"""
        return (self.up.error - self.down.error) / 2.0


def bin_index(angles: Sequence[float] | np.ndarray, bin_degrees: float) -> np.ndarray:
    """the number k of the bin of every angle (degrees): bin k covers [k * bin_degrees, (k + 1) * bin_degrees), so
    the bins are a fixed grid and those of two sweeps line up whatever their speed or span"""
    if bin_degrees <= 0:
        raise ValueError('the bins have to be wider than 0 degrees, got {}'.format(bin_degrees))
    return np.floor(np.asarray(angles, dtype=float) / bin_degrees).astype(int)


def bin_sweep(angles: Sequence[float] | np.ndarray, directions: Sequence[int] | np.ndarray,
              errors: Sequence[float] | np.ndarray, pwms: Sequence[float] | np.ndarray, bin_degrees: float,
              minimum: int = MIN_BIN_SAMPLES) -> list[SweepBin]:
    """
    pair the two directions of a sweep per bin of the angle of the joint. Pass the cruise samples only: the legs
    ramp up and down with a speed, and so a lag, of their own.
    :param angles: the measured angle of the joint of every sample, degrees
    :param directions: 1 when the ANGLE of the joint was rising at the sample, -1 when it was falling (not the
        direction of the position, which is the other way round for a servo mounted reversed); other values leave
        the sample out
    :param errors: the error of every sample, positions
    :param pwms: the measured pwm of every sample
    :param bin_degrees: the width of a bin, see bin_index; its centre is (k + 0.5) * bin_degrees
    :param minimum: the samples a direction needs in a bin (at least 1) for the bin to be returned
    :return: the bins that have at least `minimum` samples going both ways, sorted by angle
    """
    count = len(angles)
    if not len(directions) == len(errors) == len(pwms) == count:
        raise ValueError('angles, directions, errors and pwms need the same number of samples, got {}, {}, {}, {}'
                         .format(count, len(directions), len(errors), len(pwms)))
    index = bin_index(angles, bin_degrees)
    direction = np.asarray(directions)
    error = np.asarray(errors, dtype=float)
    pwm = np.asarray(pwms, dtype=float)
    minimum = max(minimum, 1)
    bins = []
    for k in np.unique(index):
        in_bin = index == k
        sides = []
        for mask in (in_bin & (direction > 0), in_bin & (direction < 0)):
            samples = int(mask.sum())
            if samples < minimum:
                break
            sides.append(BinSide(float(error[mask].mean()), float(pwm[mask].mean()), samples))
        else:
            bins.append(SweepBin((float(k) + 0.5) * bin_degrees, sides[0], sides[1]))
    return bins


# the swing stage: the arm oscillating at its own resonance while it moves

SWING_BAND = (3.0, 15.0)  # Hz, where the resonance of the arm lies (6.6 Hz on joint 2 on 20 November 2026)
SWING_RATE = 500.0  # Hz, the rate the lag of a leg is resampled to before its spectrum is taken
SWING_MIN_LEG = 0.6  # seconds of cruise a leg needs to be used, a few periods of the resonance
# the spectrum of a leg is taken over this many times its samples (zero padded): the peak of a sine that falls between
# two bins of the spectrum of the window would read up to 15 % low, the padding reads it off the peak of the window
SWING_PAD = 8
SWING_RESOLUTION = 0.1  # positions, a swing that changes by less than this has not changed
SWING_TOLERANCE = 1e-9  # for comparing a swing with the limit


class SwingSample(NamedTuple):
    """a sample of the servo during a sweep with what the commanded motion was doing at the time"""
    sample: Sample
    direction: int  # of the ANGLE of the joint: 1 when it rises, -1 when it falls (not the direction of the position)
    cruise: bool
    leg: int = 0


class SwingSide(NamedTuple):
    """the cruise legs of one direction of the angle"""
    legs: int  # the legs that were used
    frequency: float  # Hz, of the strongest component of the lag in the band, the mean over the legs
    swing: float  # positions, the amplitude of that component (a sine of amplitude A gives A), the mean over the legs
    lag_std: float  # positions, of the lag with the trend of the leg taken out, the mean over the legs
    lag: float  # positions the servo trails its trajectory (negative: it leads), the mean over the legs
    pwm: float  # pwm units in the direction of the move, the mean over the legs
    pwm_std: float  # of the pwm within a leg, the mean over the legs


class SwingStatistics(NamedTuple):
    """the swing of a sweep per direction of the ANGLE of the joint"""
    rising: SwingSide
    falling: SwingSide


class SwingPair(NamedTuple):
    """one number per direction of the angle"""
    rising: float
    falling: float


class _LegSwing(NamedTuple):
    frequency: float
    swing: float
    lag_std: float
    lag: float
    pwm: float
    pwm_std: float


def _leg_swing(rows: Sequence[Sample], sign: int, band: tuple[float, float], rate: float) -> _LegSwing:
    """the swing of the cruise samples of one leg: the lag and the pwm are resampled to a uniform rate (the servo is
    read at irregular intervals), the linear trend of the lag is taken out (the lag of the speed, the ramp into the
    leg), and the strongest component of the amplitude spectrum in the band is the swing"""
    times = np.array([r.time for r in rows])
    order = np.argsort(times, kind='stable')
    times = times[order]
    lag = np.array([r.position_trajectory - r.position for r in rows], dtype=float)[order]
    pwm = np.array([r.pwm for r in rows], dtype=float)[order]
    seconds = np.arange(0.0, times[-1] - times[0], 1.0 / rate)
    lag = np.interp(times[0] + seconds, times, lag)
    pwm = np.interp(times[0] + seconds, times, pwm)
    mean_lag = float(lag.mean())
    lag = lag - np.polyval(np.polyfit(seconds, lag, 1), seconds)
    window = np.hanning(len(lag))
    size = SWING_PAD * len(lag)
    amplitude = np.abs(np.fft.rfft(lag * window, n=size)) * 2.0 / window.sum()
    frequencies = np.fft.rfftfreq(size, 1.0 / rate)
    in_band = (frequencies >= band[0]) & (frequencies <= band[1])
    peak = int(np.argmax(amplitude[in_band]))
    return _LegSwing(frequency=float(frequencies[in_band][peak]), swing=float(amplitude[in_band][peak]),
                     lag_std=float(lag.std()), lag=sign * mean_lag, pwm=sign * float(pwm.mean()),
                     pwm_std=float(pwm.std()))


def swing_statistics(samples: Sequence[SwingSample], position_sign: int = 1, band: tuple[float, float] = SWING_BAND,
                     rate: float = SWING_RATE, minimum_leg: float = SWING_MIN_LEG) -> SwingStatistics:
    """
    the swing of the cruise samples of a sweep, per direction of the ANGLE of the joint (rising: SwingSample.direction
    1, falling: -1; the direction of the position is the other way round for a servo mounted reversed, which is what
    position_sign is for): every cruise leg is analysed on its own (the cruise parts of the legs are never joined),
    and the result of a direction is the mean over its legs
    :param position_sign: 1 when the positions of the servo rise with its angle, -1 when they fall (the script's
        position_sign): the lag and the pwm of a side are signed with the direction of the move, so that a servo that
        trails its trajectory has a positive lag in both directions, as in cruise_statistics. The swing, its
        frequency and the std don't depend on it.
    :param band: Hz, the strongest component of the lag between these is the swing
    :param rate: Hz, the samples of a leg are resampled to this rate
    :param minimum_leg: seconds, a leg whose cruise samples span less is left out, and so is a leg sampled at less
        than twice the top of the band, which would alias
    :raises ValueError: when a direction has no leg that can be used
    """
    if not 0 < band[0] < band[1] <= rate / 2 or minimum_leg * band[0] < 1:
        raise ValueError('the band {} Hz needs 0 < low < high <= half the rate {} Hz and a leg of at least a period '
                         'of the low end, got a minimum leg of {} s'.format(band, rate, minimum_leg))
    sides = []
    for direction, name in ((1, 'rising'), (-1, 'falling')):
        legs: dict[int, list[Sample]] = {}
        for s in samples:
            if s.cruise and s.direction == direction:
                legs.setdefault(s.leg, []).append(s.sample)
        used = []
        for rows in legs.values():
            duration = max(r.time for r in rows) - min(r.time for r in rows)
            if duration >= minimum_leg - 1e-9 and len(rows) >= 2 * band[1] * duration:
                used.append(_leg_swing(rows, direction * position_sign, band, rate))
        if not used:
            raise ValueError('no cruise leg with the angle {} that spans at least {:g} s sampled at {:g} Hz or more '
                             '({} cruise legs were too short or too sparse)'.format(name, minimum_leg, 2 * band[1],
                                                                                    len(legs)))
        sides.append(SwingSide(legs=len(used), frequency=float(np.mean([leg.frequency for leg in used])),
                               swing=float(np.mean([leg.swing for leg in used])),
                               lag_std=float(np.mean([leg.lag_std for leg in used])),
                               lag=float(np.mean([leg.lag for leg in used])),
                               pwm=float(np.mean([leg.pwm for leg in used])),
                               pwm_std=float(np.mean([leg.pwm_std for leg in used]))))
    return SwingStatistics(rising=sides[0], falling=sides[1])


def swing_legs(steps: Sequence[SweepStep], dt: float, minimum_leg: float = SWING_MIN_LEG) -> dict[int, int]:
    """
    the number of legs of a sweep that cruise long enough for swing_statistics, per direction of the angle (the
    SweepStep.direction): the cruise steps of a leg span from the first to the last one plus the period dt they are
    held for. Nothing moves: this is how the stage knows before it starts that it has something to measure.
    """
    legs: dict[int, list[SweepStep]] = {}
    for step in steps:
        if step.cruise:
            legs.setdefault(step.leg, []).append(step)
    counts = {1: 0, -1: 0}
    for cruise in legs.values():
        if cruise[-1].time + dt - cruise[0].time >= minimum_leg - 1e-9:
            counts[cruise[0].direction] += 1
    return counts


def swing_spread(first: SwingStatistics, last: SwingStatistics) -> SwingPair:
    """how much two runs at the same gains differ, per direction: the absolute difference of their swings in
    positions"""
    return SwingPair(abs(first.rising.swing - last.rising.swing), abs(first.falling.swing - last.falling.swing))


def swing_change(reference: Sequence[SwingStatistics], other: SwingStatistics) -> SwingPair:
    """the swing of a run minus the mean swing of the reference runs, per direction, positions"""
    return SwingPair(other.rising.swing - float(np.mean([r.rising.swing for r in reference])),
                     other.falling.swing - float(np.mean([r.falling.swing for r in reference])))


def swing_changed(change: float, spread: float, resolution: float = SWING_RESOLUTION) -> bool:
    """whether a change of the swing is beyond the spread of a repeat (and the resolution: a spread of 0 from two
    runs that happen to agree is no reason to count every change)"""
    return abs(change) > max(spread, resolution)


def recommended_swing_p(tried: Sequence[tuple[int, int, float, float]], yaml_d: int, limit: float) -> int | None:
    """
    the largest P whose swing is at or below the limit in BOTH directions at the yaml D, None when there is none
    :param tried: (p, d, swing with the angle rising, swing with the angle falling) of every setting tried; a P that
        was tried more than once (the yaml gains are run twice) has to be within the limit every time
    :param limit: positions
    """
    within: dict[int, bool] = {}
    for p, d, rising, falling in tried:
        if d == yaml_d:
            within[p] = within.get(p, True) and max(rising, falling) <= limit + SWING_TOLERANCE
    qualified = [p for p, ok in within.items() if ok]
    return max(qualified) if qualified else None
