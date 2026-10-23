"""
The arithmetic of calibration/tune_servo_gains.py: the size of a test step, the metrics of a step response and of a
rest period, the acceptance criteria that mark the knee of a gain sweep, the live abort rules, and the choice of the
gains from the trials. Pure functions on lists of samples, nothing here talks to the robot.

A Sample is one read of the state block of the servo under test (ServoHandler.read_state) with the host time and the
goal position the host last wrote. The quantities are in the units of the servo registers: positions (4096 per turn),
pwm units (885 is 100 % duty) and the 0.229 rpm velocity unit.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from src.robot_controllers.dynamixel_robot import dynamixel_x_config as cfg

# the pwm of the position controller per position of error is P / 128 according to the manual, in units that it
# doesn't give; the script measures it (pwm_per_position) and uses this to size the steps before it has
PWM_PER_POSITION_PER_P = 1.0 / 128.0


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
    """the acceptance criteria of a trial, a trial that violates one is the knee of the sweep"""
    max_overshoot_fraction: float = 0.10  # of the step
    min_overshoot_positions: float = 2.0  # an overshoot below this is quantization, never flagged
    max_ring_count: int = 1
    max_settle_time: float = 0.5  # seconds
    max_rest_pp: float = 2.0  # positions peak to peak at rest, or
    rest_pp_factor: float = 2.0  # this times the peak to peak of the baseline, whichever is larger
    pwm_std_factor: float = 1.5  # times the pwm std of the baseline
    pwm_std_floor: float = 5.0  # pwm units, a std below this is noise however it compares to the baseline
    max_saturated_fraction: float = 0.3  # of the samples of a step, a warning
    max_temperature_rise: float = 3.0  # degrees during one trial, a warning


class AbortLimits(NamedTuple):
    """the live limits, a sample beyond one aborts the trial and restores the gains at once"""
    max_temperature: float = 50.0  # degrees, at the 50 C shutoff temp of the robot, the winding is warmer than the sensor
    max_voltage_drop: float = 0.7  # volts below the voltage at the start of the trial
    max_jump: int = 50  # positions between two samples, an encoder glitch or a slipped belt
    divergence_factor: float = 3.0  # times the step, or
    divergence_minimum: int = 40  # positions of error against the trajectory, or
    divergence_sag_factor: float = 2.5  # times the sag expected at these gains, whichever is the largest
    stream_divergence: int = 120  # positions behind the trajectory during a streamed move (plus the sag term)
    oscillation_factor: float = 3.0  # times the baseline peak to peak, or
    oscillation_minimum: float = 6.0  # positions peak to peak, whichever is larger
    oscillation_window: float = 0.2  # seconds
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
    the raw pwm std of that is half the step and would fail every stable joint at a high P.
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
        flags.add('pwm_noise')
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
        window = [x.position for x in samples if x.time > s.time - limits.oscillation_window]
        pp = max(window) - min(window)
        if pp > max(limits.oscillation_factor * baseline_pp, limits.oscillation_minimum):
            return 'oscillation, {} positions peak to peak at rest'.format(pp)
    if since_goal > limits.saturation_window:
        recent = [x for x in samples if x.time > s.time - limits.saturation_window]
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


def d_values(p: int, fractions: Sequence[float] = (0.0, 0.125, 0.25, 0.5), cap: int = cfg.MAX_GAIN) -> list[int]:
    """the D gains to try at a P gain, as fractions of it"""
    values = sorted({min(cap, round_gain(p * fraction)) for fraction in fractions})
    return values


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


def choose_d(results: Sequence[tuple[int, bool, float, int, float]], pwm_std_factor: float = 1.5,
             pwm_std_floor: float = 5.0) -> int | None:
    """
    the D gain to use from the trials at the chosen P
    :param results: (d, passed, overshoot in positions, ring count, pwm residual std at rest) per trial
    :return: the passed D with the least overshoot and rings whose pwm noise stays below pwm_std_factor times that
        of D = 0 (or of the smallest D tried) and the floor, the smallest D on a tie, None when nothing passed
    """
    if not results:
        return None
    reference = min(results, key=lambda r: r[0])[4]
    limit = max(pwm_std_floor, pwm_std_factor * reference)
    candidates = [r for r in results if r[1] and r[4] <= limit]
    if not candidates:
        return None
    best = min(candidates, key=lambda r: (round(r[2], 1), r[3], r[0]))
    return best[0]
