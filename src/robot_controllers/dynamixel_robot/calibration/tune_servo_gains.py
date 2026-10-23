"""
Tunes the position P and D gains of one arm servo. Run it from the project root with the arm powered, its torque off,
nothing in its way and a hand near the power switch; one joint per run, the wrist first (6, 4, 5), then 3, 1 and 2:
    python -m src.robot_controllers.dynamixel_robot.calibration.tune_servo_gains --joint 6

For every pose (--pose, a few named poses per joint by default) the script moves there with the planner at the yaml
gains, records a baseline at rest, and then runs trials at increasing P (--factor steps from the yaml value up to
--max-factor times it, or down from it when the yaml value itself fails). A trial clamps the goal pwm of the servo
under test (a multiple of the pwm it needs to hold the pose, the only torque limit the XC430 wrist servos have),
writes the gains to that servo only, records the rest (ripple of the position and the pwm), steps the goal a few
positions up and down with the profile of the servo switched off (the response of the position loop itself:
overshoot, rings, settle time, steady state error per direction), and streams a short planned move with the
production profile (the tracking of the trajectory). The step is sized so the pwm stays below the clamp. A trial
that overshoots, rings, doesn't settle, or raises the ripple or the pwm noise at rest is the knee of the sweep; the
recommended P is 0.65 times the knee. When every pose has a recommendation, the smallest is the P of the joint, and
the D gains 0, P/8, P/4 and P/2 are tried at that P at every pose where the step overshoots; the D with the least
overshoot that doesn't add pwm noise is recommended. The other joints hold their position at their yaml gains
throughout.

Safety: every sample is checked live (temperature, voltage drop, position jumps, divergence from the trajectory,
oscillation at rest, pwm stuck at the clamp) and a violation restores the yaml gains, the profile and the goal pwm
at once. Divergence, oscillation and a pwm stuck at the clamp count as the knee; a temperature, voltage or position
jump abort says nothing about the gains and ends the run. Ctrl+C stops after the current read and restores the same;
a second Ctrl+C kills the script, the trial gains and profile then stay in the servo until the robot is started
again, a goal pwm clamp until the servos are rebooted or power cycled (the script resets a leftover clamp when it
starts). Nothing is written to the eeprom. Every write is read back, the sync write of a gain isn't acknowledged by
the servo.

Output: a csv with every sample (<output-dir>/tune_joint<N>_<time>_samples.csv), a csv with the metrics of every
trial (..._trials.csv) and the recommended yaml line per pose and for the joint. Record the static load again
after changing a P gain: the compensation is fitted in positions and scales with 1/P.
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import signal
import sys
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from types import FrameType
from typing import Any, NamedTuple

import numpy as np

from src import global_constants
from src.global_constants import WorkSpaceLimits
from src.global_objects import get_robot
from src.robot_controllers.combined_robot.combined_robot import CombinedRobot
from src.robot_controllers.dynamixel_robot import dynamixel_x_config as cfg
from src.robot_controllers.dynamixel_robot.calibration import record_static_load
from src.robot_controllers.dynamixel_robot.calibration.gain_tuning import (
    AbortLimits, Criteria, RestMetrics, Sample, StepMetrics, TrackingMetrics, abort_reason, choose_d, choose_p,
    d_values, environmental_abort, evaluate, p_values, p_values_down, pwm_clamp, pwm_per_position, rest_metrics,
    step_metrics, step_size, tracking_metrics)
from src.robot_controllers.dynamixel_robot.dynamixel_robot_controller import ARM_IDS, DynamixelRobotController
from src.robot_controllers.dynamixel_robot.dynamixel_utils import DynamixelError
from src.utils.movement import PlannedMove, check_move, plan_move_to_angles
from src.utils.movement_exception import MovementException

logger = logging.getLogger(__name__)

# named poses in degrees of joints 1 to 6 (joint 1: 90 is straight ahead; joint 2: 0 is the lower arm horizontal,
# 90 vertical; joint 2 + 3 = 90 is the upper arm horizontal, 0 is hanging down). The workspace limits are a sphere
# of 60 cm around the base (57 until 23 October 2026), so an arm stretched out horizontally is near its edge; the
# poses below keep 8 degrees of margin on every joint for the test moves.
POSES: dict[str, tuple[float, ...]] = {
    'elbow_out': (90, 90, 0, 0, 0, 0),  # lower arm vertical, upper arm and gripper horizontal: the largest load on
    #                                     joints 3 and 5, joint 2 carries the upper arm's moment without a lever
    'reach': (90, 25, 65, 0, 90, 0),  # lower arm 25 degrees up, upper arm horizontal, gripper bent 90 degrees: the
    #                                   largest lever on joint 2, the gripper's mass off the axes of the wrist rolls
    'low': (90, 60, 0, 0, 0, 0),  # upper arm 30 degrees below horizontal: less load on 3, another lever on 2
}
DEFAULT_POSES: dict[int, tuple[str, ...]] = {
    1: ('elbow_out', 'low'),
    2: ('reach', 'elbow_out', 'low'),
    3: ('elbow_out', 'low'),
    4: ('reach', 'elbow_out'),
    5: ('elbow_out', 'reach'),
    6: ('elbow_out', 'reach'),
}
SAMPLE_FIELDS = ('joint', 'pose', 'trial', 'p', 'd', 'segment', 'repeat', 'step_index', 'direction', 'time', 'tick',
                 'goal', 'position', 'position_trajectory', 'velocity', 'velocity_trajectory', 'pwm', 'current',
                 'moving_status', 'voltage', 'temperature')
TRIAL_FIELDS = ('joint', 'pose', 'pose_degrees', 'trial', 'phase', 'p', 'i', 'd', 'clamp', 'step', 'step_limited',
                'rest_pp', 'rest_pwm_mean', 'rest_pwm_std', 'rest_pwm_residual_std', 'rest_error', 'max_overshoot',
                'max_ring_count',
                'max_settle_time', 'min_rise_time', 'error_up', 'error_down', 'pwm_per_position', 'track_peak',
                'track_rms', 'track_final', 'following_error', 'flags', 'warnings', 'aborted', 'temperature_start',
                'temperature_end', 'voltage_min', 'sample_period')
STEP_SEQUENCE = ((1, 1), (-1, 0), (-1, -1), (1, 0))  # (direction, offset in steps from the pose) of a repeat
MOVE_HOLD = 1.0  # seconds of hold after a streamed move
GAIN_SETTLE = 0.3  # seconds between a gain write and the first measurement
SAMPLES_PER_ABORT_CHECK = 5
READ_ATTEMPTS = 3  # reads of the servo before a failure stops the run


class Stop(Exception):
    """raised to stop the tuning, with the reason"""


class StopFlag:
    """turns the first Ctrl+C into a request to stop after the current read, a second one kills the script
    unless the restore is running"""

    def __init__(self) -> None:
        self.requested = False
        self.protected = False  # the restore is running, a second Ctrl+C has to wait for it
        self._previous: Any = None

    def install(self) -> None:
        self._previous = signal.signal(signal.SIGINT, self._handle)

    def uninstall(self) -> None:
        if self._previous is not None:
            signal.signal(signal.SIGINT, self._previous)
            self._previous = None

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        if self.requested:
            if self.protected:
                print('\nrestoring the servo, wait')
                return
            # the second one: back to the default, which raises KeyboardInterrupt wherever the script is
            self.uninstall()
            raise KeyboardInterrupt
        self.requested = True
        print('\nstopping after the current read, the gains are restored; Ctrl+C again to kill the script')


class Settings(NamedTuple):
    factor: float = 1.3
    max_factor: float = 2.0
    tune_d: bool = True
    d_fractions: tuple[float, ...] = (0.0, 0.125, 0.25, 0.5)
    repeats: int = 2
    hold: float = 0.8  # seconds after every step
    rest: float = 3.0  # seconds of rest per trial
    baseline_rest: float = 5.0  # seconds of rest at the yaml gains per pose
    move_degrees: float = 8.0  # the streamed move, 0 disables it
    speed: float = 0.5  # rad/s of the planned moves
    clamp: int | None = None  # goal pwm clamp, None computes it from the holding pwm
    criteria: Criteria = Criteria()
    limits: AbortLimits = AbortLimits()


class TrialResult(NamedTuple):
    phase: str
    p: int
    d: int
    clamp: int
    step: int
    step_limited: bool
    rest: RestMetrics
    steps: tuple[StepMetrics, ...]
    tracking: tuple[TrackingMetrics, ...]
    pwm_slope: float | None
    flags: tuple[str, ...]
    warnings: tuple[str, ...]
    aborted: str | None
    temperature_start: int
    temperature_end: int
    voltage_min: float

    @property
    def passed(self) -> bool:
        return not self.flags and self.aborted is None

    @property
    def environmental(self) -> bool:
        """aborted for a reason that says nothing about the gains"""
        return self.aborted is not None and environmental_abort(self.aborted)

    @property
    def max_overshoot(self) -> float:
        return max((m.overshoot for m in self.steps), default=0.0)

    @property
    def max_ring_count(self) -> int:
        return max((m.ring_count for m in self.steps), default=0)


class PoseResult(NamedTuple):
    number: int  # the number of the pose in the run, as in the csv files
    pose: np.ndarray
    baseline: RestMetrics
    trials: list[TrialResult]
    recommended_p: int | None
    knee_p: int | None
    recommended_d: int | None
    d_tested: bool


def parse_pose(value: str) -> np.ndarray:
    """a pose name or 6 comma separated degrees of joints 1 to 6 into the 7 element angle array in radians"""
    if value in POSES:
        degrees = POSES[value]
    else:
        try:
            degrees = tuple(float(part) for part in value.split(','))
        except ValueError:
            raise argparse.ArgumentTypeError('a pose is one of {} or 6 comma separated degrees, got {!r}'
                                             .format(', '.join(POSES), value))
        if len(degrees) != 6:
            raise argparse.ArgumentTypeError('a pose needs the degrees of joints 1 to 6, got {!r}'.format(value))
    angles = np.zeros(7, dtype=np.float64)
    angles[1:] = np.radians(degrees)
    return angles


def pose_is_accepted(pose: np.ndarray, robot: DynamixelRobotController | CombinedRobot, joint: int,
                     excursion: float) -> str | None:
    """why the planner refuses the pose or the pose with the joint moved by the excursion either way, or None"""
    for offset in (0.0, excursion, -excursion):
        angles = pose.copy()
        angles[joint] += offset
        try:
            check_move(PlannedMove('pose', angles[np.newaxis], 0.0), robot)
        except MovementException as e:
            return str(e)
    return None


def pose_degrees(pose: np.ndarray) -> str:
    return ' '.join('{:.0f}'.format(value) for value in np.degrees(pose[1:]))


def joint_p_of(results: Sequence[PoseResult]) -> int | None:
    """the P gain of the joint: the smallest recommendation of the poses, None when a pose has none"""
    if not results or any(r.recommended_p is None for r in results):
        return None
    return min(r.recommended_p for r in results if r.recommended_p is not None)


class TuningSession:
    """the trials of one joint on the robot, see the module docstring"""

    def __init__(self, robot: DynamixelRobotController | CombinedRobot, joint: int, settings: Settings,
                 sample_writer: csv.DictWriter[str] | None = None, trial_writer: csv.DictWriter[str] | None = None,
                 clock: Callable[[], float] = time.perf_counter, sleep: Callable[[float], None] = time.sleep,
                 stop: StopFlag | None = None) -> None:
        self.robot = robot
        self.handler = robot.servo_handler
        self.joint = joint
        self.servo = self.handler.get_servo(joint)
        self.settings = settings
        self.sample_writer = sample_writer
        self.trial_writer = trial_writer
        self.clock = clock
        self.sleep = sleep
        self.stop = stop if stop is not None else StopFlag()
        self.original_goal_pwm: int | None = None
        self.results: list[PoseResult] = []  # the poses done so far, kept when the run stops early
        self.pose_index = 0
        self.trial_index = 0
        self.pose_goal = 0
        self.trial_start = 0.0
        self.last_goal_time = 0.0
        self.sag = 0.0  # the position error expected at rest at the gains of the current trial
        self.read_failures = 0
        self.current_p = self.servo.p
        self.current_d = self.servo.d

    # the robot

    def snapshot(self) -> None:
        """check the servo has the yaml gains and no goal pwm clamp left by an earlier run"""
        goal_pwm = self.handler.read_goal_pwm([self.joint])
        if goal_pwm is None:
            raise Stop('failed to read the goal pwm of servo {}'.format(self.joint))
        if goal_pwm[self.joint] != cfg.MAX_PWM:
            logger.warning('servo %d had a goal pwm of %d, a clamp left by an earlier run, setting it to %d',
                           self.joint, goal_pwm[self.joint], cfg.MAX_PWM)
            if not self.handler.set_goal_pwm(self.joint, cfg.MAX_PWM) or not self.verify_settings(cfg.MAX_PWM, None):
                raise Stop('could not reset the goal pwm of servo {}'.format(self.joint))
        self.original_goal_pwm = cfg.MAX_PWM
        if not self.verify_gains(self.servo.p, self.servo.i, self.servo.d):
            raise Stop('servo {} does not have the gains of the yaml, start the robot again'.format(self.joint))

    def verify_gains(self, p: int, i: int, d: int) -> bool:
        gains = self.handler.read_pid([self.joint])
        if gains is None:
            raise Stop('failed to read the gains of servo {}'.format(self.joint))
        if gains[self.joint] != (p, i, d):
            logger.error('servo %d has gains p=%d i=%d d=%d, not the written p=%d i=%d d=%d', self.joint,
                         *gains[self.joint], p, i, d)
            return False
        return True

    def verify_settings(self, goal_pwm: int | None, profile: tuple[int, int] | None) -> bool:
        """read the goal pwm and the profile of the servo back and compare the ones given"""
        read_pwm = self.handler.read_goal_pwm([self.joint])
        read_profile = self.handler.read_profile([self.joint])
        if read_pwm is None or read_profile is None:
            raise Stop('failed to read the goal pwm or the profile of servo {}'.format(self.joint))
        ok = True
        if goal_pwm is not None and read_pwm[self.joint] != goal_pwm:
            logger.error('servo %d has a goal pwm of %d, not the written %d', self.joint, read_pwm[self.joint],
                         goal_pwm)
            ok = False
        if profile is not None and read_profile[self.joint] != profile:
            logger.error('servo %d has the profile %s, not the written %s', self.joint, read_profile[self.joint],
                         profile)
            ok = False
        return ok

    def write_gains(self, p: int, d: int) -> None:
        """write the trial gains to the servo under test and read them back"""
        if not self.handler.set_pid_single_servo(self.joint, p, 0, d) or not self.verify_gains(p, 0, d):
            self.restore()
            raise Stop('the gain write to servo {} failed'.format(self.joint))
        self.current_p, self.current_d = p, d

    def restore(self) -> bool:
        """put the yaml gains, the yaml profile and the original goal pwm back, returns False when a read-back
        doesn't match"""
        ok = self.handler.set_pid([self.joint])
        ok &= self.handler.set_profile(self.joint, self.servo.profile_velocity, self.servo.profile_acceleration)
        if self.original_goal_pwm is not None:
            ok &= self.handler.set_goal_pwm(self.joint, self.original_goal_pwm)
        self.current_p, self.current_d = self.servo.p, self.servo.d
        gains = self.handler.read_pid([self.joint])
        profile = self.handler.read_profile([self.joint])
        goal_pwm = self.handler.read_goal_pwm([self.joint])
        if gains is None or profile is None or goal_pwm is None:
            logger.error('could not read servo %d back after restoring its settings', self.joint)
            return False
        expected = ((self.servo.p, self.servo.i, self.servo.d),
                    (self.servo.profile_velocity, self.servo.profile_acceleration), self.original_goal_pwm)
        actual = (gains[self.joint], profile[self.joint], goal_pwm[self.joint])
        if self.original_goal_pwm is None:
            actual = (actual[0], actual[1], None)
        if actual != expected:
            logger.error('servo %d has gains, profile, goal pwm %s after the restore, expected %s', self.joint,
                         actual, expected)
            return False
        return ok

    def check_servos(self) -> None:
        """stop on a hardware error, a servo warmer than the warning, or the servo under test too warm to be
        allowed to warm up by a trial"""
        try:
            record_static_load.check_servos(self.robot, self.robot.servo_config.temperature_warning)
        except record_static_load.StopRecording as e:
            raise Stop(str(e))
        temperatures = self.handler.read_temperatures([self.joint])
        if temperatures is None:
            raise Stop('failed to read the temperature of servo {}'.format(self.joint))
        limit = self.settings.limits.max_temperature - self.settings.criteria.max_temperature_rise
        if temperatures[self.joint] >= limit:
            raise Stop('servo {} is at {}C, let it cool below {:.0f}C before a trial'.format(
                self.joint, temperatures[self.joint], limit))

    def move_to_pose(self, pose: np.ndarray) -> bool:
        """plan and execute a move to the pose at the yaml gains, returns False when the planner refuses it"""
        if not self.restore():
            raise Stop('could not restore the settings of servo {} before moving'.format(self.joint))
        try:
            return record_static_load.move_to(self.robot, pose, self.settings.speed, lambda: self.stop.requested)
        except record_static_load.StopRecording as e:
            raise Stop(str(e))

    # sampling

    def sample(self, rows: list[Sample], goal: int) -> Sample:
        if self.stop.requested:
            raise Stop('interrupted')
        states = None
        for _ in range(READ_ATTEMPTS):
            states = self.handler.read_state([self.joint])
            if states is not None:
                break
            self.read_failures += 1
        if states is None:
            raise Stop('failed to read servo {} {} times in a row'.format(self.joint, READ_ATTEMPTS))
        s = states[self.joint]
        row = Sample(time=self.clock() - self.trial_start, tick=s.tick, goal=goal, position=s.position,
                     position_trajectory=s.position_trajectory, velocity=s.velocity,
                     velocity_trajectory=s.velocity_trajectory, pwm=s.pwm, current=s.current,
                     moving_status=s.moving_status, voltage=s.voltage, temperature=s.temperature)
        rows.append(row)
        return row

    def capture(self, rows: list[Sample], goal: int, duration: float, clamp: int, step: int, baseline_pp: float,
                start_voltage: float) -> str | None:
        """read the servo for the duration, checking the abort rules; returns the reason to abort or None
        :param clamp: the goal pwm the servo has now
        """
        end = self.clock() + duration
        count = 0
        while self.clock() < end:
            self.sample(rows, goal)
            count += 1
            if count % SAMPLES_PER_ABORT_CHECK == 0:
                reason = abort_reason(rows, clamp, step, baseline_pp, start_voltage, self.last_goal_time,
                                      self.settings.limits, sag=self.sag)
                if reason is not None:
                    return reason
        return None

    def write_goal(self, position: int) -> int:
        """write a goal position to the servo under test, clamped to its calibrated range, returns what was sent"""
        position = self.servo.clamp_position(position)
        if not self.handler.move_servo_to_pos(self.joint, position):
            raise Stop('failed to write the goal position of servo {}'.format(self.joint))
        self.last_goal_time = self.clock() - self.trial_start
        return position

    def write_samples(self, rows: Sequence[Sample], p: int, d: int, segment: str, repeat: int, step_index: int,
                      direction: int) -> None:
        if self.sample_writer is None:
            return
        for s in rows:
            self.sample_writer.writerow({'joint': self.joint, 'pose': self.pose_index, 'trial': self.trial_index,
                                         'p': p, 'd': d, 'segment': segment, 'repeat': repeat,
                                         'step_index': step_index, 'direction': direction,
                                         'time': '{:.4f}'.format(s.time), 'tick': s.tick, 'goal': s.goal,
                                         'position': s.position, 'position_trajectory': s.position_trajectory,
                                         'velocity': s.velocity, 'velocity_trajectory': s.velocity_trajectory,
                                         'pwm': s.pwm, 'current': s.current, 'moving_status': s.moving_status,
                                         'voltage': s.voltage, 'temperature': s.temperature})

    def write_trial(self, pose: np.ndarray, result: TrialResult) -> None:
        if self.trial_writer is None:
            return
        errors_up = [m.error for m in result.steps if m.step > 0]
        errors_down = [m.error for m in result.steps if m.step < 0]
        settled = [m.settle_time for m in result.steps if m.settle_time is not None]
        max_settle = '{:.3f}'.format(max(settled)) if settled and len(settled) == len(result.steps) else ''
        rise = [m.rise_time for m in result.steps if m.rise_time is not None]
        self.trial_writer.writerow({
            'joint': self.joint, 'pose': self.pose_index, 'pose_degrees': pose_degrees(pose),
            'trial': self.trial_index, 'phase': result.phase, 'p': result.p, 'i': 0, 'd': result.d,
            'clamp': result.clamp, 'step': result.step, 'step_limited': result.step_limited,
            'rest_pp': result.rest.position_pp, 'rest_pwm_mean': '{:.1f}'.format(result.rest.pwm_mean),
            'rest_pwm_std': '{:.2f}'.format(result.rest.pwm_std),
            'rest_pwm_residual_std': '{:.2f}'.format(result.rest.pwm_residual_std),
            'rest_error': '{:.1f}'.format(result.rest.error_mean),
            'max_overshoot': '{:.1f}'.format(result.max_overshoot), 'max_ring_count': result.max_ring_count,
            'max_settle_time': max_settle,
            'min_rise_time': '{:.3f}'.format(min(rise)) if rise else '',
            'error_up': '{:.1f}'.format(float(np.mean(errors_up))) if errors_up else '',
            'error_down': '{:.1f}'.format(float(np.mean(errors_down))) if errors_down else '',
            'pwm_per_position': '' if result.pwm_slope is None else '{:.2f}'.format(result.pwm_slope),
            'track_peak': max((t.peak_trajectory_error for t in result.tracking), default=''),
            'track_rms': '{:.1f}'.format(max(t.rms_trajectory_error for t in result.tracking)) if result.tracking else '',
            'track_final': '{:.1f}'.format(max((t.final_error for t in result.tracking), key=abs)) if result.tracking else '',
            'following_error': any(t.following_error for t in result.tracking),
            'flags': ' '.join(result.flags), 'warnings': ' '.join(result.warnings),
            'aborted': result.aborted or '', 'temperature_start': result.temperature_start,
            'temperature_end': result.temperature_end, 'voltage_min': result.voltage_min,
            'sample_period': '{:.4f}'.format(result.rest.period)})

    # trials

    def baseline(self, pose: np.ndarray) -> RestMetrics:
        """the rest at the yaml gains at the pose, the reference of the trials there"""
        self.trial_index += 1
        self.trial_start = self.clock()
        self.last_goal_time = 0.0
        self.sag = math.inf  # not known yet, the divergence rule is off
        rows: list[Sample] = []
        clamp = self.original_goal_pwm if self.original_goal_pwm is not None else cfg.MAX_PWM
        first = self.sample(rows, self.pose_goal)
        reason = self.capture(rows, self.pose_goal, self.settings.baseline_rest, clamp, 0, 0.0, first.voltage)
        self.write_samples(rows, self.servo.p, self.servo.d, 'baseline', 0, 0, 0)
        if reason is not None:
            raise Stop('the servo is not at rest at the yaml gains: {}'.format(reason))
        return rest_metrics(rows, self.servo.p)

    def abort(self, reason: str) -> bool:
        """restore the servo at once, returns whether the restore was verified"""
        print('  aborted: {}'.format(reason))
        ok = self.restore()
        self.write_goal(self.pose_goal)
        return ok

    def set_clamp(self, clamp: int) -> None:
        if not self.handler.set_goal_pwm(self.joint, clamp) or not self.verify_settings(clamp, None):
            self.restore()
            raise Stop('failed to set the goal pwm of servo {}'.format(self.joint))

    def run_trial(self, pose: np.ndarray, phase: str, p: int, d: int, baseline: RestMetrics) -> TrialResult:
        """one trial at the gains: rest, steps with the profile off, a streamed move with the profile on"""
        settings = self.settings
        self.trial_index += 1
        self.check_servos()
        self.trial_start = self.clock()
        self.last_goal_time = 0.0
        rows: list[Sample] = []
        holding = abs(baseline.pwm_mean)
        if settings.clamp is not None:
            if settings.clamp < 1.5 * holding:
                self.restore()
                raise Stop('--pwm-clamp {} is below 1.5 times the holding pwm of {:.0f} at this pose'.format(
                    settings.clamp, holding))
            clamp = settings.clamp
        else:
            clamp = pwm_clamp(holding)
        # the sag scales with 1 / P, the abort rules allow a multiple of it
        self.sag = baseline.error_mean * self.servo.p / p
        print('  trial {} ({}): p={} d={} clamp={}'.format(self.trial_index, phase, p, d, clamp))

        # the clamp first, so the output is bounded from the moment the gains change
        self.set_clamp(clamp)
        self.write_gains(p, d)
        self.write_goal(self.pose_goal)  # a new goal so the servo applies the gains
        self.sleep(GAIN_SETTLE)
        first = self.sample(rows, self.pose_goal)
        start_voltage = first.voltage
        steps: list[StepMetrics] = []
        tracking: list[TrackingMetrics] = []
        slope_points: list[tuple[float, float]] = []
        restore_ok = True
        step, limited = 0, False

        aborted = self.capture(rows, self.pose_goal, settings.rest, clamp, 0, baseline.position_pp, start_voltage)
        self.write_samples(rows, p, d, 'rest', 0, 0, 0)
        rest = rest_metrics(rows, p)
        slope_points.append((self.pose_goal - rows[-1].position, rows[-1].pwm))
        if aborted is not None:
            restore_ok = self.abort(aborted)
        else:
            # this side of the friction band may need more pwm than the baseline showed: widen the clamp
            holding = max(holding, abs(rest.pwm_mean))
            if settings.clamp is None and pwm_clamp(holding) > clamp:
                clamp = pwm_clamp(holding)
                self.set_clamp(clamp)
            step, limited = step_size(p, holding, clamp)
            if limited:
                logger.warning('P %d leaves no room for a step below the clamp of %d at this pose, using %d '
                               'positions', p, clamp, step)
            # the steps: profile off so the loop sees a step
            if not self.handler.set_profile(self.joint, 0, 0) or not self.verify_settings(None, (0, 0)):
                self.restore()
                raise Stop('failed to switch off the profile of servo {}'.format(self.joint))
            previous_goal = self.pose_goal
            for repeat in range(settings.repeats):
                for step_index, (direction, offset) in enumerate(STEP_SEQUENCE):
                    segment_start = max(0, len(rows) - SAMPLES_PER_ABORT_CHECK)
                    goal = self.write_goal(self.pose_goal + offset * step)
                    step_time = self.last_goal_time
                    aborted = self.capture(rows, goal, settings.hold, clamp, step, baseline.position_pp,
                                           start_voltage)
                    self.write_samples(rows[segment_start:], p, d, 'step', repeat, step_index, direction)
                    if aborted is not None:
                        break
                    metrics = step_metrics(rows[segment_start:], step_time, previous_goal, goal, clamp)
                    steps.append(metrics)
                    slope_points.append((goal - metrics.final, metrics.pwm_final))
                    previous_goal = goal
                if aborted is not None:
                    break
            if aborted is not None:
                restore_ok = self.abort(aborted)
            else:
                # the profile and the goal pwm back for the streamed move
                original = self.original_goal_pwm if self.original_goal_pwm is not None else cfg.MAX_PWM
                profile = (self.servo.profile_velocity, self.servo.profile_acceleration)
                if not self.handler.set_profile(self.joint, *profile) or \
                        not self.handler.set_goal_pwm(self.joint, original) or \
                        not self.verify_settings(original, profile):
                    self.restore()
                    raise Stop('failed to restore the profile or the goal pwm of servo {}'.format(self.joint))
                if settings.move_degrees > 0:
                    aborted = self.move_test(pose, rows, p, d, original, step, baseline.position_pp,
                                             start_voltage, tracking)
                    if aborted is not None:
                        restore_ok = self.abort(aborted)

        # the yaml gains back between trials, so a crash doesn't leave trial gains in the servo
        if aborted is None and not self.restore():
            raise Stop('could not restore the settings of servo {} after the trial'.format(self.joint))
        flags, warnings = evaluate(rest, steps, baseline, settings.criteria, p_ratio=p / self.servo.p,
                                   temperature_rise=float(rows[-1].temperature - first.temperature))
        if aborted is not None:
            flags = ['aborted'] + flags
        result = TrialResult(phase=phase, p=p, d=d, clamp=clamp, step=step, step_limited=limited, rest=rest,
                             steps=tuple(steps), tracking=tuple(tracking), pwm_slope=pwm_per_position(slope_points),
                             flags=tuple(flags), warnings=tuple(warnings), aborted=aborted,
                             temperature_start=first.temperature, temperature_end=rows[-1].temperature,
                             voltage_min=min(s.voltage for s in rows))
        self.write_trial(pose, result)
        print('    rest pp {} pwm {:.0f} std {:.1f} residual {:.1f} error {:.0f}, overshoot {:.1f}, rings {}, '
              'errors up {} down {}, {}{}'.format(rest.position_pp, rest.pwm_mean, rest.pwm_std,
                                                  rest.pwm_residual_std, rest.error_mean, result.max_overshoot,
                                                  result.max_ring_count,
                            ' '.join('{:+.0f}'.format(m.error) for m in steps if m.step > 0),
                            ' '.join('{:+.0f}'.format(m.error) for m in steps if m.step < 0),
                            'flags ' + ' '.join(flags) if flags else 'passed',
                            ', warnings ' + ' '.join(warnings) if warnings else ''))
        if not restore_ok:
            raise Stop('could not restore the settings of servo {} after the abort'.format(self.joint))
        return result

    def move_test(self, pose: np.ndarray, rows: list[Sample], p: int, d: int, clamp: int, step: int,
                  baseline_pp: float, start_voltage: float, tracking: list[TrackingMetrics]) -> str | None:
        """stream a planned move of the joint away from the pose and back, sampling; returns the abort reason

        The moves are planned from the commanded angles, not the measured ones, so the other joints keep their
        goal and their holding error.
        """
        excursion = np.radians(self.settings.move_degrees)
        previous = pose.copy()
        for direction, target_offset in ((1, excursion), (-1, 0.0)):
            target = pose.copy()
            target[self.joint] += target_offset
            try:
                move = plan_move_to_angles(self.robot, previous, target, angular_speed=self.settings.speed)
            except MovementException as e:
                logger.warning('skipping the move test: %s', e)
                return None
            segment_start = len(rows)
            dt = 1.0 / global_constants.steps_per_second
            next_step = self.clock()
            for angles in move.angles:
                self.robot.move_servos(angles)
                self.last_goal_time = self.clock() - self.trial_start
                self.sample(rows, self.servo.target_position)
                reason = abort_reason(rows, clamp, step, baseline_pp, start_voltage, self.last_goal_time,
                                      self.settings.limits, sag=self.sag, streaming=True)
                if reason is not None:
                    self.write_samples(rows[segment_start:], p, d, 'move', 0, 0, direction)
                    return reason
                next_step += dt
                self.sleep(max(0.0, next_step - self.clock()))
            reason = self.capture(rows, self.servo.target_position, MOVE_HOLD, clamp, step, baseline_pp,
                                  start_voltage)
            self.write_samples(rows[segment_start:], p, d, 'move', 0, 0, direction)
            if reason is not None:
                return reason
            tracking.append(tracking_metrics(rows[segment_start:]))
            previous = target
        return None

    # the sweep

    def record(self, trials: list[TrialResult], result: TrialResult) -> None:
        """keep the trial, and stop the run when it aborted for a reason that says nothing about the gains"""
        trials.append(result)
        if result.environmental:
            raise Stop('trial p={} d={} aborted: {}; that says nothing about the gains, let the servo cool down or '
                       'check the supply and run again'.format(result.p, result.d, result.aborted))

    def arrive(self, index: int, pose: np.ndarray) -> RestMetrics | None:
        """move to the pose and record the baseline there, None when the planner refuses the pose"""
        self.pose_index = index
        self.check_servos()
        if not self.move_to_pose(pose):
            return None
        self.pose_goal = self.servo.target_position
        self.sleep(self.settings.baseline_rest / 2)
        baseline = self.baseline(pose)
        print('  baseline at p={} d={}: rest pp {}, holding pwm {:.0f} std {:.1f} residual {:.1f}, error {:.0f}, '
              '{:.1f} ms per sample'.format(self.servo.p, self.servo.d, baseline.position_pp, baseline.pwm_mean,
                                            baseline.pwm_std, baseline.pwm_residual_std, baseline.error_mean,
                                            1000 * baseline.period))
        return baseline

    def run_p_phase(self, index: int, pose: np.ndarray) -> PoseResult | None:
        """the P sweep at one pose, None when the planner refuses the pose"""
        settings = self.settings
        print('pose {}: {} degrees'.format(index, pose_degrees(pose)))
        baseline = self.arrive(index, pose)
        if baseline is None:
            return None
        trials: list[TrialResult] = []
        for p in p_values(self.servo.p, settings.factor, settings.max_factor):
            result = self.run_trial(pose, 'p', p, self.servo.d, baseline)
            self.record(trials, result)
            if not result.passed:
                break
        if not trials[0].passed:
            for p in p_values_down(self.servo.p, settings.factor):
                result = self.run_trial(pose, 'p_down', p, self.servo.d, baseline)
                self.record(trials, result)
                if result.passed:
                    break
        recommended_p, knee_p = choose_p([(t.p, t.passed) for t in trials], settings.factor)
        print('  P: knee {}, recommended {}'.format(knee_p, recommended_p))
        return PoseResult(index, pose, baseline, trials, recommended_p, knee_p, None, False)

    def run_d_phase(self, result: PoseResult, joint_p: int) -> PoseResult:
        """the D sweep at the P of the joint, back at the pose of the result"""
        settings = self.settings
        print('pose {} again for D at p={}'.format(result.number, joint_p))
        baseline = self.arrive(result.number, result.pose)
        if baseline is None:
            return result
        values = d_values(joint_p, settings.d_fractions)
        if self.servo.d not in values:
            values.append(self.servo.d)
        trials: list[TrialResult] = []
        for d in sorted(set(values)):
            trial = self.run_trial(result.pose, 'd', joint_p, d, baseline)
            self.record(trials, trial)
            if d == 0 and trial.passed and trial.max_overshoot <= settings.criteria.min_overshoot_positions \
                    and self.servo.d == 0:
                print('  no overshoot to damp at d=0, skipping the other D values')
                break
        recommended_d = choose_d([(t.d, t.passed, t.max_overshoot, t.max_ring_count, t.rest.pwm_residual_std)
                                  for t in trials], settings.criteria.pwm_std_factor,
                                 settings.criteria.pwm_std_floor)
        print('  D: recommended {}'.format(recommended_d))
        return result._replace(baseline=baseline, trials=result.trials + trials, recommended_d=recommended_d,
                               d_tested=True)

    def run(self, poses: Sequence[np.ndarray]) -> list[PoseResult]:
        """the P sweep at every pose, then the D sweep at the P of the joint at every pose; the results so far
        stay in self.results when a Stop ends the run"""
        self.snapshot()
        for index, pose in enumerate(poses, start=1):
            result = self.run_p_phase(index, pose)
            if result is not None:
                self.results.append(result)
        if self.settings.tune_d:
            joint_p = joint_p_of(self.results)
            if joint_p is None:
                print('no D sweep: a pose has no P recommendation')
            else:
                for position, result in enumerate(self.results):
                    self.results[position] = self.run_d_phase(result, joint_p)
        return self.results


def recommend(results: Sequence[PoseResult], yaml_d: int,
              criteria: Criteria = Criteria()) -> tuple[int | None, int | None]:
    """
    the gains for the joint from the poses: the smallest recommended P (None when a pose has none), and the D that
    passed at every pose where it was tried with the least overshoot (the yaml D when D was not tried, None when no
    D passed)
    """
    p = joint_p_of(results)
    if p is None:
        return None, None
    tested = [r for r in results if r.d_tested]
    if not tested:
        return p, yaml_d
    by_d: dict[int, list[TrialResult]] = {}
    for r in tested:
        for t in r.trials:
            if t.phase == 'd':
                by_d.setdefault(t.d, []).append(t)
    candidates = [(d, all(t.passed for t in trials), max(t.max_overshoot for t in trials),
                   max(t.max_ring_count for t in trials), max(t.rest.pwm_residual_std for t in trials))
                  for d, trials in by_d.items()]
    return p, choose_d(candidates, criteria.pwm_std_factor, criteria.pwm_std_floor)


def print_summary(joint: int, results: Sequence[PoseResult], servo_p: int, servo_d: int,
                  stopped: bool = False) -> None:
    print('\nsummary of joint {} (yaml p={} d={}):'.format(joint, servo_p, servo_d))
    if stopped:
        print('  the run stopped early: the poses below are the ones that were finished, a pose that was not '
              'done may have a lower knee')
    for r in results:
        print('  pose {} ({} degrees): P knee {}, recommended {}, D {}'.format(
            r.number, pose_degrees(r.pose), r.knee_p, r.recommended_p,
            r.recommended_d if r.d_tested else 'not tried'))
        for t in r.trials:
            print('    {:7s} p={:5d} d={:4d} step {:3d}: overshoot {:5.1f} rings {} rest pp {} pwm residual {:5.1f} '
                  'slope {} {}{}'.format(t.phase, t.p, t.d, t.step, t.max_overshoot, t.max_ring_count,
                                         t.rest.position_pp, t.rest.pwm_residual_std,
                                         '?' if t.pwm_slope is None else '{:.1f}'.format(t.pwm_slope),
                                         t.aborted or ' '.join(t.flags) or 'passed',
                                         ' (warnings: {})'.format(' '.join(t.warnings)) if t.warnings else ''))
    p, d = recommend(results, servo_d)
    if p is None:
        print('no recommendation: a pose had no passing P gain below its knee, look at the trials')
        return
    if d is None:
        print('no D passed at every pose where it was tried, keeping the yaml d={} (not confirmed at this P, look '
              'at the D trials)'.format(servo_d))
        d = servo_d
    print('recommended for servo_config.yaml (record the static load again after changing P):')
    print('    pid: {{p: {}, i: 0, d: {}}}'.format(p, d))


def plot_results(joint: int, results: Sequence[PoseResult], samples_path: str) -> None:
    """the first step up of every trial per pose, from the samples csv"""
    import matplotlib.pyplot as plt
    with open(samples_path, newline='') as samples_file:
        rows = list(csv.DictReader(samples_file))
    for r in results:
        figure, (top, bottom) = plt.subplots(2, 1, sharex=True)
        figure.suptitle('joint {} pose {}: {} degrees'.format(joint, r.number, pose_degrees(r.pose)))
        for t in r.trials:
            trial_rows = [row for row in rows if int(row['pose']) == r.number and int(row['p']) == t.p
                          and int(row['d']) == t.d and row['segment'] == 'step' and row['repeat'] == '0'
                          and row['step_index'] == '0']
            if not trial_rows:
                continue
            start = float(trial_rows[0]['time'])
            times = [float(row['time']) - start for row in trial_rows]
            goal = int(trial_rows[-1]['goal'])
            top.plot(times, [int(row['position']) - goal for row in trial_rows],
                     label='{} p={} d={}'.format(t.phase, t.p, t.d))
            bottom.plot(times, [int(row['pwm']) for row in trial_rows])
        top.set_ylabel('position - goal')
        top.legend()
        bottom.set_ylabel('pwm')
        bottom.set_xlabel('s')
    plt.show()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='tune the position P and D gains of one arm servo')
    parser.add_argument('--joint', type=int, required=True, help='the joint to tune, 1 to 6')
    parser.add_argument('--pose', nargs='+', default=None,
                        help='poses to tune at: {} or 6 comma separated degrees of joints 1 to 6'.format(
                            ', '.join(POSES)))
    parser.add_argument('--factor', type=float, default=1.3, help='the factor between the P gains of the sweep')
    parser.add_argument('--max-factor', type=float, default=2.0, help='sweep P up to this times the yaml value')
    parser.add_argument('--no-d', action='store_true', help='only tune P')
    parser.add_argument('--d-fractions', default='0,0.125,0.25,0.5', help='the D gains to try as fractions of P')
    parser.add_argument('--repeats', type=int, default=2, help='how often the steps are repeated per trial')
    parser.add_argument('--hold', type=float, default=0.8, help='seconds after every step')
    parser.add_argument('--rest', type=float, default=3.0, help='seconds of rest per trial')
    parser.add_argument('--baseline-rest', type=float, default=5.0, help='seconds of rest at the yaml gains per pose')
    parser.add_argument('--move-degrees', type=float, default=8.0, help='the streamed move per trial, 0 disables it')
    parser.add_argument('--speed', type=float, default=0.5, help='joint speed of the planned moves in rad/s')
    parser.add_argument('--pwm-clamp', type=int, default=None,
                        help='goal pwm during the trials (885 is 100%%), default 3 times the holding pwm, at least '
                             'half the limit')
    parser.add_argument('--max-temperature', type=float, default=AbortLimits().max_temperature,
                        help='abort a trial when the servo reaches this temperature')
    parser.add_argument('--floor', type=float, default=None,
                        help='cm above the table the wrist and gripper tip stay, default WorkSpaceLimits.z_min ({})'
                        .format(WorkSpaceLimits.z_min))
    parser.add_argument('--output-dir', default='.', help='where the csv files go')
    parser.add_argument('--plot', action='store_true', help='plot the step responses at the end')
    parser.add_argument('--yes', action='store_true', help="don't ask before enabling the servos")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

    if args.joint not in ARM_IDS:
        print('the joint has to be one of {}'.format(ARM_IDS))
        return 1
    try:
        poses = [parse_pose(value) for value in (args.pose or DEFAULT_POSES[args.joint])]
        fractions = tuple(float(part) for part in args.d_fractions.split(','))
    except (argparse.ArgumentTypeError, ValueError) as e:
        print(e)
        return 1
    settings = Settings(factor=args.factor, max_factor=args.max_factor, tune_d=not args.no_d,
                        d_fractions=fractions, repeats=args.repeats, hold=args.hold, rest=args.rest,
                        baseline_rest=args.baseline_rest, move_degrees=args.move_degrees, speed=args.speed,
                        clamp=args.pwm_clamp, limits=AbortLimits(max_temperature=args.max_temperature))

    try:
        robot = get_robot()
    except DynamixelError as e:
        print(e)
        return 1
    if not isinstance(robot, CombinedRobot):
        print('the gains can only be tuned on the real robot, use_simulation is set')
        return 1
    if args.floor is not None:
        robot.workspace_limits = record_static_load.limits_with_floor(args.floor, robot.workspace_limits)
    excursion = np.radians(max(args.move_degrees, 6.0))
    accepted = []
    for pose in poses:
        refused = pose_is_accepted(pose, robot, args.joint, excursion)
        if refused is None:
            accepted.append(pose)
        else:
            print('skipping pose {} degrees: {}'.format(pose_degrees(pose), refused))
    if not accepted:
        print('no pose the planner accepts')
        return 1
    servo = robot.servo_handler.get_servo(args.joint)

    stamp = datetime.now().strftime('%Y%m%d_%H%M')
    samples_path = '{}/tune_joint{}_{}_samples.csv'.format(args.output_dir, args.joint, stamp)
    trials_path = '{}/tune_joint{}_{}_trials.csv'.format(args.output_dir, args.joint, stamp)
    print('joint {} (yaml p={} d={}), poses {}, P from {} to {} in steps of {}, D {}, {} repeats, output {}'.format(
        args.joint, servo.p, servo.d, ', '.join(pose_degrees(pose) for pose in accepted),
        servo.p, round(servo.p * args.max_factor), args.factor, 'off' if args.no_d else fractions, args.repeats,
        trials_path))
    print('the arm moves to these poses and steps joint {} by up to 5 degrees there, nothing may be in its way; '
          'the other joints hold their position'.format(args.joint))
    if not args.yes and input('type yes to enable the servos and start: ').strip().lower() != 'yes':
        return 0
    if not robot.enable_servos():
        print('failed to enable the servos')
        return 1

    stop = StopFlag()
    stop.install()
    session: TuningSession | None = None
    stopped = False
    try:
        with open(samples_path, 'w', newline='') as samples_file, open(trials_path, 'w', newline='') as trials_file:
            sample_writer = csv.DictWriter(samples_file, fieldnames=SAMPLE_FIELDS)
            sample_writer.writeheader()
            trial_writer = csv.DictWriter(trials_file, fieldnames=TRIAL_FIELDS)
            trial_writer.writeheader()
            session = TuningSession(robot, args.joint, settings, sample_writer, trial_writer, stop=stop)
            session.run(accepted)
    except Stop as e:
        print('\nstopped: {}'.format(e))
        stopped = True
    except KeyboardInterrupt:
        print('\nkilled')
        stopped = True
    finally:
        if session is not None:
            stop.protected = True
            if session.restore():
                print('the yaml gains, profile and goal pwm of servo {} are back'.format(args.joint))
            else:
                print('RESTORING SERVO {} FAILED: start the robot again for the gains and the profile, reboot or '
                      'power cycle the servos for the goal pwm'.format(args.joint))
            stop.protected = False
        stop.uninstall()
    results = session.results if session is not None else []
    if results:
        print_summary(args.joint, results, servo.p, servo.d, stopped)
    if session is not None and session.read_failures:
        print('{} reads of the servo failed and were retried'.format(session.read_failures))
    print('samples in {}, trials in {}'.format(samples_path, trials_path))
    if args.plot and results:
        plot_results(args.joint, results, samples_path)
    record_static_load.disable_torque_on_request(robot)
    return 0


if __name__ == '__main__':
    sys.exit(main())
