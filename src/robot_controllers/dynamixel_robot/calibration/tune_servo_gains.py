"""
Tunes the position P and D gains of one arm servo. Run it from the project root with the arm powered, its torque off,
nothing in its way and a hand near the power switch; one joint per run, the wrist first (6, 4, 5), then 3, 1 and 2:
    python -m src.robot_controllers.dynamixel_robot.calibration.tune_servo_gains --joint 6

The P and D gains are tuned together, because damping raises the P at which a loop starts to overshoot: for every D
of a grid (0, then --d-start and steps of --d-factor, plus the yaml D) the script sweeps P at every pose. At every
pose (--pose, a few named poses per joint by default) it moves there with the planner at the yaml gains, records a
baseline at rest, and then runs trials at increasing P (--factor steps up to --max-factor times the yaml value, from
the yaml value at the first D and from the highest P that passed at the previous D after that; down from the start
when the start itself fails). A trial clamps the goal pwm of the servo
under test (a multiple of the pwm it needs to hold the pose, the only torque limit the XC430 wrist servos have; the
full 885 on joints 1, 4 and 6, which carry no static load), writes the gains to that servo only, records the rest
(ripple of the position and the pwm), steps the goal a few positions up and down with the profile of the servo
switched off (the response of the position loop itself: overshoot, rings, settle time, steady state error per
direction; a step is written with ServoHandler.step_servo_to_pos, which is a step in the time-based profile as well),
and streams a short planned move with the production profile (the tracking of the trajectory). The step
is sized so the pwm stays below the clamp, and capped per joint (--max-step: 15 positions on joints 1 and 2, which
swing the whole arm, 60 on the others). A trial that overshoots, rings, doesn't settle, or raises the ripple at
rest is the knee of the sweep; the recommended P is 0.65 times the knee. The pwm noise at rest is reported as a
warning only: it follows where the joint parks relative to an encoder edge rather than the gain (joint 3,
13 November 2026), though a D that adds pwm noise is still held back when the candidates are compared. The smallest
recommendation of the poses is the P of the joint at that D. D is raised until --d-patience D gains in a row are no
progress: a D gain is progress when it raises the P of the joint above the best before it, or, at the same P, when
the overshoot of the trial at the knee of the limiting pose (the pose with the smallest recommendation) fell clearly
(by the larger of a position and 10 %) below the lowest at the same pose and knee P since that P was reached; P moves
in steps of 1.3, so D can damp the loop steadily without moving the knee a step (joint 4, 18 November 2026: knee P
1010 at D 250, 400 and 640, overshoot 11.7, 10.0 and 8.7 positions). Otherwise the P of the joint stays, falls or is
missing, and too much D shows as pwm noise, a slow settle, or on the belt joints as oscillation; the grid stops at
--d-max, 2000 by default (the largest D in servo_config.yaml is 550), and --d-max raises it up to 16383. A start that
fails only on its settle time (or doesn't move) is sluggish rather than too stiff, and lowering P would only make it
more so (joint 2 at the low pose oscillated in its backlash below P 1000 at D 0): the sweep doesn't go down then. The D with the highest P of the joint wins. D
gains that share it (their knees lie in the same step of the sweep, so their margins can't be told apart) are
compared at that P at every pose, and a larger D than the smallest is chosen only when it lowers the overshoot or
the resting error after the steps (the friction band the joint lands in) by a position or removes a ring, without
adding pwm noise. The recommendation is that D with the P of the joint at it. The other joints hold their position
at their yaml gains throughout.

A run that stops keeps the D gains swept at every pose so far: the grid goes D by D, every pose at a D before the
next D, so the summary can still recommend from them.

The feedforward stage (--ff after the P and D tuning, --ff-only for the stage alone) measures and verifies the
velocity lead of a joint with the time-based profile (a servo with a move_time in servo_config.yaml) at its yaml
gains, at the first pose only. The servos run a P-only position loop, pwm = P / 128 * (goal - present), and a turning
motor needs a drive in proportion to its speed (its back voltage), so a moving joint trails its trajectory by an
error that grows with the speed. The PC sends every goal ahead of the planned path by exactly that error (the
`velocity_feedforward` block of a servo in servo_config.yaml, VelocityFeedforward: lead in positions = coefficient *
velocity * 128 / P, the coefficient in pwm units per degree/s of the joint).
The joint sweeps +-15 degrees (--ff-degrees) at three speeds (--ff-speeds, degrees per second, 10 20 30), in
legs that ramp up over 0.2 s, cruise and ramp down; joints 1 and 2 sit behind 3:1 belts and sweep +-10 degrees at 5 10
15 by default and are refused above 15 degrees/s. The rounds:
  1. without a lead on the joint (the `velocity_feedforward` of its servo is set aside): the drive and the lag of the
     cruise parts against the speed give the coefficient the joint needs, by two routes (the measured pwm, and the lag
     times P / 128; their ratio is 1 when the factor 128 holds);
  2. with a lead. The coefficient is --ff-lead when given, else the one of the `velocity_feedforward` block of the
     servo in the yaml, else the one fitted in round 1 (from the lag); the clamp is the max_positions of the yaml
     `velocity_feedforward`, else --ff-max-positions (30). The lag behind the INTENDED trajectory (the trajectory of
     the servo less the lead the PC added) should now be the same at every speed, only the friction part is left;
  3. when it still changes by more than a position over the speeds, the coefficient is corrected by the slope that is
     left (times P / 128) and measured once more (never a second correction). Not when the lead was at its clamp at a
     speed: no coefficient flattens the lag then, the report says to raise the clamp and recommends nothing.
The report has per round the table of fit_velocity_lead.py (per speed the cruise samples, pwm, the error behind the
trajectory of the servo, the lead, the lag behind the intended trajectory and the std of the error and of the pwm), the
two fits of round 1, the change of the intended lag over the speeds per round, whether a lead made the joint rougher
(a std of the error more than a position, or of the pwm more than the drive of a position, above round 1), and the
`velocity_feedforward` block for servo_config.yaml: the coefficient of the last round when its lag is flat, else it
says that it is not and gives the corrected value. The numbers are those of fit_velocity_lead.py, which the stage
runs on the very rows it writes to tune_joint<N>_<time>_ff.csv (joint, pose, speed, time, direction of the
position, cruise, goal as written with the lead in it, position, position_trajectory, velocity, velocity_trajectory,
pwm, current, voltage, temperature, and the P of the servo (p), the lead the PC added to the goal in positions (lead)
and the coefficient in force (lead_gain, 0 without a lead)). Running that script on the csv gives the
same numbers. The servo has its own `velocity_feedforward` again afterwards, whatever ends the stage.

The swing stage (--swing after the other stages, --swing-only for the stage alone, not together with --ff-only; with
--ff and --swing the order is the step tests, the feedforward stage, the swing stage) looks for what the step tests
can't see: a joint that swings while it moves. On 20 November 2026 joint 2 (the shoulder, behind a 3:1 belt)
oscillated in its lag at the resonance of the arm, 6 to 7 Hz whatever the speed, and more in the direction in which
gravity does the work, while every step test passed; the swing depended strongly on P (about 2 positions at P 2500,
0.75 at P 1500 in the falling direction) and not on D. The joint sweeps --swing-degrees (10) either side of the first
pose at --swing-speed (10 degrees per second), two round trips, in legs that ramp up over 0.2 s, cruise and ramp
down, once per setting of the gains, sampling the servo as often as it can be read (about 750 Hz): the yaml gains,
the yaml P with every --swing-d, every --swing-p (default 0.8 and 0.6 times the yaml P) with the yaml D, and the
yaml gains again, so that the difference between the two runs at the yaml gains shows how much a repeat differs.
The lag behind the trajectory of every cruise leg of 0.6 s or more is resampled to 500 Hz, its trend is taken out and
the strongest component between 3 and 15 Hz of its spectrum is the swing: its amplitude in positions and its
frequency, the mean over the legs of a direction. The direction is that of the ANGLE of the joint, rising or falling.
The report has a table of the settings (swing, frequency and mean lag per direction), the spread between the two runs
at the yaml gains, which direction swings more, whether D made a difference beyond the spread (only with --swing-d),
and the recommendation: the largest P tried that swings at most --swing-limit positions (1) in both directions at the
yaml D (the worse of the runs when it was run twice), the yaml P is fine when that is the one, or, when none did,
that the belt or the mechanics should be checked before P is lowered further (a slack belt lowers the resonance and
raises the swing). The script recommends, it doesn't change servo_config.yaml. The samples of every setting go to
tune_joint<N>_<time>_swing.csv, with the direction of the angle (the direction of the _ff.csv is that of the
position) and the lead the PC added to the goal in positions (lead; the sweep runs with the velocity lead of the yaml
like every planned move, 0 without a `velocity_feedforward` block; the statistics of the stage don't use it, they are
of the band of 3 to 15 Hz of the lag behind the trajectory of the servo). The servo has its yaml gains again afterwards.

Safety: every sample is checked live (temperature, voltage drop, position jumps, divergence from the trajectory,
oscillation at rest, pwm stuck at the clamp) and a violation restores the yaml gains, the profile, the goal pwm and
the velocity lead of the servo (its own `velocity_feedforward`) at once, after which the run waits 1.5 s for the arm to
move back to the pose. Oscillation is the position going both up and down by more than max(6 positions, 3 times the
baseline ripple) within the last second, never counting the 0.5 s after a goal write; a creep in one direction is the
joint settling. Divergence, oscillation and a pwm stuck at the clamp count as the knee; a temperature, voltage or
position jump abort says nothing about the gains and ends the run. Ctrl+C stops after the current read and restores
the same; a second Ctrl+C kills the script, the trial gains and profile then stay in the servo until the robot is
started again, a goal pwm clamp until the servos are rebooted or power cycled (the script resets a leftover clamp when
it starts). Nothing is written to the eeprom. Every write is read back, the sync write of a gain isn't acknowledged by
the servo.

The run ends with the arm lowered to record_static_load.REST_POSE (where it lies on the table, a bit over 1 cm higher) at
0.3 rad/s and the torque switched off there, before the plot. The prompt to support the arm and press enter is the
fallback: when the servo under test doesn't have its yaml gains again, when a servo has a hardware error or the status
can't be read, when the planner refuses the move, and on Ctrl+C during it (the servos then hold where they are).

Output: a csv with every sample (<output-dir>/tune_joint<N>_<time>_samples.csv), a csv with the metrics of every
trial (..._trials.csv) and the recommended yaml line per pose and for the joint. Record the static load again
after changing a P gain: the compensation is fitted in positions and scales with 1/P.
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import signal
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import ExitStack
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
from src.robot_controllers.dynamixel_robot.calibration.fit_velocity_lead import (
    GroupKey, GroupResult, SpeedStatistics, analyse_group, fit_lines, intended_line, is_flat, lead_meaning,
    speed_statistics, table, yaml_block)
from src.robot_controllers.dynamixel_robot.calibration.gain_tuning import (
    FF_LAG_TOLERANCE, FF_ROUGHNESS, SWING_BAND, SWING_MIN_LEG, AbortLimits, Criteria, RestMetrics, Sample, StepMetrics,
    SweepStep, SwingPair, SwingSample, SwingStatistics, TrackingMetrics, abort_reason, choose_d, choose_p, d_ladder,
    environmental_abort, evaluate, ff_sweep, p_values, p_values_down, pwm_clamp, pwm_per_position,
    recommended_swing_p, rest_metrics, rougher_speeds, round_gain, step_metrics, step_size, swing_change,
    swing_changed, swing_legs, swing_spread, swing_statistics, tracking_metrics)
from src.robot_controllers.dynamixel_robot.dynamixel_robot_controller import ARM_IDS, DynamixelRobotController
from src.robot_controllers.dynamixel_robot.dynamixel_utils import DynamixelError
from src.robot_controllers.dynamixel_robot.velocity_feedforward import VelocityFeedforward
from src.robot_controllers.dynamixel_robot.servo import Servo
from src.utils.movement import PlannedMove, check_move, plan_move_to_angles, step_velocities
from src.utils.movement_exception import MovementException

logger = logging.getLogger(__name__)

# named poses in degrees of joints 1 to 6 (joint 1: 90 is straight ahead; joint 2: 0 is the lower arm horizontal,
# 90 vertical; joint 2 + 3 = 90 is the upper arm horizontal, 0 is hanging down). The workspace limits keep the wrist
# within 41.8 cm of the shoulder, the elbow bent at least about 11 degrees; the poses below keep 8 degrees of margin
# on every joint for the test moves ('reach' has its elbow bent 25 degrees, 17 with joint 3 moved towards straight).
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
                'max_ring_count', 'mean_abs_error',
                'max_settle_time', 'min_rise_time', 'error_up', 'error_down', 'pwm_per_position', 'track_peak',
                'track_rms', 'track_final', 'following_error', 'flags', 'warnings', 'aborted', 'temperature_start',
                'temperature_end', 'voltage_min', 'sample_period')
# p is the P gain of the servo, lead the positions the velocity lead added to the goal (signed like the position, the
# goal column has it in it), lead_gain the coefficient of the lead in force in pwm per degree/s, 0 for none: the
# columns fit_velocity_lead.py reads
FF_SAMPLE_FIELDS = ('joint', 'pose', 'speed', 'time', 'direction', 'cruise', 'goal', 'position',
                    'position_trajectory', 'velocity', 'velocity_trajectory', 'pwm', 'current', 'voltage',
                    'temperature', 'p', 'lead', 'lead_gain')
# setting is the number of the setting of the stage in the order they ran, direction is that of the ANGLE of the joint
# (1 rising, -1 falling; the direction of the _ff.csv is that of the position, which is the other way round for a servo
# mounted reversed), speed the degrees per second of the cruise, lead the positions the velocity lead added to the goal
SWING_SAMPLE_FIELDS = ('joint', 'pose', 'setting', 'p', 'd', 'speed', 'time', 'direction', 'cruise', 'leg', 'goal',
                       'position', 'position_trajectory', 'velocity', 'pwm', 'current', 'voltage', 'temperature',
                       'lead')
STEP_SEQUENCE = ((1, 1), (-1, 0), (-1, -1), (1, 0))  # (direction, offset in steps from the pose) of a repeat
MOVE_HOLD = 1.0  # seconds of hold after a streamed move
GAIN_SETTLE = 0.3  # seconds between a gain write and the first measurement
SAMPLES_PER_ABORT_CHECK = 5
ABORT_SETTLE = 1.5  # seconds after an abort before the run goes on, the arm is still moving back to the pose
READ_ATTEMPTS = 3  # reads of the servo before a failure stops the run
# the base yaw and the two wrist rolls carry no static load: a clamp of 3 times their holding pwm would sit at the
# floor of pwm_clamp and starve the steps, so they run at the full MAX_PWM unless --pwm-clamp is given
UNLOADED_JOINTS = (1, 4, 6)
# the largest step of the trials in positions per joint. The base and the shoulder swing the whole arm: a 60 position
# step with the profile off shakes the structure (joint 1, 17 November 2026). A smaller step is a gentler but
# coarser test: the overshoot criterion only counts above 2 positions.
MAX_STEP: dict[int, int] = {1: 15, 2: 15}
DEFAULT_MAX_STEP = 60  # the joints not in MAX_STEP
MIN_STEP = 8  # the smallest step of step_size, a cap below it would not be kept
# the flags of a trial that is too slow rather than too stiff: the P sweep doesn't go down from a start that fails
# on these alone
SLUGGISH_FLAGS = frozenset({'settle', 'stuck'})
# the default largest D of the grid (0, 250, 400, 640, 1020, 1640): the largest D in servo_config.yaml is 550, the pwm
# kick per encoder flip grows with D, and joint 2 gained nothing from D 250 to 2620 on 25 October 2026
DEFAULT_D_MAX = 2000
# a D gain that doesn't raise the P of the joint still helps when the overshoot of the trial at the knee falls by this
# many positions or this fraction of the lowest so far, whichever is more: P moves in steps of 1.3, so the knee can
# stay in a step while D damps the loop (joint 4, 18 November 2026: knee P 1010 at D 250, 400 and 640, overshoot 11.7,
# 10.0 and 8.7 positions)
KNEE_OVERSHOOT_DROP = 1.0
KNEE_OVERSHOOT_DROP_FRACTION = 0.10
P_SWEEP_PHASES = ('p', 'p_down')  # the phases of the trials of a P sweep, as opposed to 'd' of the comparison
# (pose number, knee P, max overshoot of the trial at the knee) of the pose that limits the joint at a D
Knee = tuple[int, int, float]
DEFAULT_FF_DEGREES = 15.0  # the sweep of the feedforward stage goes this far either side of the pose
DEFAULT_FF_SPEEDS = (10.0, 20.0, 30.0)  # degrees per second of the cruise of the sweeps
# joints 1 and 2 sit behind 3:1 belts and are not moved fast: smaller and slower sweeps, as run on 18 November 2026
BELT_FF_DEGREES = 10.0
BELT_FF_SPEEDS = (5.0, 10.0, 15.0)
BELT_JOINTS = (1, 2)
# the fastest sweep of the stage per joint in degrees per second, as TOP_SPEEDS of record_load_sweep.py is for its own
FF_TOP_SPEEDS: dict[int, float] = {joint: max(BELT_FF_SPEEDS) for joint in BELT_JOINTS}
DEFAULT_FF_MAX_POSITIONS = 30  # the clamp of the lead of a servo without a `velocity_feedforward` block, in positions
LEAD_COEFFICIENT_STEP = 0.005  # pwm per degree/s: a correction smaller than this is not measured again
MIN_FF_SPEEDS = 3  # the line of the lag against the speed needs more than two points to be a line
FF_REST = 0.5  # seconds at rest after a sweep before the next one
DEFAULT_SWING_DEGREES = 10.0  # the sweep of the swing stage goes this far either side of the pose
DEFAULT_SWING_SPEED = 10.0  # degrees per second of its cruise
DEFAULT_SWING_P_FACTORS = (0.8, 0.6)  # times the yaml P, the P gains the stage tries besides it
DEFAULT_SWING_LIMIT = 1.0  # positions, the swing a P may have in either direction to be recommended
SWING_REST = FF_REST  # seconds at rest after a sweep of the stage before the next one
SWING_ROUND_TRIPS = 2  # how often a sweep goes to both ends of its range


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
    tune_d: bool = True  # False sweeps P at the yaml D only
    d_start: int = 250  # the first D above 0 of the grid
    d_factor: float = 1.6  # between the D gains of the grid
    d_max: int = DEFAULT_D_MAX
    d_values: tuple[int, ...] | None = None  # the D gains of the grid instead of the ladder, in this order
    d_patience: int = 2  # D gains in a row that neither raise the P of the joint nor lower the knee overshoot end it
    repeats: int = 2
    hold: float = 0.8  # seconds after every step
    rest: float = 3.0  # seconds of rest per trial
    baseline_rest: float = 5.0  # seconds of rest at the yaml gains per pose
    move_degrees: float = 8.0  # the streamed move, 0 disables it
    speed: float = 0.5  # rad/s of the planned moves
    clamp: int | None = None  # goal pwm clamp, None computes it from the holding pwm
    max_step: int | None = None  # the largest step of the trials in positions, None takes MAX_STEP of the joint
    ff_degrees: float | None = None  # the feedforward sweep goes this far either side of the pose, None: per joint
    ff_speeds: tuple[float, ...] | None = None  # degrees per second of its cruise, None: per joint
    ff_lead: float | None = None  # coefficient of the lead of the second round in pwm per degree/s, None: see the stage
    # the clamp of the lead when the servo has no `velocity_feedforward` block
    ff_max_positions: int = DEFAULT_FF_MAX_POSITIONS
    swing_degrees: float = DEFAULT_SWING_DEGREES  # the swing sweep goes this far either side of the pose
    swing_speed: float = DEFAULT_SWING_SPEED  # degrees per second of its cruise
    swing_p: tuple[int, ...] | None = None  # P gains besides the yaml one, None: DEFAULT_SWING_P_FACTORS times it
    swing_d: tuple[int, ...] = ()  # D gains at the yaml P besides the yaml one
    swing_limit: float = DEFAULT_SWING_LIMIT  # positions, the swing a P may have in either direction
    criteria: Criteria = Criteria()
    limits: AbortLimits = AbortLimits()

    def ff_degrees_of(self, joint: int) -> float:
        """the degrees either side of the pose the feedforward sweep of the joint goes: --ff-degrees when given, else
        the default of the joint (smaller on the belt joints 1 and 2)"""
        if self.ff_degrees is not None:
            return self.ff_degrees
        return BELT_FF_DEGREES if joint in BELT_JOINTS else DEFAULT_FF_DEGREES

    def ff_speeds_of(self, joint: int) -> tuple[float, ...]:
        """the cruise speeds in degrees per second of the feedforward sweeps of the joint: --ff-speeds when given, else
        the default of the joint (slower on the belt joints 1 and 2)"""
        if self.ff_speeds is not None:
            return self.ff_speeds
        return BELT_FF_SPEEDS if joint in BELT_JOINTS else DEFAULT_FF_SPEEDS


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
    def sluggish(self) -> bool:
        """failed only for being slow: a step that didn't settle in time, or nothing moved. A lower P makes that
        worse, it is not a sign of too much gain"""
        return self.aborted is None and bool(self.flags) and set(self.flags) <= SLUGGISH_FLAGS

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

    @property
    def mean_abs_error(self) -> float:
        """the mean of |final - goal| of the steps: where in its friction band the joint comes to rest"""
        return float(np.mean([abs(m.error) for m in self.steps])) if self.steps else 0.0


class Sweep(NamedTuple):
    """the P sweep at one D at one pose"""
    knee_p: int | None  # the first P that failed, None when every P passed
    recommended_p: int | None  # None when nothing passed below the knee
    highest_passed: int | None


class PoseResult(NamedTuple):
    number: int  # the number of the pose in the run, as in the csv files
    pose: np.ndarray
    baseline: RestMetrics  # at the first arrival
    trials: list[TrialResult]  # every trial at the pose, in the order they ran
    sweeps: dict[int, Sweep]  # the P sweep per D, in the order the D gains ran
    recommended_d: int | None  # of the comparison of the D candidates at this pose
    d_tested: bool  # the D candidates were compared at this pose


class FeedforwardRound(NamedTuple):
    """the sweeps at every speed with one velocity lead, analysed by fit_velocity_lead.py on the rows of the csv"""
    lead_gain: float  # the coefficient of the lead in pwm per degree/s, 0 for a round without a lead
    max_positions: int  # the clamp of the lead in positions, 0 for a round without a lead
    origin: str  # where the coefficient came from, empty for the round without a lead
    analysis: GroupResult

    @property
    def clamped(self) -> bool:
        """whether the lead reached its clamp (within half a position) at some speed: the lag can't be flat there,
        whatever the coefficient"""
        return self.max_positions > 0 and any(abs(s.lead) >= self.max_positions - 0.5
                                              for s in self.analysis.statistics)


class FeedforwardResult(NamedTuple):
    """the rounds of the feedforward stage in the order they ran: without a lead, with a lead, and with the corrected
    coefficient when the lag behind the intended trajectory was not flat in the second. No lead round when the lag of
    the first does not grow with the speed and no coefficient was given."""
    speeds: tuple[float, ...]  # degrees per second
    rounds: tuple[FeedforwardRound, ...]

    @property
    def lead_rounds(self) -> tuple[FeedforwardRound, ...]:
        return self.rounds[1:]

    @property
    def flat(self) -> bool:
        """whether the lag behind the intended trajectory of the last round with a lead is flat over the speeds"""
        fit = self.lead_rounds[-1].analysis.fit if self.lead_rounds else None
        return fit is not None and is_flat(fit)

    @property
    def corrected(self) -> float | None:
        """
        the coefficient that the slope of the lag behind the intended trajectory in the last round with a lead says
        would have made it flat (the one in force plus what the lead lacks, not below 0, to 2 decimals), None without
        a round with a lead
        """
        return corrected_coefficient(self.lead_rounds[-1]) if self.lead_rounds else None

    @property
    def recommended(self) -> float | None:
        """the coefficient for the yaml: the one of the last round with a lead when its lag is flat, else the
        corrected one; None without a round with a lead, and when the lead was at its clamp in the last round
        without being flat (no coefficient can say anything then, the clamp has to be raised)"""
        if not self.lead_rounds:
            return None
        last = self.lead_rounds[-1]
        if self.flat:
            return last.lead_gain
        return None if last.clamped else self.corrected


class SwingRun(NamedTuple):
    """the sweep at one setting of the gains"""
    p: int
    d: int
    statistics: SwingStatistics


class SwingResult(NamedTuple):
    """the runs of the swing stage in the order they ran: the yaml gains first and, repeated, last"""
    runs: tuple[SwingRun, ...]
    limit: float  # positions, the swing a recommended P may have in either direction


def swing_gains(settings: Settings, yaml_p: int, yaml_d: int) -> list[tuple[int, int]]:
    """
    the (P, D) settings of the swing stage in the order they run: the yaml gains; the yaml P with every extra D; every
    extra P (swing_p, by default DEFAULT_SWING_P_FACTORS times the yaml P) with the yaml D; the yaml gains again. A
    value that is the yaml one or comes twice is run once.
    """
    extra_d = [d for d in dict.fromkeys(settings.swing_d) if d != yaml_d]
    wanted = list(settings.swing_p) if settings.swing_p is not None \
        else [round_gain(factor * yaml_p) for factor in DEFAULT_SWING_P_FACTORS]
    extra_p = [p for p in dict.fromkeys(wanted) if p != yaml_p and 0 < p <= cfg.MAX_GAIN]
    return [(yaml_p, yaml_d)] + [(yaml_p, d) for d in extra_d] + [(p, yaml_d) for p in extra_p] + [(yaml_p, yaml_d)]


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


def angles_of_sweep(pose: np.ndarray, joint: int, steps: Sequence[SweepStep]) -> np.ndarray:
    """the angles of the robot at every step of a sweep: only the joint under test moves, the other joints keep the
    angles of the pose"""
    angles = np.tile(pose, (len(steps), 1))
    angles[:, joint] += np.radians([step.offset for step in steps])
    return angles


def sweep_angles(pose: np.ndarray, joint: int, speed: float, settings: Settings) -> tuple[list[SweepStep], np.ndarray]:
    """the steps of the sweep of the feedforward stage of the joint at the speed (degrees per second) and the angles
    of every step"""
    steps = ff_sweep(settings.ff_degrees_of(joint), speed, 1.0 / global_constants.steps_per_second)
    return steps, angles_of_sweep(pose, joint, steps)


def swing_sweep_angles(pose: np.ndarray, joint: int, settings: Settings) -> tuple[list[SweepStep], np.ndarray]:
    """the steps of the sweep of the swing stage and the angles of every step: two round trips of
    settings.swing_degrees either side of the pose at settings.swing_speed degrees per second"""
    steps = ff_sweep(settings.swing_degrees, settings.swing_speed, 1.0 / global_constants.steps_per_second,
                     round_trips=SWING_ROUND_TRIPS)
    return steps, angles_of_sweep(pose, joint, steps)


class StreamedSample(NamedTuple):
    """a sample of the servo read while a sweep was streamed, with the step and the goal the host had written"""
    sample: Sample
    step: SweepStep  # its direction is the direction of the ANGLE, see position_sign
    goal: int  # the goal position of the servo as the host wrote it, with the compensation in it
    compensation: int  # the positions the static compensation of the servo added to that goal
    lead: int = 0  # the positions the velocity lead of the servo added to that goal, signed like the position


def stream_sweep(robot: DynamixelRobotController | CombinedRobot, servo: Servo, steps: Sequence[SweepStep],
                 angles: np.ndarray, read: Callable[[int, float], Sample], check: Callable[[float], str | None],
                 recorded: list[StreamedSample], clock: Callable[[], float] = time.perf_counter,
                 sleep: Callable[[float], None] = time.sleep) -> str | None:
    """
    stream the steps of a sweep to the robot against the clock, as execute_move does, and read the servo as often as
    fits between two writes: while the loop keeps up, the next step is written at its time on the clock (the time of
    every step follows from the start, not from the previous write) and the servo is read again only when a read
    still fits before that. When a step is late (a stall of the PC, a read that took too long) the next ones do not
    hurry to catch up, the loop carries on from there with one period between the steps, so the sweep ends later and
    no step is skipped. The loop knows nothing about the tuning or the recording: what a read does and what ends the
    sweep early are the callables. The velocity of every step (movement.step_velocities, zero at the first and the last
    step) goes to the robot with it, so the servos run with the velocity lead of servo_config.yaml as they do in every
    planned move; a servo without a `velocity_feedforward` block is not affected.
    :param servo: the servo under test, its target position, compensation and lead are noted with every sample
    :param steps: the steps of the sweep, made for a period of 1 / global_constants.steps_per_second
    :param angles: the angles of the robot at every step, as sweep_angles makes them
    :param read: read(goal, written) reads the servo and returns the sample; goal is the goal position the step
        wrote and written the time on the clock when it was written. It may raise to stop the sweep
    :param check: check(written) is called after every sample and returns the reason to stop the sweep, or None
    :param recorded: filled with a StreamedSample per sample as they come, so what was read is kept when a read
        raises
    :return: the reason check gave when it stopped the sweep, None when every step was streamed
    """
    dt = 1.0 / global_constants.steps_per_second
    velocities = step_velocities(angles, dt)
    next_step = clock()
    for step, step_angles, step_velocity in zip(steps, angles, velocities):
        robot.move_servos(step_angles, step_velocity)
        written = clock()
        goal = servo.target_position
        compensation = servo.compensation_positions
        lead = servo.lead_positions
        next_step += dt
        while True:
            began = clock()
            sample = read(goal, written)
            now = clock()
            recorded.append(StreamedSample(sample, step, goal, compensation, lead))
            reason = check(written)
            if reason is not None:
                return reason
            # another sample only when a read still fits before the next write is due
            if now + (now - began) >= next_step:
                break
        remaining = next_step - clock()
        if remaining > 1e-4:  # a shorter sleep is not worth it
            sleep(next_step - clock())
        elif remaining <= 0:
            # the next step is overdue: a stall must not become a burst of steps, the arm would kick and ring
            next_step = clock()
    return None


def position_sign(servo: Servo) -> int:
    """
    1 when the positions of the servo rise with its angle, -1 when they fall (a servo mounted reversed: servo_config
    has `positions: [3072, 1024]` for `angles: [-90, 90]`, which Servo keeps as min_position 3072 for min_angle). The
    lag and the pwm of the feedforward stage are in positions, so the direction of a sweep, which is one of the angle,
    has to be turned into the direction of the position before they are signed with it.
    """
    return 1 if servo.max_position > servo.min_position else -1


def feedforward_refusal(robot: DynamixelRobotController | CombinedRobot, joint: int, pose: np.ndarray,
                        settings: Settings) -> str | None:
    """
    why the feedforward stage can't run at the pose, or None; nothing moves. The servo needs the time-based profile, the
    speeds may not be above FF_TOP_SPEEDS of the joint (the belt joints 1 and 2), the pose with the joint moved by the
    sweep either way has to pass the pose check of the script, every sweep has to pass the planner's check_move (joint
    speed, range and workspace of every step), and has to cruise in both directions or there is nothing to measure.
    """
    servo = robot.servo_handler.get_servo(joint)
    if not servo.time_based:
        return 'servo {} has no move_time in servo_config.yaml, the feedforward stage needs the time-based profile' \
            .format(joint)
    degrees, speeds = settings.ff_degrees_of(joint), settings.ff_speeds_of(joint)
    top = FF_TOP_SPEEDS.get(joint)
    too_fast = [speed for speed in speeds if top is not None and speed > top]
    if top is not None and too_fast:
        return 'no feedforward stage, joint {} can not be swept faster than {:g} degrees/s (it moves behind a belt), ' \
               'got {}'.format(joint, top, ' '.join('{:g}'.format(speed) for speed in too_fast))
    refused = pose_is_accepted(pose, robot, joint, np.radians(degrees))
    if refused is not None:
        return 'no feedforward stage at pose {} degrees: {}'.format(pose_degrees(pose), refused)
    for speed in speeds:
        steps, angles = sweep_angles(pose, joint, speed, settings)
        try:
            check_move(PlannedMove('sweep at {} degrees/s'.format(speed), angles, steps[-1].time), robot)
        except MovementException as e:
            return 'no feedforward stage, the sweep at {} degrees/s is refused: {}'.format(speed, e)
        if {step.direction for step in steps if step.cruise} != {1, -1}:
            return 'no feedforward stage, a sweep of {:g} degrees either way at {:g} degrees/s has no cruise part ' \
                   'going both ways: use a larger --ff-degrees or lower --ff-speeds'.format(degrees, speed)
    return None


def corrected_coefficient(feedforward_round: FeedforwardRound) -> float:
    """
    the coefficient of the lead that would have made the lag behind the intended trajectory of the round flat: the one
    in force plus the slope the lag still has over the speeds times P / 128 (pwm per degree/s the lead lacks, negative
    when it was too large), not below 0, to 2 decimals as it goes into the yaml. The coefficient in force when the
    round has no fit.
    """
    fit = feedforward_round.analysis.fit
    leftover = fit.leftover_coefficient if fit is not None else None
    if leftover is None:
        return feedforward_round.lead_gain
    return round(max(0.0, feedforward_round.lead_gain + leftover), 2)


def lead_choice(settings: Settings, feedforward: VelocityFeedforward | None, fitted: float) -> tuple[float, int, str]:
    """
    the coefficient, the clamp and where the coefficient came from for the first round with a lead, in the order: the
    --ff-lead option, the `velocity_feedforward` block of the servo in the yaml, the coefficient fitted in the round
    without a lead (rounded to the 2 decimals of the yaml). The clamp is the one of the yaml `velocity_feedforward` when
    the servo has one, else --ff-max-positions.
    :param feedforward: the `velocity_feedforward` of the servo in the yaml, None for a servo without one
    :param fitted: the coefficient fitted from the lag in the round without a lead
    """
    max_positions = feedforward.max_positions if feedforward is not None else settings.ff_max_positions
    if settings.ff_lead is not None:
        return settings.ff_lead, max_positions, 'the --ff-lead option'
    if feedforward is not None:
        return (feedforward.pwm_per_degree_per_second, max_positions,
                'the velocity_feedforward block of servo_config.yaml')
    return round(fitted, 2), max_positions, 'the coefficient fitted from the lag in the round without a lead'


def ff_sample_row(joint: int, pose_index: int, speed: float, p: int, lead_gain: float, sign: int,
                  streamed: StreamedSample) -> dict[str, str]:
    """
    the csv row (FF_SAMPLE_FIELDS, every value as text, as it is in the file) of a sample of a feedforward sweep; the
    stage writes it to the csv and gives the very same rows to fit_velocity_lead.py for its numbers
    :param sign: position_sign of the servo, the direction in the csv is the one of the position
    """
    sample = streamed.sample
    values: dict[str, object] = {
        'joint': joint, 'pose': pose_index, 'speed': speed, 'time': '{:.4f}'.format(sample.time),
        'direction': streamed.step.direction * sign, 'cruise': int(streamed.step.cruise), 'goal': sample.goal,
        'position': sample.position, 'position_trajectory': sample.position_trajectory, 'velocity': sample.velocity,
        'velocity_trajectory': sample.velocity_trajectory, 'pwm': sample.pwm, 'current': sample.current,
        'voltage': sample.voltage, 'temperature': sample.temperature, 'p': p, 'lead': streamed.lead,
        'lead_gain': '{:g}'.format(lead_gain)}
    return {name: str(values[name]) for name in FF_SAMPLE_FIELDS}


def swing_refusal(robot: DynamixelRobotController | CombinedRobot, joint: int, pose: np.ndarray,
                  settings: Settings) -> str | None:
    """
    why the swing stage can't run at the pose, or None; nothing moves. The servo needs the time-based profile, the
    pose with the joint moved by the sweep either way has to pass the pose check of the script, the sweep has to
    pass the planner's check_move, and it has to have a cruise leg long enough for the analysis in both directions
    of the angle or there is nothing to measure.
    """
    servo = robot.servo_handler.get_servo(joint)
    if not servo.time_based:
        return 'servo {} has no move_time in servo_config.yaml, the swing stage needs the time-based profile' \
            .format(joint)
    refused = pose_is_accepted(pose, robot, joint, np.radians(settings.swing_degrees))
    if refused is not None:
        return 'no swing stage at pose {} degrees: {}'.format(pose_degrees(pose), refused)
    steps, angles = swing_sweep_angles(pose, joint, settings)
    try:
        check_move(PlannedMove('swing sweep at {:g} degrees/s'.format(settings.swing_speed), angles, steps[-1].time),
                   robot)
    except MovementException as e:
        return 'no swing stage, the sweep at {:g} degrees/s is refused: {}'.format(settings.swing_speed, e)
    legs = swing_legs(steps, 1.0 / global_constants.steps_per_second)
    if min(legs.values()) < 1:
        return 'no swing stage, a sweep of {:g} degrees either way at {:g} degrees/s has no cruise leg of at least ' \
               '{:g} s going both ways: use a larger --swing-degrees or a lower --swing-speed'.format(
                   settings.swing_degrees, settings.swing_speed, SWING_MIN_LEG)
    return None


def d_grid(settings: Settings, yaml_d: int) -> list[int]:
    """the D gains to sweep P at, in order: the ladder with the yaml D added, or settings.d_values as given; only the
    yaml D without tune_d"""
    if not settings.tune_d:
        return [yaml_d]
    if settings.d_values is not None:
        return list(dict.fromkeys(settings.d_values))
    return sorted(set(d_ladder(settings.d_start, settings.d_factor, settings.d_max)) | {yaml_d})


def swept_d_values(results: Sequence[PoseResult]) -> list[int]:
    """the D gains P was swept at at every pose, in the order they ran"""
    if not results:
        return []
    return [d for d in results[0].sweeps if all(d in r.sweeps for r in results)]


def joint_p_at(results: Sequence[PoseResult], d: int) -> int | None:
    """the P gain of the joint at the D: the smallest recommendation of the poses, None when a pose has none or was
    not swept at the D"""
    recommended = [r.sweeps[d].recommended_p if d in r.sweeps else None for r in results]
    if not recommended or any(p is None for p in recommended):
        return None
    return min(p for p in recommended if p is not None)


def d_candidates(results: Sequence[PoseResult]) -> list[int]:
    """the D gains with the best P of the joint, smallest first. The sweeps step P by a factor, so D gains with the
    same P of the joint have their knee in the same step and can't be told apart by their margin; one step more is
    the resolution of the sweep and counts"""
    joint_ps = {d: joint_p_at(results, d) for d in swept_d_values(results)}
    passed = {d: p for d, p in joint_ps.items() if p is not None}
    if not passed:
        return []
    best = max(passed.values())
    # 2 % for the rounding of the gains to tens
    return sorted(d for d, p in passed.items() if p >= 0.98 * best)


def knee_trial(result: PoseResult, d: int) -> TrialResult | None:
    """the knee trial of the pose at the D: the failing trial of the P sweep (not of the comparison of the D
    candidates) at that D whose P is the knee P of the sweep, None when the sweep has no knee"""
    sweep = result.sweeps.get(d)
    if sweep is None or sweep.knee_p is None:
        return None
    return next((t for t in result.trials if t.phase in P_SWEEP_PHASES and t.d == d and t.p == sweep.knee_p
                 and not t.passed), None)


def joint_knee_at(results: Sequence[PoseResult], d: int) -> Knee | None:
    """
    the knee of the joint at the D: (pose number, knee P, max overshoot of the knee trial) of the limiting pose,
    the pose whose recommended P is the P of the joint (the one whose knee trial overshoots most when several tie,
    a pose without a knee trial after the others). None when the joint has no P at the D, the limiting pose has no
    knee, or its knee trial is not flagged for overshoot: a knee from ringing, settling or ripple alone says nothing
    about the overshoot
    """
    p = joint_p_at(results, d)
    if p is None:
        return None
    tied = [(r, knee_trial(r, d)) for r in results if r.sweeps[d].recommended_p == p]
    pose, trial = max(tied, key=lambda pair: pair[1].max_overshoot if pair[1] is not None else -1.0)
    if trial is None or 'overshoot' not in trial.flags:
        return None
    return pose.number, trial.p, trial.max_overshoot


class DProgress(NamedTuple):
    """what a D gain of the grid did: 'p' raised the P of the joint, 'overshoot' lowered the overshoot at the knee
    (from_overshoot is the lowest before it), None neither"""
    kind: str | None
    from_overshoot: float | None = None


def d_progress(joint_ps: Sequence[int | None], knees: Sequence[Knee | None] | None = None) -> list[DProgress]:
    """
    the progress of every D gain of the grid, in order. A D gain is progress when its P of the joint is above the best
    so far (which also starts a new record of the lowest knee overshoot per (pose number, knee P), beginning with its
    own knee). At the best P it is progress when its knee has the same pose and knee P as one recorded since that P
    was reached, and an overshoot at least KNEE_OVERSHOOT_DROP and KNEE_OVERSHOOT_DROP_FRACTION of the lowest
    recorded below it. Any other D gain is no progress, its knee is recorded when it is new or lower. Without knees
    only the P of the joint counts.
    """
    if knees is None:
        knees = [None] * len(joint_ps)
    if len(knees) != len(joint_ps):
        raise ValueError('{} knees for {} D gains'.format(len(knees), len(joint_ps)))
    best: int | None = None
    lowest: dict[tuple[int, int], float] = {}
    progress: list[DProgress] = []
    for p, knee in zip(joint_ps, knees):
        key = (knee[0], knee[1]) if knee is not None else None
        if p is not None and (best is None or p > best):
            best = p
            lowest = {} if key is None or knee is None else {key: knee[2]}
            progress.append(DProgress('p'))
            continue
        step: DProgress | None = None
        if knee is not None and key is not None:
            recorded = lowest.get(key)
            # the small tolerance keeps a drop of exactly the threshold from failing on float rounding
            if p is not None and p == best and recorded is not None and knee[2] <= recorded - max(
                    KNEE_OVERSHOOT_DROP, KNEE_OVERSHOOT_DROP_FRACTION * recorded) + 1e-9:
                step = DProgress('overshoot', recorded)
            if recorded is None or knee[2] < recorded:
                lowest[key] = knee[2]
        progress.append(step if step is not None else DProgress(None))
    return progress


def grid_is_done(joint_ps: Sequence[int | None], patience: int, knees: Sequence[Knee | None] | None = None) -> bool:
    """
    whether the last patience D gains of the grid were all no progress: they neither raised the P of the joint above
    the best before them (the same P, a lower one, or none) nor lowered the overshoot at its knee clearly, see
    d_progress. More D than that only adds pwm noise (joint 2 on 25 October: the same P of the joint from D 250 to
    2620). The P moves in steps of a factor, so D can help steadily without moving the knee a whole step (joint 4 on
    18 November: the knee stayed at P 1010 from D 250 to 640 while the overshoot at it fell from 11.7 to 8.7).
    :param knees: joint_knee_at of every D gain, in the order of joint_ps; None: no knees known
    """
    progress = d_progress(joint_ps, knees)
    without_gain = 0
    for step in progress:
        without_gain = 0 if step.kind is not None else without_gain + 1
    return without_gain >= patience


class TuningSession:
    """the trials of one joint on the robot, see the module docstring"""

    def __init__(self, robot: DynamixelRobotController | CombinedRobot, joint: int, settings: Settings,
                 sample_writer: csv.DictWriter[str] | None = None, trial_writer: csv.DictWriter[str] | None = None,
                 clock: Callable[[], float] = time.perf_counter, sleep: Callable[[float], None] = time.sleep,
                 stop: StopFlag | None = None, ff_writer: csv.DictWriter[str] | None = None,
                 swing_writer: csv.DictWriter[str] | None = None) -> None:
        self.robot = robot
        self.handler = robot.servo_handler
        self.joint = joint
        self.servo = self.handler.get_servo(joint)
        self.settings = settings
        self.max_step = settings.max_step if settings.max_step is not None else MAX_STEP.get(joint, DEFAULT_MAX_STEP)
        self.sample_writer = sample_writer
        self.trial_writer = trial_writer
        self.ff_writer = ff_writer
        self.swing_writer = swing_writer
        self.clock = clock
        self.sleep = sleep
        self.stop = stop if stop is not None else StopFlag()
        self.original_goal_pwm: int | None = None
        # the velocity lead the servo has from the yaml, which the feedforward stage sets aside for its first round
        self.original_velocity_feedforward: VelocityFeedforward | None = None
        self.velocity_feedforward_saved = False
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
        """note the velocity lead (`velocity_feedforward`) the servo has, which restore puts back, and check that the
        servo has the yaml gains and no goal pwm clamp left by an earlier run"""
        self.original_velocity_feedforward = self.servo.velocity_feedforward
        self.velocity_feedforward_saved = True
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

    def restore_velocity_feedforward(self) -> None:
        """put the velocity lead (`velocity_feedforward`) the servo had from the yaml back, whatever the feedforward
        stage left there; nothing before snapshot has noted it"""
        if self.velocity_feedforward_saved:
            self.servo.velocity_feedforward = self.original_velocity_feedforward

    def restore(self) -> bool:
        """put the yaml gains, the yaml profile (the time and the acceleration time of a servo with a move_time), the
        original goal pwm and the velocity lead of the servo back, returns False when a read-back doesn't match. The
        lead is only on the PC, so it goes back first and can't be lost to a failed write."""
        self.restore_velocity_feedforward()
        ok = self.handler.set_pid([self.joint])
        ok &= self.handler.set_profile(self.joint, *self.servo.profile_registers())
        if self.original_goal_pwm is not None:
            ok &= self.handler.set_goal_pwm(self.joint, self.original_goal_pwm)
        self.current_p, self.current_d = self.servo.p, self.servo.d
        gains = self.handler.read_pid([self.joint])
        profile = self.handler.read_profile([self.joint])
        goal_pwm = self.handler.read_goal_pwm([self.joint])
        if gains is None or profile is None or goal_pwm is None:
            logger.error('could not read servo %d back after restoring its settings', self.joint)
            return False
        expected = ((self.servo.p, self.servo.i, self.servo.d), self.servo.profile_registers(), self.original_goal_pwm)
        actual = (gains[self.joint], profile[self.joint],
                  goal_pwm[self.joint] if self.original_goal_pwm is not None else None)
        if actual != expected:
            logger.error('servo %d has gains, profile, goal pwm %s after the restore, expected %s',
                         self.joint, actual, expected)
            return False
        return ok

    def check_servos(self) -> None:
        """stop on a hardware error, a servo warmer than the warning, or the servo under test too warm to be
        allowed to warm up by a trial"""
        try:
            record_static_load.check_servos(self.robot, self.robot.servo_config.temperature_warning)
        except record_static_load.StopRecording as e:
            raise Stop(str(e))
        try:
            temperatures = record_static_load.read_status(lambda: self.handler.read_temperatures([self.joint]),
                                                          'the temperature of servo {}'.format(self.joint))
        except record_static_load.StopRecording as e:
            raise Stop(str(e))
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

    def write_goal(self, position: int, step: bool = False) -> int:
        """
        write a goal position to the servo under test, clamped to its calibrated range, returns what was sent
        :param step: a deliberate step, with the profile registers at 0 (the servo jumps to the goal), in either drive
            mode; otherwise the goal follows the profile of the servo, and with a time-based servo carries its time
        """
        position = self.servo.clamp_position(position)
        written = self.handler.step_servo_to_pos(self.joint, position) if step \
            else self.handler.move_servo_to_pos(self.joint, position)
        if not written:
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
            'mean_abs_error': '{:.1f}'.format(result.mean_abs_error) if result.steps else '',
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
        self.sleep(ABORT_SETTLE)
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
        elif self.joint in UNLOADED_JOINTS:
            clamp = cfg.MAX_PWM
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
            step, limited = step_size(p, holding, clamp, maximum=self.max_step)
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
                    goal = self.write_goal(self.pose_goal + offset * step, step=True)
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
                profile = self.servo.profile_registers()
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

    def run_sweep(self, pose: np.ndarray, d: int, start: int, baseline: RestMetrics
                  ) -> tuple[list[TrialResult], Sweep]:
        """the P sweep at the D: up from start until the first trial that fails, down from start when start itself
        fails for too much gain (not when it is only sluggish); never above max_factor times the yaml P"""
        settings = self.settings
        top = self.servo.p * settings.max_factor
        trials: list[TrialResult] = []
        for p in p_values(start, settings.factor, max(1.0, top / start)):
            result = self.run_trial(pose, 'p', p, d, baseline)
            self.record(trials, result)
            if not result.passed:
                break
        if trials[0].sluggish:
            print('  p={} is too slow rather than too stiff ({}), not sweeping down'.format(
                trials[0].p, ' '.join(trials[0].flags)))
        elif not trials[0].passed:
            for p in p_values_down(start, settings.factor):
                result = self.run_trial(pose, 'p_down', p, d, baseline)
                self.record(trials, result)
                if result.passed:
                    break
        recommended_p, knee_p = choose_p([(t.p, t.passed) for t in trials], settings.factor)
        passed = [t.p for t in trials if t.passed]
        print('  D {}: P knee {}, recommended {}'.format(d, knee_p, recommended_p))
        return trials, Sweep(knee_p, recommended_p, max(passed) if passed else None)

    def sweep_pose(self, index: int, pose: np.ndarray, d: int) -> bool:
        """the P sweep at the D at the pose, added to self.results; False when the planner refuses the pose"""
        print('pose {}: {} degrees, D {}'.format(index, pose_degrees(pose), d))
        baseline = self.arrive(index, pose)
        if baseline is None:
            return False
        position = next((i for i, r in enumerate(self.results) if r.number == index), None)
        previous = self.results[position] if position is not None else None
        # the P that passed at the previous D is where this D starts: more damping rarely lowers the knee, and when
        # it does the start fails and the sweep goes down from there
        start = self.servo.p
        if previous is not None and previous.sweeps:
            last = list(previous.sweeps.values())[-1]
            if last.highest_passed is not None:
                start = last.highest_passed
        trials, sweep = self.run_sweep(pose, d, start, baseline)
        if previous is None or position is None:
            self.results.append(PoseResult(index, pose, baseline, trials, {d: sweep}, None, False))
        else:
            self.results[position] = previous._replace(trials=previous.trials + trials,
                                                       sweeps={**previous.sweeps, d: sweep})
        return True

    def compare_d(self, candidates: Sequence[int], p: int) -> None:
        """the candidate D gains at one P at every pose, so their overshoot and resting error can be compared"""
        settings = self.settings
        for position, result in enumerate(self.results):
            print('pose {} again, D {} at p={}'.format(result.number, ', '.join(str(d) for d in candidates), p))
            baseline = self.arrive(result.number, result.pose)
            if baseline is None:
                continue
            trials: list[TrialResult] = []
            for d in candidates:
                trial = self.run_trial(result.pose, 'd', p, d, baseline)
                self.record(trials, trial)
            recommended_d = choose_d([(t.d, t.passed, t.max_overshoot, t.max_ring_count, t.rest.pwm_residual_std,
                                       t.mean_abs_error) for t in trials], settings.criteria.pwm_std_factor,
                                     settings.criteria.pwm_std_floor)
            print('  D: recommended {}'.format(recommended_d))
            self.results[position] = result._replace(trials=result.trials + trials, recommended_d=recommended_d,
                                                     d_tested=True)

    def run(self, poses: Sequence[np.ndarray]) -> list[PoseResult]:
        """the P sweep at every pose for every D of the grid, one D at a time, until the grid is done; then the
        candidates are compared at one P. The results so far stay in self.results when a Stop ends the run"""
        self.snapshot()
        settings = self.settings
        joint_ps: list[int | None] = []
        knees: list[Knee | None] = []
        for d in d_grid(settings, self.servo.d):
            if not self.results:
                # the first D: the poses the planner refuses are left out of the run
                for index, pose in enumerate(poses, start=1):
                    self.sweep_pose(index, pose, d)
                if not self.results:
                    return self.results
            else:
                refused = [r.number for r in list(self.results) if not self.sweep_pose(r.number, r.pose, d)]
                if refused:
                    print('the planner refused pose {} at D {}, the grid stops'.format(refused, d))
                    break
            joint_ps.append(joint_p_at(self.results, d))
            knees.append(joint_knee_at(self.results, d))
            print('D {}: P of the joint {}'.format(d, joint_ps[-1] if joint_ps[-1] is not None
                                                   else 'none, a pose has no passing P'))
            latest = d_progress(joint_ps, knees)[-1]
            latest_knee = knees[-1]
            if latest.kind == 'overshoot' and latest_knee is not None and latest.from_overshoot is not None:
                print('D {}: the overshoot at the knee (pose {}, p={}) fell from {:.1f} to {:.1f} positions, D still '
                      'helps'.format(d, latest_knee[0], latest_knee[1], latest.from_overshoot, latest_knee[2]))
            if grid_is_done(joint_ps, settings.d_patience, knees):
                print('the last {} D gains neither raised the P of the joint nor lowered the overshoot at its knee: '
                      'the grid stops'.format(settings.d_patience))
                break
        candidates = d_candidates(self.results)
        if len(candidates) > 1:
            p = min(p for p in (joint_p_at(self.results, d) for d in candidates) if p is not None)
            print('D {} share the best P of the joint, comparing them at p={}'.format(
                ', '.join(str(d) for d in candidates), p))
            self.compare_d(candidates, p)
        return self.results

    # the feedforward stage

    def write_ff_samples(self, rows: Sequence[dict[str, str]]) -> None:
        """write the rows of ff_sample_row to the csv of the stage, if it has one"""
        if self.ff_writer is not None:
            self.ff_writer.writerows(rows)

    def stream_stage_sweep(self, steps: Sequence[SweepStep], angles: np.ndarray, stage: str,
                           write: Callable[[list[StreamedSample]], None]) -> list[StreamedSample]:
        """
        stream the steps of a sweep against the clock, as move_test does, and sample the servo as often as fits
        between two writes. An abort rule for a streamed move restores the servo and stops the stage with a Stop.
        :param stage: the name of the stage, for the reason of the Stop
        :param write: writes what was sampled to the csv; it is called with the samples so far whatever ends the
            sweep (an abort, a failed read, Ctrl+C, an error)
        """
        clamp = self.original_goal_pwm if self.original_goal_pwm is not None else cfg.MAX_PWM
        rows: list[Sample] = []
        streamed: list[StreamedSample] = []
        self.trial_start = self.clock()
        self.last_goal_time = 0.0
        self.sag = 0.0  # the divergence rule of a streamed move allows 120 positions

        def read(goal: int, written: float) -> Sample:
            self.last_goal_time = written - self.trial_start
            return self.sample(rows, goal)

        def check(written: float) -> str | None:
            return abort_reason(rows, clamp, 0, 0.0, rows[0].voltage, self.last_goal_time, self.settings.limits,
                                sag=self.sag, streaming=True)
        reason = None
        try:
            reason = stream_sweep(self.robot, self.servo, steps, angles, read, check, streamed, self.clock,
                                  self.sleep)
        finally:
            write(streamed)
        if reason is not None:
            if not self.abort(reason):
                raise Stop('could not restore the settings of servo {} after the abort'.format(self.joint))
            raise Stop('the {} stage was aborted: {}'.format(stage, reason))
        return streamed

    def feedforward_sweep(self, pose: np.ndarray, speed: float, lead_gain: float) -> list[dict[str, str]]:
        """
        stream one sweep of the joint at the speed (degrees per second) against the clock, as move_test does, and
        sample the servo as often as fits between two writes. An abort rule for a streamed move restores the servo
        and stops the stage.
        :param lead_gain: the coefficient of the velocity lead the servo has now, only for the csv
        :return: the rows of ff_sample_row of every sample; they are written to the csv whatever ends the sweep
        """
        self.check_servos()
        steps, angles = sweep_angles(pose, self.joint, speed, self.settings)
        sign = position_sign(self.servo)
        rows: list[dict[str, str]] = []

        def write(streamed: list[StreamedSample]) -> None:
            rows[:] = [ff_sample_row(self.joint, self.pose_index, speed, self.servo.p, lead_gain, sign, s)
                       for s in streamed]
            self.write_ff_samples(rows)
        self.stream_stage_sweep(steps, angles, 'feedforward', write)
        return rows

    def measure_feedforward(self, pose: np.ndarray, feedforward: VelocityFeedforward | None, label: str,
                            origin: str = '') -> FeedforwardRound:
        """
        sweep at every speed with the velocity lead `feedforward` on the servo (None: without one, the servo's own
        velocity_feedforward is set aside; restore_velocity_feedforward puts it back) and analyse the rows with
        fit_velocity_lead
        :param label: what the round is called in the progress lines
        :param origin: where the coefficient came from, for the report
        :raises Stop: when a sweep has too few cruise samples or the rows can't be fitted
        """
        self.servo.velocity_feedforward = feedforward
        lead_gain = feedforward.pwm_per_degree_per_second if feedforward is not None else 0.0
        print('  {}:'.format(label))
        rows: list[dict[str, str]] = []
        for speed in self.settings.ff_speeds_of(self.joint):
            sweep = self.feedforward_sweep(pose, speed, lead_gain)
            rows.extend(sweep)
            statistics, notes = speed_statistics(sweep)
            if not statistics:
                raise Stop('the sweep at {:g} degrees/s has too few cruise samples: {}'.format(speed, ' '.join(notes)))
            s = statistics[0]
            print('    {:g} degrees/s: error {:.1f} positions, lead {:.1f}, lag behind the intended trajectory {:.1f}, '
                  'pwm {:.0f}, error std {:.1f}'.format(speed, s.error, s.lead, s.intended, s.pwm, s.error_std))
            self.sleep(FF_REST)
        analysis = analyse_group(GroupKey('', self.joint, lead_gain), rows)
        if analysis.fit is None:
            raise Stop('the sweeps of {} can not be fitted: {}'.format(label, analysis.problem))
        max_positions = feedforward.max_positions if feedforward is not None else 0
        return FeedforwardRound(lead_gain, max_positions, origin, analysis)

    def run_feedforward(self, pose: np.ndarray) -> FeedforwardResult | None:
        """
        the feedforward stage at the pose, see the module docstring; None when the stage can't run (the servo is not
        time-based, a sweep is refused, the planner refuses the pose). The servo has its own velocity
        lead (`velocity_feedforward`) again afterwards, whatever ends the stage.
        """
        settings = self.settings
        refused = feedforward_refusal(self.robot, self.joint, pose, settings)
        if refused is not None:
            print(refused)
            return None
        if self.original_goal_pwm is None:  # without the P and D stages
            self.snapshot()
        self.pose_index = 1
        self.check_servos()
        speeds = settings.ff_speeds_of(self.joint)
        print('feedforward stage at pose {} degrees: sweeps of {:g} degrees either way at {} degrees/s, yaml p={} '
              'velocity_feedforward={}'.format(pose_degrees(pose), settings.ff_degrees_of(self.joint),
                                               ' '.join('{:g}'.format(s) for s in speeds), self.servo.p,
                                               self.servo.velocity_feedforward))
        if not self.move_to_pose(pose):
            print('no feedforward stage: the planner refuses the move to the pose')
            return None
        self.pose_goal = self.servo.target_position
        self.sleep(settings.baseline_rest / 2)
        rounds: list[FeedforwardRound] = []
        try:
            rounds.append(self.measure_feedforward(pose, None, 'round 1, without a lead'))
            self.run_lead_rounds(pose, rounds)
        finally:
            self.restore_velocity_feedforward()
        if not self.restore():
            raise Stop('could not restore the settings of servo {} after the feedforward stage'.format(self.joint))
        result = FeedforwardResult(tuple(speeds), tuple(rounds))
        print_feedforward(self.joint, self.servo, result)
        return result

    def run_lead_rounds(self, pose: np.ndarray, rounds: list[FeedforwardRound]) -> None:
        """
        the rounds with a lead after the round without one, appended to rounds: one with the coefficient of lead_choice
        and, when the lag behind the intended trajectory still changes by more than FF_LAG_TOLERANCE over the speeds,
        one with the corrected coefficient (never a second correction). No round when the lag of the first does not
        grow with the speed and no coefficient was asked for.
        """
        settings = self.settings
        fit = rounds[0].analysis.fit
        if fit is None:
            return
        coefficient, max_positions, origin = lead_choice(settings, self.original_velocity_feedforward, fit.coefficient)
        if coefficient <= 0:
            print('the lag does not grow with the speed (the fit gives {:.2f} pwm per degree/s): no lead is '
                  'measured, --ff-lead tries one'.format(fit.coefficient))
            return
        print('lead of {:g} pwm per degree/s from {}, at most {} positions'.format(coefficient, origin, max_positions))
        rounds.append(self.measure_feedforward(
            pose, VelocityFeedforward(coefficient, max_positions),
            'round 2, with a lead of {:g} pwm per degree/s'.format(coefficient), origin))
        last = rounds[-1]
        last_fit = last.analysis.fit
        if last_fit is None or is_flat(last_fit):
            return
        if last.clamped:
            # the lead can't follow a larger coefficient, the lag stays as it is: another round would only chase it
            print('the lag behind the intended trajectory changes by {:+.2f} positions over the speeds (more than '
                  '{:g}) while the lead is at its clamp of {} positions: no correction is measured, raise '
                  '--ff-max-positions (or max_positions of the velocity_feedforward block) and run again'.format(
                      last_fit.intended_change, FF_LAG_TOLERANCE, last.max_positions))
            return
        corrected = corrected_coefficient(last)
        print('the lag behind the intended trajectory still changes by {:+.2f} positions over the speeds (more than '
              '{:g}): the coefficient should be {:g} instead of {:g}'.format(
                  last_fit.intended_change, FF_LAG_TOLERANCE, corrected, last.lead_gain))
        if corrected <= 0 or abs(corrected - last.lead_gain) < LEAD_COEFFICIENT_STEP:
            print('that is no change worth measuring again')
            return
        rounds.append(self.measure_feedforward(
            pose, VelocityFeedforward(corrected, max_positions),
            'round 3, with the corrected lead of {:g} pwm per degree/s'.format(corrected),
            'the correction of round 2'))

    # the swing stage

    def write_swing_samples(self, streamed: Sequence[StreamedSample], setting: int, p: int, d: int) -> None:
        """write the samples of a swing sweep to the csv of the stage, if it has one; the direction is that of the
        angle, the lead the positions the velocity lead added to the goal"""
        if self.swing_writer is None:
            return
        for streamed_sample in streamed:
            s = streamed_sample.sample
            step = streamed_sample.step
            self.swing_writer.writerow({'joint': self.joint, 'pose': self.pose_index, 'setting': setting, 'p': p,
                                        'd': d, 'speed': self.settings.swing_speed, 'time': '{:.4f}'.format(s.time),
                                        'direction': step.direction, 'cruise': int(step.cruise), 'leg': step.leg,
                                        'goal': s.goal, 'position': s.position,
                                        'position_trajectory': s.position_trajectory, 'velocity': s.velocity,
                                        'pwm': s.pwm, 'current': s.current, 'voltage': s.voltage,
                                        'temperature': s.temperature, 'lead': streamed_sample.lead})

    @staticmethod
    def swing_samples(streamed: Sequence[StreamedSample]) -> list[SwingSample]:
        """the samples of a swing sweep with the direction of the angle of the joint, as the sweep has it"""
        return [SwingSample(s.sample, s.step.direction, s.step.cruise, s.step.leg) for s in streamed]

    def swing_sweep(self, pose: np.ndarray, setting: int, p: int, d: int) -> list[SwingSample]:
        """stream one sweep of the joint at the swing speed, sampling as feedforward_sweep does; the samples are
        written to the csv whatever ends the sweep"""
        steps, angles = swing_sweep_angles(pose, self.joint, self.settings)
        streamed = self.stream_stage_sweep(
            steps, angles, 'swing', lambda rows: self.write_swing_samples(rows, setting, p, d))
        return self.swing_samples(streamed)

    def measure_swing(self, pose: np.ndarray, setting: int, p: int, d: int) -> SwingRun:
        """write the gains, sweep once and measure the swing"""
        self.check_servos()
        self.write_gains(p, d)
        self.sleep(GAIN_SETTLE)
        recorded = self.swing_sweep(pose, setting, p, d)
        try:
            statistics = swing_statistics(recorded, position_sign(self.servo))
        except ValueError as e:
            raise Stop('the sweep at p={} d={} has too few cruise samples to measure the swing: {}'.format(p, d, e))
        up, down = statistics.rising, statistics.falling
        print('  p={} d={}: swing {:.2f} positions at {:.1f} Hz with the angle rising, {:.2f} at {:.1f} Hz falling; '
              'mean lag {:+.1f} and {:+.1f} positions'.format(p, d, up.swing, up.frequency, down.swing,
                                                              down.frequency, up.lag, down.lag))
        self.sleep(SWING_REST)
        return SwingRun(p, d, statistics)

    def run_swing(self, pose: np.ndarray) -> SwingResult | None:
        """
        the swing stage at the pose, see the module docstring; None when the stage can't run (the servo is not
        time-based, the sweep is refused, the planner refuses the pose). The servo has its yaml gains again
        afterwards.
        """
        settings = self.settings
        refused = swing_refusal(self.robot, self.joint, pose, settings)
        if refused is not None:
            print(refused)
            return None
        if self.original_goal_pwm is None:  # without the P and D stages
            self.snapshot()
        self.pose_index = 1
        self.check_servos()
        gains = swing_gains(settings, self.servo.p, self.servo.d)
        print('swing stage at pose {} degrees: sweeps of {:g} degrees either way at {:g} degrees/s, yaml p={} d={}, '
              'settings (p, d) {}'.format(pose_degrees(pose), settings.swing_degrees, settings.swing_speed,
                                          self.servo.p, self.servo.d,
                                          ' '.join('({}, {})'.format(p, d) for p, d in gains)))
        if not self.move_to_pose(pose):
            print('no swing stage: the planner refuses the move to the pose')
            return None
        self.pose_goal = self.servo.target_position
        self.sleep(settings.baseline_rest / 2)
        runs = [self.measure_swing(pose, setting, p, d) for setting, (p, d) in enumerate(gains)]
        if not self.restore():
            raise Stop('could not restore the settings of servo {} after the swing stage'.format(self.joint))
        result = SwingResult(tuple(runs), settings.swing_limit)
        print_swing(self.joint, self.servo, result)
        return result


def compared_d(results: Sequence[PoseResult], criteria: Criteria = Criteria()) -> int | None:
    """the D of choose_d over the comparison of the candidates, judged per D on its worst pose: the largest
    overshoot, ring count, pwm noise and resting error, passed only when it passed at every pose; None when the
    candidates were not compared or none passed"""
    by_d: dict[int, list[TrialResult]] = {}
    for r in results:
        if r.d_tested:
            for t in r.trials:
                if t.phase == 'd':
                    by_d.setdefault(t.d, []).append(t)
    if not by_d:
        return None
    candidates = [(d, all(t.passed for t in trials), max(t.max_overshoot for t in trials),
                   max(t.max_ring_count for t in trials), max(t.rest.pwm_residual_std for t in trials),
                   max(t.mean_abs_error for t in trials))
                  for d, trials in by_d.items()]
    return choose_d(candidates, criteria.pwm_std_factor, criteria.pwm_std_floor)


def recommend(results: Sequence[PoseResult], criteria: Criteria = Criteria()) -> tuple[int | None, int | None]:
    """
    the gains for the joint: the D of the comparison of the candidates (compared_d), or without one the candidate
    with the highest P of the joint (the smallest D on a tie), and the P of the joint at that D; (None, None) when no
    D gave a passing P at every pose
    """
    candidates = d_candidates(results)
    if not candidates:
        return None, None
    d = compared_d(results, criteria)
    if d is None or d not in candidates:
        d = max(candidates, key=lambda c: (joint_p_at(results, c) or 0, -c))
    return joint_p_at(results, d), d


def print_summary(joint: int, results: Sequence[PoseResult], servo_p: int, servo_d: int,
                  stopped: bool = False) -> None:
    print('\nsummary of joint {} (yaml p={} d={}):'.format(joint, servo_p, servo_d))
    if stopped:
        print('  the run stopped early: only the D gains swept at every pose count, a D that was not tried may have '
              'given a higher P')
    for r in results:
        print('  pose {} ({} degrees):'.format(r.number, pose_degrees(r.pose)))
        for sweep_d, sweep in r.sweeps.items():
            knee = sweep.knee_p if sweep.knee_p is not None else 'none up to {}'.format(sweep.highest_passed)
            print('    D {:5d}: P knee {}, recommended {}'.format(sweep_d, knee, sweep.recommended_p))
        if r.d_tested:
            print('    the candidates compared at one P: D {}'.format(r.recommended_d))
        for t in r.trials:
            print('    {:7s} p={:5d} d={:4d} step {:3d}: overshoot {:5.1f} rings {} mean |error| {:4.1f} rest pp {} '
                  'pwm residual {:5.1f} slope {} {}{}'.format(t.phase, t.p, t.d, t.step, t.max_overshoot,
                                                              t.max_ring_count, t.mean_abs_error, t.rest.position_pp,
                                                              t.rest.pwm_residual_std,
                                         '?' if t.pwm_slope is None else '{:.1f}'.format(t.pwm_slope),
                                         t.aborted or ' '.join(t.flags) or 'passed',
                                         ' (warnings: {})'.format(' '.join(t.warnings)) if t.warnings else ''))
    swept = swept_d_values(results)
    if swept:
        print('  P of the joint per D (the smallest recommendation of the poses): {}'.format(', '.join(
            'D {} {}'.format(d, joint_p_at(results, d) or 'none') for d in swept)))
    p, d = recommend(results)
    if p is None or d is None:
        print('no recommendation: no D gave a passing P at every pose, look at the trials')
        return
    candidates = d_candidates(results)
    if len(candidates) > 1:
        print('  D {} share the best P of the joint'.format(', '.join(str(c) for c in candidates)))
        if compared_d(results) is not None:
            print('  their comparison at one P chose D {}'.format(d))
        elif any(r.d_tested for r in results):
            print('  none of them passed the comparison at every pose, D {} has the highest P of the joint (look at '
                  'the d trials)'.format(d))
        else:
            print('  they were not compared, the run stopped first: D {} has the highest P of the joint'.format(d))
    print('recommended for servo_config.yaml (record the static load again after changing P):')
    print('    pid: {{p: {}, i: 0, d: {}}}'.format(p, d))


def roughness_lines(servo: Servo, number: int, before: Sequence[SpeedStatistics],
                    after: Sequence[SpeedStatistics]) -> list[str]:
    """
    whether the lead of round `number` made the joint rougher than round 1 (`before`, without a lead): the std of the
    error behind the trajectory of the servo more than FF_ROUGHNESS positions above it at a speed, and the std of the
    pwm more than the drive of FF_ROUGHNESS positions at the P of the servo (P / 128 pwm units per position); only the
    speeds both rounds have are compared
    """
    earlier = {s.speed: s for s in before}
    later = {s.speed: s for s in after}
    shared = sorted(set(earlier) & set(later))
    pwm_limit = FF_ROUGHNESS * servo.p * cfg.PWM_PER_POSITION_PER_P
    lines = []
    for what, values, limit, unit in (('error', lambda s: s.error_std, FF_ROUGHNESS,
                                        'position' if FF_ROUGHNESS == 1 else 'positions'),
                                      ('pwm', lambda s: s.pwm_std, pwm_limit, 'pwm units')):
        rougher = rougher_speeds([values(earlier[speed]) for speed in shared],
                                 [values(later[speed]) for speed in shared], limit)
        if rougher:
            lines.append('  warning: in round {} the std of the {} grew by more than {:.3g} {} at {} degrees/s, the '
                         'lead makes the joint rougher'.format(number, what, limit, unit,
                                                               ', '.join('{:g}'.format(shared[i]) for i in rougher)))
        else:
            lines.append('  round {}: the std of the {} is at most {:.3g} {} above round 1 at every speed, the lead '
                         'does not make the joint rougher'.format(number, what, limit, unit))
    return lines


def print_feedforward(joint: int, servo: Servo, result: FeedforwardResult) -> None:
    """
    the report of the feedforward stage, with the tables, fits and texts of fit_velocity_lead.py: per round the table
    (cruise samples, pwm, the error behind the trajectory of the servo, the lead, the lag behind the intended
    trajectory, the stds) and the change of the intended lag over the speeds, for the round without a lead the two
    fits and their ratio, whether a lead made the joint rougher, and the `velocity_feedforward` block for
    servo_config.yaml
    """
    print('\nvelocity lead of joint {} (yaml p={} d={}), at the speeds {} degrees/s:'.format(
        joint, servo.p, servo.d, ' '.join('{:g}'.format(s) for s in result.speeds)))
    first = result.rounds[0]
    for number, r in enumerate(result.rounds, start=1):
        fit = r.analysis.fit
        if r.lead_gain:
            print('round {}, with a lead of {:g} pwm per degree/s from {}, at most {} positions:'.format(
                number, r.lead_gain, r.origin, r.max_positions))
        else:
            print('round {}, without a lead:'.format(number))
        print(table(r.analysis.statistics))
        for note in r.analysis.notes:
            print('  note: ' + note)
        if fit is None:
            print('  no fit: {}'.format(r.analysis.problem))
            continue
        if not r.lead_gain:
            print('\n'.join(fit_lines(fit)))
        print(intended_line(fit))
        if r.clamped:
            print('  warning: the lead is at its clamp of {} positions at the fast speeds, the lag can not be flat '
                  'there: raise max_positions'.format(r.max_positions))
        if number > 1:
            for line in roughness_lines(servo, number, first.analysis.statistics, r.analysis.statistics):
                print(line)
    if not result.lead_rounds:
        print('no lead to recommend: the lag does not grow with the speed, or no round with a lead ran')
        return
    last = result.lead_rounds[-1]
    last_fit = last.analysis.fit
    number = len(result.rounds)
    coefficient = result.recommended
    if last.clamped and not result.flat:
        print('no coefficient to recommend: the lead was at its clamp of {} positions in round {} and the lag behind '
              'the intended trajectory is not flat, raise --ff-max-positions (or max_positions of the '
              'velocity_feedforward block) and run again'.format(last.max_positions, number))
        return
    if coefficient is None or last_fit is None:
        print('no lead to recommend: the last round could not be fitted')
        return
    if result.flat:
        print('for servo_config.yaml, the coefficient of round {} (from {}): the lag behind the intended trajectory '
              'is flat over the speeds, within {:g} position:'.format(number, last.origin, FF_LAG_TOLERANCE))
    else:
        print('the lag behind the intended trajectory is NOT flat in round {}: it changes by {:+.2f} positions over '
              'the speeds, more than {:g}. With the slope that is left taken out the coefficient would be {:g} (not '
              'measured), for servo_config.yaml to try:'.format(number, last_fit.intended_change, FF_LAG_TOLERANCE,
                                                               coefficient))
    print(yaml_block(joint, coefficient, last.max_positions).rstrip('\n'))
    print('  ' + lead_meaning(coefficient, servo.p))


def print_swing(joint: int, servo: Servo, result: SwingResult) -> None:
    """the table of the swing stage, the spread of a repeat, the direction that swings more, what D did and the P
    that stays within the limit"""
    runs = result.runs
    print('\nswing of joint {} (yaml p={} d={}), the lag behind the trajectory between {:g} and {:g} Hz, the angle '
          'rising and falling:'.format(joint, servo.p, servo.d, *SWING_BAND))
    print('  {:>3s} {:>6s} {:>6s}  {:>8s} {:>6s} {:>6s} {:>5s}  {:>8s} {:>6s} {:>6s} {:>5s}'.format(
        'run', 'p', 'd', 'swing', 'Hz', 'lag', 'legs', 'swing', 'Hz', 'lag', 'legs'))
    for number, run in enumerate(runs):
        up, down = run.statistics.rising, run.statistics.falling
        print('  {:3d} {:6d} {:6d}  {:8.2f} {:6.1f} {:+6.1f} {:5d}  {:8.2f} {:6.1f} {:+6.1f} {:5d}'.format(
            number, run.p, run.d, up.swing, up.frequency, up.lag, up.legs, down.swing, down.frequency, down.lag,
            down.legs))
    yaml_runs = [run for run in runs if (run.p, run.d) == (servo.p, servo.d)]
    reference = [run.statistics for run in yaml_runs]
    spread = swing_spread(reference[0], reference[-1])
    if len(yaml_runs) > 1:
        print('  the two runs at the yaml gains differ by {:.2f} positions with the angle rising and {:.2f} falling: '
              'that is how much a repeat differs'.format(*spread))
    else:
        print('  the yaml gains were run once, the spread of a repeat is not known')
    mean = SwingPair(float(np.mean([r.rising.swing for r in reference])),
                     float(np.mean([r.falling.swing for r in reference])))
    if swing_changed(mean.falling - mean.rising, max(spread)):
        print('  at the yaml gains the joint swings more with the angle {}: {:.2f} against {:.2f} positions'.format(
            'falling' if mean.falling > mean.rising else 'rising', max(mean), min(mean)))
    else:
        print('  at the yaml gains the joint swings alike in both directions, {:.2f} rising and {:.2f} falling, within '
              'the spread'.format(*mean))
    d_runs = [run for run in runs if run.p == servo.p and run.d != servo.d]
    if d_runs:
        beyond = []
        for run in d_runs:
            change = swing_change(reference, run.statistics)
            beyond.append(swing_changed(change.rising, spread.rising) or swing_changed(change.falling, spread.falling))
            print('  d={}: the swing changes by {:+.2f} rising and {:+.2f} falling from the yaml D, {} the spread'
                  .format(run.d, change.rising, change.falling, 'beyond' if beyond[-1] else 'within'))
        print('  D made {} difference beyond the spread of a repeat ({:.2f} rising and {:.2f} falling)'.format(
            'a' if any(beyond) else 'no', *spread))
    recommended = recommended_swing_p([(r.p, r.d, r.statistics.rising.swing, r.statistics.falling.swing)
                                       for r in runs], servo.d, result.limit)
    if recommended is None:
        print('no P tried swings at most {:g} positions in both directions at d={}: check the belt or the mechanics '
              'before lowering P further (a slack belt lowers the resonance and raises the swing)'.format(
                  result.limit, servo.d))
    elif recommended == servo.p:
        print('the yaml P {} is fine: it swings at most {:g} positions in both directions at d={}'.format(
            servo.p, result.limit, servo.d))
    else:
        print('recommended for servo_config.yaml, the largest P tried that swings at most {:g} positions in both '
              'directions at d={} (record the static load again after changing P):'.format(result.limit, servo.d))
        print('    pid: {{p: {}, i: {}, d: {}}}'.format(recommended, servo.i, servo.d))


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


def finish(robot: DynamixelRobotController | CombinedRobot, restored: bool) -> None:
    """
    end the run: lower the arm to the rest pose and switch the torque off, or ask first when that isn't safe
    :param restored: whether the servo under test has its yaml gains again; with trial gains in it the arm is not
        moved, only the prompt is shown
    """
    if restored:
        record_static_load.lower_and_disable_torque(robot)
    else:
        print('the arm is not lowered: the servo does not have its yaml gains')
        record_static_load.disable_torque_on_request(robot)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='tune the position P and D gains of one arm servo')
    parser.add_argument('--joint', type=int, required=True, help='the joint to tune, 1 to 6')
    parser.add_argument('--pose', nargs='+', default=None,
                        help='poses to tune at: {} or 6 comma separated degrees of joints 1 to 6'.format(
                            ', '.join(POSES)))
    parser.add_argument('--factor', type=float, default=1.3, help='the factor between the P gains of the sweep')
    parser.add_argument('--max-factor', type=float, default=2.0, help='sweep P up to this times the yaml value')
    parser.add_argument('--no-d', action='store_true', help='only tune P, at the yaml D')
    parser.add_argument('--d-start', type=int, default=250, help='the first D above 0 of the grid')
    parser.add_argument('--d-factor', type=float, default=1.6, help='the factor between the D gains of the grid')
    parser.add_argument('--d-max', type=int, default=DEFAULT_D_MAX,
                        help='the largest D of the grid, at most {} (default %(default)s)'.format(cfg.MAX_GAIN))
    parser.add_argument('--d-values', default=None,
                        help='comma separated D gains to sweep P at, in this order, instead of the grid')
    parser.add_argument('--d-patience', type=int, default=2,
                        help="D gains in a row that neither raise the P of the joint above the best nor clearly lower "
                             "the overshoot at its knee, that end the grid")
    parser.add_argument('--repeats', type=int, default=2, help='how often the steps are repeated per trial')
    parser.add_argument('--hold', type=float, default=0.8, help='seconds after every step')
    parser.add_argument('--rest', type=float, default=3.0, help='seconds of rest per trial')
    parser.add_argument('--baseline-rest', type=float, default=5.0, help='seconds of rest at the yaml gains per pose')
    parser.add_argument('--move-degrees', type=float, default=8.0, help='the streamed move per trial, 0 disables it')
    parser.add_argument('--speed', type=float, default=0.5, help='joint speed of the planned moves in rad/s')
    parser.add_argument('--pwm-clamp', type=int, default=None,
                        help='goal pwm during the trials (885 is 100%%), default 3 times the holding pwm, at least '
                             'half the limit, and the limit on joints {}'.format(
                                 ', '.join(str(j) for j in UNLOADED_JOINTS)))
    parser.add_argument('--max-step', type=int, default=None,
                        help='the largest step of the trials in positions, default {} for joints {} and {} for the '
                             'others'.format(MAX_STEP[1], ' and '.join(str(j) for j in MAX_STEP), DEFAULT_MAX_STEP))
    parser.add_argument('--max-temperature', type=float, default=None,
                        help='abort a trial when the servo reaches this temperature, default the temperature_limit '
                             'of servo_config.yaml')
    parser.add_argument('--floor', type=float, default=None,
                        help='cm above the table the wrist and gripper tip stay, default WorkSpaceLimits.z_min ({})'
                        .format(WorkSpaceLimits.z_min))
    parser.add_argument('--ff', action='store_true',
                        help='after the P and D tuning, measure and verify the velocity lead of the PC (a servo with '
                             'a move_time)')
    parser.add_argument('--ff-only', action='store_true',
                        help='only the feedforward stage, the P and D stages are skipped')
    parser.add_argument('--ff-degrees', type=float, default=None,
                        help='the feedforward sweep goes this many degrees either side of the pose (default {:g}, '
                             '{:g} for joints {})'.format(DEFAULT_FF_DEGREES, BELT_FF_DEGREES,
                                                          ' and '.join(str(j) for j in BELT_JOINTS)))
    parser.add_argument('--ff-speeds', type=float, nargs='+', default=None,
                        help='three or more cruise speeds of the feedforward sweeps in degrees per second (default '
                             '{}, {} for joints {}, which are not swept above {:g})'.format(
                                 ' '.join('{:g}'.format(s) for s in DEFAULT_FF_SPEEDS),
                                 ' '.join('{:g}'.format(s) for s in BELT_FF_SPEEDS),
                                 ' and '.join(str(j) for j in BELT_JOINTS), max(BELT_FF_SPEEDS)))
    parser.add_argument('--ff-lead', type=float, default=None,
                        help='the coefficient of the lead of the second round in pwm per degree/s, default the '
                             '`velocity_feedforward` of the servo in servo_config.yaml, else the one fitted in the '
                             'first round')
    parser.add_argument('--ff-max-positions', type=int, default=DEFAULT_FF_MAX_POSITIONS,
                        help='the clamp of the lead in positions for a servo without a `velocity_feedforward` block '
                             '(default %(default)s)')
    # an option of an earlier version of the stage: refused by name, argparse would otherwise
    # take it for the abbreviation of --ff-max-positions and clamp the lead at that many positions
    parser.add_argument('--ff-max', default=None, help=argparse.SUPPRESS)
    parser.add_argument('--swing', action='store_true',
                        help='after the other stages, measure the swing of the joint while it moves at the yaml gains '
                             'and at other P gains (a servo with a move_time)')
    parser.add_argument('--swing-only', action='store_true',
                        help='only the swing stage, the P and D stages are skipped; not with --ff or --ff-only')
    parser.add_argument('--swing-degrees', type=float, default=DEFAULT_SWING_DEGREES,
                        help='the swing sweep goes this many degrees either side of the pose (default %(default)s)')
    parser.add_argument('--swing-speed', type=float, default=DEFAULT_SWING_SPEED,
                        help='the cruise speed of the swing sweep in degrees per second (default %(default)s)')
    parser.add_argument('--swing-p', type=int, nargs='+', default=None,
                        help='P gains to try besides the yaml one, at the yaml D (default {} times the yaml P), '
                             'between 1 and {}'.format(' and '.join('{:g}'.format(f) for f in DEFAULT_SWING_P_FACTORS),
                                                       cfg.MAX_GAIN))
    parser.add_argument('--swing-d', type=int, nargs='+', default=[],
                        help='D gains to try at the yaml P besides the yaml one, between 0 and {} (default none)'
                        .format(cfg.MAX_GAIN))
    parser.add_argument('--swing-limit', type=float, default=DEFAULT_SWING_LIMIT,
                        help='positions, the swing a P may have in either direction to be recommended (default '
                             '%(default)s)')
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
        d_values = tuple(int(part) for part in args.d_values.split(',')) if args.d_values else None
        if d_values is not None and any(not 0 <= d <= cfg.MAX_GAIN for d in d_values):
            raise ValueError('the D gains have to be between 0 and {}'.format(cfg.MAX_GAIN))
        d_ladder(args.d_start, args.d_factor, args.d_max)  # raises on a bad start or factor
        if args.max_step is not None and args.max_step < MIN_STEP:
            raise ValueError('--max-step has to be at least {} positions, the smallest step of a trial'.format(
                MIN_STEP))
        if args.ff_max is not None:
            raise ValueError('--ff-max is gone: the stage measures the velocity lead of the '
                             'PC (--ff-max-positions clamps the lead, --ff-lead sets its coefficient)')
        if args.ff_speeds is not None and (len(set(args.ff_speeds)) < MIN_FF_SPEEDS
                                           or any(not 0 < speed < math.inf for speed in args.ff_speeds)):
            raise ValueError('--ff-speeds needs at least {} different positive speeds in degrees per second'.format(
                MIN_FF_SPEEDS))
        if args.ff_degrees is not None and not 0 < args.ff_degrees < math.inf:
            raise ValueError('--ff-degrees has to be above 0')
        if args.ff_lead is not None and not 0 < args.ff_lead < math.inf:
            raise ValueError('--ff-lead has to be above 0 pwm per degree/s')
        if args.ff_max_positions < 1:
            raise ValueError('--ff-max-positions has to be at least 1')
        if args.swing_only and (args.ff_only or args.ff):
            raise ValueError('--swing-only runs the swing stage alone, not together with --ff or --ff-only (use --ff '
                             '--swing for both stages after the P and D tuning)')
        if args.ff_only and args.swing:
            raise ValueError('--ff-only runs the feedforward stage alone, not together with --swing (use --ff '
                             '--swing for both stages after the P and D tuning)')
        if args.swing_degrees <= 0 or args.swing_speed <= 0:
            raise ValueError('--swing-degrees and --swing-speed have to be above 0')
        if args.swing_limit <= 0:
            raise ValueError('--swing-limit has to be above 0 positions')
        if args.swing_p is not None and any(not 0 < p <= cfg.MAX_GAIN for p in args.swing_p):
            raise ValueError('--swing-p has to be between 1 and {}'.format(cfg.MAX_GAIN))
        if any(not 0 <= d <= cfg.MAX_GAIN for d in args.swing_d):
            raise ValueError('--swing-d has to be between 0 and {}'.format(cfg.MAX_GAIN))
    except (argparse.ArgumentTypeError, ValueError) as e:
        print(e)
        return 1
    try:
        robot = get_robot()
    except DynamixelError as e:
        print(e)
        return 1
    if not isinstance(robot, CombinedRobot):
        print('the gains can only be tuned on the real robot, use_simulation is set')
        return 1
    max_temperature = args.max_temperature if args.max_temperature is not None \
        else robot.servo_config.temperature_limit
    settings = Settings(factor=args.factor, max_factor=args.max_factor, tune_d=not args.no_d,
                        d_start=args.d_start, d_factor=args.d_factor, d_max=args.d_max, d_values=d_values,
                        d_patience=args.d_patience, repeats=args.repeats, hold=args.hold, rest=args.rest,
                        baseline_rest=args.baseline_rest, move_degrees=args.move_degrees, speed=args.speed,
                        clamp=args.pwm_clamp, max_step=args.max_step, ff_degrees=args.ff_degrees,
                        ff_speeds=tuple(args.ff_speeds) if args.ff_speeds is not None else None,
                        ff_lead=args.ff_lead, ff_max_positions=args.ff_max_positions,
                        swing_degrees=args.swing_degrees, swing_speed=args.swing_speed,
                        swing_p=tuple(args.swing_p) if args.swing_p is not None else None,
                        swing_d=tuple(args.swing_d), swing_limit=args.swing_limit,
                        limits=AbortLimits(max_temperature=max_temperature))
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
    run_pd = not (args.ff_only or args.swing_only)
    run_ff = args.ff or args.ff_only
    run_swing = args.swing or args.swing_only
    if run_ff:
        # nothing moves yet: a servo without the time-based profile or a refused sweep is known before the servos
        # are enabled
        refused_ff = feedforward_refusal(robot, args.joint, accepted[0], settings)
        if refused_ff is not None:
            print(refused_ff)
            if args.ff_only:
                return 1
            run_ff = False
    if run_swing:
        refused_swing = swing_refusal(robot, args.joint, accepted[0], settings)
        if refused_swing is not None:
            print(refused_swing)
            if args.swing_only:
                return 1
            run_swing = False

    # before the servos are enabled, so a bad directory doesn't leave the arm under torque; relative to the working
    # directory, which a PyCharm run configuration may set to this file's folder
    output_dir = os.path.abspath(args.output_dir)
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print('can not create the output directory {}: {}'.format(output_dir, e))
        return 1
    stamp = datetime.now().strftime('%Y%m%d_%H%M')
    samples_path = os.path.join(output_dir, 'tune_joint{}_{}_samples.csv'.format(args.joint, stamp))
    trials_path = os.path.join(output_dir, 'tune_joint{}_{}_trials.csv'.format(args.joint, stamp))
    ff_path = os.path.join(output_dir, 'tune_joint{}_{}_ff.csv'.format(args.joint, stamp))
    swing_path = os.path.join(output_dir, 'tune_joint{}_{}_swing.csv'.format(args.joint, stamp))
    if run_pd:
        print("joint {} (yaml p={} d={}), poses {}, P up to {} in steps of {}, at D {} until {} D gains in a row "
              "don't raise the P of the joint, {} repeats, steps of at most {} positions, output {}".format(
                  args.joint, servo.p, servo.d, ', '.join(pose_degrees(pose) for pose in accepted),
                  round(servo.p * args.max_factor), args.factor, ' '.join(str(d) for d in d_grid(settings, servo.d)),
                  args.d_patience, args.repeats,
                  args.max_step if args.max_step is not None else MAX_STEP.get(args.joint, DEFAULT_MAX_STEP),
                  trials_path))
        print('the arm moves to these poses and steps joint {} by up to 5 degrees there, nothing may be in its way; '
              'the other joints hold their position'.format(args.joint))
    if run_ff:
        print('feedforward stage (velocity lead): joint {} (yaml p={} velocity_feedforward={}) at pose {} degrees, '
              'sweeps of {:g} degrees either way at {} degrees/s, first without a lead, then with {}, output {}'.format(
                  args.joint, servo.p, servo.velocity_feedforward, pose_degrees(accepted[0]),
                  settings.ff_degrees_of(args.joint),
                  ' '.join('{:g}'.format(s) for s in settings.ff_speeds_of(args.joint)),
                  'a coefficient of {:g} pwm per degree/s'.format(args.ff_lead) if args.ff_lead is not None
                  else 'the velocity_feedforward block of the yaml' if servo.velocity_feedforward is not None
                  else 'the coefficient fitted in the first round', ff_path))
        print('joint {} sweeps up to {:g} degrees either way at the speeds above, nothing may be in its way; '
              'the other joints hold their position'.format(args.joint, settings.ff_degrees_of(args.joint)))
    if run_swing:
        print('swing stage: joint {} (yaml p={} d={}) at pose {} degrees, sweeps of {:g} degrees either way at {:g} '
              'degrees/s with the gains (p, d) {}, recommending the largest P that swings at most {:g} positions, '
              'output {}'.format(args.joint, servo.p, servo.d, pose_degrees(accepted[0]), args.swing_degrees,
                                 args.swing_speed,
                                 ' '.join('({}, {})'.format(p, d) for p, d in swing_gains(settings, servo.p, servo.d)),
                                 args.swing_limit, swing_path))
        print('joint {} sweeps up to {:g} degrees either way at {:g} degrees/s once per setting, nothing may be in its '
              'way; the other joints hold their position'.format(args.joint, args.swing_degrees, args.swing_speed))
    if not args.yes and input('type yes to enable the servos and start: ').strip().lower() != 'yes':
        return 0
    if not robot.enable_servos():
        print('failed to enable the servos')
        return 1

    stop = StopFlag()
    stop.install()
    session: TuningSession | None = None
    stopped = False
    restored = True  # without a session nothing was changed
    try:
        with ExitStack() as files:
            print('temperatures: {}'.format(robot.get_servo_temperatures()))
            sample_writer = trial_writer = ff_writer = swing_writer = None
            if run_pd:
                samples_file = files.enter_context(open(samples_path, 'w', newline=''))
                trials_file = files.enter_context(open(trials_path, 'w', newline=''))
                sample_writer = csv.DictWriter(samples_file, fieldnames=SAMPLE_FIELDS)
                sample_writer.writeheader()
                trial_writer = csv.DictWriter(trials_file, fieldnames=TRIAL_FIELDS)
                trial_writer.writeheader()
            if run_ff:
                ff_writer = csv.DictWriter(files.enter_context(open(ff_path, 'w', newline='')),
                                           fieldnames=FF_SAMPLE_FIELDS)
                ff_writer.writeheader()
            if run_swing:
                swing_writer = csv.DictWriter(files.enter_context(open(swing_path, 'w', newline='')),
                                              fieldnames=SWING_SAMPLE_FIELDS)
                swing_writer.writeheader()
            session = TuningSession(robot, args.joint, settings, sample_writer, trial_writer, stop=stop,
                                    ff_writer=ff_writer, swing_writer=swing_writer)
            if run_pd:
                session.run(accepted)
            # a P and D run that stopped raised Stop and never gets here; a Ctrl+C that came after the last read
            # of it doesn't start the stage either
            if run_ff and not stop.requested:
                session.run_feedforward(accepted[0])
            if run_swing and not stop.requested:
                session.run_swing(accepted[0])
            print('temperatures: {}'.format(robot.get_servo_temperatures()))
    except Stop as e:
        print('\nstopped: {}'.format(e))
        stopped = True
    except KeyboardInterrupt:
        print('\nkilled')
        stopped = True
    finally:
        if session is not None:
            stop.protected = True
            restored = session.restore()
            if restored:
                print('the yaml gains, profile, goal pwm and velocity lead of servo {} are back'.format(args.joint))
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
    handler = robot.servo_handler
    if handler.failed_reads:
        print('{} sync reads of a bus failed and were tried again ({}); {} at the packet timeout (no reply, or not '
              'all of it, in time: the request or the reply lost on the wire, or the pc or usb late) and {} before '
              'it (a reply with wrong bytes: the wire, the ground or the adapter)'.format(
                  handler.failed_reads, ', '.join('servos {}: {}'.format(list(bus), count)
                                                  for bus, count in handler.failed_reads_by_bus.items()),
                  handler.late_reads, handler.failed_reads - handler.late_reads))
    print('; '.join(([] if not run_pd else ['samples in {}, trials in {}'.format(samples_path, trials_path)]) +
                     ([] if not run_ff else ['feedforward samples in {}'.format(ff_path)]) +
                     ([] if not run_swing else ['swing samples in {}'.format(swing_path)])))
    finish(robot, restored)  # before the plot, which blocks until its windows are closed
    if args.plot and results:
        plot_results(args.joint, results, samples_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
