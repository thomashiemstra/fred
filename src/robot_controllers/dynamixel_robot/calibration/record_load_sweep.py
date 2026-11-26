"""
Records the static load of one arm joint by sweeping it: the joint is driven slowly at constant speed through a span,
there and back, while the other joints hold their angles, and the error of the servo is read all the way. Run it from
the project root with the arm powered, its torque off and nothing in its way, one joint per run:
    python -m src.robot_controllers.dynamixel_robot.calibration.record_load_sweep --joint 3

A joint holds a load with a steady position error (the servos run a P-only position loop), but where it comes to rest
inside its friction band depends on how the move ended, which makes the static measurement of record_static_load.py
scatter. In a sweep the friction is measured instead of suffered: at every angle the joint passes in both directions,
the mean of the two errors is the gravity sag, and half the difference is the friction plus the lag of the speed,
which also flips sign with the direction and so cancels in the mean. Only the joint under test moves, so its error
says nothing about the others; their angles are logged so the fit knows the pose.

The sweeps run with the velocity lead of servo_config.yaml (a `velocity_feedforward` block of the servo), as every
planned move of the arm does, so what is recorded is the arm as it moves. The lead is added to the goal and makes the
joint trail its trajectory by less, so the error of a bin is measured against the goal of the angle itself, without
the compensation and without the lead (Servo.trajectory_position): the lead is not load, and the lag of the speed it
takes out would otherwise be taken for it. The mean lead of the bin and direction is logged in the lead column. With a
lead the half difference of the two directions shrinks by the lag the lead takes out: it is the friction, plus what the lead
does not cover. Without a velocity_feedforward block the lead is 0 and nothing changes.

For every pose (--pose: a pose name of tune_servo_gains or 6 comma separated degrees of joints 1 to 6, which gives the
angles of the OTHER joints; by default the default sweeps of the joint, DEFAULT_SWEEPS) and every speed (--speeds,
degrees per second, 5 and 10 by default) the script moves to the middle of the span (--span LOW:HIGH degrees, by
default the span of the default sweep) with the planner and streams one round trip at the speed with the time-based
profile of the servo: from the middle up to HIGH, down to LOW and back to the middle. The servo of the joint is read as
often as fits between two writes. The speed has a top per joint, 10 degrees/s for joint 2, which carries the whole arm
through a belt and must never be driven fast, and 30 for the others; a higher --speeds value is refused before anything
else happens. Before the servos are enabled (nothing moves until you type yes) the servo is checked to have a move_time
in servo_config.yaml (the time-based profile), and every sweep is checked against the planner: workspace limits, joint
ranges and speeds at every step, and it needs cruise steps in both directions. A refused sweep is printed with its
reason and skipped. --floor replaces WorkSpaceLimits.z_min as in record_static_load.py.

The samples of the cruise parts (the speed has been constant for a moment) are binned by the angle the servo measures:
a fixed grid of --bin degrees (5 by default), so the bins of the sweeps at two speeds line up. A bin counts when it has
at least MIN_BIN_SAMPLES samples in both directions. After every sweep the script prints the number of bins and, over
the bins, the range of the gravity sag (mean of the up and the down error, in positions), of the measured pwm and the
mean half difference of the two directions (friction plus the lag of the speed). After the last speed of a pose the sweeps are compared: a bin whose mean error
differs by more than 1 position between speeds gets a warning, the friction is not the same in both directions there
and the slower sweep is the one to trust. The speeds also show how much of the half difference is lag (it grows with
the speed) and how much is friction.

The robot stops before every pose when a servo is warmer than the warning temperature or has a hardware error, and
during a sweep on the abort rules of the gain tuning (temperature, voltage drop, position jumps, divergence from the
trajectory, pwm stuck at the clamp). An abort, three failed reads in a row, a move the planner refuses or Ctrl+C (which
stops at the next sample) ends the recording. Whatever ends it, the arm is then lowered to the rest pose and the torque
switched off, with the same prompt as record_static_load.py when that is not possible. What was recorded stays in the
csv, which is flushed after every sweep.

Output: a csv with the columns of record_static_load.FIELDS (mode is 'sweep', speed the degrees per second), written
per bin: a new pose number per bin and for each of the two directions (up: the angle was rising) a row per arm servo.
The row of the swept servo has the mean error (present position minus the goal of the angle itself, without the
compensation and without the velocity lead, in positions, as in record_static_load.py: the whole load is error minus
compensation), the compensation, the lead (both means of the samples of the bin and direction, signed like the
position), the mean present position, the measured pwm, the current, the number of samples, the gains, the highest
temperature and the lowest voltage of the bin. The rows of the other servos only have the angle of their joint and
the gains: those joints stood still, their error says nothing. The angle of the swept joint in every row of a bin is
the centre of the bin. fit_static_load.py reads this csv together with the ones of record_static_load.py (every file is its own set of poses); record the static
load at rest as well, as the check. fit_static_load.py converts every row with the P gain of its p column to the holding
drive in pwm units, and the compensation converts that drive back to positions with the P gain the servo runs with, so a
recording stays valid when a P gain is retuned; it has to be repeated after a mechanical change of the arm (another
gripper, a payload, a changed link) or a change of the operating mode of a servo.
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any, NamedTuple

import numpy as np

from src import global_constants
from src.global_constants import WorkSpaceLimits
from src.global_objects import get_robot
from src.robot_controllers.combined_robot.combined_robot import CombinedRobot
from src.robot_controllers.dynamixel_robot import dynamixel_x_config as cfg
from src.robot_controllers.dynamixel_robot.calibration import record_static_load
from src.robot_controllers.dynamixel_robot.calibration.gain_tuning import (
    MIN_BIN_SAMPLES, AbortLimits, Sample, SweepBin, SweepStep, abort_reason, bin_index, bin_sweep, ff_sweep)
from src.robot_controllers.dynamixel_robot.calibration.load_directions import DOWN, UP
from src.robot_controllers.dynamixel_robot.calibration.record_static_load import StopRecording
from src.robot_controllers.dynamixel_robot.calibration.tune_servo_gains import (
    POSES, StopFlag, StreamedSample, parse_pose, pose_degrees, stream_sweep)
from src.robot_controllers.dynamixel_robot.dynamixel_robot_controller import ARM_IDS, DynamixelRobotController
from src.robot_controllers.dynamixel_robot.dynamixel_utils import DynamixelError
from src.robot_controllers.dynamixel_robot.servo import Servo
from src.robot_controllers.dynamixel_robot.static_compensation import TERMS_OF_JOINT
from src.utils.movement import PlannedMove, check_move
from src.utils.movement_exception import MovementException

MODE = 'sweep'  # the value of the mode column for the rows of this script
DEFAULT_SPEEDS = (5.0, 10.0)  # degrees per second
DEFAULT_BIN = 5.0  # degrees
# the fastest sweep per joint in degrees per second: the shoulder carries the whole arm through a belt and must never
# be driven fast
TOP_SPEEDS: dict[int, float] = {2: 10.0}
DEFAULT_TOP_SPEED = 30.0
SETTLE = 1.0  # seconds at the start of a pose before the first sweep
SWEEP_REST = 0.5  # seconds at rest after a sweep
# the mean error of a bin at two speeds may differ by this many positions before the friction of the joint counts as
# different in the two directions
DISAGREEMENT = 1.0

# the sweeps of every joint: (pose in degrees of joints 1 to 6, (low, high) span in degrees of the swept joint, its own
# value in the pose is ignored). Poses with the elbow out carry the largest load on joints 3 and 5, a wrist bent
# either way changes the lever of the gripper on joint 3 and joint 4 turns it about the forearm; the spans are as wide
# as the planner allows (WorkSpaceLimits and the ranges of servo_config.yaml, a test checks every sweep): the wrist
# bent 60 degrees puts the tip closer than y_min to the base beyond 35 degrees of joint 3 (bent one way) or below -35
# (the other way). Since 26 November 2026 the wrist centre has to stay 12.5cm from the base axis (wrist_radius_min) and
# the gripper is 15cm (y_min 15.1): the spans of joint 2 at 45 and at -30 degrees of joint 3, of joint 3 at the
# upright upper arm, and of joint 5 with the upper arm raised 40 degrees are narrower than the recordings of 19 to 24
# November were made with (40 to 100, 70 to 120, -60 to 60 and -80 to 55 degrees), by a few degrees short of what the
# planner refuses now (joint 2 beyond about 94 and 115 degrees, joint 3 beyond about 54, joint 5 beyond about 53)
DEFAULT_SWEEPS: dict[int, list[tuple[tuple[float, ...], tuple[float, float]]]] = {
    2: [((90, 0, 0, 0, 0, 0), (50, 110)),
        ((90, 0, 45, 0, 0, 0), (40, 90)),
        ((90, 0, -30, 0, 0, 0), (70, 110))],
    3: [((90, 90, 0, 0, 0, 0), (-50, 50)),
        ((90, 60, 0, 0, 0, 0), (-20, 70)),
        ((90, 90, 0, 0, 60, 0), (-45, 35)),
        ((90, 90, 0, 0, -60, 0), (-35, 45))],
    4: [((90, 90, 0, 0, 60, 0), (-80, 80)),
        ((90, 90, 0, 0, -60, 0), (-80, 80))],
    5: [((90, 90, 0, 0, 0, 0), (-80, 80)),
        ((90, 90, 0, 60, 0, 0), (-80, 80)),
        ((90, 90, 40, 0, 0, 0), (-80, 50))],
}


class PoseSpan(NamedTuple):
    """a pose of the other joints and the span of the swept joint at it"""
    pose: np.ndarray  # 7 element angle array in radians, the angle of the swept joint is the middle of the span
    low: float  # degrees
    high: float


class SweepPlan(NamedTuple):
    """one sweep of a pose at a speed: the steps of the commanded motion and the angles of the robot at every step"""
    speed: float  # degrees per second
    steps: list[SweepStep]
    angles: np.ndarray

    @property
    def duration(self) -> float:
        return self.steps[-1].time


class PosePlan(NamedTuple):
    """a pose with the sweeps at it that the planner accepted"""
    start: PoseSpan
    sweeps: list[SweepPlan]


class Options(NamedTuple):
    bin_degrees: float = DEFAULT_BIN
    move_speed: float = record_static_load.REST_SPEED  # rad/s of the moves between the poses
    limits: AbortLimits = AbortLimits()  # of the swept servo during a sweep
    settle: float = SETTLE
    rest: float = SWEEP_REST


class BinDetails(NamedTuple):
    """what the csv row of a direction in a bin has besides the error and the pwm of the SweepBin"""
    compensation: float  # mean, positions
    lead: float  # mean positions the velocity lead added to the goal, signed like the position
    position: float  # mean present position
    current: float  # mean of the current register
    temperature: int  # highest
    voltage: float  # lowest


class SweepResult(NamedTuple):
    speed: float
    bins: list[SweepBin]


class Disagreement(NamedTuple):
    """where the sweeps of a pose at different speeds differ most"""
    angle: float  # degrees, the centre of the bin
    difference: float  # positions, the largest minus the smallest mean error of the bin
    slowest: float  # degrees per second of the slowest sweep


class Recording:
    """what a recording got done, kept when it stops early"""

    def __init__(self) -> None:
        self.sweeps = 0  # finished sweeps
        self.bins = 0  # bins written, which is also the number of poses in the csv
        self.read_failures = 0  # reads of the servo that failed and were tried again
        self.stopped: str | None = None  # why the recording ended early


def top_speed(joint: int) -> float:
    """the fastest sweep of the joint in degrees per second"""
    return TOP_SPEEDS.get(joint, DEFAULT_TOP_SPEED)


def parse_span(value: str) -> tuple[float, float]:
    """a LOW:HIGH span in degrees"""
    try:
        low, high = (float(part) for part in value.split(':'))
    except ValueError:
        raise argparse.ArgumentTypeError('a span looks like -60:60 (degrees), got {!r}'.format(value))
    if not (math.isfinite(low) and math.isfinite(high)) or low >= high:
        raise argparse.ArgumentTypeError('the span has to go from a low to a higher angle, got {!r}'.format(value))
    return low, high


def positive_float(value: str) -> float:
    """argparse type of a number above 0"""
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError('should be a number, got {!r}'.format(value))
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('should be above 0, got {}'.format(value))
    return number


def default_poses(joint: int, span: tuple[float, float] | None = None) -> list[PoseSpan]:
    """the default sweeps of the joint, all with the span when one is given"""
    poses = []
    for degrees, default_span in DEFAULT_SWEEPS[joint]:
        angles = np.zeros(7, dtype=np.float64)
        angles[1:] = np.radians(degrees)
        low, high = span if span is not None else default_span
        poses.append(pose_span(angles, joint, low, high))
    return poses


def pose_span(pose: np.ndarray, joint: int, low: float, high: float) -> PoseSpan:
    """the pose with the swept joint at the middle of the span"""
    start = np.asarray(pose, dtype=np.float64).copy()
    start[joint] = np.radians((low + high) / 2.0)
    return PoseSpan(start, low, high)


def plan_sweep(span: PoseSpan, joint: int, speed: float) -> SweepPlan:
    """the sweep of the pose at the speed (degrees per second): one round trip around the middle of the span, and the
    angles of every step, the pose with the swept joint at its middle plus the offset of the step"""
    dt = 1.0 / global_constants.steps_per_second
    steps = ff_sweep((span.high - span.low) / 2.0, speed, dt, round_trips=1)
    angles = np.tile(span.pose, (len(steps), 1))
    angles[:, joint] += np.radians([step.offset for step in steps])
    return SweepPlan(speed, steps, angles)


def sweep_refusal(robot: DynamixelRobotController | CombinedRobot, plan: SweepPlan) -> str | None:
    """why the planner refuses the sweep, or None: every step has to pass check_move (joint speed, range and workspace)
    and the sweep has to cruise in both directions or there is nothing to pair"""
    try:
        check_move(PlannedMove('sweep at {:g} degrees/s'.format(plan.speed), plan.angles, plan.duration), robot)
    except MovementException as e:
        return str(e)
    if {step.direction for step in plan.steps if step.cruise} != {1, -1}:
        return 'no cruise part going both ways: use a wider span or a lower speed'
    return None


def plan_poses(robot: DynamixelRobotController | CombinedRobot, joint: int, spans: Sequence[PoseSpan],
               speeds: Sequence[float]) -> list[PosePlan]:
    """the sweeps of every pose at every speed that the planner accepts; a refused sweep is printed with the reason
    and left out, and so is a pose without a sweep left. Nothing moves."""
    poses = []
    for span in spans:
        sweeps = []
        for speed in speeds:
            plan = plan_sweep(span, joint, speed)
            refused = sweep_refusal(robot, plan)
            if refused is None:
                sweeps.append(plan)
            else:
                print('skipping the sweep of joint {} from {:g} to {:g} degrees at pose {} degrees at {:g} degrees/s: '
                      '{}'.format(joint, span.low, span.high, pose_degrees(span.pose), speed, refused))
        if sweeps:
            poses.append(PosePlan(span, sweeps))
    return poses


def time_based_refusal(robot: DynamixelRobotController | CombinedRobot, joint: int) -> str | None:
    """why the joint can't be swept, or None: its servo needs the time-based profile, the one the robot moves with"""
    if not robot.servo_handler.get_servo(joint).time_based:
        return 'servo {} has no move_time in servo_config.yaml, the sweep needs the time-based profile'.format(joint)
    return None


def estimated_seconds(poses: Sequence[PosePlan], options: Options = Options()) -> float:
    """about how long the recording takes: the sweeps, the rests, and the moves between the poses (not the first move
    and the lowering at the end, the arm may be anywhere when it starts)"""
    total = 0.0
    previous: np.ndarray | None = None
    for pose in poses:
        if previous is not None:
            largest = float(np.abs((pose.start.pose - previous)[1:]).max())
            total += max(largest / options.move_speed, 1.0)
        previous = pose.start.pose
        total += options.settle + sum(plan.duration + options.rest for plan in pose.sweeps)
    return total


# binning


def bin_cruise_samples(servo: Servo, streamed: Sequence[StreamedSample], bin_degrees: float
                       ) -> list[tuple[SweepBin, BinDetails, BinDetails]]:
    """
    the bins of the cruise samples of a sweep with what the csv rows of their two directions have: for every bin that
    was passed in both directions the SweepBin and the BinDetails going up and going down
    The angle of a sample is the one the servo measures (not the commanded one, which is where the joint would be
    without the load), the direction the one of the ANGLE of its step, and the error is the position minus the goal
    of the angle itself: the position of the trajectory less the compensation and less the velocity lead that were
    added to it. With a lead the joint trails the goal that was sent by less, and the error of the two directions
    keeps its gravity part and loses the lag of the speed the lead took away.
    """
    cruise = [s for s in streamed if s.step.cruise]
    if not cruise:
        return []
    angles = np.array([servo.degrees_at_position(s.sample.position) for s in cruise])
    directions = np.array([s.step.direction for s in cruise])
    positions = np.array([s.sample.position for s in cruise], dtype=float)
    compensations = np.array([s.compensation for s in cruise], dtype=float)
    leads = np.array([s.lead for s in cruise], dtype=float)
    errors = positions - (np.array([s.sample.position_trajectory for s in cruise], dtype=float) - compensations - leads)
    pwms = np.array([s.sample.pwm for s in cruise], dtype=float)
    currents = np.array([s.sample.current for s in cruise], dtype=float)
    temperatures = np.array([s.sample.temperature for s in cruise])
    voltages = np.array([s.sample.voltage for s in cruise])
    index = bin_index(angles, bin_degrees)
    result = []
    for found in bin_sweep(angles, directions, errors, pwms, bin_degrees):
        k = int(math.floor(found.angle / bin_degrees))
        details = []
        for direction in (1, -1):
            mask = (index == k) & (directions == direction)
            details.append(BinDetails(float(compensations[mask].mean()), float(leads[mask].mean()),
                                      float(positions[mask].mean()), float(currents[mask].mean()),
                                      int(temperatures[mask].max()), float(voltages[mask].min())))
        result.append((found, details[0], details[1]))
    return result


def bin_rows(pose: np.ndarray, joint: int, speed: float, bins: Sequence[tuple[SweepBin, BinDetails, BinDetails]],
             servos: dict[int, Servo], models: dict[int, str], first_pose: int, now: float) -> list[dict[str, Any]]:
    """
    the csv rows of the bins: a pose number per bin (from first_pose) and for each direction a row per arm servo
    :param pose: the angles of the sweep, the angle of the swept joint is replaced by the centre of every bin
    :param servos: mapping of servo id to Servo, for the gains
    :param models: mapping of servo id to model name, for the unit of the current
    """
    rows = []
    for number, (found, up, down) in enumerate(bins, start=first_pose):
        angles = pose.copy()
        angles[joint] = np.radians(found.angle)
        for direction, side, details in ((UP, found.up, up), (DOWN, found.down, down)):
            for servo_id in ARM_IDS:
                servo = servos[servo_id]
                row: dict[str, Any] = {name: '' for name in record_static_load.FIELDS}
                row.update({'time': '{:.2f}'.format(now), 'pose': number, 'direction': direction, 'servo': servo_id,
                            'angle': '{:.6f}'.format(angles[servo_id]), 'mode': MODE, 'speed': '{:g}'.format(speed),
                            'p': servo.p, 'i': servo.i, 'd': servo.d})
                if servo_id == joint:
                    current, unit = record_static_load.current_in_units(details.current, models.get(servo_id))
                    row.update({'goal_position': '{:.2f}'.format(details.position - side.error),
                                'compensation': '{:.2f}'.format(details.compensation),
                                'lead': '{:.2f}'.format(details.lead),
                                'present_position': '{:.2f}'.format(details.position),
                                'error': '{:.2f}'.format(side.error), 'current_raw': '{:.1f}'.format(details.current),
                                'current': '' if current is None else '{:.2f}'.format(current), 'current_unit': unit,
                                'temperature': details.temperature, 'voltage': details.voltage,
                                'samples': side.samples, 'pwm': '{:.1f}'.format(side.pwm)})
                rows.append(row)
    return rows


# reporting


def describe_sweep(speed: float, bins: Sequence[SweepBin], lead: float = 0.0) -> str:
    """the line after a sweep: the bins, the range of the gravity sag and of the measured gravity pwm over the bins
    (the means of the two directions) and the mean absolute half difference of the two directions, which is the
    friction plus the lag of the speed (it grows with the speed)
    :param lead: the mean absolute velocity lead of the bins in positions; above 0 the line says that the errors are
        against the goal without it and that the half difference is smaller by the lag the lead took out"""
    if not bins:
        return 'speed {:g} degrees/s: no bin has {} samples in both directions, use a lower speed, a wider span or ' \
               'check the reads'.format(speed, MIN_BIN_SAMPLES)
    sags = [b.sag for b in bins]
    pwms = [(b.up.pwm + b.down.pwm) / 2.0 for b in bins]
    half = float(np.mean([abs(b.half_difference) for b in bins]))
    line = 'speed {:g} degrees/s: {} bins, gravity sag {:.1f} to {:.1f} positions, measured gravity pwm {:.0f} to ' \
           '{:.0f}, half the up/down difference {:.1f} positions (friction plus the lag of the speed)'.format(
               speed, len(bins), min(sags), max(sags), min(pwms), max(pwms), half)
    if lead > 0:
        line += '; the velocity lead of {:.1f} positions is not in the errors (they are against the goal without it) ' \
                'and took part of the lag of the speed out of the half difference'.format(lead)
    return line


def speed_disagreement(results: Sequence[SweepResult]) -> Disagreement | None:
    """
    where the mean error of the bins that every sweep has differs most between two speeds, None with fewer than two
    sweeps or without a bin they share. The lag of the speed and the friction cancel in the mean of the two
    directions when they are the same both ways, so a mean that moves with the speed says they are not.
    """
    if len(results) < 2:
        return None
    sags = [{round(b.angle, 6): b.sag for b in result.bins} for result in results]
    shared = set(sags[0]).intersection(*sags[1:])
    if not shared:
        return None
    angle, difference = max(((a, max(s[a] for s in sags) - min(s[a] for s in sags)) for a in sorted(shared)),
                            key=lambda pair: pair[1])
    return Disagreement(angle, difference, min(result.speed for result in results))


def disagreement_warning(disagreement: Disagreement | None, speeds: Sequence[float]) -> str | None:
    """the warning for a disagreement above DISAGREEMENT, None otherwise"""
    if disagreement is None or disagreement.difference <= DISAGREEMENT:
        return None
    return 'warning: at {:.1f} degrees the mean error differs by {:.1f} positions between the speeds {} degrees/s: ' \
           'the friction of the joint is not the same in both directions there, so the mean of the two is less ' \
           'reliable; trust the sweep at {:g} degrees/s'.format(
               disagreement.angle, disagreement.difference, ', '.join('{:g}'.format(s) for s in speeds),
               disagreement.slowest)


# the recording


def read_servo(robot: DynamixelRobotController | CombinedRobot, joint: int, recording: Recording,
               attempts: int = record_static_load.READ_ATTEMPTS) -> Sample:
    """the state of the servo as a Sample whose time and goal the caller fills in; raises StopRecording when every
    attempt failed (a corrupt packet now and then happens on the bus, the read is harmless to repeat)"""
    for _ in range(attempts):
        states = robot.servo_handler.read_state([joint])
        if states is not None:
            s = states[joint]
            return Sample(time=0.0, tick=s.tick, goal=0, position=s.position,
                          position_trajectory=s.position_trajectory, velocity=s.velocity,
                          velocity_trajectory=s.velocity_trajectory, pwm=s.pwm, current=s.current,
                          moving_status=s.moving_status, voltage=s.voltage, temperature=s.temperature)
        recording.read_failures += 1
    raise StopRecording('failed to read servo {} {} times in a row'.format(joint, attempts))


def stream_one_sweep(robot: DynamixelRobotController | CombinedRobot, joint: int, plan: SweepPlan,
                     options: Options, recording: Recording, stop: StopFlag, clock: Callable[[], float],
                     sleep: Callable[[float], None]) -> list[StreamedSample]:
    """stream the sweep with the shared loop of the gain tuning and read the servo of the joint; raises StopRecording
    on a stop request, a read that failed or an abort rule of the gain tuning"""
    servo = robot.servo_handler.get_servo(joint)
    rows: list[Sample] = []
    streamed: list[StreamedSample] = []
    start = clock()

    def read(goal: int, written: float) -> Sample:
        if stop.requested:
            raise StopRecording('interrupted')
        state = read_servo(robot, joint, recording)
        sample = state._replace(time=clock() - start, goal=goal)
        rows.append(sample)
        return sample

    def check(written: float) -> str | None:
        return abort_reason(rows, cfg.MAX_PWM, 0, 0.0, rows[0].voltage, written - start, options.limits, sag=0.0,
                            streaming=True)
    reason = stream_sweep(robot, servo, plan.steps, plan.angles, read, check, streamed, clock, sleep)
    if reason is not None:
        raise StopRecording('aborted: {}'.format(reason))
    return streamed


def record_sweeps(robot: DynamixelRobotController | CombinedRobot, joint: int, poses: Sequence[PosePlan],
                  writer: csv.DictWriter[str], options: Options = Options(), stop: StopFlag | None = None,
                  clock: Callable[[], float] = time.perf_counter, sleep: Callable[[float], None] = time.sleep,
                  wall_clock: Callable[[], float] = time.time, flush: Callable[[], None] | None = None) -> Recording:
    """
    sweep the joint at every pose and speed and write the bins of every finished sweep to the csv. A stop (an abort
    rule, a failed read, a servo that is warm or has a hardware error, a refused move, Ctrl+C) ends the recording
    with the reason in Recording.stopped; the unfinished sweep is not written, the finished ones are kept.
    :param robot: the robot of global_objects.get_robot with its servos enabled
    :param poses: see plan_poses; every move is checked by the planner against the workspace limits of the robot
    :param writer: csv.DictWriter with record_static_load.FIELDS
    :param clock: the clock the writes of the sweep follow, perf_counter
    :param wall_clock: the time column of the csv
    :param flush: called after every sweep to flush the file
    """
    stop = stop if stop is not None else StopFlag()
    recording = Recording()
    servos = {servo_id: robot.servo_handler.get_servo(servo_id) for servo_id in ARM_IDS}
    models = robot.servo_config.models
    servo = servos[joint]
    try:
        for number, pose in enumerate(poses, start=1):
            print('pose {} of {}: {} degrees, joint {} from {:g} to {:g} degrees'.format(
                number, len(poses), pose_degrees(pose.start.pose), joint, pose.start.low, pose.start.high))
            record_static_load.check_servos(robot, robot.servo_config.temperature_warning)
            if not record_static_load.move_to(robot, pose.start.pose, options.move_speed, lambda: stop.requested):
                raise StopRecording('the planner refuses the move to the start of the sweeps at pose {}'.format(
                    number))
            sleep(options.settle)
            results: list[SweepResult] = []
            for plan in pose.sweeps:
                if stop.requested:
                    raise StopRecording('interrupted')
                streamed = stream_one_sweep(robot, joint, plan, options, recording, stop, clock, sleep)
                bins = bin_cruise_samples(servo, streamed, options.bin_degrees)
                rows = bin_rows(pose.start.pose, joint, plan.speed, bins, servos, models, recording.bins, wall_clock())
                writer.writerows(rows)
                if flush is not None:
                    flush()
                recording.sweeps += 1
                recording.bins += len(bins)
                results.append(SweepResult(plan.speed, [found for found, _, _ in bins]))
                lead = float(np.mean([abs(details.lead) for _, up, down in bins for details in (up, down)])) \
                    if bins else 0.0
                print('  ' + describe_sweep(plan.speed, results[-1].bins, lead))
                sleep(options.rest)
            warning = disagreement_warning(speed_disagreement(results), [plan.speed for plan in pose.sweeps])
            if warning is not None:
                print('  ' + warning)
    except StopRecording as e:
        recording.stopped = str(e)
    except KeyboardInterrupt:
        recording.stopped = 'killed'
    return recording


# the script


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='record the static load of an arm joint by sweeping it slowly '
                                                 'through a span in both directions')
    parser.add_argument('--joint', type=int, required=True,
                        help='the joint to sweep, one of {}'.format(', '.join(str(j) for j in TERMS_OF_JOINT)))
    parser.add_argument('--pose', nargs='+', type=parse_pose, default=None,
                        help='poses of the other joints to sweep at: {} or 6 comma separated degrees of joints 1 to '
                             '6 (the angle of the swept joint is ignored), default the default sweeps of the joint; '
                             'needs --span'.format(', '.join(POSES)))
    parser.add_argument('--span', type=parse_span, default=None, metavar='LOW:HIGH',
                        help='the span of the swept joint in degrees, replaces the span of every pose; --span -60:60 '
                             'and --span=-60:60 both work')
    parser.add_argument('--speeds', type=positive_float, nargs='+', default=list(DEFAULT_SPEEDS),
                        help='degrees per second, every pose is swept once at each speed (default {}), at most {:g} '
                             'for joint 2 and {:g} for the others'.format(
                                 ' '.join('{:g}'.format(s) for s in DEFAULT_SPEEDS), TOP_SPEEDS[2],
                                 DEFAULT_TOP_SPEED))
    parser.add_argument('--bin', type=positive_float, default=DEFAULT_BIN,
                        help='width of the bins of the angle in degrees (default %(default)s)')
    parser.add_argument('--floor', type=float, default=None,
                        help='cm above the table the wrist and gripper tip stay, default WorkSpaceLimits.z_min ({})'
                        .format(WorkSpaceLimits.z_min))
    parser.add_argument('--move-speed', type=positive_float, default=record_static_load.REST_SPEED,
                        help='rad/s of the moves between the sweeps (default %(default)s)')
    parser.add_argument('--output', default=None,
                        help='the csv file, default load_sweep_joint<N>_<date>_<time>.csv')
    parser.add_argument('--yes', action='store_true', help="don't ask before enabling the servos")
    return parser


def join_span_argument(argv: Sequence[str]) -> list[str]:
    """argv with a `--span -60:60` written as `--span=-60:60`: argparse takes a value that starts with a minus for the
    next option when it is not a plain number, which a span with a negative low end is not"""
    joined: list[str] = []
    for argument in argv:
        if joined and joined[-1] == '--span' and argument[:1] == '-' and argument[1:2].isdigit():
            joined[-1] = '--span=' + argument
        else:
            joined.append(argument)
    return joined


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(join_span_argument(sys.argv[1:] if argv is None else argv))


def main(argv: list[str] | None = None, clock: Callable[[], float] = time.perf_counter,
         sleep: Callable[[float], None] = time.sleep) -> int:
    args = parse_arguments(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

    joint: int = args.joint
    if joint not in TERMS_OF_JOINT:
        print('only joints {} have a static load to record, not {}'.format(
            ', '.join(str(j) for j in TERMS_OF_JOINT), joint))
        return 1
    too_fast = [speed for speed in args.speeds if speed > top_speed(joint)]
    if too_fast:
        print('joint {} can not be swept faster than {:g} degrees/s, got {}'.format(
            joint, top_speed(joint), ', '.join('{:g}'.format(s) for s in too_fast)))
        return 1
    if args.pose is not None and args.span is None:
        print('--pose needs --span, the span of the swept joint at those poses')
        return 1
    if args.pose is not None:
        spans = [pose_span(pose, joint, *args.span) for pose in args.pose]
    else:
        spans = default_poses(joint, args.span)
    output_path = args.output or 'load_sweep_joint{}_{}.csv'.format(joint, datetime.now().strftime('%Y%m%d_%H%M'))

    try:
        robot = get_robot()
    except DynamixelError as e:
        print(e)
        return 1
    if not isinstance(robot, CombinedRobot):
        print('the load can only be swept on the real robot, use_simulation is set')
        return 1
    refused = time_based_refusal(robot, joint)
    if refused is not None:
        print(refused)
        return 1
    if args.floor is not None:
        robot.workspace_limits = record_static_load.limits_with_floor(args.floor, robot.workspace_limits)
    poses = plan_poses(robot, joint, spans, args.speeds)
    if not poses:
        print('no sweep the planner accepts')
        return 1

    options = Options(args.bin, args.move_speed, AbortLimits(max_temperature=robot.servo_config.temperature_limit))
    servo = robot.servo_handler.get_servo(joint)
    sweep_count = sum(len(pose.sweeps) for pose in poses)
    print('joint {} (yaml p={} d={}), {} sweeps at {} poses, bins of {:g} degrees, output {}'.format(
        joint, servo.p, servo.d, sweep_count, len(poses), args.bin, output_path))
    for number, pose in enumerate(poses, start=1):
        print('  pose {}: {} degrees, joint {} from {:g} to {:g} degrees at {} degrees/s'.format(
            number, pose_degrees(pose.start.pose), joint, pose.start.low, pose.start.high,
            ' '.join('{:g}'.format(plan.speed) for plan in pose.sweeps)))
    print('about {:.1f} minutes, not counting the first move and the lowering at the end'.format(
        estimated_seconds(poses, options) / 60.0))
    print('the arm moves to these poses and sweeps joint {} through the spans above, nothing may be in its way; the '
          'other joints hold their position'.format(joint))
    if not args.yes:
        try:
            answer = input('type yes to enable the servos and start: ')
        except EOFError:
            print('there is no input to answer the question, use --yes for an unattended run')
            return 1
        if answer.strip().lower() != 'yes':
            return 0

    # the file before the servos are enabled, so a bad path doesn't leave the arm under torque
    try:
        output = open(output_path, 'w', newline='')
    except OSError as e:
        print('can not write {}: {}'.format(output_path, e))
        return 1
    with output:
        writer = csv.DictWriter(output, fieldnames=record_static_load.FIELDS)
        writer.writeheader()
        output.flush()
        if not robot.enable_servos():
            print('failed to enable the servos')
            return 1
        stop = StopFlag()
        stop.install()
        try:
            recording = record_sweeps(robot, joint, poses, writer, options, stop, clock, sleep, flush=output.flush)
        finally:
            # a Ctrl+C while the arm is lowered has to reach lower_and_disable_torque as a KeyboardInterrupt
            stop.uninstall()
            record_static_load.lower_and_disable_torque(robot)  # falls back to the prompt by itself
    if recording.stopped is not None:
        print('\nstopped: {}'.format(recording.stopped))
    if recording.read_failures:
        print('{} reads of the servo failed and were retried'.format(recording.read_failures))
    print('{} sweeps, {} bins recorded in {}'.format(recording.sweeps, recording.bins, output_path))
    return 0 if recording.sweeps else 1


if __name__ == '__main__':
    sys.exit(main())
