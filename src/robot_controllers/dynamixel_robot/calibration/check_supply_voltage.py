"""
Checks the supply voltage along the servo chain: the input voltage every servo (1-7) sees at rest with the torque
off, holding a few poses, and during a planned move, and where along the chain it is lost. Run it from the project
root with the arm powered and its torque off:
    python -m src.robot_controllers.dynamixel_robot.calibration.check_supply_voltage

Three phases, each skippable (--no-rest, --no-hold, --no-motion):
1. rest: the torque off, every servo is sampled for --duration seconds (default 5).
2. hold: after asking, the servos are enabled and the arm moves with the planner (checked like every other move)
   to the named poses of tune_servo_gains.POSES (--pose, default elbow_out, reach, low), settles for --settle
   seconds and is sampled for --duration at each.
3. motion: a planned move between the first two of those poses and back at --speed (default 0.5 rad/s), sampled
   as fast as the bus allows while the move streams (about 10 ms per sample of all servos).

A sample is one sync read per bus of the state block of all servos (ServoHandler.read_state): the input voltage,
the present current (XM, XL servos) or load (XC430) and the temperature go to the csv. The report per phase and
servo: min, mean and max voltage, the drop from servo 1 for the 12 V servos 1-6 (the gripper has its own 5 V rail),
the difference to the reference of 21 October 2026 (11.6, 11.4, 11.3, 11.3, 11.3, 11.2 V for servos 1-6 and 5.1 V
for the gripper, torque off, after a power cycle), and a flag for a reading outside the rated range of the model
(XM430 and XM540 10.0-14.8 V, XC430 6.5-12.0 V, XL330 3.7-6.0 V). The conclusion: a servo 1 that is already low
at rest points at the supply or its lead, a drop along the chain that grows under load points at the leads or the
connectors, and the hop with the largest drop is named. Background: during the gain tuning of joint 6 on 21 October
servo 6 read 10.5 V at rest and dipped to 9.8 V during the steps, 0.7 V below what it read earlier that day.

Safety: it asks before enabling the servos (--yes skips), Ctrl+C stops after the current read, the sampling stops
when a servo passes the temperature warning of the yaml, nothing is written to the eeprom or to the gains, and at
the end it offers to disable the torque (support the arm, it drops).

Output: <output-dir>/supply_voltage_<time>.csv with every sample (phase, pose, time, servo, voltage, current or
load, temperature) and the console summary.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime
from typing import NamedTuple

import numpy as np

from src import global_constants
from src.global_constants import WorkSpaceLimits
from src.global_objects import get_robot
from src.robot_controllers.combined_robot.combined_robot import CombinedRobot
from src.robot_controllers.dynamixel_robot.calibration import record_static_load
from src.robot_controllers.dynamixel_robot.calibration.tune_servo_gains import POSES, StopFlag, parse_pose, \
    pose_degrees
from src.robot_controllers.dynamixel_robot.dynamixel_robot_controller import GRIPPER_ID, DynamixelRobotController
from src.robot_controllers.dynamixel_robot.dynamixel_utils import DynamixelError
from src.robot_controllers.dynamixel_robot.servo_handler import ServoHandler
from src.utils.movement import PlannedMove, check_move, plan_move_to_angles
from src.utils.movement_exception import MovementException

logger = logging.getLogger(__name__)

# the voltages read on 21 October 2026 with the torque off after a power cycle
REFERENCE_VOLTAGES = {1: 11.6, 2: 11.4, 3: 11.3, 4: 11.3, 5: 11.3, 6: 11.2, 7: 5.1}
# the input voltage every model is rated for, from the ROBOTIS manuals
RATED_RANGES = {
    'XM430-W350': (10.0, 14.8),
    'XM540-W270': (10.0, 14.8),
    'XC430-W240': (6.5, 12.0),
    'XC430-W150': (6.5, 12.0),
    'XL330-M288': (3.7, 6.0),
}
CHAIN = (1, 2, 3, 4, 5, 6)  # the 12 V servos in the order of the chain, the gripper has its own 5 V rail
FIELDS = ('phase', 'pose', 'time', 'servo', 'voltage', 'current_raw', 'current', 'current_unit', 'temperature')
DEFAULT_POSES = ('elbow_out', 'reach', 'low')
SUPPLY_LOW = 0.4  # volts below the reference at servo 1 at rest that point at the supply
LOAD_DROP = 0.3  # volts the drop along the chain has to grow under load to point at the leads or the connectors
READ_ATTEMPTS = 3  # reads of the servos before a failure stops the run


class Stop(Exception):
    """raised to stop the check, with the reason"""


class Reading(NamedTuple):
    phase: str
    pose: str
    time: float  # seconds since the start of the phase
    servo: int
    voltage: float
    current_raw: int
    temperature: int


class VoltageStats(NamedTuple):
    servo: int
    samples: int
    minimum: float
    mean: float
    maximum: float


def statistics(readings: Iterable[Reading]) -> dict[int, VoltageStats]:
    """min, mean and max of the voltage per servo"""
    voltages: dict[int, list[float]] = {}
    for reading in readings:
        voltages.setdefault(reading.servo, []).append(reading.voltage)
    return {servo: VoltageStats(servo, len(values), min(values), float(np.mean(values)), max(values))
            for servo, values in sorted(voltages.items())}


def chain_drops(stats: Mapping[int, VoltageStats], chain: Sequence[int] = CHAIN) -> dict[int, float]:
    """the mean voltage of the first servo of the chain minus that of every other servo of the chain"""
    present = [servo for servo in chain if servo in stats]
    if not present:
        return {}
    first = stats[present[0]].mean
    return {servo: first - stats[servo].mean for servo in present[1:]}


def hop_drops(stats: Mapping[int, VoltageStats], chain: Sequence[int] = CHAIN) -> dict[tuple[int, int], float]:
    """the mean voltage lost between consecutive servos of the chain"""
    present = [servo for servo in chain if servo in stats]
    return {(a, b): stats[a].mean - stats[b].mean for a, b in zip(present, present[1:])}


def largest_hop(hops: Mapping[tuple[int, int], float]) -> tuple[tuple[int, int], float] | None:
    if not hops:
        return None
    hop = max(hops, key=lambda key: hops[key])
    return hop, hops[hop]


def out_of_range(stats: Mapping[int, VoltageStats], models: Mapping[int, str],
                 ranges: Mapping[str, tuple[float, float]] = RATED_RANGES) -> dict[int, str]:
    """servo id to the reason, for the servos with a reading outside the rated range of their model"""
    flags = {}
    for servo, stat in stats.items():
        model = models.get(servo)
        if model is None or model not in ranges:
            continue
        low, high = ranges[model]
        if stat.minimum < low:
            flags[servo] = '{:.1f} V is below the rated {:.1f}-{:.1f} V of the {}'.format(stat.minimum, low, high,
                                                                                         model)
        elif stat.maximum > high:
            flags[servo] = '{:.1f} V is above the rated {:.1f}-{:.1f} V of the {}'.format(stat.maximum, low, high,
                                                                                         model)
    return flags


def against_reference(stats: Mapping[int, VoltageStats],
                      reference: Mapping[int, float] = REFERENCE_VOLTAGES) -> dict[int, float]:
    """the mean voltage minus the reference, per servo that has one"""
    return {servo: stat.mean - reference[servo] for servo, stat in stats.items() if servo in reference}


def conclusions(by_phase: Mapping[str, Mapping[int, VoltageStats]], models: Mapping[int, str],
                reference: Mapping[int, float] = REFERENCE_VOLTAGES, supply_low: float = SUPPLY_LOW,
                load_drop: float = LOAD_DROP, ranges: Mapping[str, tuple[float, float]] = RATED_RANGES) -> list[str]:
    """
    what the phases say about the supply, the chain and the gripper rail
    :param by_phase: phase name to the statistics of that phase, the phases are 'rest', 'hold' and 'motion'
    """
    lines = []
    first, last = CHAIN[0], CHAIN[-1]
    rest = by_phase.get('rest', {})
    loaded = {phase: stats for phase, stats in by_phase.items() if phase != 'rest' and stats}

    if first in rest:
        below = reference[first] - rest[first].mean
        if below > supply_low:
            lines.append('servo {} reads {:.1f} V at rest with the torque off, {:.1f} V below the reference of {:.1f} '
                         'V: the supply or its lead to the first servo is low before any load'.format(
                             first, rest[first].mean, below, reference[first]))
        else:
            lines.append('servo {} reads {:.1f} V at rest, {:+.1f} V against the reference: the supply is fine at '
                         'rest'.format(first, rest[first].mean, -below))

    rest_drop = chain_drops(rest).get(last)
    loaded_drops: dict[str, float] = {}
    for phase, stats in loaded.items():
        drop = chain_drops(stats).get(last)
        if drop is not None:
            loaded_drops[phase] = drop
    if loaded_drops:
        worst_phase = max(loaded_drops, key=lambda phase: loaded_drops[phase])
        worst_drop = loaded_drops[worst_phase]
        hop = largest_hop(hop_drops(loaded[worst_phase]))
        hop_text = ', the largest hop is servo {} to {} with {:.1f} V'.format(hop[0][0], hop[0][1], hop[1]) \
            if hop is not None else ''
        if rest_drop is not None and worst_drop - rest_drop > load_drop:
            lines.append('the drop from servo {} to {} grows from {:.1f} V at rest to {:.1f} V during {}: '
                         'resistance in the leads or the connectors{}'.format(first, last, rest_drop, worst_drop,
                                                                              worst_phase, hop_text))
        elif rest_drop is not None:
            lines.append('the drop from servo {} to {} is {:.1f} V at rest and {:.1f} V during {}: the chain '
                         'itself is fine{}'.format(first, last, rest_drop, worst_drop, worst_phase, hop_text))
        else:
            lines.append('the drop from servo {} to {} is {:.1f} V during {} (no rest phase to compare with){}'
                         .format(first, last, worst_drop, worst_phase, hop_text))
    elif rest_drop is not None:
        hop = largest_hop(hop_drops(rest))
        lines.append('the drop from servo {} to {} is {:.1f} V at rest{}'.format(
            first, last, rest_drop, ', the largest hop is servo {} to {} with {:.1f} V'.format(hop[0][0], hop[0][1],
                                                                                              hop[1])
            if hop is not None else ''))

    for phase, stats in loaded.items():
        dips = [(servo, stat.minimum) for servo, stat in stats.items() if servo in CHAIN
                and models.get(servo) in ranges and stat.minimum < ranges[models[servo]][0]]
        if dips:
            servo, minimum = min(dips, key=lambda dip: dip[1])
            lines.append('the voltage dips to {:.1f} V at servo {} during {}, below the rated minimum of its '
                         'model'.format(minimum, servo, phase))

    gripper = [stats[GRIPPER_ID] for stats in by_phase.values() if GRIPPER_ID in stats]
    if gripper and GRIPPER_ID in reference:
        means = [stat.mean for stat in gripper]
        lines.append('the gripper rail reads {:.1f}-{:.1f} V (reference {:.1f} V)'.format(
            min(means), max(means), reference[GRIPPER_ID]))
    return lines


def format_phase(phase: str, pose: str, stats: Mapping[int, VoltageStats], models: Mapping[int, str],
                 duration: float, reference: Mapping[int, float] = REFERENCE_VOLTAGES) -> str:
    """the table of one phase (and pose) for the console"""
    drops = chain_drops(stats)
    deltas = against_reference(stats, reference)
    flags = out_of_range(stats, models)
    lines = ['{}{}: {:.1f} s, {} samples per servo'.format(
        phase, ' at ' + pose if pose else '', duration, max((s.samples for s in stats.values()), default=0))]
    lines.append('  servo  model        min   mean   max  drop from 1  vs 21 Oct')
    for servo, stat in stats.items():
        drop = '{:5.2f}'.format(drops[servo]) if servo in drops else ('    -' if servo in CHAIN else ' 5 V rail')
        delta = '{:+5.2f}'.format(deltas[servo]) if servo in deltas else '    ?'
        lines.append('  {:<5d}  {:<11s} {:5.1f}  {:5.2f} {:5.1f}  {:>11s}  {:>8s}  {}'.format(
            servo, models.get(servo, '?'), stat.minimum, stat.mean, stat.maximum, drop, delta,
            'OUTSIDE THE RATED RANGE: ' + flags[servo] if servo in flags else ''))
    return '\n'.join(lines)


class Sampler:
    """reads all servos in one packet per bus and keeps the readings, see the module docstring"""

    def __init__(self, handler: ServoHandler, models: Mapping[int, str], temperature_warning: float,
                 writer: csv.DictWriter[str] | None = None, clock: Callable[[], float] = time.perf_counter,
                 stop: StopFlag | None = None) -> None:
        self.handler = handler
        self.models = models
        self.temperature_warning = temperature_warning
        self.writer = writer
        self.clock = clock
        self.stop = stop if stop is not None else StopFlag()
        self.readings: list[Reading] = []
        self.phase_start = 0.0
        self.failed_reads: dict[str, int] = {}  # phase to the number of reads that failed and were retried

    def start_phase(self) -> None:
        self.phase_start = self.clock()

    def read(self, phase: str, pose: str = '') -> list[Reading]:
        """one read of all servos, appended to the readings and the csv; stops on Ctrl+C, READ_ATTEMPTS failed
        reads in a row or a servo above the temperature warning

        A read fails on a timeout or a corrupt status packet (the sdk reports a packet that arrived late or in part
        as corrupt too); one such packet under load is worth knowing but not worth ending the run, so the read is
        retried and the failures are counted per phase for the report.
        """
        if self.stop.requested:
            raise Stop('interrupted')
        states = None
        for _ in range(READ_ATTEMPTS):
            states = self.handler.read_state()
            if states is not None:
                break
            self.failed_reads[phase] = self.failed_reads.get(phase, 0) + 1
        if states is None:
            raise Stop('failed to read the servos {} times in a row'.format(READ_ATTEMPTS))
        now = self.clock() - self.phase_start
        readings = [Reading(phase, pose, now, servo_id, state.voltage, state.current, state.temperature)
                    for servo_id, state in sorted(states.items())]
        self.readings.extend(readings)
        if self.writer is not None:
            for reading in readings:
                current, unit = record_static_load.current_in_units(reading.current_raw,
                                                                    self.models.get(reading.servo))
                self.writer.writerow({'phase': phase, 'pose': pose, 'time': '{:.4f}'.format(now),
                                      'servo': reading.servo, 'voltage': reading.voltage,
                                      'current_raw': reading.current_raw,
                                      'current': '' if current is None else '{:.2f}'.format(current),
                                      'current_unit': unit, 'temperature': reading.temperature})
        warm = [r for r in readings if r.temperature > self.temperature_warning]
        if warm:
            raise Stop('servo {} is at {}C, above the warning of {}C'.format(
                warm[0].servo, warm[0].temperature, self.temperature_warning))
        return readings

    def capture(self, phase: str, pose: str, duration: float) -> None:
        """read for the duration, as fast as the bus allows"""
        end = self.clock() + duration
        while self.clock() < end:
            self.read(phase, pose)

    def stats(self, phase: str, pose: str | None = None) -> dict[int, VoltageStats]:
        return statistics(r for r in self.readings if r.phase == phase and (pose is None or r.pose == pose))


def pose_refusal(pose: np.ndarray, robot: DynamixelRobotController | CombinedRobot) -> str | None:
    """why the planner refuses the pose, or None"""
    try:
        check_move(PlannedMove('pose', pose[np.newaxis], 0.0), robot)
    except MovementException as e:
        return str(e)
    return None


def run_rest(sampler: Sampler, duration: float) -> None:
    """the rest phase: the servos as they are, which should be with the torque off"""
    torque = sampler.handler.read_torque_enabled()
    if torque is None:
        raise Stop('failed to read the torque state of the servos')
    holding = sorted(servo_id for servo_id, enabled in torque.items() if enabled)
    if holding:
        logger.warning('the rest phase expects the torque off, servos %s are holding, their readings are under load',
                       holding)
    print('rest: sampling for {:.0f} s'.format(duration))
    sampler.start_phase()
    sampler.capture('rest', '', duration)


def move_to(robot: DynamixelRobotController | CombinedRobot, pose: np.ndarray, speed: float,
            stop: StopFlag) -> bool:
    try:
        return record_static_load.move_to(robot, pose, speed, lambda: stop.requested)
    except record_static_load.StopRecording as e:
        raise Stop(str(e))


def check_servos(robot: DynamixelRobotController | CombinedRobot) -> None:
    try:
        record_static_load.check_servos(robot, robot.servo_config.temperature_warning)
    except record_static_load.StopRecording as e:
        raise Stop(str(e))


def run_hold(sampler: Sampler, robot: DynamixelRobotController | CombinedRobot, poses: Sequence[tuple[str, np.ndarray]],
             settle: float, duration: float, speed: float, sleep: Callable[[float], None] = time.sleep) -> None:
    """the hold phase: every pose, settled, sampled"""
    for name, pose in poses:
        check_servos(robot)
        print('hold at {} ({} degrees)'.format(name, pose_degrees(pose)))
        if not move_to(robot, pose, speed, sampler.stop):
            continue
        sleep(settle)
        sampler.start_phase()
        sampler.capture('hold', name, duration)


def run_motion(sampler: Sampler, robot: DynamixelRobotController | CombinedRobot,
               poses: Sequence[tuple[str, np.ndarray]], speed: float, sleep: Callable[[float], None] = time.sleep,
               clock: Callable[[], float] = time.perf_counter) -> None:
    """the motion phase: a planned move from the first pose to the second and back, sampled while it streams"""
    if len(poses) < 2:
        print('motion: skipped, it needs two poses the planner accepts')
        return
    (name_a, pose_a), (name_b, pose_b) = poses[0], poses[1]
    check_servos(robot)
    if not move_to(robot, pose_a, speed, sampler.stop):
        print('motion: skipped, the planner refuses the move to {}'.format(name_a))
        return
    sampler.start_phase()
    dt = 1.0 / global_constants.steps_per_second
    for start, target, label in ((pose_a, pose_b, name_a + '-' + name_b), (pose_b, pose_a, name_b + '-' + name_a)):
        try:
            move = plan_move_to_angles(robot, start, target, angular_speed=speed)
        except MovementException as e:
            print('motion: skipped, {}'.format(e))
            return
        print('motion: {} in {:.1f} s'.format(label, move.duration))
        next_step = clock()
        for angles in move.angles:
            robot.move_servos(angles)
            sampler.read('motion', label)
            next_step += dt
            sleep(max(0.0, next_step - clock()))


def print_report(sampler: Sampler, models: Mapping[int, str], phases: Sequence[tuple[str, str, float]]) -> None:
    """
    the tables per phase and pose and the conclusions
    :param phases: (phase, pose, duration) of every capture that was done; an empty pose takes every reading of
        the phase (the motion phase labels its readings with the move, e.g. 'elbow_out-reach')
    """
    by_phase: dict[str, dict[int, VoltageStats]] = {}
    for phase, pose, duration in phases:
        stats = sampler.stats(phase, pose or None)
        if not stats:
            continue
        print()
        print(format_phase(phase, pose, stats, models, duration))
        by_phase[phase] = sampler.stats(phase)
    print()
    for line in conclusions(by_phase, models):
        print(line)
    for phase, failures in sampler.failed_reads.items():
        total = max((s.samples for s in by_phase.get(phase, {}).values()), default=0) + failures
        print('{} of {} reads failed during {} and were retried (a timeout or a corrupt status packet); failures '
              'only under load point at noise or a supply dip on the bus'.format(failures, total, phase))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='check the supply voltage along the servo chain')
    parser.add_argument('--duration', type=float, default=5.0, help='seconds of sampling per phase and pose')
    parser.add_argument('--settle', type=float, default=2.0, help='seconds to wait at a pose before sampling')
    parser.add_argument('--speed', type=float, default=0.5, help='joint speed of the planned moves in rad/s')
    parser.add_argument('--pose', nargs='+', default=list(DEFAULT_POSES),
                        help='poses of the hold phase, the first two are the motion phase: {} or 6 comma separated '
                             'degrees'.format(', '.join(POSES)))
    parser.add_argument('--no-rest', action='store_true', help='skip the rest phase')
    parser.add_argument('--no-hold', action='store_true', help='skip the hold phase')
    parser.add_argument('--no-motion', action='store_true', help='skip the motion phase')
    parser.add_argument('--floor', type=float, default=None,
                        help='cm above the table the wrist and gripper tip stay, default WorkSpaceLimits.z_min ({})'
                        .format(WorkSpaceLimits.z_min))
    parser.add_argument('--output-dir', default='.', help='where the csv goes')
    parser.add_argument('--yes', action='store_true', help="don't ask before enabling the servos")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

    try:
        poses = [(name, parse_pose(name)) for name in args.pose]
    except argparse.ArgumentTypeError as e:
        print(e)
        return 1
    try:
        robot = get_robot()
    except DynamixelError as e:
        print(e)
        return 1
    if not isinstance(robot, CombinedRobot):
        print('the supply can only be checked on the real robot, use_simulation is set')
        return 1
    if args.floor is not None:
        robot.workspace_limits = record_static_load.limits_with_floor(args.floor, robot.workspace_limits)
    config = robot.servo_config
    accepted = []
    for name, pose in poses:
        refused = pose_refusal(pose, robot)
        if refused is None:
            accepted.append((name, pose))
        else:
            print('skipping pose {} ({} degrees): {}'.format(name, pose_degrees(pose), refused))
    moving = (not args.no_hold and accepted) or (not args.no_motion and len(accepted) >= 2)

    path = '{}/supply_voltage_{}.csv'.format(args.output_dir, datetime.now().strftime('%Y%m%d_%H%M'))
    stop = StopFlag()
    stop.install()
    enabled = False
    phases: list[tuple[str, str, float]] = []
    sampler: Sampler | None = None
    try:
        with open(path, 'w', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=FIELDS)
            writer.writeheader()
            sampler = Sampler(robot.servo_handler, config.models, config.temperature_warning, writer, stop=stop)
            if not args.no_rest:
                run_rest(sampler, args.duration)
                phases.append(('rest', '', args.duration))
            if moving:
                print('the arm will move to {}, nothing may be in its way'.format(
                    ', '.join('{} ({} degrees)'.format(name, pose_degrees(pose)) for name, pose in accepted)))
                if not args.yes and input('type yes to enable the servos and move: ').strip().lower() != 'yes':
                    raise Stop('not enabling the servos')
                if not robot.enable_servos():
                    raise Stop('failed to enable the servos')
                enabled = True
            if not args.no_hold and accepted:
                run_hold(sampler, robot, accepted, args.settle, args.duration, args.speed)
                phases.extend(('hold', name, args.duration) for name, _ in accepted)
            if not args.no_motion and len(accepted) >= 2:
                run_motion(sampler, robot, accepted, args.speed)
                phases.append(('motion', '', 0.0))
    except Stop as e:
        print('\nstopped: {}'.format(e))
    except KeyboardInterrupt:
        print('\nkilled')
    finally:
        stop.uninstall()
    if sampler is not None and sampler.readings:
        # the motion phase has no fixed duration, report what it took
        phases = [(phase, pose, duration if phase != 'motion' else
                   max((r.time for r in sampler.readings if r.phase == 'motion'), default=0.0))
                  for phase, pose, duration in phases]
        print_report(sampler, config.models, phases)
    print('samples in {}'.format(path))
    if enabled:
        record_static_load.disable_torque_on_request(robot)
    return 0


if __name__ == '__main__':
    sys.exit(main())
