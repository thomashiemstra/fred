"""
Measures what one servo of the wrist needs for its moves, to choose its goal current and profile so it doesn't
overload: the wrist roll (servo 6) by default, or the gripper (--servo 7). Made for the XL330-M288 on the 5 V line
(November 2026) and the XC330-T288-T on the 12 V bus that replaced it; the model is read from the servo and sets the
voltage limits and the largest goal current. Run it from the project root with the arm powered and nothing in the way
of the gripper; only the servo under test is enabled:
    python -m src.robot_controllers.dynamixel_robot.calibration.wrist_roll_limits [--servo 6]
        [--deltas 45 -45 90 -90 180] [--goal-currents 150 250 400 600] [--profiles 200,50 300,100]
        [--range 1000 3000] [--output wrist_roll_limits.csv] [--yes]

The servo runs in current-based position mode, so it never draws more than its goal current. For every profile
(velocity,acceleration in the units of the servo, default the one of servo_config.yaml) and every goal current, from
low to high, it turns by every delta (degrees) from where it starts and back, and records per move: whether it
arrived, how long it took, the peak and mean current, the lowest input voltage, the temperature and the hardware
errors. A move is stopped, the servo holding where it is, when the voltage drops below the model's minimum (a sagging
supply), when it stops making progress before the target (the goal current is too low for the move, or the gripper or
its cable hits something) or when it takes much longer than its profile. A hardware error or a servo warmer than
MAX_TEMPERATURE ends the run.

The deltas have to stay inside the range of the servo in servo_config.yaml; --range MIN MAX (positions) replaces it
for a servo that isn't calibrated yet. For the gripper use small deltas and low goal currents until its open and closed
positions are known: a delta beyond a mechanical end stalls there, at the goal current.

At the end it prints per profile the lowest goal current at which every move arrived, the peak current the moves
took and the lowest voltage, and a goal current for servo_config.yaml: 1.5 times the peak current the moves needed,
at least the lowest goal current that worked, and not above what the supply held at the model's healthy voltage.
Then it puts the configured profile and goal current back and disables the torque of the servo (the roll and the
gripper carry no load, nothing drops). Ctrl+C holds the servo where it is and does the same.

Check the cable of the gripper before turning the roll by 180 degrees: the deltas are from where the servo starts.
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import time
from collections.abc import Sequence
from typing import Any, NamedTuple

from src.robot_controllers.dynamixel_robot import dynamixel_x_config as cfg
from src.robot_controllers.dynamixel_robot.dynamixel_utils import setup_dynamixel_handlers
from src.robot_controllers.dynamixel_robot.servo_config import load_servo_config
from src.robot_controllers.dynamixel_robot.servo_handler import ServoHandler, ServoState

logger = logging.getLogger(__name__)


class ModelLimits(NamedTuple):
    name: str
    min_voltage: float  # volts, a move is stopped below this
    healthy_voltage: float  # volts, the recommended goal current keeps the supply above this
    max_current: int  # mA, the current limit of the model
    supply: str  # what feeds it, for the messages
    goal_currents: tuple[int, ...]  # mA, the default goal currents to try


# by model number; the XL330 shuts down below 3.5 V, the XC330-T288 is rated for 6.5 to 12 V (both 1 mA per unit)
MODELS = {
    1200: ModelLimits('XL330-M288', 4.5, 4.75, 1750, 'the 5 V line', (150, 250, 400, 600, 800)),
    1220: ModelLimits('XC330-T288', 10.5, 11.0, 910, 'the 12 V supply', (150, 250, 400, 600)),
}
XL330_M288 = 1200
SERVO_ID = 6
POSITIONS_PER_DEGREE = 4096 / 360.0
DEFAULT_DELTAS = (45.0, -45.0, 90.0, -90.0, 180.0)  # degrees
MIN_VOLTAGE = MODELS[XL330_M288].min_voltage
HEALTHY_VOLTAGE = MODELS[XL330_M288].healthy_voltage
MAX_TEMPERATURE = 55  # degrees celsius, the run ends above this; both models shut down at their limit of 70
ARRIVED = 10  # positions from the target, about 0.9 degrees
STAYED = 20  # positions: a move only arrived when the servo stays this close to the target while it settles
STALL_TIME = 0.5  # seconds without progress before the target that stop a move
STALL_POSITIONS = 2  # positions of movement that count as progress
# seconds after arriving, the holding current is measured then and the servo has to stay at the target: with gains
# that are too high it passes the target and swings around it (the XC330 at servo 6 with P 1200 and D 0, 11 November)
SETTLE = 0.5
READ_ATTEMPTS = 3  # failed reads of the servo in a row that end the run
MARGIN = 1.5  # the recommended goal current over the peak current the moves needed
RPM_PER_VELOCITY_UNIT = 0.229
RPM_PER_MINUTE_PER_ACCELERATION_UNIT = 214.577
FIELDS = ('velocity', 'acceleration', 'goal_current', 'delta', 'start', 'target', 'arrived', 'reason', 'duration',
          'expected', 'peak_current', 'mean_current', 'holding_current', 'min_voltage', 'start_voltage',
          'max_temperature', 'final_error')


class Stop(Exception):
    """ends the run, with the reason"""


class MoveResult(NamedTuple):
    velocity: int
    acceleration: int
    goal_current: int
    delta: float  # degrees, signed; the move back has the opposite sign
    start: int  # positions
    target: int
    arrived: bool
    reason: str | None  # why the move was stopped, None when it arrived
    duration: float  # seconds
    expected: float  # seconds, the time of the profile
    peak_current: int  # mA, absolute
    mean_current: float
    holding_current: float  # mA, mean absolute current after arriving
    min_voltage: float
    start_voltage: float
    max_temperature: int
    final_error: int  # positions, target - present at the end


def profile_time(delta_degrees: float, velocity: int, acceleration: int) -> float:
    """seconds a move of the delta takes with the profile (a trapezoid, or a triangle when it is too short to reach
    the velocity); 0 velocity or acceleration means unlimited, the time is then unknown and 0 is returned"""
    if velocity <= 0 or acceleration <= 0:
        return 0.0
    revolutions = abs(delta_degrees) / 360.0
    v = velocity * RPM_PER_VELOCITY_UNIT / 60.0  # revolutions per second
    a = acceleration * RPM_PER_MINUTE_PER_ACCELERATION_UNIT / 3600.0  # revolutions per second squared
    if revolutions < v * v / a:  # never reaches the velocity
        return 2.0 * math.sqrt(revolutions / a)
    return revolutions / v + v / a


def move_timeout(expected: float) -> float:
    return 2.0 * expected + 1.5 if expected > 0 else 4.0


def recommend(results: Sequence[MoveResult], healthy_voltage: float = HEALTHY_VOLTAGE,
              supply: str = 'the 5 V line') -> tuple[int | None, str]:
    """the goal current for servo_config.yaml from the moves of one profile, and how it was chosen; None when no
    goal current let every move arrive"""
    by_goal: dict[int, list[MoveResult]] = {}
    for r in results:
        by_goal.setdefault(r.goal_current, []).append(r)
    working = sorted(g for g, moves in by_goal.items() if all(m.arrived for m in moves))
    if not working:
        return None, 'no goal current let every move arrive'
    lowest = working[0]
    # what the moves need: the peak current at the highest goal current where every move arrived; a peak close to
    # that goal current means the moves were still limited by it
    highest = working[-1]
    needed = max(m.peak_current for m in by_goal[highest])
    limited = needed >= 0.9 * highest
    healthy = [g for g, moves in by_goal.items() if min(m.min_voltage for m in moves) >= healthy_voltage]
    goal = max(lowest, int(math.ceil(MARGIN * needed / 50.0)) * 50)
    note = 'the moves took at most {} mA at a goal current of {} mA, {} times that is {} mA'.format(
        needed, highest, MARGIN, int(math.ceil(MARGIN * needed / 50.0)) * 50)
    if limited:
        note += '; that peak is close to the goal current, so the moves were still limited by it: test higher'
    ceiling = max((g for g in healthy if g >= lowest), default=None)
    if ceiling is not None and goal > ceiling:
        note += '; capped at {} mA, the highest goal current at which {} stayed above {} V'.format(
            ceiling, supply, healthy_voltage)
        goal = ceiling
    elif ceiling is None:
        note += '; {} dropped below {} V at every goal current that worked, check it'.format(supply, healthy_voltage)
    return goal, note


class WristRollTest:
    """the moves of one servo, see the module docstring"""

    def __init__(self, handler: ServoHandler, clock: Any = time.perf_counter, sleep: Any = time.sleep,
                 servo_id: int = SERVO_ID, limits: ModelLimits = MODELS[XL330_M288],
                 position_range: tuple[int, int] | None = None) -> None:
        """
        :param position_range: (min, max) positions the moves have to stay in, default the range of the servo in
            servo_config.yaml
        """
        self.handler = handler
        self.servo_id = servo_id
        self.servo = handler.get_servo(servo_id)
        self.limits = limits
        self.position_range = position_range if position_range is not None else (self.servo.min_position,
                                                                                 self.servo.max_position)
        self.clock = clock
        self.sleep = sleep
        self.read_failures = 0
        self.results: list[MoveResult] = []  # the moves so far, kept when a Stop ends the run

    def state(self) -> ServoState:
        for _ in range(READ_ATTEMPTS):
            states = self.handler.read_state([self.servo_id])
            if states is not None:
                return states[self.servo_id]
            self.read_failures += 1
        raise Stop('failed to read servo {} {} times in a row'.format(self.servo_id, READ_ATTEMPTS))

    def check_errors(self) -> None:
        errors = self.handler.read_hardware_errors([self.servo_id])
        if errors is None:
            raise Stop('failed to read the hardware error status of servo {}'.format(self.servo_id))
        if errors:
            raise Stop('servo {} shut down: {}; reboot it (or power cycle it) before running again'.format(
                self.servo_id, ', '.join(errors[self.servo_id])))

    def hold(self) -> int:
        """hold where the servo is, returns the position"""
        position = self.state().position
        self.handler.move_servo_to_pos(self.servo_id, position)
        return position

    def back_to(self, start: int, velocity: int, acceleration: int, delta: float) -> None:
        """after a move that didn't arrive: back to the start, so the next move starts where it should"""
        if not self.handler.move_servo_to_pos(self.servo_id, start):
            raise Stop('failed to send servo {} back to its start'.format(self.servo_id))
        deadline = self.clock() + move_timeout(profile_time(delta, velocity, acceleration))
        while self.clock() < deadline:
            state = self.state()
            if abs(start - state.position) <= ARRIVED:
                return
            if state.voltage < self.limits.min_voltage:
                self.hold()
                raise Stop('the voltage dropped to {:.1f} V on the way back to the start'.format(state.voltage))
        self.hold()
        raise Stop('servo {} did not get back to its start at position {}, it holds at {}'.format(
            self.servo_id, start, self.state().position))

    def move(self, start: int, target: int, velocity: int, acceleration: int, goal_current: int,
             delta: float) -> MoveResult:
        expected = profile_time(delta, velocity, acceleration)
        timeout = move_timeout(expected)
        first = self.state()
        currents: list[int] = []
        min_voltage, max_temperature = first.voltage, first.temperature
        reason: str | None = None
        arrived = False
        if not self.handler.move_servo_to_pos(self.servo_id, target):
            raise Stop('failed to send the goal position to servo {}'.format(self.servo_id))
        begin = self.clock()
        progress_position, progress_time = first.position, begin
        state = first
        while True:
            state = self.state()
            now = self.clock()
            currents.append(abs(state.current))
            min_voltage = min(min_voltage, state.voltage)
            max_temperature = max(max_temperature, state.temperature)
            error = target - state.position
            if abs(state.position - progress_position) >= STALL_POSITIONS:
                progress_position, progress_time = state.position, now
            profile_done = not state.moving_status & cfg.MOVING_STATUS_PROFILE_ONGOING
            if abs(error) <= ARRIVED and profile_done:
                arrived = True
                break
            if state.voltage < self.limits.min_voltage:
                reason = 'the voltage dropped to {:.1f} V'.format(state.voltage)
                break
            if now - progress_time > STALL_TIME:
                reason = 'stalled {} positions before the target (the goal current is too low, or it hits ' \
                         'something)'.format(error)
                break
            if now - begin > timeout:
                reason = 'not there after {:.1f} s ({:.2f} s expected)'.format(now - begin, expected)
                break
        duration = self.clock() - begin
        if not arrived:
            self.hold()
        # the holding current once it is there, and whether it stays there
        holding: list[int] = []
        furthest = 0
        settle_end = self.clock() + SETTLE
        while self.clock() < settle_end:
            settled = self.state()
            holding.append(abs(settled.current))
            min_voltage = min(min_voltage, settled.voltage)
            furthest = max(furthest, abs(target - settled.position))
            state = settled
        if arrived and furthest > STAYED:
            arrived = False
            reason = 'reached the target but did not stay: up to {} positions from it in the {} s after (it swings ' \
                     'around the target, the gains are too high or the damping too low)'.format(furthest, SETTLE)
        return MoveResult(velocity=velocity, acceleration=acceleration, goal_current=goal_current, delta=delta,
                          start=start, target=target, arrived=arrived, reason=reason, duration=duration,
                          expected=expected, peak_current=max(currents, default=0),
                          mean_current=sum(currents) / max(len(currents), 1),
                          holding_current=sum(holding) / max(len(holding), 1), min_voltage=min_voltage,
                          start_voltage=first.voltage, max_temperature=max_temperature,
                          final_error=target - state.position)

    def run(self, deltas: Sequence[float], goal_currents: Sequence[int], profiles: Sequence[tuple[int, int]],
            writer: csv.DictWriter[str] | None = None) -> list[MoveResult]:
        """every profile, every goal current from low to high, every delta out and back; the results so far stay in
        self.results when a Stop ends the run"""
        self.results = []
        start = self.hold()
        low, high = self.position_range
        targets = []
        for delta in deltas:
            target = start + int(round(delta * POSITIONS_PER_DEGREE))
            if not low <= target <= high:
                print('skipping {:+.0f} degrees: position {} is outside the range of servo {} [{}, {}]'.format(
                    delta, target, self.servo_id, low, high))
                continue
            targets.append((delta, target))
        if not targets:
            raise Stop('no delta stays inside the range of servo {} [{}, {}] from position {}'.format(
                self.servo_id, low, high, start))
        for velocity, acceleration in profiles:
            if not self.handler.set_profile(self.servo_id, velocity, acceleration):
                raise Stop('failed to set the profile of servo {}'.format(self.servo_id))
            for goal_current in sorted(goal_currents):
                if not self.handler.set_goal_current(self.servo_id, goal_current):
                    raise Stop('failed to set the goal current of servo {}'.format(self.servo_id))
                print('profile {},{} goal current {} mA'.format(velocity, acceleration, goal_current))
                sagged = False
                for delta, target in targets:
                    for out_and_back in ((start, target, delta), (target, start, -delta)):
                        result = self.move(*out_and_back[:2], velocity, acceleration, goal_current, out_and_back[2])
                        self.results.append(result)
                        if writer is not None:
                            writer.writerow(result._asdict())
                        print('  {:+6.0f} deg: {} in {:.2f} s ({:.2f} expected), current peak {} mean {:.0f} holding '
                              '{:.0f} mA, voltage {:.2f} -> {:.2f} V, {} C'.format(
                                  result.delta, 'arrived' if result.arrived else result.reason, result.duration,
                                  result.expected, result.peak_current, result.mean_current, result.holding_current,
                                  result.start_voltage, result.min_voltage, result.max_temperature))
                        self.check_errors()
                        if result.max_temperature > MAX_TEMPERATURE:
                            raise Stop('servo {} is at {} C, let it cool down'.format(self.servo_id,
                                                                                     result.max_temperature))
                        sagged |= result.reason is not None and result.reason.startswith('the voltage')
                        if not result.arrived:
                            self.back_to(start, velocity, acceleration, delta)
                            break
                if sagged:
                    print('  {} sagged below {} V, not trying higher goal currents with this profile'.format(
                        self.limits.supply, self.limits.min_voltage))
                    break
        return self.results


def print_summary(results: Sequence[MoveResult], limits: ModelLimits = MODELS[XL330_M288]) -> None:
    profiles = list(dict.fromkeys((r.velocity, r.acceleration) for r in results))
    for velocity, acceleration in profiles:
        moves = [r for r in results if (r.velocity, r.acceleration) == (velocity, acceleration)]
        print('\nprofile velocity {}, acceleration {}:'.format(velocity, acceleration))
        for goal_current in sorted({r.goal_current for r in moves}):
            at = [r for r in moves if r.goal_current == goal_current]
            print('  goal current {:4d} mA: {}/{} moves arrived, current peak {} mA, holding {:.0f} mA, lowest voltage '
                  '{:.2f} V{}'.format(goal_current, sum(r.arrived for r in at), len(at),
                                      max(r.peak_current for r in at), max(r.holding_current for r in at),
                                      min(r.min_voltage for r in at),
                                      '' if all(r.arrived for r in at)
                                      else ', ' + str(next(r.reason for r in at if not r.arrived))))
        goal, note = recommend(moves, limits.healthy_voltage, limits.supply)
        if goal is None:
            print('  no recommendation: {}'.format(note))
        else:
            print('  recommended: goal_current: {} with profile: {{velocity: {}, acceleration: {}}} ({})'.format(
                goal, velocity, acceleration, note))


def parse_profiles(values: Sequence[str]) -> list[tuple[int, int]]:
    profiles = []
    for value in values:
        try:
            velocity, acceleration = (int(part) for part in value.split(','))
        except ValueError:
            raise argparse.ArgumentTypeError('a profile is velocity,acceleration, got {!r}'.format(value))
        if velocity < 0 or acceleration < 0:
            raise argparse.ArgumentTypeError('the profile velocity and acceleration can not be negative')
        profiles.append((velocity, acceleration))
    return profiles


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='measure what a servo of the wrist needs for its moves')
    parser.add_argument('--servo', type=int, default=SERVO_ID, help='the servo id, 6 (wrist roll) or 7 (gripper)')
    parser.add_argument('--deltas', type=float, nargs='+', default=list(DEFAULT_DELTAS),
                        help='degrees to turn from where the servo starts, and back')
    parser.add_argument('--goal-currents', type=int, nargs='+', default=None,
                        help='mA, tried from low to high, default a few up to about two thirds of the model limit')
    parser.add_argument('--profiles', nargs='+', default=None,
                        help='velocity,acceleration pairs in the units of the servo, default the yaml profile')
    parser.add_argument('--range', type=int, nargs=2, default=None, metavar=('MIN', 'MAX'),
                        help='positions the moves stay in instead of the range in the yaml, for an uncalibrated servo')
    parser.add_argument('--output', default=None, help='a csv file with every move')
    parser.add_argument('--yes', action='store_true', help="don't ask before enabling the servo")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    servo_id = args.servo

    config = load_servo_config()
    if servo_id not in config.servos:
        print('servo {} is not in servo_config.yaml'.format(servo_id))
        return 1
    servo = config.servos[servo_id]
    try:
        profiles = parse_profiles(args.profiles) if args.profiles else [(servo.profile_velocity,
                                                                          servo.profile_acceleration)]
    except argparse.ArgumentTypeError as e:
        print(e)
        return 1
    if args.range is not None and args.range[0] >= args.range[1]:
        print('--range is MIN MAX with MIN below MAX')
        return 1
    port_handler, packet_handler = setup_dynamixel_handlers(None, cfg, (servo_id,))
    handler = ServoHandler({servo_id: servo}, cfg, port_handler, packet_handler, buses=[(servo_id,)])
    model, comm_result, _ = packet_handler.ping(port_handler, servo_id)
    if comm_result != 0:
        print('servo {} does not answer'.format(servo_id))
        return 1
    if model not in MODELS:
        print('servo {} is model {}, this script knows {}'.format(servo_id, model, ', '.join(
            '{} ({})'.format(m.name, number) for number, m in MODELS.items())))
        return 1
    limits = MODELS[model]
    goal_currents = args.goal_currents if args.goal_currents is not None else list(limits.goal_currents)
    if any(not 0 < g <= limits.max_current for g in goal_currents):
        print('the goal currents have to be between 1 and {} mA, the current limit of the {}'.format(
            limits.max_current, limits.name))
        return 1
    test = WristRollTest(handler, servo_id=servo_id, limits=limits,
                         position_range=(args.range[0], args.range[1]) if args.range is not None else None)
    try:
        test.check_errors()
    except Stop as e:
        print(e)
        return 1
    first = test.state()
    print('servo {} ({}): position {}, {:.1f} V, {} C, range [{}, {}]; deltas {} degrees, goal currents {} mA, '
          'profiles {}'.format(servo_id, limits.name, first.position, first.voltage, first.temperature,
                               *test.position_range, ' '.join('{:+.0f}'.format(d) for d in args.deltas),
                               ' '.join(str(g) for g in sorted(goal_currents)),
                               ' '.join('{},{}'.format(*p) for p in profiles)))
    print('servo {} turns by up to {:.0f} degrees from where it is now and back; check the cable of the gripper and '
          'keep your fingers clear'.format(servo_id, max(abs(d) for d in args.deltas)))
    if not args.yes and input('type yes to enable servo {} and start: '.format(servo_id)).strip().lower() != 'yes':
        return 0

    output = open(args.output, 'w', newline='') if args.output else None
    writer = None
    if output is not None:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
    results: list[MoveResult] = []
    try:
        # the operating mode can only be changed with the torque off
        if not (handler.set_torque(False, [servo_id]) and handler.set_configured_operating_mode()
                and handler.set_pid() and handler.set_configured_goal_current()
                and handler.set_profile_velocity_and_acceleration()):
            print('failed to set up servo {}'.format(servo_id))
            return 1
        test.hold()
        if not handler.set_torque(True, [servo_id]):
            print('failed to enable servo {}'.format(servo_id))
            return 1
        test.run(args.deltas, goal_currents, profiles, writer)
    except Stop as e:
        print('\nstopped: {}'.format(e))
    except KeyboardInterrupt:
        print('\nstopped')
        try:
            test.hold()
        except Stop:
            pass
    finally:
        results = test.results
        restored = handler.set_profile_velocity_and_acceleration() & handler.set_configured_goal_current()
        # the writes aren't acknowledged, the torque is read back
        torque_off = handler.set_torque(False, [servo_id]) and handler.read_torque_enabled([servo_id]) == {
            servo_id: False}
        if restored and torque_off:
            print('the profile and goal current of the yaml are back, the torque of servo {} is off'.format(servo_id))
        else:
            print('COULD NOT {} of servo {}: it does not answer. Power cycle it; until then it may hold its '
                  'position with the test settings'.format(
                      'turn off the torque' if not torque_off else 'restore the profile and goal current', servo_id))
        port_handler.closePort()
        if output is not None:
            output.close()
    if results:
        print_summary(results, limits)
    if test.read_failures or handler.failed_reads:
        print('{} reads of servo {} failed and were tried again'.format(test.read_failures + handler.failed_reads,
                                                                        servo_id))
    return 0


if __name__ == '__main__':
    sys.exit(main())
