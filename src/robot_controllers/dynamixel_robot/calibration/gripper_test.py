"""
Test script that moves the physical gripper (servo 7), the other servos are not enabled. Run it from the project root.
There are two modes:

    python -m src.robot_controllers.dynamixel_robot.calibration.gripper_test --calibrate [--current 60] [--step 10]
        [--max-travel 1500] [--closing up|down] [--write] [--yes]
    python -m src.robot_controllers.dynamixel_robot.calibration.gripper_test [--yes]

--calibrate finds the closed position of the gripper, which servo_config.yaml needs before the gripper can be driven
by state (the XC330-T288-T that replaced the XL330-M288 on 11 November 2026 has not been calibrated). Put the gripper
open where you want its open position, with the torque off and nothing between the fingers. It reads the open
position, enables only servo 7 with a low goal current (--current mA, at most the goal_current of the yaml), closes it
in steps of --step positions every 50 ms until it stops against its mechanical end (it is stopped when the goal is
over 40 positions ahead and it advanced less than 2 positions in 0.5 s), opens it again and disables the torque. It
reports the open and the closed position, the travel and the current, and the line for the yaml:
    positions: [<open>, <closed>]
--write puts that line in servo_config.yaml (asking first unless --yes is given), nothing else in the file changes.
--closing is the direction of the closed position, by default that of the positions in the yaml. Ctrl+C, an error
or a warm servo stop it: the servo holds where it is and the torque is disabled.

Without --calibrate it runs the interactive tests, which need calibrated positions (they refuse to start when the
gripper stands more than 50 positions outside the range of the yaml; run --calibrate first):
1. closes the gripper without an object, to check whether it pushes against its mechanical end at the closed end
2. optionally squeezes an object with increasing goal currents (up to the current limit of the model, 910mA for the
   XC330-T288-T) and reports the current, the position, the temperature and the hardware errors. It stops at a
   hardware error or when the gripper gets warm
3. squeezes an object with the target a few states beyond the object, like the xbox controller does, to check how
   much of the goal current that uses
4. disables the torque of the gripper

The gripper is open at state 0 and closed at state 100.

Ctrl+C disables the torque of the gripper and exits.
"""
import argparse
import logging
import re
import sys
from collections.abc import Callable, Sequence
from statistics import median
from time import sleep, perf_counter
from typing import NamedTuple, TypeVar

import yaml

from src.robot_controllers.dynamixel_robot import dynamixel_x_config as cfg
from src.robot_controllers.dynamixel_robot.calibration.wrist_roll_limits import MODELS
from src.robot_controllers.dynamixel_robot.dynamixel_robot_controller import ARM_IDS, GRIPPER_ID
from src.robot_controllers.dynamixel_robot.dynamixel_utils import setup_dynamixel_handlers, to_signed
from src.robot_controllers.dynamixel_robot.servo import Servo
from src.robot_controllers.dynamixel_robot.servo_config import load_servo_config, parse_servo_config, \
    DEFAULT_SERVO_CONFIG_PATH
from src.robot_controllers.dynamixel_robot.servo_handler import ServoHandler

# mA, the goal currents of the squeeze tests; they are capped at the current limit of the model of the gripper (see
# squeeze_currents), which is 910mA for the XC330-T288-T that is in the gripper now (the XL330-M288 before it took 1750)
GOAL_CURRENTS = [200, 400, 600, 800, 1000, 1200, 1500]
# the current limit in mA of models that wrist_roll_limits.MODELS doesn't have, by the start of the model name; the
# values are from the e-manuals of the XC330 and XL330 families, check them before a new model is used. A model that
# is in neither table gets the lowest limit here
OTHER_CURRENT_LIMITS = {'XC330-M288': 910, 'XC330-M181': 910, 'XL330-M077': 1750}
DEFAULT_CURRENT_LIMIT = 910  # mA
# the squeeze tests stop when the gripper gets warmer than this, it shuts down at 70C
MAX_TEMPERATURE = 55  # degrees celsius
# how far the target is beyond the object, the xbox controller uses GRIPPER_MAX_LEAD
LEADS = [10, 20, 25]  # gripper states
SQUEEZE_TIME = 3.0  # seconds
MOVE_TIMEOUT = 5.0  # seconds
OPEN_STATE = 0
CLOSED_STATE = 100
# the interactive tests refuse to start when the gripper stands further outside the positions of the yaml than this,
# which is how an uncalibrated gripper looks
CALIBRATION_MARGIN = 50  # positions

DEFAULT_CALIBRATION_CURRENT = 60  # mA, gentler than the goal current of a grip
DEFAULT_STEP = 10  # positions the goal moves every STEP_TIME while closing
DEFAULT_MAX_TRAVEL = 1500  # positions from the open position
STEP_TIME = 0.05  # seconds between goal steps
CONTACT_LAG = 40  # positions the goal has to be ahead of the gripper before a stop counts as contact
STALL_TIME = 0.5  # seconds the gripper has to advance less than STALL_POSITIONS in to be stopped
STALL_POSITIONS = 2
MIN_TRAVEL = 30  # positions, a gripper that stops sooner did not move in this direction
SETTLE_TIME = 0.3  # seconds after the goal is back at the present position, before the closed position is read
POSITION_SAMPLES = 5  # reads the median of which is the open or the closed position
READ_ATTEMPTS = 3  # failed reads in a row that stop the calibration
CHECK_EVERY = 10  # steps between the checks of the hardware errors and the temperature
LIMIT_WAIT = 5.0  # seconds the gripper may take to catch up with the goal at the end of the travel
OPEN_TIMEOUT = 3.0  # seconds to wait for the gripper to be back at the open position
ARRIVED = 10  # positions from the open position that count as back
TORQUE_OFF_ATTEMPTS = 3
POSITIONS_PER_TURN = 4096
CONTACT = 'contact'
NO_MOVE = 'no move'
NO_CONTACT = 'no contact'

T = TypeVar('T')


class Stop(Exception):
    """ends the calibration, with the reason"""


class Status(NamedTuple):
    position: int
    current: int  # mA, absolute
    temperature: int  # degrees celsius
    voltage: float
    errors: list[str]  # names of the hardware errors
    torque: bool


class ContactResult(NamedTuple):
    outcome: str  # CONTACT, NO_MOVE (stopped within MIN_TRAVEL positions) or NO_CONTACT (the whole travel was free)
    position: int  # the closed position (median of reads) after a stop, else the last position read
    goal: int  # the last goal that was written
    peak_current: int  # mA, absolute, while closing
    contact_current: int | None  # mA, absolute, when the gripper was stopped, None without a stop


def read_present_current(handler: ServoHandler) -> int | None:
    current, comm_result, error = handler.packet_handler.read2ByteTxRx(
        handler.port_handler, GRIPPER_ID, cfg.ADDR_PRESENT_CURRENT)
    if comm_result != 0:
        return None
    return to_signed(current, cfg.LEN_PRESENT_CURRENT)


def gripper_state(handler: ServoHandler) -> float | None:
    if not handler.read_current_pos():
        return None
    servo = handler.get_servo(GRIPPER_ID)
    return servo.get_angle_from_position(servo.current_position)


def print_status(handler: ServoHandler, name: str) -> dict[int, list[str]] | None:
    state = gripper_state(handler)
    current = read_present_current(handler)
    errors = handler.read_hardware_errors()
    temperature_and_voltage = handler.read_temperature_and_voltage() or {}
    temperature, voltage = temperature_and_voltage.get(GRIPPER_ID, (None, None))
    print('{}: state={} position={} current={}mA temperature={}C voltage={}V hardware errors={}'.format(
        name, None if state is None else round(state, 1), handler.get_servo(GRIPPER_ID).current_position, current,
        temperature, voltage, errors))
    return errors


def read_temperature(handler: ServoHandler) -> int | None:
    temperature_and_voltage = handler.read_temperature_and_voltage()
    if temperature_and_voltage is None:
        return None
    return temperature_and_voltage[GRIPPER_ID][0]


def move_and_wait(handler: ServoHandler, state: float) -> int:
    """move the gripper to the state and wait until it stops moving, returns the peak current while moving"""
    handler.set_angle(GRIPPER_ID, state)
    handler.move_to_angles()
    peak_current = 0
    previous_position = None
    stable_since = perf_counter()
    start = perf_counter()
    while perf_counter() - start < MOVE_TIMEOUT:
        sleep(0.05)
        current = read_present_current(handler)
        if current is not None:
            peak_current = max(peak_current, abs(current))
        if not handler.read_current_pos():
            continue
        position = handler.get_servo(GRIPPER_ID).current_position
        if position != previous_position:
            previous_position = position
            stable_since = perf_counter()
        elif perf_counter() - stable_since > 0.3:
            break
    return peak_current


def measure_current(handler: ServoHandler, duration: float) -> tuple[float, int]:
    """returns the average and peak absolute current over the duration"""
    currents = []
    start = perf_counter()
    while perf_counter() - start < duration:
        sleep(0.1)
        current = read_present_current(handler)
        if current is not None:
            currents.append(abs(current))
    return sum(currents) / max(len(currents), 1), max(currents, default=0)


def closed_end_test(handler: ServoHandler) -> bool:
    """returns False if the gripper shut down"""
    goal_current = handler.get_servo(GRIPPER_ID).goal_current
    move_and_wait(handler, OPEN_STATE)
    input('\nremove any object from the gripper and press enter ')
    print('closing the gripper without an object')
    peak_current = move_and_wait(handler, CLOSED_STATE)
    average_current, _ = measure_current(handler, 1.0)
    print('peak current while closing: {}mA, current when closed: {:.0f}mA'.format(peak_current, average_current))
    errors = print_status(handler, 'closed')
    if average_current > 0.5 * goal_current:  # type: ignore[operator]  # the gripper has a goal current in the servo config
        print('the gripper pushes against its mechanical end when it is closed, the closed end of its range '
              '(position {}) is further than the gripper can close'.format(handler.get_servo(GRIPPER_ID).max_position))
    move_and_wait(handler, OPEN_STATE)
    return not errors


def model_current_limit(model: str | None) -> int:
    """
    the current limit in mA of a servo model, from wrist_roll_limits.MODELS or OTHER_CURRENT_LIMITS
    :param model: the model name of the yaml, like XC330-T288-T, or None when the yaml doesn't give one
    """
    if model is not None:
        for limits in MODELS.values():
            if model.startswith(limits.name):
                return limits.max_current
        for name, limit in OTHER_CURRENT_LIMITS.items():
            if model.startswith(name):
                return limit
    return DEFAULT_CURRENT_LIMIT


def squeeze_currents(current_limit: int) -> list[int]:
    """the goal currents of the squeeze tests: GOAL_CURRENTS, none above the limit, which is itself the last one"""
    return sorted({min(goal_current, current_limit) for goal_current in GOAL_CURRENTS})


def distance_outside_range(servo: Servo, position: int) -> int:
    """the positions the raw position is beyond the calibrated range of the servo, 0 inside it"""
    lower, upper = servo.calibrated_range()
    return max(lower - position, position - upper, 0)


def gripper_is_calibrated(handler: ServoHandler) -> bool:
    """False, with a message, when the gripper stands outside the positions of the yaml by more than the margin"""
    position = handler.read_current_pos_single_servo(GRIPPER_ID)
    if position is None:
        print('failed to read the position of the gripper')
        return False
    servo = handler.get_servo(GRIPPER_ID)
    outside = distance_outside_range(servo, position)
    if outside > CALIBRATION_MARGIN:
        print('the gripper is at position {}, {} positions outside the range {} of servo_config.yaml: it is not '
              'calibrated. Put it open by hand and run it with --calibrate first'.format(
                  position, outside, list(servo.calibrated_range())))
        return False
    return True


def squeeze_tests(handler: ServoHandler, goal_currents: Sequence[int]) -> bool:
    """squeeze with increasing goal currents, returns False if the gripper shut down"""
    for goal_current in goal_currents:
        if input('\nsqueeze with a goal current of {}mA? (enter to continue, s to stop) '.format(goal_current)) \
                .strip().lower() == 's':
            break
        handler.set_goal_current(GRIPPER_ID, goal_current)
        handler.set_angle(GRIPPER_ID, CLOSED_STATE)
        handler.move_to_angles()
        average_current, peak_current = measure_current(handler, SQUEEZE_TIME)
        print('current while squeezing: average {:.0f}mA, peak {}mA'.format(average_current, peak_current))
        errors = print_status(handler, 'squeezing')
        temperature = read_temperature(handler)
        move_and_wait(handler, OPEN_STATE)
        if errors:
            print('the gripper shut down, stopping the squeeze tests')
            return False
        if temperature is not None and temperature > MAX_TEMPERATURE:
            print('the gripper is warmer than {}C, stopping the squeeze tests'.format(MAX_TEMPERATURE))
            break
    handler.set_configured_goal_current()
    return True


def lead_tests(handler: ServoHandler) -> None:
    """squeeze with the target a few states beyond the object, like the xbox controller does"""
    goal_current = handler.get_servo(GRIPPER_ID).goal_current
    print('\nsqueezing with the target a few states beyond the object, with the configured goal current of {}mA'
          .format(goal_current))
    move_and_wait(handler, CLOSED_STATE)
    contact_state = gripper_state(handler)
    if contact_state is None:
        print('failed to read the gripper state')
        return
    print('the object stops the gripper at state {:.1f}'.format(contact_state))

    for lead in LEADS:
        target = min(contact_state + lead, CLOSED_STATE)
        handler.set_angle(GRIPPER_ID, target)
        handler.move_to_angles()
        sleep(0.5)
        average_current, peak_current = measure_current(handler, 1.5)
        print('target {} beyond the object: average {:.0f}mA ({:.0f}% of the goal current), peak {}mA'.format(
            lead, average_current, 100 * average_current / goal_current, peak_current))  # type: ignore[operator]  # the gripper has a goal current in the servo config
        if print_status(handler, 'squeezing'):
            print('the gripper shut down, stopping the lead tests')
            return
    move_and_wait(handler, OPEN_STATE)


def read_with_retries(read: Callable[[], T | None], what: str) -> T:
    """the result of read, which gives None when it fails, tried READ_ATTEMPTS times; raises Stop when it never works"""
    for _ in range(READ_ATTEMPTS):
        value = read()
        if value is not None:
            return value
    raise Stop('failed to read the {} of the gripper {} times in a row'.format(what, READ_ATTEMPTS))


def read_position(handler: ServoHandler) -> int:
    """the present raw position of the gripper"""
    return read_with_retries(lambda: handler.read_current_pos_single_servo(GRIPPER_ID), 'position')


def read_abs_current(handler: ServoHandler) -> int:
    """the present current of the gripper in mA, without the sign"""
    def read() -> int | None:
        currents = handler.read_present_current([GRIPPER_ID])
        return None if currents is None else abs(currents[GRIPPER_ID])
    return read_with_retries(read, 'current')


def read_hardware_errors(handler: ServoHandler) -> list[str]:
    """the names of the hardware errors of the gripper, an empty list when it has none"""
    errors = read_with_retries(lambda: handler.read_hardware_errors([GRIPPER_ID]), 'hardware errors')
    return errors.get(GRIPPER_ID, [])


def read_gripper_temperature(handler: ServoHandler) -> tuple[int, float]:
    """the temperature in degrees celsius and the input voltage of the gripper"""
    values = read_with_retries(lambda: handler.read_temperature_and_voltage([GRIPPER_ID]), 'temperature')
    return values[GRIPPER_ID]


def read_status(handler: ServoHandler) -> Status:
    """everything the calibration checks before it enables the servo, raises Stop when a read fails"""
    position = read_position(handler)
    current = read_abs_current(handler)
    errors = read_hardware_errors(handler)
    temperature, voltage = read_gripper_temperature(handler)
    torque = read_with_retries(lambda: handler.read_torque_enabled([GRIPPER_ID]), 'torque state')[GRIPPER_ID]
    return Status(position, current, temperature, voltage, errors, torque)


def check_health(handler: ServoHandler) -> None:
    """raises Stop for a hardware error (the servo shuts down and ignores goals), a warm servo or a failed read"""
    errors = read_hardware_errors(handler)
    if errors:
        raise Stop('the gripper has a hardware error ({})'.format(', '.join(errors)))
    temperature, _ = read_gripper_temperature(handler)
    if temperature > MAX_TEMPERATURE:
        raise Stop('the gripper is warmer than {}C ({}C)'.format(MAX_TEMPERATURE, temperature))


def median_position(handler: ServoHandler, sleep: Callable[[float], None] = sleep) -> int:
    """the median of POSITION_SAMPLES reads of the position, a little apart"""
    positions = []
    for i in range(POSITION_SAMPLES):
        if i:
            sleep(0.02)
        positions.append(read_position(handler))
    return int(median(positions))


def write_goal(handler: ServoHandler, position: int) -> None:
    """
    write the raw goal position of the gripper and nothing else, raises Stop when it can't be sent. This is
    move_servo_to_pos: move_to_angles maps a state to a position and clamps it to the positions of the yaml, which are
    the old ones here, and step_servo_to_pos rewrites the profile, which we want kept
    """
    if not handler.move_servo_to_pos(GRIPPER_ID, position):
        raise Stop('failed to write the goal position {}'.format(position))


def closing_direction(servo: Servo, closing: str | None) -> int:
    """
    +1 when the gripper closes towards higher positions, -1 towards lower ones
    :param closing: 'up' or 'down', None for the direction of the yaml: from the position of state 0 (open) to that of
        the highest state (closed)
    """
    if closing is not None:
        return {'up': 1, 'down': -1}[closing]
    return 1 if servo.max_position >= servo.min_position else -1


def room_to_close(open_position: int, direction: int) -> int:
    """the positions between the open position and the end of the range of the servo in the direction"""
    return cfg.MAX_POSITION - open_position if direction > 0 else open_position - cfg.MIN_POSITION


def check_calibration_arguments(servo: Servo, current: int, step: int, max_travel: int, model_limit: int) -> str | None:
    """the reason the arguments are refused, None when they are fine"""
    if current < 1:
        return 'the current has to be at least 1 mA, not {}'.format(current)
    if servo.goal_current is not None and current > servo.goal_current:
        return 'the current of {} mA is above the goal current of {} mA of servo_config.yaml: the calibration is ' \
               'meant to be gentler than a grip'.format(current, servo.goal_current)
    if current > model_limit:
        return 'the current of {} mA is above the {} mA limit of the model'.format(current, model_limit)
    if step < 1:
        return 'the step has to be at least 1 position, not {}'.format(step)
    if max_travel <= MIN_TRAVEL:
        return 'the maximum travel has to be above {} positions, not {}'.format(MIN_TRAVEL, max_travel)
    return None


def advance_over_window(history: list[tuple[float, int]], now: float, direction: int) -> int | None:
    """
    the positions the gripper advanced in its closing direction over the last STALL_TIME seconds, None while the
    history doesn't reach back that far; drops the samples that are no longer needed
    """
    while len(history) > 1 and history[1][0] <= now - STALL_TIME:
        history.pop(0)
    start_time, start_position = history[0]
    if start_time > now - STALL_TIME:
        return None
    return (history[-1][1] - start_position) * direction


def find_closed_position(handler: ServoHandler, open_position: int, direction: int, step: int, max_travel: int,
                         sleep: Callable[[float], None] = sleep,
                         clock: Callable[[], float] = perf_counter) -> ContactResult:
    """
    close the torque-enabled gripper in small goal steps until it stops against something. The goal moves by step
    positions every STEP_TIME seconds from the open position, at most max_travel positions; the gripper has stopped
    when the goal is more than CONTACT_LAG positions ahead of it and it advanced less than STALL_POSITIONS positions
    in the last STALL_TIME seconds. The goal is then set to where the gripper is (it stops pushing), and after
    SETTLE_TIME the median of its position is the closed position.
    :param direction: +1 to close towards higher positions, -1 towards lower ones
    :raises Stop: when a read or a write fails, on a hardware error or a servo that is too warm
    """
    limit = min(max_travel, room_to_close(open_position, direction))  # the goal stays in the range of the servo
    goal = open_position
    history: list[tuple[float, int]] = []
    peak_current = 0
    steps = 0
    limit_reached_at: float | None = None
    while True:
        travel = abs(goal - open_position)
        if travel < limit:
            goal += direction * min(step, limit - travel)
            write_goal(handler, goal)
        sleep(STEP_TIME)
        position = read_position(handler)
        current = read_abs_current(handler)
        peak_current = max(peak_current, current)
        now = clock()
        history.append((now, position))
        steps += 1
        if steps % CHECK_EVERY == 0:
            check_health(handler)
        ahead = (goal - position) * direction
        advance = advance_over_window(history, now, direction)
        if ahead > CONTACT_LAG and advance is not None and advance < STALL_POSITIONS:
            check_health(handler)  # a servo that shut down stops too
            write_goal(handler, position)  # stop pushing
            sleep(SETTLE_TIME)
            closed = median_position(handler, sleep)
            outcome = NO_MOVE if abs(closed - open_position) < MIN_TRAVEL else CONTACT
            return ContactResult(outcome, closed, position, peak_current, current)
        if abs(goal - open_position) >= limit:
            if ahead <= CONTACT_LAG:
                return ContactResult(NO_CONTACT, position, goal, peak_current, None)
            limit_reached_at = now if limit_reached_at is None else limit_reached_at
            if now - limit_reached_at > LIMIT_WAIT:  # it creeps on without stopping or arriving
                raise Stop('the gripper does not follow the goal at the end of the travel')


def open_again(handler: ServoHandler, goal: int, open_position: int, step: int,
               sleep: Callable[[float], None] = sleep, clock: Callable[[], float] = perf_counter) -> bool:
    """
    move the goal from where it is back to the open position in the same small steps, then wait until the gripper is
    there or OPEN_TIMEOUT seconds have passed, returns whether it arrived
    """
    while goal != open_position:
        goal += max(-step, min(step, open_position - goal))
        write_goal(handler, goal)
        sleep(STEP_TIME)
    deadline = clock() + OPEN_TIMEOUT
    while abs(read_position(handler) - open_position) > ARRIVED:
        if clock() >= deadline:
            return False
        sleep(STEP_TIME)
    return True


def shut_down(handler: ServoHandler, sleep: Callable[[float], None] = sleep) -> bool:
    """
    hold the gripper where it is (the goal is its present position, when that can be read) and switch its torque off,
    whatever went wrong before; returns False when the torque could not be confirmed off
    """
    try:
        handler.hold_current_position([GRIPPER_ID])
    except Exception as e:  # nothing here may stop the torque from being switched off
        print('could not hold the gripper: {}'.format(e))
    for _ in range(TORQUE_OFF_ATTEMPTS):
        try:
            handler.set_torque(False, [GRIPPER_ID])
            if handler.read_torque_enabled([GRIPPER_ID]) == {GRIPPER_ID: False}:
                return True
        except Exception as e:
            print('could not switch the torque off: {}'.format(e))
        sleep(0.1)
    print('COULD NOT CONFIRM THAT THE TORQUE OF THE GRIPPER IS OFF: power cycle the servo')
    return False


def read_goal_current(handler: ServoHandler) -> int | None:
    """the goal current register of the gripper in mA, None when it can't be read"""
    value, comm_result, error = handler.packet_handler.read2ByteTxRx(
        handler.port_handler, GRIPPER_ID, cfg.ADDR_GOAL_CURRENT)
    if comm_result != 0 or error != 0:
        return None
    return to_signed(value, cfg.LEN_GOAL_CURRENT)


def replace_positions(yaml_text: str, servo_id: int, open_position: int, closed_position: int,
                      note: str = '') -> str:
    """
    return the yaml text with the positions line of the servo replaced by [open_position, closed_position], everything
    else (other lines, comments, line endings) is kept as it is, as is a comment after the positions
    :param note: text for the comment after the positions, when the line has no comment of its own
    :raises ValueError: when the servo or its positions line is not in the yaml
    """
    lines = yaml_text.splitlines(keepends=True)
    in_servo = False
    replaced = False
    for i, line in enumerate(lines):
        header = re.match(r'^  (\d+):\s*(#.*)?$', line)
        if header:
            in_servo = int(header.group(1)) == servo_id
            continue
        match = re.match(r'^(\s+positions:\s*)\[[^\]]*\](.*?)(\r?\n?)$', line)
        if in_servo and match:
            if replaced:
                raise ValueError('servo {} has two positions lines'.format(servo_id))
            comment = match.group(2) if match.group(2).strip() else '  # {}'.format(note) if note else ''
            lines[i] = '{}[{}, {}]{}{}'.format(match.group(1), open_position, closed_position, comment, match.group(3))
            replaced = True
    if not replaced:
        raise ValueError('the positions of servo {} are not in the yaml'.format(servo_id))
    return ''.join(lines)


def write_positions(config_path: str, servo_id: int, open_position: int, closed_position: int) -> None:
    """
    put the positions in the yaml, keeping its comments, layout and line endings; the new text is parsed before it is
    written and the file is loaded again afterwards, so a broken result fails loudly
    :raises ValueError: when the positions are not in the yaml, or the result would not give these positions
    """
    with open(config_path, newline='') as config_file:  # newline='' keeps the CRLF line endings as they are
        text = config_file.read()
    text = replace_positions(text, servo_id, open_position, closed_position,
                             'open, closed: measured with gripper_test --calibrate')
    servo = parse_servo_config(yaml.safe_load(text), config_path).servos[servo_id]
    if (servo.min_position, servo.max_position) != (open_position, closed_position):
        raise ValueError('with these positions state 0 would not be at the open position, check the states of servo '
                         '{} in the yaml'.format(servo_id))
    with open(config_path, 'w', newline='') as config_file:
        config_file.write(text)
    load_servo_config(config_path)  # fails loudly when the written yaml is broken


def report(open_position: int, result: ContactResult) -> str:
    """the report of a calibration that found the closed position, the last line is for the yaml"""
    travel = abs(result.position - open_position)
    return ('open position {}, closed position {}\n'
            'travel {} positions, {:.1f} degrees\n'
            'peak current while closing {} mA, current at contact {} mA\n'
            'for servo_config.yaml:\n'
            '    positions: [{}, {}]').format(
                open_position, result.position, travel, travel * 360.0 / POSITIONS_PER_TURN, result.peak_current,
                result.contact_current, open_position, result.position)


def confirm_move(yes: bool, ask: Callable[[str], str]) -> bool:
    """ask whether the gripper may move, True without asking when yes is given"""
    if yes:
        return True
    return ask('\nthe gripper will move, keep your fingers clear. type yes to continue: ').strip().lower() == 'yes'


def calibrate(handler: ServoHandler, current: int = DEFAULT_CALIBRATION_CURRENT, step: int = DEFAULT_STEP,
              max_travel: int = DEFAULT_MAX_TRAVEL, closing: str | None = None, write: bool = False,
              yes: bool = False, config_path: str = DEFAULT_SERVO_CONFIG_PATH,
              model_limit: int = DEFAULT_CURRENT_LIMIT, sleep: Callable[[float], None] = sleep,
              clock: Callable[[], float] = perf_counter, ask: Callable[[str], str] = input) -> int:
    """
    find the closed position of the gripper, which stands open where it should be with its torque off, see the
    docstring of the module; returns 0 when it found the closed position, else 1. Whatever goes wrong after the torque
    was enabled, the gripper holds where it is and the torque is switched off.
    :param current: the goal current in mA while closing, at most the goal current of the yaml
    :param closing: 'up' or 'down', None for the direction of the yaml
    :param write: put the positions in the yaml at config_path, after asking unless yes is given
    :param model_limit: the current limit of the model in mA
    :param ask: gives the answer to a question, input() for a person
    """
    servo = handler.get_servo(GRIPPER_ID)
    problem = check_calibration_arguments(servo, current, step, max_travel, model_limit)
    if problem:
        print(problem)
        return 1
    try:
        status = read_status(handler)
    except Stop as e:
        print(e)
        return 1
    print('gripper: position {} current {}mA temperature {}C voltage {}V hardware errors {} torque {}'.format(
        status.position, status.current, status.temperature, status.voltage, status.errors or None,
        'on' if status.torque else 'off'))
    if status.errors:
        print('the gripper has a hardware error ({}), reboot the servo first'.format(', '.join(status.errors)))
        return 1
    if status.torque:
        print('the torque of the gripper is on: switch it off and put the gripper open by hand, the position it has '
              'now is taken as the open position')
        return 1

    try:
        open_position = median_position(handler, sleep)
    except Stop as e:
        print(e)
        return 1
    direction = closing_direction(servo, closing)
    if room_to_close(open_position, direction) <= MIN_TRAVEL:
        print('the open position {} is at the end of the range of the servo in the closing direction ({}), use '
              '--closing {}'.format(open_position, 'up' if direction > 0 else 'down',
                                    'down' if direction > 0 else 'up'))
        return 1
    print('open position {}; the gripper closes towards {} positions, at most {} positions, with {} mA, in steps of '
          '{} positions'.format(open_position, 'higher' if direction > 0 else 'lower', max_travel, current, step))
    if not confirm_move(yes, ask):
        print('not started')
        return 0

    result: ContactResult | None = None
    torque_off = True
    enabled = False
    try:
        if not (handler.set_configured_operating_mode([GRIPPER_ID]) and
                handler.set_profile_velocity_and_acceleration([GRIPPER_ID]) and handler.set_pid([GRIPPER_ID])):
            print('failed to set up the gripper')
            return 1
        # the goal current register holds the maximum of the model until it is written, so it is written (and read
        # back, the write isn't acknowledged) before the torque goes on
        if not handler.set_goal_current(GRIPPER_ID, current):
            print('failed to set the goal current')
            return 1
        if read_goal_current(handler) != current:
            print('the goal current of the gripper is not {} mA after writing it, not enabling the torque'.format(
                current))
            return 1
        if not handler.hold_current_position([GRIPPER_ID]):
            print('failed to hold the gripper')
            return 1
        enabled = True  # from here on the torque may be on, also when it fails to confirm
        if not handler.set_torque(True, [GRIPPER_ID]):
            print('failed to enable the gripper')
            return 1
        result = find_closed_position(handler, open_position, direction, step, max_travel, sleep, clock)
        if not open_again(handler, result.goal, open_position, step, sleep, clock):
            print('warning: the gripper did not get back to the open position within {:.0f} s'.format(OPEN_TIMEOUT))
    except Stop as e:
        print('\nstopped: {}'.format(e))
        return 1
    except KeyboardInterrupt:
        print('\nstopped')
        return 1
    finally:
        if enabled:
            torque_off = shut_down(handler, sleep)
    if not torque_off or result is None:
        return 1

    if result.outcome == NO_MOVE:
        print('the gripper did not move in that direction (it stopped after {} positions): is --closing right, or is '
              'it already closed or blocked?'.format(abs(result.position - open_position)))
        return 1
    if result.outcome == NO_CONTACT:
        print('the gripper did not stop within {} positions of travel: nothing closed it, raise --max-travel or check '
              'the direction'.format(max_travel))
        return 1
    print(report(open_position, result))
    if not write:
        print('nothing was written, add --write to put the positions in {}'.format(config_path))
        return 0
    if not yes and ask('write these positions to {}? type yes: '.format(config_path)).strip().lower() != 'yes':
        print('not writing')
        return 0
    write_positions(config_path, GRIPPER_ID, open_position, result.position)
    print('written to {}, the gripper can be driven by state now'.format(config_path))
    return 0


def run_interactive_tests(handler: ServoHandler, goal_currents: Sequence[int], yes: bool) -> int:
    """the interactive tests of the gripper, see the docstring of the module"""
    if print_status(handler, 'gripper'):
        print('the gripper has a hardware error, reboot the servos first')
        return 1
    if not gripper_is_calibrated(handler):
        return 1
    if not confirm_move(yes, input):
        return 0

    try:
        if not (handler.set_configured_operating_mode() and handler.set_profile_velocity_and_acceleration()
                and handler.set_pid() and handler.set_configured_goal_current()):
            print('failed to set up the gripper')
            return 1
        if not (handler.hold_current_position() and handler.set_torque(True)):
            print('failed to enable the gripper')
            return 1

        if closed_end_test(handler):
            input('\nput an object, like a pen, in the gripper and press enter ')
            if input('test squeezing with increasing goal currents? (y/n) ').strip().lower() != 'y' \
                    or squeeze_tests(handler, goal_currents):
                lead_tests(handler)
    except KeyboardInterrupt:
        print('\nstopped')
    finally:
        handler.set_torque(False)
        print_status(handler, 'torque disabled')
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='test the gripper (servo 7): --calibrate finds its closed position, without it the interactive '
                    'tests run')
    parser.add_argument('--calibrate', action='store_true', help='find the closed position of the gripper')
    parser.add_argument('--current', type=int, default=None,
                        help='--calibrate: goal current in mA while closing, default {}, at most the goal_current of '
                             'the yaml'.format(DEFAULT_CALIBRATION_CURRENT))
    parser.add_argument('--step', type=int, default=None,
                        help='--calibrate: positions the goal moves every {:.0f} ms, default {}'.format(
                            1000 * STEP_TIME, DEFAULT_STEP))
    parser.add_argument('--max-travel', type=int, default=None,
                        help='--calibrate: most positions to close from the open position, default {}'.format(
                            DEFAULT_MAX_TRAVEL))
    parser.add_argument('--closing', choices=('up', 'down'), default=None,
                        help='--calibrate: whether the closed position is at higher (up) or lower (down) positions, '
                             'default the direction of the positions in the yaml')
    parser.add_argument('--write', action='store_true', help='--calibrate: write the positions to the yaml')
    parser.add_argument('--yes', action='store_true', help="don't ask before the gripper moves or the yaml is written")
    parser.add_argument('--config', default=DEFAULT_SERVO_CONFIG_PATH, help='the yaml to read and write')
    args = parser.parse_args(argv)
    if not args.calibrate:
        only_calibrating = [name for name, value in (('--current', args.current), ('--step', args.step),
                                                     ('--max-travel', args.max_travel), ('--closing', args.closing),
                                                     ('--write', args.write or None)) if value is not None]
        if only_calibrating:
            parser.error('{} only work with --calibrate'.format(', '.join(only_calibrating)))
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

    config = load_servo_config(args.config)
    gripper = config.servos[GRIPPER_ID]
    model_limit = model_current_limit(config.models.get(GRIPPER_ID))
    current = DEFAULT_CALIBRATION_CURRENT if args.current is None else args.current
    step = DEFAULT_STEP if args.step is None else args.step
    max_travel = DEFAULT_MAX_TRAVEL if args.max_travel is None else args.max_travel
    if args.calibrate:  # refused before the port is opened
        problem = check_calibration_arguments(gripper, current, step, max_travel, model_limit)
        if problem:
            print(problem)
            return 1

    port_handler, packet_handler = setup_dynamixel_handlers(None, cfg, ARM_IDS)
    handler = ServoHandler({GRIPPER_ID: gripper}, cfg, port_handler, packet_handler, buses=[(GRIPPER_ID,)])
    try:
        if args.calibrate:
            return calibrate(handler, current, step, max_travel, args.closing, args.write, args.yes, args.config,
                             model_limit, sleep, perf_counter)
        return run_interactive_tests(handler, squeeze_currents(model_limit), args.yes)
    finally:
        port_handler.closePort()


if __name__ == '__main__':
    sys.exit(main())
