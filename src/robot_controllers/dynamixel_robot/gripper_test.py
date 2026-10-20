"""
Test script that moves the physical gripper (servo 7), the other servos are not enabled. Run it from the project root:
    python -m src.robot_controllers.dynamixel_robot.gripper_test

1. closes the gripper without an object, to check whether it pushes against its mechanical end at the closed end
2. optionally squeezes an object with increasing goal currents (up to 1500mA) and reports the current, the
   position, the temperature and the hardware errors. It stops at a hardware error or when the gripper gets warm
3. squeezes an object with the target a few states beyond the object, like the xbox controller does, to check how
   much of the goal current that uses
4. disables the torque of the gripper

The gripper is open at state 0 and closed at state 100.

Ctrl+C disables the torque of the gripper and exits.
"""
import logging
import sys
from time import sleep, perf_counter

from src.robot_controllers.dynamixel_robot import dynamixel_x_config as cfg
from src.robot_controllers.dynamixel_robot.dynamixel_robot_controller import ARM_IDS, GRIPPER_ID
from src.robot_controllers.dynamixel_robot.dynamixel_utils import setup_dynamixel_handlers, to_signed
from src.robot_controllers.dynamixel_robot.servo_config import load_servo_config
from src.robot_controllers.dynamixel_robot.servo_handler import ServoHandler

# mA, the current limit of the XL330-M288 is 1750mA and it draws about 1.5A when it is blocked at full power
GOAL_CURRENTS = [200, 400, 600, 800, 1000, 1200, 1500]
# the squeeze tests stop when the gripper gets warmer than this, it shuts down at 70C
MAX_TEMPERATURE = 55  # degrees celsius
# how far the target is beyond the object, the xbox controller uses GRIPPER_MAX_LEAD
LEADS = [10, 20, 25]  # gripper states
SQUEEZE_TIME = 3.0  # seconds
MOVE_TIMEOUT = 5.0  # seconds
OPEN_STATE = 0
CLOSED_STATE = 100


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


def squeeze_tests(handler: ServoHandler) -> bool:
    """squeeze with increasing goal currents, returns False if the gripper shut down"""
    for goal_current in GOAL_CURRENTS:
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


def main() -> int:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

    gripper = load_servo_config().servos[GRIPPER_ID]
    port_handler, packet_handler = setup_dynamixel_handlers(None, cfg, ARM_IDS)
    handler = ServoHandler({GRIPPER_ID: gripper}, cfg, port_handler, packet_handler, buses=[(GRIPPER_ID,)])

    if print_status(handler, 'gripper'):
        print('the gripper has a hardware error, reboot the servos first')
        return 1
    if input('\nthe gripper will move, keep your fingers clear. type yes to continue: ').strip().lower() != 'yes':
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
                    or squeeze_tests(handler):
                lead_tests(handler)
    except KeyboardInterrupt:
        print('\nstopped')
    finally:
        handler.set_torque(False)
        print_status(handler, 'torque disabled')
        port_handler.closePort()
    return 0


if __name__ == '__main__':
    sys.exit(main())
