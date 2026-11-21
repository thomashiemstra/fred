"""
Puts the arm down after a run of one of the calibration scripts died (its process was killed, the usb cable came
out, the computer went to sleep) and left the arm holding a pose in the air with the torque on. Run it from the
project root:
    python -m src.robot_controllers.dynamixel_robot.calibration.lower_arm [--attempts 15]

It starts the robot (a bus read fails now and then on this robot, so that is tried again), reads the torque of the arm
servos and, when any is on, lowers the arm to the rest pose just above the table with
record_static_load.lower_to_rest and switches the torque off there. Starting the robot writes the settings of the
servos (gains, profile) but does not switch the torque off or move the arm, so the servos keep holding while it
starts. A failed lowering is tried again, without asking anything. When every attempt failed the servos still hold,
and the question to support the arm and switch the torque off comes once (without input it only prints that the servos
keep holding). When the torque is off already nothing is moved or written. At the end it prints the torque, the hardware
errors and the temperatures of the servos. The return code is 0 when the arm is down (or was), 1 otherwise.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from src.global_objects import get_robot
from src.robot_controllers.combined_robot.combined_robot import CombinedRobot
from src.robot_controllers.dynamixel_robot.calibration import record_static_load
from src.robot_controllers.dynamixel_robot.dynamixel_robot_controller import ARM_IDS, DynamixelRobotController
from src.robot_controllers.dynamixel_robot.dynamixel_utils import DynamixelError

if TYPE_CHECKING:
    from src.robot_controllers.simulated_robot.simulated_robot_controller import SimulatedRobotController

DEFAULT_ATTEMPTS = 15
PAUSE = 0.3  # seconds between the attempts
STATUS_READS = 10  # reads of the torque, the errors and the temperatures before one is given up as unreadable

T = TypeVar('T')


def read_with_retries(read: Callable[[], T | None], attempts: int = STATUS_READS) -> T | None:
    """the result of the read, tried up to attempts times; None when every attempt failed (a corrupt packet)"""
    for _ in range(attempts):
        value = read()
        if value is not None:
            return value
    return None


def start_robot(attempts: int, sleep: Callable[[float], None] = time.sleep,
                pause: float = PAUSE) -> SimulatedRobotController | CombinedRobot | None:
    """get_robot() tried up to attempts times, returns None when every attempt raised a DynamixelError"""
    for attempt in range(1, attempts + 1):
        try:
            return get_robot()
        except DynamixelError as e:
            print('start attempt {} of {}: {}'.format(attempt, attempts, e))
            if attempt < attempts:
                sleep(pause)
    return None


def torque_of_arm(robot: DynamixelRobotController | CombinedRobot) -> dict[int, bool] | None:
    """mapping of arm servo id to whether its torque is on, None when it can't be read"""
    return read_with_retries(lambda: robot.servo_handler.read_torque_enabled(ARM_IDS))


def lower(robot: DynamixelRobotController | CombinedRobot, attempts: int,
          sleep: Callable[[float], None] = time.sleep, pause: float = PAUSE) -> bool:
    """
    record_static_load.lower_to_rest tried up to attempts times, every reason it gave is printed; never asks anything
    :return: True when the arm was lowered and the torque is off
    """
    for attempt in range(1, attempts + 1):
        why = record_static_load.lower_to_rest(robot, sleep=sleep)
        if why is None:
            return True
        print('lowering attempt {} of {} did not go through: {}'.format(attempt, attempts, why))
        if attempt < attempts:
            sleep(pause)
    return False


def print_status(robot: DynamixelRobotController | CombinedRobot) -> None:
    """print the torque, the hardware errors and the temperatures of the servos, None for a read that kept failing"""
    handler = robot.servo_handler
    for name, read in (('torque', handler.read_torque_enabled), ('hardware errors', robot.get_hardware_errors),
                       ('temperatures', robot.get_servo_temperatures)):
        print('{}: {}'.format(name, read_with_retries(read)))


def lower_arm(robot: DynamixelRobotController | CombinedRobot, attempts: int,
              sleep: Callable[[float], None] = time.sleep, pause: float = PAUSE) -> int:
    """lower the arm when its torque is on, then print the status; returns the return code of main"""
    torque = torque_of_arm(robot)
    result = 0
    if torque is not None and not any(torque.values()):
        print('the torque of the arm servos is off already, nothing is moved')
    else:
        if torque is None:
            print('the torque of the arm servos could not be read, lowering the arm anyway (with the torque off '
                  'that moves nothing)')
        else:
            print('the torque is on for servos {}'.format(sorted(servo_id for servo_id, on in torque.items() if on)))
        if lower(robot, attempts, sleep, pause):
            print('the arm is lowered and the torque is off')
        else:
            print('the arm is NOT lowered after {} attempts, the servos still hold'.format(attempts))
            record_static_load.disable_torque_on_request(robot)
            result = 1
    print_status(robot)
    return result


def main(argv: list[str] | None = None, sleep: Callable[[float], None] = time.sleep) -> int:
    parser = argparse.ArgumentParser(description='lower the arm to the table and switch the torque off, after a '
                                                 'calibration run died with the arm in the air')
    parser.add_argument('--attempts', type=record_static_load.positive_int, default=DEFAULT_ATTEMPTS,
                        help='how often starting the robot and lowering the arm are tried (default %(default)s)')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.ERROR, format='%(levelname)s %(name)s: %(message)s')

    robot = start_robot(args.attempts, sleep)
    if robot is None:
        print('could not start the robot after {} attempts, the servos still hold'.format(args.attempts))
        return 1
    if not isinstance(robot, CombinedRobot):
        print('the arm can only be lowered on the real robot, use_simulation is set')
        return 1
    return lower_arm(robot, args.attempts, sleep)


if __name__ == '__main__':
    sys.exit(main())
