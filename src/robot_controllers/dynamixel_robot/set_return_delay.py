"""
Sets the return delay time (eeprom, 2 us per unit, 250 = 500 us by default) of the connected servos, a servo that
replies at once answers every read faster. Run it from the project root:
    python -m src.robot_controllers.dynamixel_robot.set_return_delay [--value N] [--max-id N] [--dry-run] [--yes]

Every servo that answers on a dynamixel usb adapter is listed with its model, return delay time and torque. The
torque is only read, never written: the eeprom accepts a write with the torque off only, so a servo with the torque on
is skipped (disable its torque first) and the exit code is 1. Servos that already have the value are not written.
After a "type yes" (skipped with --yes) the value is written and read back; the exit code is 0 only when every
servo ends at the value. --dry-run only lists the servos.
"""
import argparse
import sys
from contextlib import ExitStack
from typing import Any

import dynamixel_sdk as dynamixel
import serial

from src.robot_controllers.dynamixel_robot import dynamixel_x_config as cfg
from src.robot_controllers.dynamixel_robot.dynamixel_utils import (
    list_dynamixel_adapters, get_port_handler, read_1_byte, write_1_byte, DynamixelError)
from src.robot_controllers.dynamixel_robot.scan_servos import MODEL_NAMES

US_PER_UNIT = 2


class FoundServo:
    """a servo that answered the ping, with what was read from its registers (None when the read failed)"""

    def __init__(self, port_handler: Any, packet_handler: Any, servo_id: int, model_number: int) -> None:
        self.port_handler = port_handler
        self.packet_handler = packet_handler
        self.servo_id = servo_id
        self.model_number = model_number
        self.torque = read_1_byte(packet_handler, port_handler, cfg.ADDR_TORQUE_ENABLE, servo_id)
        self.delay = read_1_byte(packet_handler, port_handler, cfg.ADDR_RETURN_DELAY_TIME, servo_id)


def delay_text(delay: int | None) -> str:
    return 'unknown' if delay is None else '%d (%d us)' % (delay, delay * US_PER_UNIT)


def int_in_range(low: int, high: int | None = None) -> Any:
    """argparse type for an integer between low and high"""
    def parse(text: str) -> int:
        try:
            number = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError('%r is not a number' % text)
        if number < low or (high is not None and number > high):
            raise argparse.ArgumentTypeError('%d is outside %d..%s' % (number, low, high if high is not None else ''))
        return number
    return parse


def find_servos(stack: ExitStack, max_servo_id: int) -> list[FoundServo]:
    """ping the ids on every adapter, the ports stay open until the stack is closed"""
    servos = []
    for port, name in list_dynamixel_adapters():
        print("%s: %s" % (port, name))
        try:
            port_handler = get_port_handler(port, cfg.BAUDRATE)
        except (serial.SerialException, DynamixelError) as e:
            print("  could not open the port, is it in use by another program? %s" % e)
            continue
        stack.callback(port_handler.closePort)
        packet_handler = dynamixel.PacketHandler(cfg.PROTOCOL_VERSION)
        found = []
        for servo_id in range(1, max_servo_id + 1):
            model_number, comm_result, error = packet_handler.ping(port_handler, servo_id)
            if comm_result == dynamixel.COMM_SUCCESS and error == 0:
                found.append(FoundServo(port_handler, packet_handler, servo_id, model_number))
        if not found:
            print("  no servos respond with ids 1-%d at %d bps, is the power on?" % (max_servo_id, cfg.BAUDRATE))
        servos.extend(found)
    return servos


def write_delay(servo: FoundServo, value: int) -> bool:
    """write the value, read it back and print the result, returns True when the servo has the value"""
    old = delay_text(servo.delay)
    if not write_1_byte(servo.packet_handler, servo.port_handler, cfg.ADDR_RETURN_DELAY_TIME, servo.servo_id, value):
        print("  id %d: writing failed (see the log), the torque must be off" % servo.servo_id)
        return False
    read_back = read_1_byte(servo.packet_handler, servo.port_handler, cfg.ADDR_RETURN_DELAY_TIME, servo.servo_id)
    if read_back != value:
        print("  id %d: read back %s instead of %d" % (servo.servo_id, delay_text(read_back), value))
        return False
    print("  id %d: %s -> %s" % (servo.servo_id, old, delay_text(value)))
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='set the return delay time of the connected servos')
    parser.add_argument('--value', type=int_in_range(0, cfg.MAX_RETURN_DELAY_TIME), default=0,
                        help='return delay time to write in units of 2 us, 0 to %d, default 0'
                        % cfg.MAX_RETURN_DELAY_TIME)
    parser.add_argument('--max-id', type=int_in_range(1), default=20, help='highest servo id to ping, default 20')
    parser.add_argument('--dry-run', action='store_true', help='only list the servos')
    parser.add_argument('--yes', action='store_true', help="don't ask before writing")
    args = parser.parse_args(argv)

    if not list_dynamixel_adapters():
        print("no dynamixel usb adapter found")
        return 1

    with ExitStack() as stack:
        servos = find_servos(stack, args.max_id)
        if not servos:
            print("no servo responds")
            return 1

        ok = True
        to_write = []
        for servo in servos:
            line = "id %d: %s, return delay time %s, torque %s" % (
                servo.servo_id, MODEL_NAMES.get(servo.model_number, 'unknown'), delay_text(servo.delay),
                'unknown' if servo.torque is None else 'on' if servo.torque else 'off')
            if servo.torque is None or servo.delay is None:
                print(line + " -> skipped, could not read its registers")
                ok = False
            elif servo.torque:
                print(line + " -> skipped, disable the torque first")
                ok = False
            elif servo.delay == args.value:
                print(line + " -> already %d" % args.value)
            else:
                print(line)
                to_write.append(servo)
        if args.dry_run:
            return 0
        if not to_write:
            print("nothing to write")
            return 0 if ok else 1

        if not args.yes and input("write return delay time %s to servo(s) %s? type yes: " % (
                delay_text(args.value), [servo.servo_id for servo in to_write])).strip().lower() != 'yes':
            print("not writing")
            return 0
        for servo in to_write:
            ok = write_delay(servo, args.value) and ok
        return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
