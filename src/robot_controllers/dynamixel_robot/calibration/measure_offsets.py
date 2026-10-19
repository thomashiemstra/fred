"""
Measures the offset of servos held at a known angle by a jig and writes it to servo_config.yaml. Run it from the
project root with the torque off and the jig mounted; by default servo 1 is held at 90 degrees, servo 2 at 0 and
servo 4 at 0 (DEFAULT_JIG):
    python -m src.robot_controllers.dynamixel_robot.calibration.measure_offsets
    python -m src.robot_controllers.dynamixel_robot.calibration.measure_offsets --jig 1=90 2=0 --samples 10
    python -m src.robot_controllers.dynamixel_robot.calibration.measure_offsets --dry-run

The raw position of a servo is the calibrated position of the angle (interpolated over positions/angles of the yaml)
plus its offset, so offset = raw position at the jig angle - calibrated position of that angle. Only the servos in the
jig are read, the others may be unpowered. Nothing is written to the servos and the torque is never enabled; the
script refuses when the torque of a jig servo is on. It asks before writing the yaml unless --yes is given. The
comments and the layout of the yaml are kept, only the offset lines change.
"""
import argparse
import logging
import re
import sys
from statistics import median

import numpy as np

from src.robot_controllers.dynamixel_robot import dynamixel_x_config as cfg
from src.robot_controllers.dynamixel_robot.dynamixel_utils import setup_dynamixel_handlers, to_signed, DynamixelError
from src.robot_controllers.dynamixel_robot.servo_config import load_servo_config, DEFAULT_SERVO_CONFIG_PATH
from src.robot_controllers.dynamixel_robot.servo_handler import ServoHandler

DEFAULT_JIG = {1: 90.0, 2: 0.0, 4: 0.0}  # servo id -> angle in degrees the jig holds it at
MAX_SPREAD = 2  # positions, the readings have to agree this well, otherwise the joint isn't seated in the jig


def calibrated_position(servo, angle_degrees):
    """the position of the angle without the offset of the servo"""
    return int(np.rint(np.interp(np.radians(angle_degrees), [servo.min_angle, servo.max_angle],
                                 [servo.min_position, servo.max_position])))


def new_offset(servo, angle_degrees, raw_position):
    """offset = raw position at a known angle - calibrated position of that angle"""
    return int(raw_position) - calibrated_position(servo, angle_degrees)


def raw_range(servo, offset):
    """the raw positions of the ends of the calibrated range with the given offset"""
    lower = min(servo.min_position, servo.max_position) + offset
    upper = max(servo.min_position, servo.max_position) + offset
    return lower, upper


def replace_offsets(yaml_text, offsets, note=''):
    """
    return the yaml text with the offset of every servo in offsets replaced (or added after its angles line),
    everything else is kept as it is
    :param offsets: mapping of servo id to the new offset
    :param note: text for the comment after the offset, or a mapping of servo id to text
    """
    lines = yaml_text.splitlines(keepends=True)
    newline = '\r\n' if lines and lines[0].endswith('\r\n') else '\n'
    notes = note if isinstance(note, dict) else {servo_id: note for servo_id in offsets}
    comments = {servo_id: '  # ' + text if text else '' for servo_id, text in notes.items()}

    # which lines belong to which servo, and which servos already have an offset line
    servo_of_line = {}
    servo_id = None
    has_offset = set()
    for i, line in enumerate(lines):
        header = re.match(r'^  (\d+):\s*(#.*)?$', line)
        if header:
            servo_id = int(header.group(1))
            continue
        servo_of_line[i] = servo_id
        if servo_id in offsets and re.match(r'^\s+offset:', line):
            has_offset.add(servo_id)

    done = set()
    for i in reversed(range(len(lines))):  # backwards, so inserting doesn't shift the lines still to visit
        servo_id = servo_of_line.get(i)
        if servo_id not in offsets:
            continue
        new_line = '    offset: {}{}{}'.format(offsets[servo_id], comments.get(servo_id, ''), newline)
        if servo_id in has_offset:
            if re.match(r'^\s+offset:', lines[i]):
                lines[i] = new_line
                done.add(servo_id)
        elif re.match(r'^\s+angles:', lines[i]):
            lines.insert(i + 1, new_line)
            done.add(servo_id)
    missing = set(offsets) - done
    if missing:
        raise ValueError('servos {} not found in the yaml'.format(sorted(missing)))
    return ''.join(lines)


def parse_jig(values):
    jig = {}
    for value in values:
        try:
            servo_id, angle = value.split('=')
            jig[int(servo_id)] = float(angle)
        except ValueError:
            raise argparse.ArgumentTypeError('jig entries look like 1=90, got {!r}'.format(value))
    return jig


def read_registers(handler, servo_ids):
    """returns {servo id: (torque enabled, homing offset)}, or None when a servo doesn't answer"""
    registers = {}
    for servo_id in servo_ids:
        torque, result, error = handler.packet_handler.read1ByteTxRx(handler.port_handler, servo_id,
                                                                     cfg.ADDR_TORQUE_ENABLE)
        homing, result2, error2 = handler.packet_handler.read4ByteTxRx(handler.port_handler, servo_id,
                                                                       cfg.ADDR_HOMING_OFFSET)
        if result != 0 or error != 0 or result2 != 0 or error2 != 0:
            print('servo {} does not answer, is it powered and connected?'.format(servo_id))
            return None
        registers[servo_id] = (torque, to_signed(homing, cfg.LEN_HOMING_OFFSET))
    return registers


def read_positions(handler, servo_ids, samples):
    """returns {servo id: [raw positions]} or None when a read fails"""
    readings = {servo_id: [] for servo_id in servo_ids}
    for _ in range(samples):
        if not handler.read_current_pos(servo_ids):
            print('failed to read the positions')
            return None
        for servo_id in servo_ids:
            readings[servo_id].append(handler.get_servo(servo_id).current_position)
    return readings


def main(argv=None):
    parser = argparse.ArgumentParser(description='measure the servo offsets with the jig and write them to the yaml')
    default_jig = ['{}={:g}'.format(servo_id, angle) for servo_id, angle in DEFAULT_JIG.items()]
    parser.add_argument('--jig', nargs='+', default=default_jig,
                        help='servo=angle in degrees, default {}'.format(' '.join(default_jig)))
    parser.add_argument('--samples', type=int, default=10, help='number of readings per servo')
    parser.add_argument('--config', default=DEFAULT_SERVO_CONFIG_PATH, help='the yaml to read and write')
    parser.add_argument('--dry-run', action='store_true', help='only print the offsets')
    parser.add_argument('--yes', action='store_true', help="don't ask before writing the yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.ERROR)
    jig = parse_jig(args.jig)
    servo_ids = sorted(jig)

    config = load_servo_config(args.config)
    unknown = set(servo_ids) - set(config.servos)
    if unknown:
        print('servos {} are not in {}'.format(sorted(unknown), args.config))
        return 1
    servos = {servo_id: config.servos[servo_id] for servo_id in servo_ids}
    buses = [tuple(servo_id for servo_id in bus if servo_id in servos) for bus in config.buses]
    buses = [bus for bus in buses if bus]

    try:
        port_handler, packet_handler = setup_dynamixel_handlers(None, cfg, servo_ids)
    except DynamixelError as e:
        print(e)
        return 1
    try:
        handler = ServoHandler(servos, cfg, port_handler, packet_handler, buses)
        registers = read_registers(handler, servo_ids)
        if registers is None:
            return 1
        for servo_id, (torque, homing) in registers.items():
            if torque:
                print('the torque of servo {} is enabled, disable it and seat the joint in the jig first'.format(servo_id))
                return 1
            if homing:
                print('warning: servo {} has a homing offset of {} in its eeprom, the yaml offset only holds while '
                      'it stays'.format(servo_id, homing))
        readings = read_positions(handler, servo_ids, args.samples)
        if readings is None:
            return 1
    finally:
        port_handler.closePort()

    offsets = {}
    raw_positions = {}
    ok = True
    for servo_id in servo_ids:
        servo = servos[servo_id]
        values = readings[servo_id]
        spread = max(values) - min(values)
        raw = raw_positions[servo_id] = int(median(values))
        offset = new_offset(servo, jig[servo_id], raw)
        lower, upper = raw_range(servo, offset)
        print('servo {} ({}) at {:g} degrees: raw {} (spread {}), calibrated {}, offset {} -> {}, raw range [{}, {}]'
              .format(servo_id, config.names[servo_id], jig[servo_id], raw, spread,
                      calibrated_position(servo, jig[servo_id]), servo.constant_offset, offset, lower, upper))
        if spread > MAX_SPREAD:
            print('  the readings differ by {} positions, is the joint seated in the jig?'.format(spread))
            ok = False
        if lower < cfg.MIN_POSITION or upper > cfg.MAX_POSITION:
            if upper - lower >= cfg.MAX_POSITION - cfg.MIN_POSITION:
                # a full turn can't fit with any offset, the servo handler clamps the goal to 0-4095, so only the
                # last positions at one end of the range are lost
                print('  the calibrated range is a full turn, the {} positions at the {} end are unreachable'.format(
                    abs(offset), 'lower' if lower < cfg.MIN_POSITION else 'upper'))
            else:
                print('  the raw range leaves {}-{}, the servo can not cross its zero in position mode: rotate the '
                      'magnet or use the homing offset of the servo'.format(cfg.MIN_POSITION, cfg.MAX_POSITION))
                ok = False
        offsets[servo_id] = offset
    if not ok:
        print('not writing the yaml')
        return 1
    if args.dry_run:
        return 0

    if not args.yes and input('write these offsets to {}? type yes: '.format(args.config)).strip().lower() != 'yes':
        print('not writing')
        return 0
    with open(args.config) as config_file:
        text = config_file.read()
    text = replace_offsets(text, offsets, {servo_id: 'measured in the jig at {:g} degrees, raw {}'.format(
        jig[servo_id], raw_positions[servo_id]) for servo_id in servo_ids})
    with open(args.config, 'w') as config_file:
        config_file.write(text)
    load_servo_config(args.config)  # fails loudly when the written yaml is broken
    print('written, run the tests and robot_move_test')
    return 0


if __name__ == '__main__':
    sys.exit(main())
