"""Loads the servo configuration of the robot from a yaml file, see servo_config.yaml for the format."""
import os

import numpy as np
import yaml

from src.global_constants import POSITION_CONTROL_MODE, EXTENDED_POSITION_CONTROL, CURRENT_BASED_POSITION_CONTROL_MODE
from src.robot_controllers.dynamixel_robot.servo import Servo

DEFAULT_SERVO_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'servo_config.yaml')

OPERATING_MODES = {
    'position': POSITION_CONTROL_MODE,
    'extended_position': EXTENDED_POSITION_CONTROL,
    'current_based_position': CURRENT_BASED_POSITION_CONTROL_MODE,
}

_REQUIRED_SERVO_FIELDS = {'name', 'bus', 'mode', 'positions', 'profile', 'pid'}
_OPTIONAL_SERVO_FIELDS = {'model', 'angles', 'states', 'offset', 'goal_current'}
_REQUIRED_FIELDS = {'temperature_warning', 'temperature_limit', 'servos'}


class ServoConfigError(ValueError):
    """raised when the servo configuration is not valid"""


class ServoConfig:
    """the servos of the robot and the settings that apply to all of them"""

    def __init__(self, servos, names, buses, temperature_warning, temperature_limit):
        self.servos = servos  # mapping of servo id to Servo, ordered by servo id
        self.names = names  # mapping of servo id to name
        self.buses = buses  # tuple of tuples of the servo ids on the same bus
        self.temperature_warning = temperature_warning  # degrees celsius
        self.temperature_limit = temperature_limit  # degrees celsius


def load_servo_config(path=DEFAULT_SERVO_CONFIG_PATH):
    """
    read the servo configuration, every call returns new Servo objects
    :raises ServoConfigError: if the configuration is not valid
    """
    try:
        with open(path) as config_file:
            data = yaml.safe_load(config_file)
    except (OSError, yaml.YAMLError) as e:
        raise ServoConfigError('can not read the servo configuration {}: {}'.format(path, e)) from e
    return parse_servo_config(data, path)


def parse_servo_config(data, source='servo configuration'):
    """create the servo configuration from the parsed yaml data"""
    _check_fields(data, _REQUIRED_FIELDS, set(), source)
    temperature_warning = _number(data['temperature_warning'], 'temperature_warning', source)
    temperature_limit = _number(data['temperature_limit'], 'temperature_limit', source)
    if temperature_warning > temperature_limit:
        raise ServoConfigError('{}: temperature_warning is higher than temperature_limit'.format(source))

    if not isinstance(data['servos'], dict) or not data['servos']:
        raise ServoConfigError('{}: servos should be a mapping of servo id to servo'.format(source))

    servos, names, buses = {}, {}, {}
    for servo_id in sorted(data['servos'], key=lambda key: (not isinstance(key, int), str(key))):
        if not isinstance(servo_id, int) or isinstance(servo_id, bool) or not 0 <= servo_id <= 252:
            raise ServoConfigError('{}: servo id {!r} should be a number from 0 to 252'.format(source, servo_id))
        servo_data = data['servos'][servo_id]
        servo_source = '{}, servo {}'.format(source, servo_id)
        servos[servo_id] = _parse_servo(servo_data, servo_source)
        names[servo_id] = str(servo_data['name'])
        buses.setdefault(str(servo_data['bus']), []).append(servo_id)

    return ServoConfig(servos, names, tuple(tuple(bus) for bus in buses.values()),
                       temperature_warning, temperature_limit)


def _parse_servo(data, source):
    _check_fields(data, _REQUIRED_SERVO_FIELDS, _OPTIONAL_SERVO_FIELDS, source)

    if data['mode'] not in OPERATING_MODES:
        raise ServoConfigError('{}: unknown mode {!r}, use one of {}'.format(
            source, data['mode'], ', '.join(OPERATING_MODES)))

    first_position, last_position = _pair(data['positions'], 'positions', source, integer=True)
    if first_position == last_position:
        raise ServoConfigError('{}: the positions of the ends of the range are the same'.format(source))

    if ('angles' in data) == ('states' in data):
        raise ServoConfigError('{}: should have either angles (in degrees) or states'.format(source))
    if 'angles' in data:
        first, last = (float(np.radians(angle)) for angle in _pair(data['angles'], 'angles', source))
    else:
        first, last = _pair(data['states'], 'states', source)

    profile = data['profile']
    _check_fields(profile, {'velocity', 'acceleration'}, set(), source + ', profile')
    pid = data['pid']
    _check_fields(pid, {'p', 'i', 'd'}, set(), source + ', pid')

    goal_current = data.get('goal_current')
    if goal_current is not None:
        goal_current = _integer(goal_current, 'goal_current', source)

    # Servo expects the smallest angle first, the positions follow the angles
    if first > last:
        first, last = last, first
        first_position, last_position = last_position, first_position

    return Servo(first_position, last_position, first, last, OPERATING_MODES[data['mode']],
                 profile_velocity=_integer(profile['velocity'], 'profile velocity', source),
                 profile_acceleration=_integer(profile['acceleration'], 'profile acceleration', source),
                 p=_integer(pid['p'], 'p', source), i=_integer(pid['i'], 'i', source),
                 d=_integer(pid['d'], 'd', source),
                 offset=_integer(data.get('offset', 0), 'offset', source), goal_current=goal_current)


def _check_fields(data, required, optional, source):
    if not isinstance(data, dict):
        raise ServoConfigError('{}: should be a mapping with {}'.format(source, ', '.join(sorted(required))))
    missing = required - set(data)
    if missing:
        raise ServoConfigError('{}: missing {}'.format(source, ', '.join(sorted(missing))))
    unknown = set(data) - required - optional
    if unknown:
        raise ServoConfigError('{}: unknown {}'.format(source, ', '.join(sorted(str(key) for key in unknown))))


def _pair(value, name, source, integer=False):
    if not isinstance(value, list) or len(value) != 2:
        raise ServoConfigError('{}: {} should be [first, last]'.format(source, name))
    convert = _integer if integer else _number
    return convert(value[0], name, source), convert(value[1], name, source)


def _number(value, name, source):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ServoConfigError('{}: {} should be a number, not {!r}'.format(source, name, value))
    return value


def _integer(value, name, source):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ServoConfigError('{}: {} should be a whole number, not {!r}'.format(source, name, value))
    return value
