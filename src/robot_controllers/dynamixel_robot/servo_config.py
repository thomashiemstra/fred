"""Loads the servo configuration of the robot from a yaml file, see servo_config.yaml for the format."""
import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import yaml

from src.global_constants import POSITION_CONTROL_MODE, EXTENDED_POSITION_CONTROL, CURRENT_BASED_POSITION_CONTROL_MODE
from src.robot_controllers.dynamixel_robot import dynamixel_x_config as cfg
from src.robot_config import load_robot_configs
from src.robot_controllers.dynamixel_robot.servo import Servo
from src.robot_controllers.dynamixel_robot.static_compensation import SpringGeometry, StaticCompensation, \
    term_names, SPRING_TERMS, UP, DOWN

DEFAULT_SERVO_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'servo_config.yaml')

OPERATING_MODES = {
    'position': POSITION_CONTROL_MODE,
    'extended_position': EXTENDED_POSITION_CONTROL,
    'current_based_position': CURRENT_BASED_POSITION_CONTROL_MODE,
}

_REQUIRED_SERVO_FIELDS = {'name', 'bus', 'mode', 'positions', 'profile', 'pid'}
_OPTIONAL_SERVO_FIELDS = {'model', 'angles', 'states', 'offset', 'goal_current', 'compensation'}
_REQUIRED_FIELDS = {'temperature_warning', 'temperature_limit', 'servos'}
_OPTIONAL_FIELDS = {'spring'}
_REQUIRED_COMPENSATION_FIELDS = {'max_positions', 'terms'}
_OPTIONAL_COMPENSATION_FIELDS = {UP, DOWN}

# the register sizes in bytes of the values that are written to the servos, and whether they are signed
_POSITION_REGISTER = (cfg.LEN_GOAL_POSITION, True)
_PROFILE_ACCELERATION_REGISTER = (cfg.LEN_PROFILE_ACCELERATION, False)
_PROFILE_VELOCITY_REGISTER = (cfg.LEN_PROFILE_VELOCITY, False)
_PID_REGISTER = (cfg.LEN_PID, False, cfg.MAX_GAIN)  # the servos accept 0..16383 and drop a larger gain silently
_GOAL_CURRENT_REGISTER = (cfg.LEN_GOAL_CURRENT, True)


class ServoConfigError(ValueError):
    """raised when the servo configuration is not valid"""


class ServoConfig:
    """the servos of the robot and the settings that apply to all of them"""

    def __init__(self, servos: dict[int, Servo], names: dict[int, str], buses: tuple[tuple[int, ...], ...],
                 temperature_warning: float, temperature_limit: float, models: dict[int, str] | None = None,
                 spring: SpringGeometry | None = None) -> None:
        self.servos = servos  # mapping of servo id to Servo, ordered by servo id
        self.names = names  # mapping of servo id to name
        self.models = models if models is not None else {}  # mapping of servo id to model name, if given
        self.buses = buses  # tuple of tuples of the servo ids on the same bus
        self.temperature_warning = temperature_warning  # degrees celsius
        self.temperature_limit = temperature_limit  # degrees celsius
        self.spring = spring  # SpringGeometry of the spring between the servo 2 and 3 housings, or None


def load_servo_config(path: str = DEFAULT_SERVO_CONFIG_PATH, a2: float | None = None) -> ServoConfig:
    """
    read the servo configuration, every call returns new Servo objects
    :param a2: the length of the lower arm in cm for the spring geometry, by default from robot_config.yaml
    :raises ServoConfigError: if the configuration is not valid
    """
    try:
        with open(path) as config_file:
            data = yaml.safe_load(config_file)
    except (OSError, yaml.YAMLError) as e:
        raise ServoConfigError('can not read the servo configuration {}: {}'.format(path, e)) from e
    return parse_servo_config(data, path, a2)


def parse_servo_config(data: Any, source: str = 'servo configuration', a2: float | None = None) -> ServoConfig:
    """
    create the servo configuration from the parsed yaml data
    :param a2: the length of the lower arm in cm for the spring geometry, by default from robot_config.yaml
    """
    _check_fields(data, _REQUIRED_FIELDS, _OPTIONAL_FIELDS, source)
    temperature_warning = _number(data['temperature_warning'], 'temperature_warning', source)
    temperature_limit = _number(data['temperature_limit'], 'temperature_limit', source)
    if temperature_warning > temperature_limit:
        raise ServoConfigError('{}: temperature_warning is higher than temperature_limit'.format(source))

    spring = None
    if 'spring' in data:
        if a2 is None:
            a2 = load_robot_configs()['dynamixel_robot'].a2
        spring = _parse_spring(data['spring'], source + ', spring', a2)

    if not isinstance(data['servos'], dict) or not data['servos']:
        raise ServoConfigError('{}: servos should be a mapping of servo id to servo'.format(source))

    buses: dict[str, list[int]]
    servos, names, models, buses = {}, {}, {}, {}
    # whole numbers first in numeric order, anything else after them (it is rejected below)
    for servo_id in sorted(data['servos'],
                           key=lambda key: (not isinstance(key, int), key if isinstance(key, int) else 0, str(key))):
        if not isinstance(servo_id, int) or isinstance(servo_id, bool) or not 0 <= servo_id <= 252:
            raise ServoConfigError('{}: servo id {!r} should be a number from 0 to 252'.format(source, servo_id))
        servo_data = data['servos'][servo_id]
        servo_source = '{}, servo {}'.format(source, servo_id)
        servos[servo_id] = _parse_servo(servo_data, servo_source, servo_id, spring)
        names[servo_id] = str(servo_data['name'])
        if 'model' in servo_data:
            models[servo_id] = str(servo_data['model'])
        buses.setdefault(str(servo_data['bus']), []).append(servo_id)

    return ServoConfig(servos, names, tuple(tuple(bus) for bus in buses.values()),
                       temperature_warning, temperature_limit, models, spring)


def _parse_servo(data: Any, source: str, servo_id: int, spring: SpringGeometry | None) -> Servo:
    _check_fields(data, _REQUIRED_SERVO_FIELDS, _OPTIONAL_SERVO_FIELDS, source)

    if data['mode'] not in OPERATING_MODES:
        raise ServoConfigError('{}: unknown mode {!r}, use one of {}'.format(
            source, data['mode'], ', '.join(OPERATING_MODES)))

    first_position, last_position = _pair(data['positions'], 'positions', source, integer=True)
    for position in (first_position, last_position):
        _register(position, 'positions', source, *_POSITION_REGISTER)
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
        goal_current = _register(goal_current, 'goal_current', source, *_GOAL_CURRENT_REGISTER)

    compensation = None
    if 'compensation' in data:
        if 'angles' not in data:
            raise ServoConfigError('{}: only a servo with angles can have a compensation'.format(source))
        compensation = _parse_compensation(data['compensation'], source + ', compensation', servo_id, spring)

    # Servo expects the smallest angle first, the positions follow the angles
    if first > last:
        first, last = last, first
        first_position, last_position = last_position, first_position

    return Servo(first_position, last_position, first, last, OPERATING_MODES[data['mode']],
                 profile_velocity=_register(profile['velocity'], 'profile velocity', source,
                                            *_PROFILE_VELOCITY_REGISTER),
                 profile_acceleration=_register(profile['acceleration'], 'profile acceleration', source,
                                                *_PROFILE_ACCELERATION_REGISTER),
                 p=_register(pid['p'], 'p', source, *_PID_REGISTER), i=_register(pid['i'], 'i', source, *_PID_REGISTER),
                 d=_register(pid['d'], 'd', source, *_PID_REGISTER),
                 offset=_register(data.get('offset', 0), 'offset', source, *_POSITION_REGISTER),
                 goal_current=goal_current,
                 compensation=compensation)


def _parse_spring(data: Any, source: str, a2: float) -> SpringGeometry:
    _check_fields(data, {'base_anchor', 'arm_anchor'}, set(), source)
    return SpringGeometry(_pair(data['base_anchor'], 'base_anchor', source),
                          _pair(data['arm_anchor'], 'arm_anchor', source), a2)


def _parse_compensation(data: Any, source: str, servo_id: int, spring: SpringGeometry | None) -> StaticCompensation:
    _check_fields(data, _REQUIRED_COMPENSATION_FIELDS, _OPTIONAL_COMPENSATION_FIELDS, source)
    try:
        names = term_names(servo_id)
    except ValueError as e:
        raise ServoConfigError('{}: {}'.format(source, e)) from e
    max_positions = _integer(data['max_positions'], 'max_positions', source)
    if max_positions <= 0:
        raise ServoConfigError('{}: max_positions should be positive, not {}'.format(source, max_positions))
    if (UP in data) != (DOWN in data):
        raise ServoConfigError('{}: up and down have to be given together'.format(source))

    terms = {}
    for field in ('terms', UP, DOWN):
        if field not in data:
            continue
        terms[field] = _terms(data[field], '{}, {}'.format(source, field), names, spring)
    return StaticCompensation(servo_id, terms['terms'], max_positions, terms.get(UP), terms.get(DOWN), spring)


def _terms(data: Any, source: str, names: Sequence[str], spring: SpringGeometry | None) -> dict[str, float]:
    if not isinstance(data, dict):
        raise ServoConfigError('{}: should be a mapping of term to coefficient in positions'.format(source))
    unknown = set(data) - set(names)
    if unknown:
        raise ServoConfigError('{}: unknown terms {}, the terms are {}'.format(
            source, ', '.join(sorted(str(name) for name in unknown)), ', '.join(names)))
    if spring is None and any(name in SPRING_TERMS for name in data):
        raise ServoConfigError('{}: the spring terms need a spring section'.format(source))
    return {name: float(_number(value, name, source)) for name, value in data.items()}


def _check_fields(data: Any, required: set[str], optional: set[str], source: str) -> None:
    if not isinstance(data, dict):
        raise ServoConfigError('{}: should be a mapping with {}'.format(source, ', '.join(sorted(required))))
    missing = required - set(data)
    if missing:
        raise ServoConfigError('{}: missing {}'.format(source, ', '.join(sorted(missing))))
    unknown = set(data) - required - optional
    if unknown:
        raise ServoConfigError('{}: unknown {}'.format(source, ', '.join(sorted(str(key) for key in unknown))))


def _pair(value: Any, name: str, source: str, integer: bool = False) -> tuple[Any, Any]:
    if not isinstance(value, list) or len(value) != 2:
        raise ServoConfigError('{}: {} should be [first, last]'.format(source, name))
    convert = _integer if integer else _number
    return convert(value[0], name, source), convert(value[1], name, source)


def _number(value: Any, name: str, source: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ServoConfigError('{}: {} should be a number, not {!r}'.format(source, name, value))
    return value


def _integer(value: Any, name: str, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ServoConfigError('{}: {} should be a whole number, not {!r}'.format(source, name, value))
    return value


def _register(value: Any, name: str, source: str, length: int, signed: bool, maximum: int | None = None) -> int:
    """a whole number that has to fit in a register of the given size in bytes, and below the maximum if given"""
    value = _integer(value, name, source)
    bits = 8 * length
    lowest, highest = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
    if maximum is not None:
        highest = min(highest, maximum)
    if not lowest <= value <= highest:
        raise ServoConfigError('{}: {} should be from {} to {}, not {}'.format(source, name, lowest, highest, value))
    return value
