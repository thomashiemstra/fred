"""Loads the link lengths of the robots from a yaml file, see robot_config.yaml for the format."""
import os

import yaml

from src.kinematics.kinematics_utils import RobotConfig

DEFAULT_ROBOT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'robot_config.yaml')

LENGTHS = ('d1', 'a2', 'd4', 'd6')
ROBOTS = ('dynamixel_robot', 'simulated_robot')


class RobotConfigError(ValueError):
    """raised when the robot configuration is not valid"""


def load_robot_configs(path=DEFAULT_ROBOT_CONFIG_PATH):
    """
    read the link lengths of the robots, returns a mapping of robot name ('dynamixel_robot', 'simulated_robot') to a
    new RobotConfig. Every call returns new objects.
    :raises RobotConfigError: if the configuration is not valid
    """
    try:
        with open(path) as config_file:
            data = yaml.safe_load(config_file)
    except (OSError, yaml.YAMLError) as e:
        raise RobotConfigError('can not read the robot configuration {}: {}'.format(path, e)) from e
    return parse_robot_configs(data, path)


def parse_robot_configs(data, source='robot configuration'):
    """
    create the RobotConfig objects from the parsed yaml data: a mapping of robot name to the lengths d1, a2, d4 and d6
    in cm. The dynamixel robot is required, the simulated robot gets the same lengths when it has no section.
    """
    if not isinstance(data, dict):
        raise RobotConfigError('{}: should be a mapping with {}'.format(source, ', '.join(ROBOTS)))
    unknown = set(data) - set(ROBOTS)
    if unknown:
        raise RobotConfigError('{}: unknown {}'.format(source, ', '.join(sorted(str(key) for key in unknown))))
    if 'dynamixel_robot' not in data:
        raise RobotConfigError('{}: missing dynamixel_robot'.format(source))

    configs = {}
    for robot in ROBOTS:
        lengths = data.get(robot, data['dynamixel_robot'])
        configs[robot] = _parse_lengths(lengths, '{}, {}'.format(source, robot))
    return configs


def _parse_lengths(data, source):
    if not isinstance(data, dict):
        raise RobotConfigError('{}: should be a mapping with {}'.format(source, ', '.join(LENGTHS)))
    missing = set(LENGTHS) - set(data)
    if missing:
        raise RobotConfigError('{}: missing {}'.format(source, ', '.join(sorted(missing))))
    unknown = set(data) - set(LENGTHS)
    if unknown:
        raise RobotConfigError('{}: unknown {}'.format(source, ', '.join(sorted(str(key) for key in unknown))))

    lengths = {}
    for name in LENGTHS:
        value = data[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RobotConfigError('{}: {} should be a number in cm, not {!r}'.format(source, name, value))
        if value < 0:
            raise RobotConfigError('{}: {} should not be negative, got {}'.format(source, name, value))
        lengths[name] = float(value)
    return RobotConfig(**lengths)
