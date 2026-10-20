import copy
import os
import tempfile
import unittest

import yaml
from numpy import pi

from src.global_constants import POSITION_CONTROL_MODE, CURRENT_BASED_POSITION_CONTROL_MODE
from src.robot_controllers.dynamixel_robot.servo_config import load_servo_config, parse_servo_config, \
    ServoConfigError, DEFAULT_SERVO_CONFIG_PATH

VALID_CONFIG = {
    'temperature_warning': 45,
    'temperature_limit': 50,
    'servos': {
        1: {'name': 'base', 'bus': 'rs485', 'mode': 'position', 'positions': [1024, 3072], 'angles': [0, 180],
            'offset': 354, 'profile': {'velocity': 250, 'acceleration': 100}, 'pid': {'p': 1500, 'i': 0, 'd': 500}},
        2: {'name': 'elbow', 'bus': 'rs485', 'mode': 'position', 'positions': [3072, 1024], 'angles': [-90, 90],
            'profile': {'velocity': 150, 'acceleration': 50}, 'pid': {'p': 1500, 'i': 0, 'd': 0}},
        7: {'name': 'gripper', 'bus': 'ttl', 'mode': 'current_based_position', 'positions': [1024, 2048],
            'states': [0, 100], 'goal_current': 400, 'profile': {'velocity': 50, 'acceleration': 50},
            'pid': {'p': 500, 'i': 0, 'd': 0}},
    },
}


def config_with(**changes):
    """a copy of the valid config, the changes are (servo id or None, field, value), a value of None removes it"""
    config = copy.deepcopy(VALID_CONFIG)
    for servo_id, field, value in changes.values():
        target = config if servo_id is None else config['servos'][servo_id]
        if value is None:
            del target[field]
        else:
            target[field] = value
    return config


class ParseServoConfigTests(unittest.TestCase):

    def test_valid_config(self):
        config = parse_servo_config(VALID_CONFIG)

        self.assertEqual([1, 2, 7], list(config.servos))
        self.assertEqual(((1, 2), (7,)), config.buses)
        self.assertEqual({1: 'base', 2: 'elbow', 7: 'gripper'}, config.names)
        self.assertEqual((45, 50), (config.temperature_warning, config.temperature_limit))

        base = config.servos[1]
        self.assertEqual((1024, 3072, 0, pi), (base.min_position, base.max_position, base.min_angle, base.max_angle))
        self.assertEqual((354, 250, 100), (base.constant_offset, base.profile_velocity, base.profile_acceleration))
        self.assertEqual((1500, 0, 500, POSITION_CONTROL_MODE, None), (base.p, base.i, base.d, base.operating_mode,
                                                                       base.goal_current))

        gripper = config.servos[7]
        self.assertEqual((0, 100), (gripper.min_angle, gripper.max_angle))
        self.assertEqual((CURRENT_BASED_POSITION_CONTROL_MODE, 400, 0),
                         (gripper.operating_mode, gripper.goal_current, gripper.constant_offset))

    def test_reversed_servo(self):
        elbow = parse_servo_config(VALID_CONFIG).servos[2]
        self.assertEqual((3072, 1024, -pi / 2, pi / 2),
                         (elbow.min_position, elbow.max_position, elbow.min_angle, elbow.max_angle))

    def test_descending_angles_keep_the_positions_with_the_angles(self):
        config = config_with(change=(2, 'angles', [90, -90]))
        elbow = parse_servo_config(config).servos[2]
        self.assertEqual((1024, 3072, -pi / 2, pi / 2),
                         (elbow.min_position, elbow.max_position, elbow.min_angle, elbow.max_angle))

    def test_every_load_creates_new_servos(self):
        self.assertIsNot(parse_servo_config(VALID_CONFIG).servos[1], parse_servo_config(VALID_CONFIG).servos[1])

    def assert_invalid(self, config, message):
        with self.assertRaisesRegex(ServoConfigError, message):
            parse_servo_config(config)

    def test_missing_field(self):
        self.assert_invalid(config_with(change=(1, 'pid', None)), 'servo 1: missing pid')
        self.assert_invalid(config_with(change=(None, 'temperature_limit', None)), 'missing temperature_limit')

    def test_unknown_field(self):
        self.assert_invalid(config_with(change=(1, 'ofset', 3)), 'servo 1: unknown ofset')
        self.assert_invalid(config_with(change=(1, 'pid', {'p': 1, 'i': 0, 'd': 0, 'f': 2})), 'pid: unknown f')

    def test_unknown_mode(self):
        self.assert_invalid(config_with(change=(1, 'mode', 'velocity')), "unknown mode 'velocity'")

    def test_angles_or_states(self):
        self.assert_invalid(config_with(change=(1, 'states', [0, 100])), 'either angles')
        self.assert_invalid(config_with(change=(1, 'angles', None)), 'either angles')

    def test_invalid_values(self):
        self.assert_invalid(config_with(change=(1, 'positions', [1024])), r'positions should be \[first, last\]')
        self.assert_invalid(config_with(change=(1, 'positions', [1024, 1024])), 'are the same')
        self.assert_invalid(config_with(change=(1, 'positions', [1024.5, 3072])), 'whole number')
        self.assert_invalid(config_with(change=(1, 'offset', 'a lot')), 'offset should be a whole number')
        self.assert_invalid(config_with(change=(1, 'goal_current', True)), 'goal_current should be a whole number')
        self.assert_invalid(config_with(change=(None, 'temperature_warning', 60)), 'higher than temperature_limit')

    def test_invalid_servo_id(self):
        for servo_id in ('one', -1, 253, True, 1.0):
            config = copy.deepcopy(VALID_CONFIG)
            config['servos'][servo_id] = config['servos'].pop(1)
            self.assert_invalid(config, 'servo id {!r}'.format(servo_id))

    def test_highest_and_lowest_servo_ids_are_valid(self):
        config = copy.deepcopy(VALID_CONFIG)
        config['servos'][0] = config['servos'].pop(1)
        config['servos'][252] = config['servos'].pop(2)
        self.assertEqual({0, 7, 252}, set(parse_servo_config(config).servos))

    def test_servos_are_ordered_by_id(self):
        config = copy.deepcopy(VALID_CONFIG)
        config['servos'] = {servo_id: config['servos'][servo_id] for servo_id in (7, 2, 1)}
        self.assertEqual([1, 2, 7], list(parse_servo_config(config).servos))

    def test_no_servos(self):
        self.assert_invalid(config_with(change=(None, 'servos', {})), 'servos should be a mapping')
        self.assert_invalid(config_with(change=(None, 'servos', [1, 2])), 'servos should be a mapping')

    def test_config_that_is_not_a_mapping(self):
        self.assert_invalid(None, 'should be a mapping')
        self.assert_invalid(config_with(change=(None, 'servos', {1: 'base'})), 'servo 1: should be a mapping')
        self.assert_invalid(config_with(change=(1, 'profile', 250)), 'profile: should be a mapping')

    def test_unknown_top_level_field(self):
        self.assert_invalid(config_with(change=(None, 'baudrate', 57600)), 'unknown baudrate')

    def test_invalid_temperatures(self):
        self.assert_invalid(config_with(change=(None, 'temperature_limit', '50C')), 'temperature_limit should be a number')
        self.assert_invalid(config_with(change=(None, 'temperature_warning', True)), 'temperature_warning should be')
        # equal temperatures and fractional temperatures are fine
        config = parse_servo_config(config_with(warning=(None, 'temperature_warning', 49.5),
                                                limit=(None, 'temperature_limit', 49.5)))
        self.assertEqual((49.5, 49.5), (config.temperature_warning, config.temperature_limit))

    def test_invalid_profile_and_pid(self):
        self.assert_invalid(config_with(change=(1, 'profile', {'velocity': 250})), 'profile: missing acceleration')
        self.assert_invalid(config_with(change=(1, 'profile', {'velocity': 2.5, 'acceleration': 1})),
                            'profile velocity should be a whole number')
        self.assert_invalid(config_with(change=(1, 'pid', {'p': 1500, 'i': None, 'd': 0})), 'i should be a whole number')

    def test_invalid_pairs(self):
        self.assert_invalid(config_with(change=(1, 'positions', 1024)), r'positions should be \[first, last\]')
        self.assert_invalid(config_with(change=(1, 'positions', [0, 1024, 2048])), r'positions should be \[first, last\]')
        self.assert_invalid(config_with(change=(1, 'angles', [0, 'pi'])), 'angles should be a number')

    def test_descending_states_keep_the_positions_with_the_states(self):
        gripper = parse_servo_config(config_with(change=(7, 'states', [100, 0]))).servos[7]
        self.assertEqual((2048, 1024, 0, 100),
                         (gripper.min_position, gripper.max_position, gripper.min_angle, gripper.max_angle))

    def test_servos_on_the_same_bus_are_grouped(self):
        config = copy.deepcopy(VALID_CONFIG)
        config['servos'][2]['bus'] = 'ttl'
        self.assertEqual(((1,), (2, 7)), parse_servo_config(config).buses)


class LoadServoConfigTests(unittest.TestCase):

    def test_robot_config_is_valid(self):
        # only checks what the robot controller needs, not the calibrated values or the wiring
        config = load_servo_config(DEFAULT_SERVO_CONFIG_PATH)
        self.assertEqual(list(range(1, 8)), list(config.servos))
        self.assertEqual(sorted(config.servos), sorted(servo_id for bus in config.buses for servo_id in bus))

    def test_load_from_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'servos.yaml')
            with open(path, 'w') as config_file:
                yaml.safe_dump(VALID_CONFIG, config_file)
            self.assertEqual([1, 2, 7], list(load_servo_config(path).servos))

    def test_missing_file(self):
        with self.assertRaisesRegex(ServoConfigError, 'can not read the servo configuration'):
            load_servo_config('does_not_exist.yaml')

    def test_invalid_yaml_mentions_the_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'servos.yaml')
            with open(path, 'w') as config_file:
                config_file.write('servos: [1, 2\n')
            with self.assertRaisesRegex(ServoConfigError, 'servos.yaml'):
                load_servo_config(path)


class CompensationConfigTests(unittest.TestCase):
    """the compensation of a servo and the spring section"""

    SPRING = {'base_anchor': [-4.0, -6.0], 'arm_anchor': [2.0, 3.0]}

    def parse(self, **changes):
        return parse_servo_config(config_with(**changes), a2=20.0)

    def test_compensation_is_parsed(self):
        config = self.parse(c=(2, 'compensation', {'max_positions': 60, 'terms': {'cos_q2': 12.0, 'sin_phi': 5}}))
        compensation = config.servos[2].compensation
        self.assertEqual(2, compensation.joint)
        self.assertEqual({'cos_q2': 12.0, 'sin_phi': 5.0}, compensation.terms)
        self.assertEqual(60, compensation.max_positions)
        self.assertEqual({}, compensation.directional)
        self.assertIsNone(config.servos[1].compensation)
        self.assertIsNone(config.spring)

    def test_directional_compensation(self):
        config = self.parse(c=(2, 'compensation', {'max_positions': 60, 'terms': {'cos_q2': 12.0},
                                                   'up': {'cos_q2': 14.0}, 'down': {'cos_q2': 10.0}}))
        self.assertEqual({'up': {'cos_q2': 14.0}, 'down': {'cos_q2': 10.0}}, config.servos[2].compensation.directional)

    def test_spring_and_spring_terms(self):
        config = self.parse(s=(None, 'spring', self.SPRING),
                            c=(2, 'compensation', {'max_positions': 60, 'terms': {'spring_dl': 1.5, 'cos_q2': -0.1}}))
        self.assertEqual([-4.0, -6.0], list(config.spring.base_anchor))
        self.assertEqual([2.0, 3.0], list(config.spring.arm_anchor))
        self.assertEqual(20.0, config.spring.a2)
        self.assertIs(config.spring, config.servos[2].compensation.spring)

    def test_models_are_kept(self):
        config = self.parse(m=(1, 'model', 'XM430-W350'))
        self.assertEqual({1: 'XM430-W350'}, config.models)

    def test_invalid_compensations(self):
        invalid = [
            {'terms': {'cos_q2': 1.0}},                                         # no max_positions
            {'max_positions': 0, 'terms': {'cos_q2': 1.0}},
            {'max_positions': 60.5, 'terms': {'cos_q2': 1.0}},
            {'max_positions': 60},                                              # no terms
            {'max_positions': 60, 'terms': {'nope': 1.0}},
            {'max_positions': 60, 'terms': {'cos_q2': 'one'}},
            {'max_positions': 60, 'terms': [1.0]},
            {'max_positions': 60, 'terms': {'cos_q2': 1.0}, 'up': {'cos_q2': 1.0}},  # up without down
            {'max_positions': 60, 'terms': {'spring_dl': 1.0}},                 # no spring section
            {'max_positions': 60, 'terms': {'cos_q2': 1.0}, 'sideways': {}},
        ]
        for compensation in invalid:
            with self.assertRaises(ServoConfigError, msg=repr(compensation)):
                self.parse(c=(2, 'compensation', compensation))
        # joint 1 has no terms, the gripper has states instead of angles
        with self.assertRaises(ServoConfigError):
            self.parse(c=(1, 'compensation', {'max_positions': 60, 'terms': {}}))
        with self.assertRaises(ServoConfigError):
            self.parse(c=(7, 'compensation', {'max_positions': 60, 'terms': {}}))

    def test_invalid_spring(self):
        for spring in ({'base_anchor': [0, 0]}, {'base_anchor': [0, 0], 'arm_anchor': [1]},
                       {'base_anchor': [0, 0], 'arm_anchor': [1, 'a']}, [1, 2],
                       {'base_anchor': [0, 0], 'arm_anchor': [1, 2], 'k': 3}):
            with self.assertRaises(ServoConfigError, msg=repr(spring)):
                self.parse(s=(None, 'spring', spring))

    def test_spring_takes_a2_from_the_robot_config_by_default(self):
        from src.robot_config import load_robot_configs
        config = parse_servo_config(config_with(s=(None, 'spring', self.SPRING)))
        self.assertEqual(load_robot_configs()['dynamixel_robot'].a2, config.spring.a2)


if __name__ == '__main__':
    unittest.main()
