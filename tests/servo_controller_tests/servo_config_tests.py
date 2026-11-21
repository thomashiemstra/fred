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

    def test_servos_are_ordered_by_number_not_by_text(self):
        config = copy.deepcopy(VALID_CONFIG)
        config['servos'][10] = config['servos'].pop(1)
        config['servos'] = {servo_id: config['servos'][servo_id] for servo_id in (10, 7, 2)}
        parsed = parse_servo_config(config)
        self.assertEqual([2, 7, 10], list(parsed.servos))
        self.assertEqual(((2, 10), (7,)), parsed.buses)

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

    def test_values_that_do_not_fit_their_register(self):
        # the register sizes are in dynamixel_x_config.py: pid 2 bytes but at most MAX_GAIN, profile 4 bytes, goal
        # current 2 bytes signed
        for servo_id, field, value, message in (
                (1, 'pid', {'p': 70000, 'i': 0, 'd': 0}, 'servo 1: p should be from 0 to 16383, not 70000'),
                (1, 'pid', {'p': 1, 'i': -1, 'd': 0}, 'servo 1: i should be from 0 to 16383, not -1'),
                (2, 'pid', {'p': 1, 'i': 0, 'd': 16384}, 'servo 2: d should be from 0 to 16383, not 16384'),
                (1, 'profile', {'velocity': 2 ** 32, 'acceleration': 1}, 'servo 1: profile velocity should be from 0'),
                (1, 'profile', {'velocity': 1, 'acceleration': -5}, 'servo 1: profile acceleration should be from 0'),
                (7, 'goal_current', 40000, 'servo 7: goal_current should be from -32768 to 32767, not 40000'),
                (1, 'offset', 2 ** 31, 'servo 1: offset should be from -2147483648'),
                (1, 'positions', [1024, 2 ** 31], 'servo 1: positions should be from -2147483648')):
            with self.subTest(servo_id=servo_id, field=field, value=value):
                self.assert_invalid(config_with(change=(servo_id, field, value)), message)

    def test_values_at_the_edge_of_their_register_are_valid(self):
        config = parse_servo_config(config_with(
            pid=(1, 'pid', {'p': 16383, 'i': 0, 'd': 0}),
            profile=(1, 'profile', {'velocity': 2 ** 32 - 1, 'acceleration': 0}),
            current=(7, 'goal_current', -32768)))
        self.assertEqual((16383, 2 ** 32 - 1, -32768), (config.servos[1].p, config.servos[1].profile_velocity,
                                                        config.servos[7].goal_current))

    def test_profile_with_and_without_move_time(self):
        config = parse_servo_config(config_with(
            time=(1, 'profile', {'velocity': 250, 'acceleration': 100, 'move_time': 15}),
            edge=(2, 'profile', {'velocity': 150, 'acceleration': 50, 'move_time': 32737})))
        self.assertEqual((15, True), (config.servos[1].move_time, config.servos[1].time_based))
        self.assertEqual((250, 100), (config.servos[1].profile_velocity, config.servos[1].profile_acceleration))
        self.assertEqual(32737, config.servos[2].move_time)
        self.assertEqual((None, False), (config.servos[7].move_time, config.servos[7].time_based))
        self.assertEqual(1, parse_servo_config(config_with(
            one=(1, 'profile', {'velocity': 250, 'acceleration': 100, 'move_time': 1}))).servos[1].move_time)

    def test_invalid_move_time(self):
        for move_time in (0, -5, 32738, 2 ** 32, 12.5, '15', True, None):
            with self.subTest(move_time=move_time):
                profile = {'velocity': 250, 'acceleration': 100, 'move_time': move_time}
                self.assert_invalid(config_with(change=(1, 'profile', profile)), 'servo 1: profile move_time')

    def test_move_time_error_gives_the_range(self):
        self.assert_invalid(config_with(change=(1, 'profile', {'velocity': 250, 'acceleration': 100, 'move_time': 0})),
                            'move_time should be from 1 to 32737, not 0')
        self.assert_invalid(config_with(change=(1, 'profile', {'velocity': 250, 'acceleration': 100,
                                                               'move_time': 32738})),
                            'move_time should be from 0 to 32737, not 32738')

    def test_unknown_profile_key_is_still_refused(self):
        self.assert_invalid(config_with(change=(1, 'profile', {'velocity': 250, 'acceleration': 100, 'time': 15})),
                            'profile: unknown time')

    def test_an_ff1_key_under_pid_is_an_unknown_key(self):
        for value in (0, 300, 16383, None):
            with self.subTest(value=value):
                self.assert_invalid(config_with(change=(1, 'pid', {'p': 1500, 'i': 0, 'd': 500, 'ff1': value})),
                                    'servo 1, pid: unknown ff1')

    def test_unknown_pid_key_is_still_refused(self):
        self.assert_invalid(config_with(change=(1, 'pid', {'p': 1500, 'i': 0, 'd': 0, 'gain': 5})),
                            'pid: unknown gain')

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

    def test_an_empty_bus_is_refused(self):
        # config_with removes a field set to None, so the value is put in directly
        for value in (None, '', ' ', ' \t\n'):
            with self.subTest(value=value):
                config = copy.deepcopy(VALID_CONFIG)
                config['servos'][2]['bus'] = value
                self.assert_invalid(config, 'servo configuration, servo 2: bus is empty, give the connector the servo '
                                            r'is on \(servos with the same bus are read together\)')

    def test_an_empty_bus_in_yaml_is_refused(self):
        data = yaml.safe_load('temperature_warning: 45\ntemperature_limit: 50\nservos:\n  3:\n    name: wrist\n'
                              '    bus:\n    mode: position\n    positions: [1024, 3072]\n    angles: [0, 180]\n'
                              '    profile: {velocity: 250, acceleration: 100}\n    pid: {p: 1500, i: 0, d: 0}\n')
        self.assert_invalid(data, 'servo 3: bus is empty')

    def test_a_missing_bus_is_still_refused(self):
        self.assert_invalid(config_with(change=(1, 'bus', None)), 'servo 1: missing bus')

    def test_any_non_empty_bus_label_is_accepted(self):
        for label in ('ttl', 'usb adapter 2', 'x', 485):
            with self.subTest(label=label):
                config = copy.deepcopy(VALID_CONFIG)
                config['servos'][7]['bus'] = label
                self.assertEqual(((1, 2), (7,)), parse_servo_config(config).buses)

    def test_an_empty_name_is_refused(self):
        for value in (None, '', '   '):
            with self.subTest(value=value):
                config = copy.deepcopy(VALID_CONFIG)
                config['servos'][2]['name'] = value
                self.assert_invalid(config, 'servo configuration, servo 2: name is empty$')

    def test_an_empty_model_is_refused(self):
        for value in (None, '', '   '):
            with self.subTest(value=value):
                config = copy.deepcopy(VALID_CONFIG)
                config['servos'][2]['model'] = value
                self.assert_invalid(config, 'servo configuration, servo 2: model is empty, give the dynamixel model '
                                            'or leave the line out')

    def test_the_model_is_optional(self):
        self.assertEqual({}, parse_servo_config(VALID_CONFIG).models)

    def test_whitespace_around_a_name_and_a_model_is_ignored(self):
        config = copy.deepcopy(VALID_CONFIG)
        config['servos'][1]['name'] = '  base '
        config['servos'][1]['model'] = ' XC430-W240 \n'
        parsed = parse_servo_config(config)
        self.assertEqual('base', parsed.names[1])
        self.assertEqual({1: 'XC430-W240'}, parsed.models)

    def test_an_empty_goal_current_is_refused(self):
        for servo_id in (1, 7):
            with self.subTest(servo_id=servo_id):
                config = copy.deepcopy(VALID_CONFIG)
                config['servos'][servo_id]['goal_current'] = None
                self.assert_invalid(config, 'servo configuration, servo {}: goal_current is empty, give the current in '
                                            'mA or leave the line out'.format(servo_id))

    def test_a_current_based_position_servo_needs_a_goal_current(self):
        self.assert_invalid(config_with(change=(7, 'goal_current', None)),
                            r'servo configuration, servo 7: a servo in current_based_position mode needs a goal_current '
                            r'\(the most current in mA it may use\)')

    def test_a_servo_in_another_mode_may_leave_the_goal_current_out(self):
        for mode in ('position', 'extended_position'):
            with self.subTest(mode=mode):
                config = copy.deepcopy(VALID_CONFIG)
                config['servos'][1]['mode'] = mode
                self.assertIsNone(parse_servo_config(config).servos[1].goal_current)

    def test_a_goal_current_on_a_position_servo_is_still_parsed(self):
        config = copy.deepcopy(VALID_CONFIG)
        config['servos'][1]['goal_current'] = 300
        self.assertEqual(300, parse_servo_config(config).servos[1].goal_current)

    def test_whitespace_around_a_bus_label_is_ignored(self):
        config = copy.deepcopy(VALID_CONFIG)
        config['servos'][1]['bus'] = 'ttl'
        config['servos'][2]['bus'] = ' ttl '
        config['servos'][7]['bus'] = 'rs485'
        self.assertEqual(((1, 2), (7,)), parse_servo_config(config).buses)


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
    """the compensation of a servo"""

    def parse(self, **changes):
        return parse_servo_config(config_with(**changes))

    def test_compensation_is_parsed(self):
        config = self.parse(c=(2, 'compensation', {'max_positions': 60, 'terms': {'cos_q2': 12.0, 'sin_phi': 5}}))
        compensation = config.servos[2].compensation
        self.assertEqual(2, compensation.joint)
        self.assertEqual({'cos_q2': 12.0, 'sin_phi': 5.0}, compensation.terms)
        self.assertEqual(60, compensation.max_positions)
        self.assertIsNone(config.servos[1].compensation)

    def test_per_direction_blocks_are_refused_with_an_explanation(self):
        blocks = {'up': {'cos_q2': 14.0}, 'down': {'cos_q2': 10.0}}
        for given in (['up'], ['down'], ['up', 'down']):
            compensation = {'max_positions': 60, 'terms': {'cos_q2': 12.0}, **{name: blocks[name] for name in given}}
            with self.assertRaisesRegex(ServoConfigError, 'per-direction blocks.*removed.*is for gravity only',
                                        msg=repr(given)) as raised:
                self.parse(c=(2, 'compensation', compensation))
            for name in given:
                self.assertIn(name, str(raised.exception))
        # also when the rest of the block is wrong: the removed option is the first thing to say
        with self.assertRaisesRegex(ServoConfigError, 'per-direction blocks'):
            self.parse(c=(2, 'compensation', {'max_positions': 60, 'up': {'cos_q2': 1.0}}))

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
            {'max_positions': 60, 'terms': {'spring_dl': 1.0}},                 # the term of the removed spring
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

    def test_a_spring_section_is_rejected_since_the_model_was_removed(self):
        spring = {'base_anchor': [-4.0, -6.0], 'arm_anchor': [2.0, 3.0]}
        for section in (spring, {}, [1, 2]):
            with self.assertRaisesRegex(ServoConfigError, 'spring model was removed.*delete the spring section',
                                        msg=repr(section)):
                self.parse(s=(None, 'spring', section))
        # a term of the spring in a compensation is an unknown term like any other
        with self.assertRaisesRegex(ServoConfigError, 'unknown terms spring_dl'):
            self.parse(c=(2, 'compensation', {'max_positions': 60, 'terms': {'spring_dl': 1.0}}))
        with self.assertRaisesRegex(ServoConfigError, 'servo_config.yaml'):
            parse_servo_config(dict(config_with(), spring=spring), 'servo_config.yaml')


class VelocityFeedforwardConfigTests(unittest.TestCase):
    """the velocity feedforward (velocity lead) of a servo"""

    def parse(self, **changes):
        return parse_servo_config(config_with(**changes))

    def test_velocity_feedforward_is_parsed(self):
        config = self.parse(m=(2, 'velocity_feedforward', {'max_positions': 30, 'pwm_per_degree_per_second': 14.2}))
        velocity_feedforward = config.servos[2].velocity_feedforward
        self.assertEqual(14.2, velocity_feedforward.pwm_per_degree_per_second)
        self.assertEqual(30, velocity_feedforward.max_positions)

    def test_a_whole_number_and_zero_are_valid_coefficients(self):
        for coefficient in (14, 0):
            config = self.parse(m=(2, 'velocity_feedforward',
                                   {'max_positions': 30, 'pwm_per_degree_per_second': coefficient}))
            self.assertEqual(float(coefficient), config.servos[2].velocity_feedforward.pwm_per_degree_per_second)

    def test_a_servo_without_the_block_has_no_velocity_feedforward(self):
        config = self.parse(m=(2, 'velocity_feedforward', {'max_positions': 30, 'pwm_per_degree_per_second': 14.2}))
        self.assertIsNone(config.servos[1].velocity_feedforward)
        self.assertIsNone(config.servos[7].velocity_feedforward)
        self.assertIsNone(parse_servo_config(VALID_CONFIG).servos[2].velocity_feedforward)

    def test_velocity_feedforward_and_compensation_go_together(self):
        config = self.parse(c=(2, 'compensation', {'max_positions': 60, 'terms': {'cos_q2': 12.0}}),
                            m=(2, 'velocity_feedforward', {'max_positions': 30, 'pwm_per_degree_per_second': 14.2}))
        self.assertEqual({'cos_q2': 12.0}, config.servos[2].compensation.terms)
        self.assertEqual(30, config.servos[2].velocity_feedforward.max_positions)

    def test_velocity_feedforward_is_shown_in_the_servo(self):
        config = self.parse(m=(2, 'velocity_feedforward', {'max_positions': 30, 'pwm_per_degree_per_second': 14.2}))
        self.assertIn('VelocityFeedforward(pwm_per_degree_per_second=14.2, max_positions=30)', str(config.servos[2]))

    def test_invalid_velocity_feedforwards(self):
        invalid = [
            {'pwm_per_degree_per_second': 14.2},                        # no max_positions
            {'max_positions': 30},                                      # no pwm_per_degree_per_second
            {},
            {'max_positions': 30, 'pwm_per_degree_per_second': -1.0},
            {'max_positions': 30, 'pwm_per_degree_per_second': -0.001},
            {'max_positions': 30, 'pwm_per_degree_per_second': float('nan')},
            {'max_positions': 30, 'pwm_per_degree_per_second': 'fast'},
            {'max_positions': 30, 'pwm_per_degree_per_second': True},
            {'max_positions': 30, 'pwm_per_degree_per_second': None},
            {'max_positions': 0, 'pwm_per_degree_per_second': 14.2},
            {'max_positions': -3, 'pwm_per_degree_per_second': 14.2},
            {'max_positions': 30.5, 'pwm_per_degree_per_second': 14.2},
            {'max_positions': '30', 'pwm_per_degree_per_second': 14.2},
            {'max_positions': True, 'pwm_per_degree_per_second': 14.2},
            {'max_positions': 30, 'pwm_per_degree_per_second': 14.2, 'friction': 1.0},   # unknown key
            {'max_positions': 30, 'velocity': 14.2},                    # the old key
            [30, 14.2],
            14.2,
        ]
        for velocity_feedforward in invalid:
            with self.assertRaises(ServoConfigError, msg=repr(velocity_feedforward)):
                self.parse(m=(2, 'velocity_feedforward', velocity_feedforward))

    def test_the_errors_name_the_servo_and_the_problem(self):
        source = 'servo 2, velocity_feedforward: '
        with self.assertRaisesRegex(ServoConfigError, source + 'pwm_per_degree_per_second .* should be 0 or more, not -1.0'):
            self.parse(m=(2, 'velocity_feedforward', {'max_positions': 30, 'pwm_per_degree_per_second': -1.0}))
        with self.assertRaisesRegex(ServoConfigError, source + 'max_positions should be positive, not 0'):
            self.parse(m=(2, 'velocity_feedforward', {'max_positions': 0, 'pwm_per_degree_per_second': 14.2}))
        with self.assertRaisesRegex(ServoConfigError, source + 'max_positions should be a whole number'):
            self.parse(m=(2, 'velocity_feedforward', {'max_positions': 30.5, 'pwm_per_degree_per_second': 14.2}))
        with self.assertRaisesRegex(ServoConfigError, source + 'missing pwm_per_degree_per_second'):
            self.parse(m=(2, 'velocity_feedforward', {'max_positions': 30}))
        with self.assertRaisesRegex(ServoConfigError, source + 'unknown friction'):
            self.parse(m=(2, 'velocity_feedforward',
                          {'max_positions': 30, 'pwm_per_degree_per_second': 14.2, 'friction': 1}))

    def test_the_old_key_velocity_is_refused_and_the_error_names_the_keys_to_write(self):
        with self.assertRaisesRegex(ServoConfigError,
                                    'servo 2, velocity_feedforward: unknown velocity, the keys of a '
                                    'velocity_feedforward block are max_positions and pwm_per_degree_per_second'):
            self.parse(m=(2, 'velocity_feedforward', {'max_positions': 30, 'velocity': 14.2}))

    def test_a_motion_key_on_the_servo_is_refused_like_any_unknown_field(self):
        with self.assertRaisesRegex(ServoConfigError, 'servo 2.*unknown.*motion'):
            self.parse(m=(2, 'motion', {'max_positions': 30, 'pwm_per_degree_per_second': 14.2}))

    def test_a_servo_without_angles_can_not_have_a_velocity_feedforward(self):
        with self.assertRaisesRegex(ServoConfigError,
                                    'servo 7.*only a servo with angles can have a velocity_feedforward'):
            self.parse(m=(7, 'velocity_feedforward', {'max_positions': 30, 'pwm_per_degree_per_second': 14.2}))

    def test_the_velocity_feedforward_block_in_a_yaml_file_is_loaded(self):
        data = config_with()
        data['servos'][2]['velocity_feedforward'] = {'max_positions': 30, 'pwm_per_degree_per_second': 14.2}
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'servos.yaml')
            with open(path, 'w') as config_file:
                yaml.safe_dump(data, config_file)
            config = load_servo_config(path)
        self.assertEqual(14.2, config.servos[2].velocity_feedforward.pwm_per_degree_per_second)

    def test_the_real_servo_configuration_still_loads_with_or_without_velocity_feedforward(self):
        config = load_servo_config(DEFAULT_SERVO_CONFIG_PATH)
        self.assertEqual(list(range(1, 8)), list(config.servos))
        for servo in config.servos.values():  # a servo with one has a VelocityFeedforward, the others have none
            if servo.velocity_feedforward is not None:
                self.assertGreater(servo.velocity_feedforward.max_positions, 0)


if __name__ == '__main__':
    unittest.main()
