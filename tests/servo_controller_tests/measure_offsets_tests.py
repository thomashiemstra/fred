import os
import unittest
from unittest import mock

from numpy import pi

from src.global_constants import POSITION_CONTROL_MODE
from src.robot_controllers.dynamixel_robot.calibration import measure_offsets
from src.robot_controllers.dynamixel_robot.servo import Servo
from src.robot_controllers.dynamixel_robot.servo_config import load_servo_config, parse_servo_config
import yaml

TEST_SERVO_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_servo_config.yaml')


class OffsetArithmeticTests(unittest.TestCase):

    def test_offset_is_raw_minus_calibrated_position(self):
        servo = Servo(1024, 3072, 0, pi, POSITION_CONTROL_MODE, offset=-464)
        # the jig holds the joint at 0 degrees and the servo reads 457
        self.assertEqual(measure_offsets.calibrated_position(servo, 0), 1024)
        self.assertEqual(measure_offsets.new_offset(servo, 0, 457), -567)
        # at 90 degrees reading 2394
        self.assertEqual(measure_offsets.new_offset(servo, 90, 2394), 346)

    def test_new_offset_makes_the_servo_report_the_jig_angle(self):
        servo = Servo(1024, 3072, 0, pi, POSITION_CONTROL_MODE, offset=0)
        servo.constant_offset = measure_offsets.new_offset(servo, 0, 457)
        self.assertAlmostEqual(servo.get_angle_from_position(457), 0.0)
        servo.set_target_position_from_angle(0)
        self.assertEqual(servo.target_position, 457)

    def test_reversed_servo(self):
        servo = Servo(3072, 1024, -pi / 2, pi / 2, POSITION_CONTROL_MODE)
        self.assertEqual(measure_offsets.new_offset(servo, 0, 2000), -48)
        self.assertEqual(measure_offsets.raw_range(servo, -48), (976, 3024))


class ReplaceOffsetsTests(unittest.TestCase):

    def test_offsets_are_replaced_and_everything_else_is_kept(self):
        with open(TEST_SERVO_CONFIG_PATH) as config_file:
            original = config_file.read()
        before = parse_servo_config(yaml.safe_load(original))

        text = measure_offsets.replace_offsets(original, {1: 346, 2: -567}, 'measured')
        after = parse_servo_config(yaml.safe_load(text))

        self.assertEqual(after.servos[1].constant_offset, 346)
        self.assertEqual(after.servos[2].constant_offset, -567)
        for servo_id in after.servos:
            if servo_id not in (1, 2):
                self.assertEqual(after.servos[servo_id].constant_offset, before.servos[servo_id].constant_offset)
            self.assertEqual(after.servos[servo_id].min_position, before.servos[servo_id].min_position)
            self.assertEqual((after.servos[servo_id].p, after.servos[servo_id].profile_velocity),
                             (before.servos[servo_id].p, before.servos[servo_id].profile_velocity))
        self.assertEqual(after.temperature_limit, before.temperature_limit)
        # every other line, comments included, survives; the new offset lines carry the note
        for line in original.splitlines():
            if 'offset:' not in line:
                self.assertIn(line, text.splitlines())
        self.assertIn('    offset: 346  # measured', text.splitlines())
        self.assertIn('    offset: -567  # measured', text.splitlines())

    def test_offset_line_is_added_when_missing(self):
        text = "temperature_warning: 45\ntemperature_limit: 50\nservos:\n  3:\n    name: elbow\n    bus: rs485\n" \
               "    mode: position\n    positions: [3072, 1024]\n    angles: [-90, 90]\n" \
               "    profile: {velocity: 150, acceleration: 50}\n    pid: {p: 1500, i: 0, d: 0}\n"
        result = measure_offsets.replace_offsets(text, {3: -25})
        self.assertIn('    angles: [-90, 90]\n    offset: -25\n', result)
        self.assertEqual(parse_servo_config(yaml.safe_load(result)).servos[3].constant_offset, -25)

    def test_unknown_servo_is_an_error(self):
        with open(TEST_SERVO_CONFIG_PATH) as config_file:
            original = config_file.read()
        with self.assertRaises(ValueError):
            measure_offsets.replace_offsets(original, {42: 1})

    def test_real_config_round_trips(self):
        config = load_servo_config()
        with open(measure_offsets.DEFAULT_SERVO_CONFIG_PATH) as config_file:
            original = config_file.read()
        offsets = {servo_id: servo.constant_offset for servo_id, servo in config.servos.items() if servo_id in (1, 2)}
        self.assertEqual(parse_servo_config(yaml.safe_load(measure_offsets.replace_offsets(original, offsets))).servos[2]
                         .constant_offset, config.servos[2].constant_offset)


class MainTests(unittest.TestCase):

    REGISTERS = {1: (0, 0), 2: (0, 0), 4: (0, 0)}  # torque off, homing offset 0

    def run_main(self, arguments, readings, registers=REGISTERS):
        """run main with the bus faked, returns (exit code, printed lines)"""
        with mock.patch.object(measure_offsets, 'setup_dynamixel_handlers', return_value=(mock.Mock(), mock.Mock())), \
                mock.patch.object(measure_offsets, 'ServoHandler'), \
                mock.patch.object(measure_offsets, 'read_registers', return_value=registers), \
                mock.patch.object(measure_offsets, 'read_positions', return_value=readings) as read_positions, \
                mock.patch('builtins.print') as printed:
            result = measure_offsets.main(['--config', TEST_SERVO_CONFIG_PATH] + arguments)
        self.read_positions = read_positions
        return result, [str(call.args[0]) for call in printed.call_args_list]

    def test_default_jig_holds_servos_1_2_and_4(self):
        self.assertEqual(measure_offsets.DEFAULT_JIG, {1: 90.0, 2: 0.0, 4: 0.0})
        readings = {1: [2394] * 3, 2: [457] * 3, 4: [2036] * 3}
        result, output = self.run_main(['--dry-run'], readings)
        self.assertEqual(result, 0)
        self.assertEqual(self.read_positions.call_args.args[1], [1, 2, 4])
        self.assertTrue(any('servo 4' in line and 'offset -12 -> -12' in line for line in output), output)

    def test_refuses_when_the_torque_is_enabled(self):
        result, _ = self.run_main(['--yes'], {}, registers={1: (1, 0), 2: (0, 0), 4: (0, 0)})
        self.assertEqual(result, 1)
        self.read_positions.assert_not_called()

    def test_dry_run_prints_the_offsets_and_writes_nothing(self):
        with open(TEST_SERVO_CONFIG_PATH) as config_file:
            original = config_file.read()
        readings = {1: [2394] * 3, 2: [457, 457, 458]}
        result, output = self.run_main(['--jig', '1=90', '2=0', '--dry-run'], readings)
        self.assertEqual(result, 0)
        with open(TEST_SERVO_CONFIG_PATH) as config_file:
            self.assertEqual(config_file.read(), original)
        self.assertTrue(any('offset 354 -> 346' in line for line in output), output)
        self.assertTrue(any('offset -464 -> -567' in line for line in output), output)

    def test_unstable_readings_are_refused(self):
        readings = {1: [2394, 2400, 2394], 2: [457] * 3}
        result, _ = self.run_main(['--jig', '1=90', '2=0', '--yes'], readings)
        self.assertEqual(result, 1)

    def test_full_turn_servo_only_loses_the_end_of_its_range(self):
        # servo 4 covers [0, 4096], so any offset puts one end outside 0-4095; that is reported, not refused
        readings = {4: [2036] * 3}
        result, output = self.run_main(['--jig', '4=0', '--dry-run'], readings)
        self.assertEqual(result, 0)
        self.assertTrue(any('full turn' in line and '12 positions at the lower end' in line for line in output), output)

    def test_partial_range_outside_the_servo_range_is_refused(self):
        # servo 1 covers [1024, 3072], a reading of 900 at 90 degrees would put its range below 0
        readings = {1: [900] * 3}
        result, output = self.run_main(['--jig', '1=90', '--yes'], readings)
        self.assertEqual(result, 1)
        self.assertTrue(any('raw range leaves' in line for line in output), output)


if __name__ == '__main__':
    unittest.main()
