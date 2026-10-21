import copy
import unittest
from unittest import mock

import numpy as np

from src.robot_controllers.dynamixel_robot.servo_config import load_servo_config
from src.utils import robot_controller_utils as utils


def no_input(*args, **kwargs):
    raise AssertionError('the safety check should not read from stdin')


def positions_with(servo_id, position):
    positions = np.full(7, 2048, dtype=np.int_)
    positions[servo_id] = position
    return positions


class TestServoChecks(unittest.TestCase):
    """the thresholds are the raw positions 0 and 2500 of the servo_config.yaml of the robot"""

    def setUp(self):
        self.servos = load_servo_config().servos
        patcher = mock.patch('builtins.input', side_effect=no_input)
        patcher.start()
        self.addCleanup(patcher.stop)

    def check(self, servo_id, position, servos=None):
        servos = servos or self.servos
        check = utils.servo_1_check if servo_id == 1 else utils.servo_2_check
        return check(positions_with(servo_id, position), servos[servo_id])

    def test_positions_inside_the_range_pass(self):
        for servo_id in (1, 2):
            for position in (0, 1, 2048, 2499, 2500):
                with self.subTest(servo=servo_id, position=position):
                    self.assertTrue(self.check(servo_id, position))

    def test_positions_outside_the_range_are_refused_without_reading_stdin(self):
        for servo_id in (1, 2):
            for position in (-1, -500, 2501, 4000):
                with self.subTest(servo=servo_id, position=position), self.assertLogs(utils.logger, 'ERROR') as logs:
                    self.assertFalse(self.check(servo_id, position))
                self.assertIn('servo {} '.format(servo_id), logs.output[0])
                self.assertIn('raw position {}'.format(position), logs.output[0])
                self.assertIn('by hand', logs.output[0])

    def test_other_servos_do_not_matter(self):
        positions = positions_with(2, 4000)
        self.assertTrue(utils.servo_1_check(positions, self.servos[1]))
        positions = positions_with(1, 4000)
        self.assertTrue(utils.servo_2_check(positions, self.servos[2]))

    def test_a_recalibration_moves_the_raw_positions_not_the_angles(self):
        for servo_id in (1, 2):
            shifted = copy.deepcopy(self.servos)
            shifted[servo_id].constant_offset += 100
            for position, expected in ((100, True), (101, True), (2600, True), (2601, False), (99, False)):
                with self.subTest(servo=servo_id, position=position):
                    if expected:
                        self.assertTrue(self.check(servo_id, position, shifted))
                    else:
                        with self.assertLogs(utils.logger, 'ERROR'):
                            self.assertFalse(self.check(servo_id, position, shifted))

    def test_position_to_degrees_is_not_clamped(self):
        servo = self.servos[1]
        self.assertAlmostEqual(0, utils.position_to_degrees(servo, 1024 + servo.constant_offset))
        self.assertAlmostEqual(180, utils.position_to_degrees(servo, 3072 + servo.constant_offset))
        self.assertAlmostEqual(-180, utils.position_to_degrees(servo, 1024 - 2048 + servo.constant_offset))
        # the elbow is mounted the other way around
        elbow = self.servos[3]
        self.assertAlmostEqual(-90, utils.position_to_degrees(elbow, 3072 + elbow.constant_offset))
        self.assertAlmostEqual(90, utils.position_to_degrees(elbow, 1024 + elbow.constant_offset))
        self.assertAlmostEqual(270, utils.position_to_degrees(elbow, -1024 + elbow.constant_offset))


if __name__ == '__main__':
    unittest.main()
