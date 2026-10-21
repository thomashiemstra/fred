import copy
import math
import os
import unittest
from unittest import mock

import numpy as np

from src.robot_controllers.dynamixel_robot.servo_config import load_servo_config
from src.utils import robot_controller_utils as utils

# the servo configuration of the tests, so a new calibration of the robot doesn't change them
TEST_SERVO_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'servo_controller_tests',
                                      'test_servo_config.yaml')
SAFE_ANGLES = {1: utils.SERVO_1_SAFE_ANGLES, 2: utils.SERVO_2_SAFE_ANGLES}


def no_input(*args, **kwargs):
    raise AssertionError('the safety check should not read from stdin')


def positions_with(servo_id, position):
    positions = np.full(7, 2048, dtype=np.int_)
    positions[servo_id] = position
    return positions


def raw_position(servo, degrees):
    """the raw position of the angle with the calibration of the servo, the inverse of position_to_degrees"""
    angle_per_position = (servo.max_angle - servo.min_angle) / (servo.max_position - servo.min_position)
    return servo.min_position + servo.constant_offset + (math.radians(degrees) - servo.min_angle) / angle_per_position


def safe_edges(servo_id, servo):
    """the lowest and the highest raw position inside the safe range of the servo"""
    low, high = sorted(raw_position(servo, degrees) for degrees in SAFE_ANGLES[servo_id])
    return math.ceil(low - 1e-6), math.floor(high + 1e-6)


class TestServoChecks(unittest.TestCase):
    """the safe ranges are angles, so where they are in raw positions depends on the calibration of the servo"""

    def setUp(self):
        self.servos = load_servo_config(TEST_SERVO_CONFIG_PATH).servos
        patcher = mock.patch('builtins.input', side_effect=no_input)
        patcher.start()
        self.addCleanup(patcher.stop)

    def check(self, servo_id, position, servos=None):
        servos = servos or self.servos
        check = utils.servo_1_check if servo_id == 1 else utils.servo_2_check
        return check(positions_with(servo_id, position), servos[servo_id])

    def test_with_the_calibration_of_20_october_the_range_is_raw_0_to_2500(self):
        servos = copy.deepcopy(self.servos)
        servos[1].constant_offset = 346
        servos[2].constant_offset = -567
        for servo_id in (1, 2):
            with self.subTest(servo=servo_id):
                self.assertEqual((0, 2500), safe_edges(servo_id, servos[servo_id]))

    def test_positions_inside_the_range_pass(self):
        for servo_id in (1, 2):
            lowest, highest = safe_edges(servo_id, self.servos[servo_id])
            for position in (lowest, lowest + 1, (lowest + highest) // 2, highest - 1, highest):
                with self.subTest(servo=servo_id, position=position):
                    self.assertTrue(self.check(servo_id, position))

    def test_positions_outside_the_range_are_refused_without_reading_stdin(self):
        for servo_id in (1, 2):
            lowest, highest = safe_edges(servo_id, self.servos[servo_id])
            for position in (lowest - 1, lowest - 500, highest + 1, highest + 1500):
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
            lowest, highest = safe_edges(servo_id, self.servos[servo_id])
            shifted = copy.deepcopy(self.servos)
            shifted[servo_id].constant_offset += 100
            self.assertEqual((lowest + 100, highest + 100), safe_edges(servo_id, shifted[servo_id]))
            for position, expected in ((lowest + 100, True), (highest + 100, True), (lowest + 99, False),
                                       (highest + 101, False)):
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
