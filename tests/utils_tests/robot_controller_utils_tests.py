import os
import unittest

from src.robot_controllers.dynamixel_robot.servo_config import load_servo_config
from src.utils import robot_controller_utils as utils

# the servo configuration of the tests, so a new calibration of the robot doesn't change them
TEST_SERVO_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'servo_controller_tests',
                                      'test_servo_config.yaml')


class TestPositionToDegrees(unittest.TestCase):

    def setUp(self):
        self.servos = load_servo_config(TEST_SERVO_CONFIG_PATH).servos

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
