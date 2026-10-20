import unittest

import numpy as np

from src.kinematics.kinematics_utils import RobotConfig
from src.robot_controllers.combined_robot.combined_robot import CombinedRobot


class FakeDynamixelRobot:

    def __init__(self):
        self.robot_config = RobotConfig(d1=13.92, a2=20, d4=22, d6=12)
        self.servo_config = 'the servo config'
        self.moved = []

    def move_servos(self, angles):
        self.moved.append(np.asarray(angles))
        return 'moved'

    def get_present_currents(self, servo_ids=(1, 2)):
        return {servo_id: 10 for servo_id in servo_ids}


class FakeSimulatedRobot:
    physics_client = 1
    body_id = 2
    control_points = []

    def __init__(self):
        self.moved = []

    def move_servos(self, angles):
        self.moved.append(np.asarray(angles))


class CombinedRobotTests(unittest.TestCase):

    def setUp(self):
        self.dynamixel = FakeDynamixelRobot()
        self.simulated = FakeSimulatedRobot()
        self.robot = CombinedRobot(self.dynamixel, self.simulated)

    def test_moves_go_to_both_robots(self):
        angles = np.array([0, 1, 2, 3, 4, 5, 6], dtype=float)
        self.assertEqual('moved', self.robot.move_servos(angles))
        np.testing.assert_array_equal(angles, self.dynamixel.moved[0])
        np.testing.assert_array_equal(angles, self.simulated.moved[0])

    def test_everything_else_is_the_physical_robot(self):
        # the servo level of the physical robot, used by the calibration scripts
        self.assertEqual({3: 10}, self.robot.get_present_currents([3]))
        self.assertEqual('the servo config', self.robot.servo_config)

    def test_unknown_attributes_raise(self):
        with self.assertRaises(AttributeError):
            self.robot.no_such_thing
        self.assertFalse(hasattr(self.robot, 'no_such_thing'))

    def test_a_half_built_robot_does_not_recurse(self):
        robot = CombinedRobot.__new__(CombinedRobot)
        self.assertFalse(hasattr(robot, 'dynamixel_robot'))
        self.assertFalse(hasattr(robot, 'anything'))


if __name__ == '__main__':
    unittest.main()
