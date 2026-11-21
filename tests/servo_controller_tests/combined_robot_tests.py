import unittest
from unittest import mock

import numpy as np

from src.kinematics.kinematics_utils import RobotConfig
from src.robot_controllers.combined_robot.combined_robot import CombinedRobot
from src.robot_controllers.simulated_robot import simulated_robot_controller
from src.robot_controllers.simulated_robot.simulated_robot_controller import SimulatedRobotController


class FakeDynamixelRobot:

    def __init__(self):
        self.robot_config = RobotConfig(d1=13.92, a2=20, d4=22, d6=12)
        self.servo_config = 'the servo config'
        self.moved = []
        self.velocities = []

    def move_servos(self, angles, velocities=None):
        self.moved.append(np.asarray(angles))
        self.velocities.append(velocities)
        return 'moved'

    def get_present_currents(self, servo_ids=(1, 2)):
        return {servo_id: 10 for servo_id in servo_ids}

    def get_present_pwms(self, servo_ids=(1, 2)):
        return {servo_id: -20 for servo_id in servo_ids}


class FakeSimulatedRobot:
    physics_client = 1
    body_id = 2
    control_points = []

    def __init__(self):
        self.moved = []

    def move_servos(self, angles):  # as the real simulated robot's, the velocities are not passed to it
        self.moved.append(np.asarray(angles))


class SimulatedRobotVelocitiesTests(unittest.TestCase):

    def test_the_simulated_robot_ignores_the_velocities(self):
        angles = np.array([0, 1, 2, 3, 4, 5, 6], dtype=float)
        robot = SimulatedRobotController.__new__(SimulatedRobotController)  # no physics server in a test
        robot.body_id, robot.physics_client, robot._motors = 2, 1, list(range(6))
        with mock.patch.object(simulated_robot_controller, 'p') as pybullet:
            robot.move_servos(angles)
            robot.move_servos(angles, np.array([0, 9, 9, 9, 9, 9, 9], dtype=float))
        first, second = pybullet.setJointMotorControlArray.call_args_list
        for call in (first, second):
            np.testing.assert_array_equal(angles[1:7], call.kwargs['targetPositions'])
            self.assertEqual((2, list(range(6))), call.args[:2])
        self.assertEqual(first.kwargs.keys(), second.kwargs.keys())


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
        self.assertIsNone(self.dynamixel.velocities[0])

    def test_the_velocities_go_to_the_physical_robot_only(self):
        angles = np.array([0, 1, 2, 3, 4, 5, 6], dtype=float)
        velocities = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        self.assertEqual('moved', self.robot.move_servos(angles, velocities))
        np.testing.assert_array_equal(velocities, self.dynamixel.velocities[0])
        np.testing.assert_array_equal(angles, self.simulated.moved[0])

    def test_everything_else_is_the_physical_robot(self):
        # the servo level of the physical robot, used by the calibration scripts
        self.assertEqual({3: 10}, self.robot.get_present_currents([3]))
        self.assertEqual({3: -20}, self.robot.get_present_pwms([3]))
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
