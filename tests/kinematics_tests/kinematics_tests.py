import unittest

from src.kinematics.kinematics import inverse_kinematics
from src.kinematics.kinematics_utils import RobotConfig, Pose
from numpy import pi, sin, cos


test_config = RobotConfig(d1=10.0, a2=15.0, d4=25.0, d6=5.0)


class InverseKinematicsTests(unittest.TestCase):

    def test_1(self):
        x = 0
        y = test_config.d4 + test_config.d6
        z = test_config.d1 + test_config.a2 - 0.1
        pose = Pose(x, y, z)

        angles = inverse_kinematics(pose, test_config)

        expected = [0, pi/2, pi/2, 0.0, 0.0, 0.0, 0.0]
        self.assertIsNotNone(angles)
        self.assert_angles(angles, expected)

    def test_2(self):
        x = 0
        y = test_config.d4 * sin(pi/4) + test_config.d6
        z = (test_config.d1 + test_config.a2) - test_config.d4 * cos(pi/4)
        pose = Pose(x, y, z)

        angles = inverse_kinematics(pose, test_config)

        expected = [0, pi / 2, pi / 2, -pi/4, 0.0, pi/4, 0.0]
        self.assertIsNotNone(angles)
        self.assert_angles(angles, expected)

    def test_3(self):
        x = test_config.d4 * sin(pi/4)
        y = test_config.d6
        z = (test_config.d1 + test_config.a2) - test_config.d4 * cos(pi/4)
        pose = Pose(x, y, z)

        angles = inverse_kinematics(pose, test_config)

        expected = [0, 0, pi / 2, -pi/4, -pi/2, pi/2, pi/4]
        self.assertIsNotNone(angles)
        self.assert_angles(angles, expected)

    def assert_angles(self, actual_angles, expected):
        if len(actual_angles) != len(expected):
            self.fail("different sizes of angle arrays")
        for i in range(len(actual_angles)):
            self.assertAlmostEqual(actual_angles[i], expected[i], places=2,
                                   msg="angle {} does not match, actual:{}, expected:{}"
                                   .format(i, actual_angles[i], expected[i]))




