import unittest

import numpy as np
from numpy import pi

from src.kinematics.kinematics import inverse_kinematics, forward_kinematics
from src.kinematics.kinematics_utils import Pose, RobotConfig, calculate_euler_matrix_from_angles, \
    calculate_angles_from_euler_matrix

test_config = RobotConfig(d1=13.92, a2=20, d4=22, d6=12)


class EulerAnglesTests(unittest.TestCase):

    def test_angles_from_matrix_is_the_inverse_of_matrix_from_angles(self):
        rng = np.random.default_rng(0)
        for _ in range(500):
            alpha, beta, gamma = rng.uniform(-pi, pi), rng.uniform(-pi / 2 + 0.01, pi / 2 - 0.01), rng.uniform(-pi, pi)
            angles = calculate_angles_from_euler_matrix(calculate_euler_matrix_from_angles(alpha, beta, gamma))
            np.testing.assert_allclose((alpha, beta, gamma), angles, atol=1e-9)

    def test_gimbal_lock_gives_the_same_matrix(self):
        for beta in (pi / 2, -pi / 2):
            matrix = calculate_euler_matrix_from_angles(0.3, beta, 0.4)
            angles = calculate_angles_from_euler_matrix(matrix)
            np.testing.assert_allclose(matrix, calculate_euler_matrix_from_angles(*angles), atol=1e-9)


class ForwardKinematicsTests(unittest.TestCase):

    def assert_same_pose(self, expected, actual):
        np.testing.assert_allclose([expected.x, expected.y, expected.z], [actual.x, actual.y, actual.z], atol=1e-9)
        np.testing.assert_allclose(expected.get_euler_matrix(), actual.get_euler_matrix(), atol=1e-9)
        self.assertEqual(expected.flip, actual.flip)

    def test_forward_kinematics_is_the_inverse_of_inverse_kinematics(self):
        rng = np.random.default_rng(1)
        for flip in (False, True):
            for _ in range(200):
                pose = Pose(rng.uniform(-20, 20), rng.uniform(15, 30), rng.uniform(0, 20), flip,
                            rng.uniform(-1, 1), rng.uniform(-0.5, 0.5), rng.uniform(-1, 1))
                self.assert_same_pose(pose, forward_kinematics(inverse_kinematics(pose, test_config), test_config))

    def test_inverse_kinematics_of_forward_kinematics_gives_the_same_angles(self):
        angles = inverse_kinematics(Pose(30, 22, 4), test_config)
        np.testing.assert_allclose(angles, inverse_kinematics(forward_kinematics(angles, test_config), test_config),
                                   atol=1e-9)

    def test_start_pose(self):
        pose = forward_kinematics(inverse_kinematics(Pose(30, 22, 4), test_config), test_config)
        self.assert_same_pose(Pose(30, 22, 4), pose)
        self.assertAlmostEqual(0, pose.alpha)
        self.assertAlmostEqual(0, pose.beta)
        self.assertAlmostEqual(0, pose.gamma)


if __name__ == '__main__':
    unittest.main()
