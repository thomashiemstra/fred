import unittest

from src.kinematics.kinematics_utils import Pose
from src.reinforcementlearning.robot_env import RobotEnv
import numpy as np

from src.reinforcementlearning.robot_env_utils import get_attractive_force_world, get_target_points


class TestRobotEnvUtils(unittest.TestCase):

    def test_get_attractive_force_world_1(self):
        control_point = np.array([0, 0, 0])
        target_point = np.array([1, 0, 0])

        forces, distance = get_attractive_force_world(np.array([control_point]), np.array([target_point]), 1)
        self.assertIsNotNone(forces, "forces should not be None")
        self.assertIsNotNone(distance, "distance should not be None")
        vector_res = forces[0]
        expected_vector = np.array([1, 0, 0])

        self.assert_vectors(expected_vector, vector_res)
        self.assertEqual(1, distance, "Distance should be 1")

    def test_get_attractive_force_world_2(self):
        control_point = np.array([0, 0, 0])
        target_point = np.array([-5, 0, 0])

        forces, distance = get_attractive_force_world(np.array([control_point]), np.array([target_point]), 2)
        self.assertIsNotNone(forces, "forces should not be None")
        self.assertIsNotNone(distance, "distance should not be None")
        vector_res = forces[0]
        expected_vector = np.array([-2, 0, 0])

        self.assert_vectors(expected_vector, vector_res)
        self.assertEqual(5, distance, "Distance should be 1")

    def test_get_attractive_force_world_3(self):
        control_point = np.array([0, 0, 0])
        target_point = np.array([0.5, 0, 0])

        forces, distance = get_attractive_force_world(np.array([control_point]), np.array([target_point]), 1)
        self.assertIsNotNone(forces, "forces should not be None")
        self.assertIsNotNone(distance, "distance should not be None")
        vector_res = forces[0]
        expected_vector = np.array([0.5, 0, 0])

        self.assert_vectors(expected_vector, vector_res)
        self.assertEqual(0.5, distance, "Distance should be 1")

    def test_get_target_points_1(self):
        pose = Pose(0, 20, 10)

        point_1, point_2, point_3 = get_target_points(pose, 5)
        self.assertIsNone(point_1, "point_1 should be None")
        self.assertIsNotNone(point_2, "point_2 should not be None")
        self.assertIsNotNone(point_3, "point_3 should not be None")

        expected_point_2 = np.array([0, 15, 10])
        expected_point_3 = np.array([0, 20, 10])

        self.assert_vectors(expected_point_2, point_2)
        self.assert_vectors(expected_point_3, point_3)

    def assert_vectors(self, expected_vector, actual_vector):
        if len(actual_vector) != len(expected_vector):
            self.fail("vectors have different sizes")
        for i in range(len(actual_vector)):
            self.assertAlmostEqual(actual_vector[i], expected_vector[i], places=2,
                                   msg="vector index {} does not match, actual:{}, expected:{}"
                                   .format(i, actual_vector[i], expected_vector[i]))
