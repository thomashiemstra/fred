import unittest

import numpy as np
import pybullet as p

from src.kinematics.kinematics_utils import Pose
from src.reinforcementlearning.robot_env_utils import get_attractive_force_world, get_target_points, \
    get_repulsive_forces_world, control_point_ids
from src.simulation.simulation_utils import start_simulated_robot
from src.utils.obstacle import BoxObstacle


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


class ObstacleIntegrationTests(unittest.TestCase):

    def setUp(self):
        self.simulated_robot = start_simulated_robot(use_gui=False)
        self.physics_client = self.simulated_robot.physics_client

    def tearDown(self):
        p.disconnect()

    def test_repulsive_vectors(self):
        obstacle = BoxObstacle(self.physics_client, [10, 100, 50], [-31, 0, 0])

        self.simulated_robot.reset_to_pose(Pose(-20, 15, 10))
        p.stepSimulation(self.physics_client)

        obstacles = np.array([obstacle.obstacle_id])
        rep_forces = get_repulsive_forces_world(self.simulated_robot.body_id, control_point_ids, obstacles, self.physics_client)

        control_point_1_vec = rep_forces[0]
        control_point_2_vec = rep_forces[1]
        control_point_3_vec = rep_forces[2]

        # control point 1 is too far away from the obstacle to have a repulsive vector
        self.assertAlmostEqual(control_point_1_vec[0], 0, places=2, msg="control point 1 should have a null vector")
        self.assertAlmostEqual(control_point_1_vec[1], 0, places=2, msg="control point 1 should have a null vector")
        self.assertAlmostEqual(control_point_1_vec[2], 0, places=2, msg="control point 1 should have a null vector")

        # control point 2 and 3 are close enough to the obstacle to have a repulsive vector
        self.assertTrue(control_point_2_vec[0] > 0,
                        "control point 2 should have a repulsive vector pointing in the x-direction")
        self.assertTrue(control_point_3_vec[0] > 0,
                        "control point 3 should have a repulsive vector pointing in the x-direction")

        self.assertAlmostEqual(control_point_2_vec[1], 0, places=2,
                               msg="control point 2 should only point in the x direction, not also in the y direction")
        self.assertAlmostEqual(control_point_2_vec[2], 0, places=2,
                               msg="control point 2 should only point in the x direction, not also in the z direction")
        self.assertAlmostEqual(control_point_3_vec[1], 0, places=2,
                               msg="control point 2 should only point in the x direction, not also in the y direction")
        self.assertAlmostEqual(control_point_3_vec[2], 0, places=2,
                               msg="control point 2 should only point in the x direction, not also in the z direction")


        print(obstacle)
