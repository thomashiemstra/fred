import unittest

import numpy as np
from numpy import pi
import pybullet as p

from src.kinematics.kinematics_utils import Pose
from src.reinforcementlearning.environment.robot_env_utils import get_attractive_force_world, get_target_points, \
    get_repulsive_forces_world, get_clipped_state, get_normalized_current_angles, get_de_normalized_current_angles
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

    def test_get_attractive_force_world_multiple_vectors(self):
        control_point_1 = np.array([0, 0, 0])
        control_point_2 = np.array([1, 0, 0])

        target_point_1 = np.array([0.5, 0, 0])
        target_point_2 = np.array([1.5, 0, 0])

        forces, distance = get_attractive_force_world(np.array([control_point_1, control_point_2]),
                                                      np.array([target_point_1, target_point_2]), 1)
        self.assertIsNotNone(forces, "forces should not be None")
        self.assertIsNotNone(distance, "distance should not be None")
        vector_res_1 = forces[0]
        expected_vector_1 = np.array([0.5, 0, 0])
        vector_res_2 = forces[1]
        expected_vector_2 = np.array([0.5, 0, 0])

        self.assert_vectors(expected_vector_1, vector_res_1)
        self.assert_vectors(expected_vector_2, vector_res_2)
        self.assertEqual(1, distance, "Distance should be 1")

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
        obstacle = BoxObstacle([10, 100, 50], [-35, 0, 0])
        obstacle.build(self.physics_client)

        self.simulated_robot.reset_to_pose(Pose(-20, 15, 10))
        for _ in range(10):
            p.stepSimulation(self.physics_client)
        c1, c2, c3 = self.simulated_robot.control_points

        obstacles = np.array([obstacle.obstacle_id])
        rep_forces = get_repulsive_forces_world(self.simulated_robot.body_id, np.array([c1, c2, c3]),
                                                obstacles, self.physics_client, repulsive_cutoff_distance=10)

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


class AnglesTest(unittest.TestCase):

    def test_clipping_upper_bounds(self):
        angles = [0, 10, 10, 10, 10, 10, 10]
        clipped_angels = get_clipped_state(angles)

        for angle in clipped_angels:
            self.assertTrue(angle < 10, msg='angle should have been clipped to something smaller')

    def test_clipping_lower_bounds(self):
        angles = [0, -10, -10, -10, -10, -10, -10]
        clipped_angels = get_clipped_state(angles)

        for angle in clipped_angels:
            self.assertTrue(angle > -10, msg='angle should have been clipped to something bigger')

    def test_no_clipping(self):
        angles = [0, 0, 0, 0, 0, 0, 0]
        clipped_angels = get_clipped_state(angles)

        for angle in clipped_angels:
            self.assertEqual(0, angle, msg='angle should have been clipped to something bigger')

    def test_get_normalized_angles_upper_bound(self):
        angles = [pi,  pi,  2 * pi / 3, pi, 3 * pi / 4]

        normalized_angles = get_normalized_current_angles(angles)

        self.assertEqual(1, normalized_angles[0], msg="angle 1 should have been normalized to 1")
        self.assertEqual(1, normalized_angles[1], msg="angle 2 should have been normalized to 1")
        self.assertEqual(1, normalized_angles[2], msg="angle 3 should have been normalized to 1")
        self.assertEqual(1, normalized_angles[3], msg="angle 4 should have been normalized to 1")
        self.assertEqual(1, normalized_angles[4], msg="angle 5 should have been normalized to 1")

    def test_get_normalized_angles_lower_bound(self):
        angles = [0, 0,  -pi / 3, -pi, -3 * pi / 4]

        normalized_angles = get_normalized_current_angles(angles)

        self.assertEqual(-1, normalized_angles[0], msg="angle 0 should have been normalized to -1")
        self.assertEqual(-1, normalized_angles[1], msg="angle 1 should have been normalized to -1")
        self.assertEqual(-1, normalized_angles[2], msg="angle 2 should have been normalized to -1")
        self.assertEqual(-1, normalized_angles[3], msg="angle 3 should have been normalized to -1")
        self.assertEqual(-1, normalized_angles[4], msg="angle 4 should have been normalized to -1")

    def test_get_normalized_angles_5_angles_upper(self):
        angles = [pi, pi, pi/2 * (1 + 1/3), pi, (3 * pi / 4)]
        normalized_angles = get_normalized_current_angles(angles)

        self.assertEqual(1, normalized_angles[0], msg="got a wrong value for angle 0")
        self.assertEqual(1, normalized_angles[1], msg="got a wrong value for angle 1")
        self.assertEqual(1, normalized_angles[2], msg="got a wrong value for angle 2")
        self.assertEqual(1, normalized_angles[3], msg="got a wrong value for angle 3")
        self.assertEqual(1, normalized_angles[4], msg="got a wrong value for angle 4")

    def test_get_normalized_angles_5_angles_lower(self):
        angles = [0, 0, - 2 * pi / 6, -pi, -(3 * pi / 4)]
        normalized_angles = get_normalized_current_angles(angles)

        self.assertEqual(-1, normalized_angles[0], msg="got a wrong value for angle 0")
        self.assertEqual(-1, normalized_angles[1], msg="got a wrong value for angle 1")
        self.assertEqual(-1, normalized_angles[2], msg="got a wrong value for angle 2")
        self.assertEqual(-1, normalized_angles[3], msg="got a wrong value for angle 3")
        self.assertEqual(-1, normalized_angles[4], msg="got a wrong value for angle 4")

    def test_get_de_normalized_current_angles_5_angles_upper(self):
        normalized_angles = [1, 1, 1, 1, 1, 1]

        angles = get_de_normalized_current_angles(normalized_angles)

        self.assertAlmostEqual(pi, angles[0], places=2, msg="Got a wrong value for angle 0")
        self.assertAlmostEqual(pi, angles[1], places=2, msg="Got a wrong value for angle 1")
        self.assertAlmostEqual(pi/2 * (1 + 1/3), angles[2], places=2, msg="Got a wrong value for angle 2")
        self.assertAlmostEqual(pi, angles[3], places=2, msg="Got a wrong value for angle 3")
        self.assertAlmostEqual((3 * pi / 4), angles[4], places=2, msg="Got a wrong value for angle 4")

    def test_get_de_normalized_current_angles_6_angles_upper(self):
        normalized_angles = [1, 1, 1, 1, 1, 1, 1]

        angles = get_de_normalized_current_angles(normalized_angles)

        self.assertAlmostEqual(pi, angles[0], places=2, msg="Got a wrong value for angle 0")
        self.assertAlmostEqual(pi, angles[1], places=2, msg="Got a wrong value for angle 1")
        self.assertAlmostEqual(pi/2 * (1 + 1/3), angles[2], places=2, msg="Got a wrong value for angle 2")
        self.assertAlmostEqual(pi, angles[3], places=2, msg="Got a wrong value for angle 3")
        self.assertAlmostEqual((3 * pi / 4), angles[4], places=2, msg="Got a wrong value for angle 4")
        self.assertAlmostEqual(pi, angles[5], places=2, msg="Got a wrong value for angle 5")

    def test_get_de_normalized_current_angles_5_angles_lowerr(self):
        normalized_angles = [-1, -1, -1, -1, -1, -1]

        angles = get_de_normalized_current_angles(normalized_angles)

        self.assertAlmostEqual(0, angles[0], places=2, msg="Got a wrong value for angle 0")
        self.assertAlmostEqual(0, angles[1], places=2, msg="Got a wrong value for angle 1")
        self.assertAlmostEqual(- 2 * pi / 6, angles[2], places=2, msg="Got a wrong value for angle 2")
        self.assertAlmostEqual(-pi, angles[3], places=2, msg="Got a wrong value for angle 3")
        self.assertAlmostEqual(-(3 * pi / 4), angles[4], places=2, msg="Got a wrong value for angle 4")

    def test_get_de_normalized_current_angles_6_angles_lower(self):
        normalized_angles = [-1, -1, -1, -1, -1, -1, -1]

        angles = get_de_normalized_current_angles(normalized_angles)

        self.assertAlmostEqual(0, angles[0], places=2, msg="Got a wrong value for angle 0")
        self.assertAlmostEqual(0, angles[1], places=2, msg="Got a wrong value for angle 1")
        self.assertAlmostEqual(- 2 * pi / 6, angles[2], places=2, msg="Got a wrong value for angle 2")
        self.assertAlmostEqual(-pi, angles[3], places=2, msg="Got a wrong value for angle 3")
        self.assertAlmostEqual(-(3 * pi / 4), angles[4], places=2, msg="Got a wrong value for angle 4")
        self.assertAlmostEqual(-pi, angles[5], places=2, msg="Got a wrong value for angle 5")