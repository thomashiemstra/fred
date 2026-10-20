import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from numpy import pi
import pybullet as p

from src.kinematics.kinematics_utils import Pose
from src.reinforcementlearning.environment import robot_env_utils
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

    def test_get_attractive_force_world_weights_and_zero_distance(self):
        control_points = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
        target_points = np.array([[0, 0, 4], [0, 1, 0], [1, 1, 1]], dtype=np.float64)

        forces, distance = get_attractive_force_world(control_points, target_points, 2, weights=np.array([0.5, 3, 1]))

        self.assert_vectors(np.array([0, 0, 1]), forces[0])  # clipped to length 2, then weighted
        self.assert_vectors(np.array([0, 3, 0]), forces[1])  # within cutoff, weighted
        self.assert_vectors(np.array([0, 0, 0]), forces[2])  # already at the target
        self.assertAlmostEqual(5, distance, msg="total distance should be the unweighted sum of distances")

    def test_get_attractive_force_world_dimension_mismatch(self):
        with self.assertRaises(Exception):
            get_attractive_force_world(np.zeros((2, 3)), np.zeros((3, 3)))

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

    @unittest.skip('fails since the simulation loads fred_new.urdf (commit cfdc00d): control point 2 gets no repulsive '
                   'vector in x, the control points of the new urdf have to be checked')
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


class RepulsiveForceTests(unittest.TestCase):
    """get_repulsive_forces_world with the pybullet distance query replaced by a fake"""

    @staticmethod
    def control_point(point_id, radius=1, weight=1):
        return SimpleNamespace(point_id=point_id, radius=radius, weight=weight)

    def get_forces(self, distances, control_points, obstacle_ids, cutoff=10, clip_force=None):
        """distances: dict (control_point_id, obstacle_id) -> (normal, distance in cm)"""
        def fake_normal_and_distance(robot_body_id, obstacle_id, control_point_id, physics_client_id, cutoff_distance):
            self.assertEqual(cutoff, cutoff_distance)
            return distances.get((control_point_id, obstacle_id), ([0, 0, 1], 10000000))

        with mock.patch.object(robot_env_utils, 'get_normal_and_distance', side_effect=fake_normal_and_distance):
            return get_repulsive_forces_world(0, np.array(control_points), obstacle_ids, 0,
                                              repulsive_cutoff_distance=cutoff, clip_force=clip_force)

    def test_closest_obstacle_determines_force(self):
        control_points = [self.control_point(1), self.control_point(2, weight=2), self.control_point(3)]
        distances = {
            (2, 'far'): ([0, 1, 0], 6),     # distance to surface of control point: 5
            (2, 'close'): ([1, 0, 0], 3),   # distance to surface of control point: 2
            (3, 'far'): ([0, 0, 1], 20),    # beyond the cutoff
        }

        forces = self.get_forces(distances, control_points, ['far', 'close'])

        self.assert_vectors([0, 0, 0], forces[0])
        expected_magnitude = 2 * (1 / 2 - 1 / 10) / (2 * 2)
        self.assert_vectors([expected_magnitude, 0, 0], forces[1])
        self.assert_vectors([0, 0, 0], forces[2])

    def test_overlap_and_clipping(self):
        control_points = [self.control_point(1), self.control_point(2), self.control_point(3)]
        distances = {(1, 'obs'): ([0, 0, 1], 0.5), (2, 'obs'): ([0, 0, 1], 0.5)}  # inside the control point radius

        unclipped = self.get_forces(distances, control_points, ['obs'])
        clipped = self.get_forces(distances, control_points, ['obs'], clip_force=50)

        # an overlapping control point is treated as being 0.1 cm away
        self.assert_vectors([0, 0, (1 / 0.1 - 1 / 10) / (0.1 * 0.1)], unclipped[0])
        self.assert_vectors([0, 0, 50], clipped[0])
        self.assert_vectors([0, 0, 50], clipped[1])
        self.assert_vectors([0, 0, 0], clipped[2])

    def assert_vectors(self, expected_vector, actual_vector):
        np.testing.assert_allclose(actual_vector, expected_vector, atol=1e-6)


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

    def test_normalize_de_normalize_round_trip(self):
        angles = [0.3, 2.9, 1.7, -2.5, 0.4, -1.1]

        round_trip = get_de_normalized_current_angles(get_normalized_current_angles(angles))

        np.testing.assert_allclose(round_trip, angles, atol=1e-9)

    def test_clipping_values(self):
        angles = [5, -1, 4, 3, -4, 3, -4]

        clipped = get_clipped_state(angles)

        # index 0 is not a joint (angles are 1-indexed) and is always 0
        np.testing.assert_allclose(clipped, [0, 0, pi, 2 * pi / 3, -pi, 3 * pi / 4, -pi], atol=1e-9)
