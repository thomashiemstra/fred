import unittest

from src.kinematics.kinematics_utils import Pose
from src.utils import linalg_utils
import numpy as np
from numpy import pi
import numpy.testing as test

from src.utils.movement_exception import MovementException
from src.utils.movement_utils import b_spline_curve, b_spline_curve_calculate_only


class GetCentreTests(unittest.TestCase):

    def test_right_angle_lines_1(self):
        pose1 = Pose(0, 0, 0,)
        pose2 = Pose(10, 10, 0, alpha=pi/2)

        centre = linalg_utils.get_center(pose1, pose2)
        self.assert_centre_equals([0, 10, 0], centre)

    def test_right_angle_lines_2(self):
        pose1 = Pose(0, 0, 0,)
        pose2 = Pose(10, 10, 0, alpha=-pi/2)

        centre = linalg_utils.get_center(pose1, pose2)
        self.assert_centre_equals([0, 10, 0], centre)

    def test_parallel_1(self):
        pose1 = Pose(0, 0, 0,  alpha=-pi/2)
        pose2 = Pose(10, 10, 0, alpha=-pi/2)

        centre = linalg_utils.get_center(pose1, pose2)
        self.assertIsNone(centre)

    def test_parallel_2(self):
        pose1 = Pose(0, 0, 0,  alpha=pi/2)
        pose2 = Pose(10, 10, 0, alpha=-pi/2)

        centre = linalg_utils.get_center(pose1, pose2)
        self.assertIsNone(centre)

    def test_same_y_for_points_1(self):
        pose1 = Pose(0, 0, 0, alpha=pi / 4)
        pose2 = Pose(10, 0, 0, alpha=-pi / 4)

        centre = linalg_utils.get_center(pose1, pose2)
        print(centre)
        self.assert_centre_equals([5, -5, 0], centre)

    def test_same_y_for_points_2(self):
        pose1 = Pose(0, 0, 0, alpha=-pi / 4)
        pose2 = Pose(10, 0, 0, alpha=pi / 4)

        centre = linalg_utils.get_center(pose1, pose2)
        print(centre)
        self.assert_centre_equals([5, 5, 0], centre)

    def test_no_points_at_origin_1(self):
        pose1 = Pose(5, 5, 0, alpha=pi / 8)
        pose2 = Pose(10, 10, 0, alpha=-pi / 8)

        centre = linalg_utils.get_center(pose1, pose2)
        print(centre)
        self.assert_centre_equals([6.46, 1.46, 0], centre)

    def assert_centre_equals(self, expected_centre, actual_centre):
        self.assertAlmostEqual(expected_centre[0], actual_centre[0], places=1, msg='x does not match')
        self.assertAlmostEqual(expected_centre[1], actual_centre[1], places=1, msg='y does not match')
        self.assertAlmostEqual(expected_centre[2], actual_centre[2], places=1, msg='z does not match')


class GetRotationMatrixParamsTests(unittest.TestCase):

    def test_no_rotation(self):
        pose1 = Pose(0, 0, 0)
        pose2 = Pose(1, 0, 0)
        center = [1, 1, 0]

        r, p0_prime, p1_prime, pc_prime = linalg_utils.get_rotation_matrix_params(pose1, pose2, center)
        self.assertIsNotNone(r)

        expected = np.eye(3)
        test.assert_allclose(r, expected)
        test.assert_allclose(p0_prime, [0, 0, 0], err_msg='x_prime mismatch')
        test.assert_allclose(p1_prime, [1, 0, 0], err_msg='y_prime mismatch')
        test.assert_allclose(pc_prime, [1, 1, 0], err_msg='z_prime mismatch')

    def test_pi_rotation(self):
        pose1 = Pose(1, 0, 0)
        pose2 = Pose(0, 0, 0)
        center = [1, 1, 0]

        r, p0_prime, p1_prime, pc_prime = linalg_utils.get_rotation_matrix_params(pose1, pose2, center)
        self.assertIsNotNone(r)

        expected = np.array([
            [-1, 0,  0],
            [0,  1,  0],
            [0,  0, -1]
        ])
        test.assert_allclose(r, expected)
        test.assert_allclose(p0_prime, [0, 0, 0], err_msg='x_prime mismatch')
        test.assert_allclose(p1_prime, [-1, 0, 0], err_msg='y_prime mismatch')
        test.assert_allclose(pc_prime, [0, 1, 0], err_msg='z_prime mismatch')

    def test_center_above_xy(self):
        pose1 = Pose(0, 0, 0)
        pose2 = Pose(1, 0, 0)
        center = [0, 0, 1]

        r, p0_prime, p1_prime, pc_prime = linalg_utils.get_rotation_matrix_params(pose1, pose2, center)
        self.assertIsNotNone(r)

        expected = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        test.assert_allclose(r, expected)
        test.assert_allclose(p0_prime, [0, 0, 0], err_msg='x_prime mismatch')
        test.assert_allclose(p1_prime, [1, 0, 0], err_msg='y_prime mismatch')
        test.assert_allclose(pc_prime, [0, 0, 1], err_msg='z_prime mismatch')

    def test_3d_points_1(self):
        pose1 = Pose(1, 1, 1)
        pose2 = Pose(2, 2, 2)
        center = [-1, -1, -1]

        res = linalg_utils.get_rotation_matrix_params(pose1, pose2, center)
        self.assertIsNone(res)

    def test_3d_points_2(self):
        pose1 = Pose(1, 1, 1)
        pose2 = Pose(1, 2, 1)
        center = [1, 1, 2]

        r, p0_prime, p1_prime, pc_prime = linalg_utils.get_rotation_matrix_params(pose1, pose2, center)
        self.assertIsNotNone(r)

        expected = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])
        test.assert_allclose(r, expected)
        test.assert_allclose(p0_prime, [0, 0, 0], err_msg='x_prime mismatch')
        test.assert_allclose(p1_prime, [0, 1, 0], err_msg='y_prime mismatch')
        test.assert_allclose(pc_prime, [0, 0, 1], err_msg='z_prime mismatch')


class DummyWorkspaceLimits:
    x_min = -40
    x_max = 40
    y_min = 16
    y_max = 40
    z_min = 5
    z_max = 40


class BSplineCurveTests(unittest.TestCase):

    def test_outside_workspace_limits(self):
        pose1 = Pose(0, 0, 0)
        pose2 = Pose(100, 100, 100)
        poses = [pose1, pose2]

        self.assertRaises(MovementException, lambda: b_spline_curve_calculate_only(poses, 2, DummyWorkspaceLimits))

    # With these poses the b-spline should start and stop exactly as required
    def test_spline_perfect_fit(self):
        pose1 = Pose(-20, 20, 5)
        pose2 = Pose(0, 30, 10)
        pose3 = Pose(20, 20, 5)

        poses = [pose1, pose2, pose3]

        stop_pose = b_spline_curve_calculate_only(poses, 2, DummyWorkspaceLimits)
        self.assertIsNotNone(stop_pose)
        self.assertEqual(pose3, stop_pose)

    # With these poses the b-spline should not exactly end at the last pose
    def test_spline_non_perfect_fit(self):
        pose1 = Pose(-20, 20, 5)
        pose2 = Pose(-10, 30, 10)
        pose3 = Pose(0, 20, 10)
        pose4 = Pose(10, 30, 10)
        pose5 = Pose(20, 20, 5)

        poses = [pose1, pose2, pose3, pose4, pose5]

        stop_pose = b_spline_curve_calculate_only(poses, 2, DummyWorkspaceLimits)
        self.assertIsNotNone(stop_pose)
        self.assertNotEqual(pose5, stop_pose)
        self.assertAlmostEqual(stop_pose.x, pose5.x, places=1)
        self.assertAlmostEqual(stop_pose.y, pose5.y, places=1)
        self.assertAlmostEqual(stop_pose.z, pose5.z, places=1)

