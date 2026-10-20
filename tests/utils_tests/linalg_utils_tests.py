import unittest

import numpy as np
from numpy import pi
import numpy.testing as test

from src.kinematics.kinematics_utils import Pose
from src.utils import linalg_utils


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
        self.assert_centre_equals([5, -5, 0], centre)

    def test_same_y_for_points_2(self):
        pose1 = Pose(0, 0, 0, alpha=-pi / 4)
        pose2 = Pose(10, 0, 0, alpha=pi / 4)

        centre = linalg_utils.get_center(pose1, pose2)
        self.assert_centre_equals([5, 5, 0], centre)

    def test_no_points_at_origin_1(self):
        pose1 = Pose(5, 5, 0, alpha=pi / 8)
        pose2 = Pose(10, 10, 0, alpha=-pi / 8)

        centre = linalg_utils.get_center(pose1, pose2)
        self.assert_centre_equals([6.46, 1.46, 0], centre)
        # the centre is where the lines along the direction of both poses cross
        for pose in (pose1, pose2):
            self.assert_on_the_line_of(pose, centre)

    def test_different_heights_have_no_centre(self):
        self.assertIsNone(linalg_utils.get_center(Pose(0, 0, 0), Pose(10, 10, 5, alpha=pi / 2)))

    def test_poses_not_pointing_horizontally_have_no_centre(self):
        self.assertIsNone(linalg_utils.get_center(Pose(0, 0, 0, gamma=0.5), Pose(10, 10, 0, alpha=pi / 2)))

    def assert_on_the_line_of(self, pose, centre):
        # alpha turns the end effector, which points along +y, around the z-axis
        direction = Pose(0, 0, 0, alpha=pose.alpha).get_euler_matrix()[:2, 2]
        to_centre = np.asarray(centre[:2]) - [pose.x, pose.y]
        self.assertAlmostEqual(0.0, np.cross(direction, to_centre))

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

    def test_aligned_points_have_no_plane(self):
        pose1 = Pose(1, 1, 1)
        pose2 = Pose(2, 2, 2)
        center = [-1, -1, -1]

        res = linalg_utils.get_rotation_matrix_params(pose1, pose2, center)
        self.assertIsNone(res)

    def test_3d_points(self):
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


if __name__ == '__main__':
    unittest.main()
