import unittest

from src.kinematics.kinematics_utils import Pose
from src.utils import movement_utils
from numpy import pi


class GetCentreTests(unittest.TestCase):

    def test_right_angle_lines_1(self):
        pose1 = Pose(0, 0, 0,)
        pose2 = Pose(10, 10, 0, alpha=pi/2)

        centre = movement_utils.get_center(pose1, pose2)
        self.assert_centre_equals([0, 10, 0], centre)

    def test_right_angle_lines_2(self):
        pose1 = Pose(0, 0, 0,)
        pose2 = Pose(10, 10, 0, alpha=-pi/2)

        centre = movement_utils.get_center(pose1, pose2)
        self.assert_centre_equals([0, 10, 0], centre)

    def test_parallel_1(self):
        pose1 = Pose(0, 0, 0,  alpha=-pi/2)
        pose2 = Pose(10, 10, 0, alpha=-pi/2)

        centre = movement_utils.get_center(pose1, pose2)
        self.assertIsNone(centre)

    def test_parallel_2(self):
        pose1 = Pose(0, 0, 0,  alpha=pi/2)
        pose2 = Pose(10, 10, 0, alpha=-pi/2)

        centre = movement_utils.get_center(pose1, pose2)
        self.assertIsNone(centre)

    def test_same_y_for_points_1(self):
        pose1 = Pose(0, 0, 0, alpha=pi / 4)
        pose2 = Pose(10, 0, 0, alpha=-pi / 4)

        centre = movement_utils.get_center(pose1, pose2)
        print(centre)
        self.assert_centre_equals([5, 5, 0], centre)

    def test_same_y_for_points_2(self):
        pose1 = Pose(0, 0, 0, alpha=-pi / 4)
        pose2 = Pose(10, 0, 0, alpha=pi / 4)

        centre = movement_utils.get_center(pose1, pose2)
        print(centre)
        self.assert_centre_equals([5, -5, 0], centre)

    def test_no_points_at_origin_1(self):
        pose1 = Pose(5, 5, 0, alpha=pi / 8)
        pose2 = Pose(10, 10, 0, alpha=-pi / 8)

        centre = movement_utils.get_center(pose1, pose2)
        print(centre)
        self.assert_centre_equals([8.53, 13.53, 0], centre)

    def assert_centre_equals(self, expected_centre, actual_centre):
        self.assertAlmostEqual(expected_centre[0], actual_centre[0], places=1, msg='x does not match')
        self.assertAlmostEqual(expected_centre[1], actual_centre[1], places=1, msg='y does not match')
        self.assertAlmostEqual(expected_centre[2], actual_centre[2], places=1, msg='z does not match')


class FindEllipseRadiiTests(unittest.TestCase):

    def test_same_y_1(self):
        first_point = [1, 0]
        second_point = [-1, 0]
        centre = [0, 0]

        a, b = movement_utils.find_ellipse_radii(first_point, second_point, centre)
        self.assertEqual(a, 1)
        self.assertEqual(a, b)

    # if the 2 points have different x coordinates the first point is taken for the circle radius
    def test_same_y_2(self):
        first_point = [2, 0]
        second_point = [-5, 0]
        centre = [0, 0]

        a, b = movement_utils.find_ellipse_radii(first_point, second_point, centre)
        self.assertEqual(a, 2)
        self.assertEqual(a, b)

    def test_same_point(self):
        first_point = [2, 0]

        centre = [0, 0]

        a, b = movement_utils.find_ellipse_radii(first_point, first_point, centre)
        self.assertEqual(a, 0)
        self.assertEqual(b, 0)

    def test_shifted_centre_1(self):
        first_point = [2, 1]
        second_point = [-5, 3]
        centre = [4, 11]

        a, b = movement_utils.find_ellipse_radii(first_point, second_point, centre)
        self.assertAlmostEqual(a, 14.76, 1)
        self.assertAlmostEqual(b, 10.09, 1)

    def test_centre_between_points(self):
        first_point = [1, 1]
        second_point = [-1, 1]
        centre = [0, 0]

        a, b = movement_utils.find_ellipse_radii(first_point, second_point, centre)
        print(a, b)
        self.assertAlmostEqual(a, 1.41, 1)
        self.assertAlmostEqual(b, 1.41, 1)

