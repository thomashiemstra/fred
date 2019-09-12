import unittest
import numpy as np

from src.reinforcementlearning.occupancy_grid_util import is_point_in_obstacle, get_height_tallest_obstacle, \
    create_hilbert_curve_from_obstacles
from src.utils.obstacle import BoxObstacle


class TestPointInObstacle(unittest.TestCase):

    def test_point_and_obstacle_at_origin(self):
        point_location = np.array([0, 0, 0])
        obstacle_location = np.array([0, 0, 0])
        obstacle_half_extends = np.array([1, 1, 1])
        obstacle_alpha = 0

        self.assertTrue(is_point_in_obstacle(point_location, obstacle_location, obstacle_half_extends, obstacle_alpha),
                        "Point should be in the obstacle")

    def test_obstacle_at_origin(self):
        point_location = np.array([1, 1, 0])
        obstacle_location = np.array([0, 0, 0])
        obstacle_half_extends = np.array([2, 2, 2])
        obstacle_alpha = 0

        self.assertTrue(is_point_in_obstacle(point_location, obstacle_location, obstacle_half_extends, obstacle_alpha),
                        "Point should be in the obstacle")

    def test_rotated_obstacle(self):
        point_location = np.array([2, 2, 0])
        obstacle_location = np.array([0, 0, 0])
        obstacle_half_extends = np.array([2, 2, 2])
        obstacle_alpha = np.pi/4

        self.assertFalse(is_point_in_obstacle(point_location, obstacle_location, obstacle_half_extends, obstacle_alpha),
                         "Point should not be in the obstacle")

    def test_point_negative_coordinates(self):
        point_location = np.array([-2, -2, 0])
        obstacle_location = np.array([0, 0, 0])
        obstacle_half_extends = np.array([2, 2, 2])
        obstacle_alpha = np.pi/4

        self.assertFalse(is_point_in_obstacle(point_location, obstacle_location, obstacle_half_extends, obstacle_alpha),
                         "Point should not be in the obstacle")


class TestGetHeightTallestObstacle(unittest.TestCase):

    def test_no_obstacles(self):
        res = get_height_tallest_obstacle([])
        self.assertEqual(0, res, "should have gotten 0 for no obstacles")

    def test_single_obstacle(self):
        height = 10
        obstacle = BoxObstacle([1, 1, height], [0, 0, 0])
        res = get_height_tallest_obstacle([obstacle])
        self.assertEqual(height, res, "should have gotten the height of the provided obstacle")

    def test_multiple_obstacles(self):

        height = 10
        obstacle1 = BoxObstacle([1, 1, height/2], [0, 0, 0])
        obstacle2 = BoxObstacle([1, 1, height], [0, 0, 0])
        res = get_height_tallest_obstacle([obstacle1, obstacle2])

        self.assertEqual(height, res, "should have gotten the height of tallest obstacle provided")


class TestHilbertCurve(unittest.TestCase):

    def test_empty_curve_no_obstacles(self):
        curve = create_hilbert_curve_from_obstacles([])

        for element in curve:
            self.assertEqual(0, element, "curve should have all elements set to zero when no obstacles are provided")

    def test_some_elements_non_zero_for_obstacles(self):
        obstacle = BoxObstacle([10, 10, 10], [0, 20, 0])

        curve = create_hilbert_curve_from_obstacles([obstacle], grid_len_x = 40, grid_len_y=40, iteration=4)

        all_zero = True
        for element in curve:
            if element > 0:
                all_zero = False
        self.assertFalse(all_zero, "At least some elements should be non zero when an obstacle is provided")

