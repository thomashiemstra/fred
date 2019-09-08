import unittest
import numpy as np

from src.reinforcementlearning.occupancy_grid_util import is_point_in_obstacle


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
