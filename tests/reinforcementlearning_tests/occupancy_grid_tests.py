import unittest
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve

from src.reinforcementlearning.environment.occupancy_grid_util import is_point_in_obstacle, get_height_tallest_obstacle, \
    create_hilbert_curve_from_obstacles, create_occupancy_grid_from_obstacles
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


class OccupancyGridTest(unittest.TestCase):

    def test_empty_grid(self):
        grid = create_occupancy_grid_from_obstacles(None, 10, 10, 1)

        expected_grid = np.zeros((10, 10))

        self.assertTrue(np.array_equal(grid, expected_grid),
                        "did not get the expected array. Expected: {}, actual: {}".format(expected_grid, grid))

    def test_simple_obstacle(self):
        obstacle = BoxObstacle([1, 1, 40], [0, 2, 0])
        grid = create_occupancy_grid_from_obstacles([obstacle], 4, 4, 1)

        expected_grid = np.array(
            [[0, 0, 0, 0],
             [0, 1, 1, 0],
             [0, 1, 1, 0],
             [0, 0, 0, 0]]
        )

        self.assertTrue(np.array_equal(grid, expected_grid),
                        "did not get the expected array. Expected: {}, actual: {}".format(expected_grid, grid))

    def test_obstacle_out_of_field(self):
        obstacle = BoxObstacle([1, 1, 40], [10, 10, 0])
        grid = create_occupancy_grid_from_obstacles([obstacle], 4, 4, 1)

        expected_grid = np.zeros((4, 4))

        self.assertTrue(np.array_equal(grid, expected_grid),
                        "did not get the expected array. Expected: {}, actual: {}".format(expected_grid, grid))

    def test_height_is_scaled_and_tallest_obstacle_wins(self):
        low_obstacle = BoxObstacle([2, 4, 10], [-1, 2, 0])
        high_obstacle = BoxObstacle([2, 2, 20], [0, 2, 0])
        too_high_obstacle = BoxObstacle([1, 1, 80], [1.5, 0.5, 0])
        grid = create_occupancy_grid_from_obstacles([low_obstacle, high_obstacle, too_high_obstacle], 4, 4, 1)

        # rows are x (-2..2), columns are y (0..4); heights are divided by 40 and capped at 1
        expected_grid = np.array(
            [[0.25, 0.25, 0.25, 0.25],
             [0.25, 0.5, 0.5, 0.25],
             [0, 0.5, 0.5, 0],
             [1, 0, 0, 0]]
        )

        np.testing.assert_allclose(grid, expected_grid)

    def test_rotated_obstacle(self):
        # a 1x6 wall rotated by 90 degrees lies along the x axis at y = 2.5
        obstacle = BoxObstacle([1, 6, 40], [0, 2.5, 0], alpha=np.pi / 2)
        grid = create_occupancy_grid_from_obstacles([obstacle], 6, 4, 1)

        expected_grid = np.zeros((6, 4))
        expected_grid[:, 2] = 1

        np.testing.assert_allclose(grid, expected_grid)

    def test_invalid_grid_size(self):
        with self.assertRaises(ValueError):
            create_occupancy_grid_from_obstacles(None, 10, 10, 3)
        with self.assertRaises(ValueError):
            create_occupancy_grid_from_obstacles(None, 10, 9, 5)


class HilbertCurveMatchesGridTest(unittest.TestCase):

    def test_curve_is_the_grid_in_hilbert_order(self):
        obstacles = [BoxObstacle([10, 20, 20], [5, 20, 0], alpha=np.pi / 4), BoxObstacle([6, 6, 40], [-10, 10, 0])]
        iteration = 3
        # 2**3 = 8 cells per side of 5 cm, the same cells as a 40x40 grid with grid_size 5
        grid = create_occupancy_grid_from_obstacles(obstacles, 40, 40, 5)
        curve = create_hilbert_curve_from_obstacles(obstacles, grid_len_x=40, grid_len_y=40, iteration=iteration)

        self.assertEqual(64, len(curve))
        self.assertTrue(np.any(curve > 0), "the obstacles should show up in the curve")
        hilbert_curve = HilbertCurve(iteration, 2)
        for distance in range(len(curve)):
            x, y = hilbert_curve.coordinates_from_distance(distance)
            self.assertEqual(grid[x][y], curve[distance],
                             "curve element {} should match grid cell ({}, {})".format(distance, x, y))
