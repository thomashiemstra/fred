import unittest
import pybullet as p

from src.utils.obstacle import BoxObstacle


class ObstacleTests(unittest.TestCase):

    def setUp(self):
        self.physics_client = p.connect(p.DIRECT)

    def tearDown(self):
        p.disconnect()

    def test_box_obstacle(self):
        obstacle = BoxObstacle(self.physics_client, [1, 1, 1], [0, 0, 0])
        self.assertIsNotNone(obstacle.obstacle_id,
                             "obstacle should have been created resulting in an obstacle id")
        self.assertEqual(0, p.getNumJoints(obstacle.obstacle_id),
                         "should not find any joins since it's a single obstacle")

    def test_box_obstacle_zero_dimensions(self):
        obstacle = BoxObstacle(self.physics_client, [0, 0, 0], [0, 0, 0])
        self.assertIsNotNone(obstacle.obstacle_id,
                             "obstacle should have been created resulting in an obstacle id")
        self.assertEqual(0, p.getNumJoints(obstacle.obstacle_id),
                         "should not find any joins since it's a single obstacle")
