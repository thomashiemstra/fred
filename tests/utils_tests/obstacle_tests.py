import unittest
import numpy as np
import pybullet as p

from src.utils.obstacle import BoxObstacle, SphereObstacle


class ObstacleTests(unittest.TestCase):

    def setUp(self):
        self.physics_client = p.connect(p.DIRECT)

    def tearDown(self):
        p.disconnect(physicsClientId=self.physics_client)

    def test_box_obstacle(self):
        obstacle = BoxObstacle([1, 1, 1], [0, 0, 0])
        obstacle.build(self.physics_client)
        self.assertIsNotNone(obstacle.obstacle_id,
                             "obstacle should have been created resulting in an obstacle id")
        self.assertEqual(0, p.getNumJoints(obstacle.obstacle_id, physicsClientId=self.physics_client),
                         "should not find any joins since it's a single obstacle")

    def test_box_obstacle_zero_dimensions(self):
        obstacle = BoxObstacle([0, 0, 0], [0, 0, 0])
        obstacle.build(self.physics_client)
        self.assertIsNotNone(obstacle.obstacle_id,
                             "obstacle should have been created resulting in an obstacle id")
        self.assertEqual(0, p.getNumJoints(obstacle.obstacle_id, physicsClientId=self.physics_client),
                         "should not find any joins since it's a single obstacle")

    def test_box_stands_on_its_base_and_is_turned_around_the_z_axis(self):
        # dimensions and positions are in cm, pybullet works in m
        obstacle = BoxObstacle([10, 20, 30], [5, 35, 2], alpha=np.pi / 4)
        obstacle.build(self.physics_client)

        self.assertEqual([5, 35, 2 + 15], obstacle.base_center_position)
        position, orientation = p.getBasePositionAndOrientation(obstacle.obstacle_id,
                                                                physicsClientId=self.physics_client)
        np.testing.assert_allclose([0.05, 0.35, 0.17], position, atol=1e-6)
        yaw = p.getEulerFromQuaternion(orientation)[2]
        self.assertAlmostEqual(np.pi / 4, yaw, places=6)

        aabb_min, aabb_max = p.getAABB(obstacle.obstacle_id, physicsClientId=self.physics_client)
        # the bottom and the top of the box, turning around the z-axis does not change them
        self.assertAlmostEqual(0.02, aabb_min[2], places=2)
        self.assertAlmostEqual(0.32, aabb_max[2], places=2)

    def test_sphere_is_placed_at_its_center(self):
        obstacle = SphereObstacle(5, [0, 30, 10])
        obstacle.build(self.physics_client)

        position, _ = p.getBasePositionAndOrientation(obstacle.obstacle_id, physicsClientId=self.physics_client)
        np.testing.assert_allclose([0, 0.30, 0.10], position, atol=1e-6)
        aabb_min, aabb_max = p.getAABB(obstacle.obstacle_id, physicsClientId=self.physics_client)
        self.assertAlmostEqual(0.10, aabb_max[0] - aabb_min[0], places=2)

    def test_copy_is_an_unbuilt_obstacle_at_the_same_place(self):
        box = BoxObstacle([10, 20, 30], [5, 35, 2], alpha=0.3, color=[1, 0, 0, 1])
        box.build(self.physics_client)
        box_copy = box.copy()
        self.assertIsNone(box_copy.obstacle_id)
        self.assertEqual(box.base_center_position, box_copy.base_center_position)
        self.assertEqual((box.dimensions, box.alpha, box.color), (box_copy.dimensions, box_copy.alpha, box_copy.color))

        sphere_copy = SphereObstacle(5, [0, 30, 10]).copy()
        self.assertEqual((5, [0, 30, 10]), (sphere_copy.radius, sphere_copy.center_position))

    def test_destroy_removes_the_obstacle(self):
        for obstacle in (BoxObstacle([10, 10, 10], [0, 30, 0]), SphereObstacle(5, [0, 30, 10])):
            obstacle.build(self.physics_client)
            bodies = p.getNumBodies(physicsClientId=self.physics_client)
            obstacle.destroy(self.physics_client)
            self.assertEqual(bodies - 1, p.getNumBodies(physicsClientId=self.physics_client))

    def test_destroy_before_build_does_nothing(self):
        for obstacle in (BoxObstacle([10, 10, 10], [0, 30, 0]), SphereObstacle(5, [0, 30, 10])):
            obstacle.destroy(self.physics_client)
        self.assertEqual(0, p.getNumBodies(physicsClientId=self.physics_client))


if __name__ == '__main__':
    unittest.main()
