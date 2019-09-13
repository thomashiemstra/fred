from src.kinematics.kinematics_utils import Pose
from src.utils.obstacle import BoxObstacle
import numpy as np
import pybullet as p

red = [1, 0, 0, 1]


class Scenario:

    def __init__(self, obstacles, start_pose, target_pose):
        self.obstacles = obstacles
        self.target_pose = target_pose
        self.start_pose = start_pose

    def build_scenario(self, physics_client):
        for obstacle in self.obstacles:
            obstacle.build(physics_client)

    def destroy_scenario(self):
        for obstacle in self.obstacles:
            p.removeBody(obstacle.obstacle_id)


if __name__ == '__main__':
    scenarios = [Scenario([BoxObstacle([10, 10, 20], [0, 35, 0], alpha=np.pi / 4),
                           BoxObstacle([10, 10, 20], [0, 35, 0], alpha=np.pi / 4)],
                          Pose(-20, 15, 10), Pose(20, 15, 10)),
                 Scenario([BoxObstacle([20, 20, 20], [0, 35, 0], alpha=np.pi / 4),
                           BoxObstacle([10, 10, 20], [0, 35, 0], alpha=np.pi / 4)],
                          Pose(-20, 15, 10), Pose(20, 15, 10))
                 ]
