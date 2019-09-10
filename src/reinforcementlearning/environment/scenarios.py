from src.kinematics.kinematics_utils import Pose
from src.utils.obstacle import BoxObstacle
import numpy as np

red = [1, 0, 0, 1]


class Scenario:

    def __init__(self, obstacles, target_pose, start_pose):
        self.obstacles = obstacles
        self.target_pose = target_pose
        self.start_pose = start_pose

    def build_scenario(self, physics_client):
        for obstacle in self.obstacles:
            obstacle.build(physics_client)


scenarios = [Scenario([BoxObstacle([10, 10, 20], [0, 35, 0], alpha=np.pi / 4),
                       BoxObstacle([10, 10, 20], [0, 35, 0], alpha=np.pi / 4)],
                      Pose(-20, 15, 10), Pose(20, 15, 10)),
             Scenario([BoxObstacle([20, 20, 20], [0, 35, 0], alpha=np.pi / 4),
                       BoxObstacle([10, 10, 20], [0, 35, 0], alpha=np.pi / 4)],
                      Pose(-20, 15, 10), Pose(20, 15, 10))
             ]
