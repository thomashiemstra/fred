from src.kinematics.kinematics_utils import Pose
from src.utils.obstacle import BoxObstacle
import numpy as np

red = [1, 0, 0, 1]


class Scenario:

    def __init__(self, obstacles, start_pose, target_pose):
        self.obstacles = obstacles
        self.target_pose = target_pose
        self.start_pose = start_pose

    def build_scenario(self, physics_client):
        for obstacle in self.obstacles:
            obstacle.build(physics_client)

    def destroy_scenario(self, physics_client):
        for obstacle in self.obstacles:
            obstacle.destroy(physics_client)


scenarios_no_obstacles = [Scenario([],
                      Pose(-25, 35, 10), Pose(25, 35, 10)),
             Scenario([],
                      Pose(-30, 20, 10), Pose(20, 40, 20)),
             Scenario([],
                      Pose(-35, 15, 10), Pose(25, 30, 30)),
             Scenario([],
                      Pose(0, 20, 15), Pose(0, 35, 40)),
             Scenario([],
                      Pose(-25, 35, 10), Pose(0, 35, 30)),
             Scenario([],
                      Pose(0, 35, 30), Pose(30, 30, 10)),
             Scenario([],
                      Pose(0, 35, 30), Pose(-30, 30, 10)),
             Scenario([],
                      Pose(20, 40, 15), Pose(-20, 30, 30)),
             Scenario([],
                      Pose(0, 35, 10), Pose(-25, 30, 30)),
             Scenario([],
                      Pose(0, 35, 10), Pose(25, 30, 30)),

             Scenario([BoxObstacle([20, 25, 40], [0, 35, 0], alpha=np.pi / 4)],
                      Pose(-25, 25, 10), Pose(25, 25, 10))]

scenarios_obstacles = [Scenario([BoxObstacle([20, 25, 40], [0, 35, 0], alpha=np.pi / 4)],
                      Pose(-25, 25, 10), Pose(25, 25, 10)),

             Scenario([BoxObstacle([10, 10, 30], [-5, 35, 0], alpha=0),
                       BoxObstacle([10, 20, 20], [5, 35, 0], alpha=np.pi / 4)],
                      Pose(-25, 20, 10), Pose(30, 30, 10)),

             Scenario([BoxObstacle([10, 20, 20], [-10, 38, 0], alpha=-np.pi / 4),
                       BoxObstacle([10, 20, 20], [10, 38, 0], alpha=np.pi / 4)],
                      Pose(-25, 20, 10), Pose(25, 20, 10)),

             Scenario([BoxObstacle([10, 40, 25], [0, 35, 0], alpha=0)],
                      Pose(-25, 30, 10), Pose(25, 30, 10)),
             Scenario([BoxObstacle([10, 30, 20], [0, 30, 0], alpha=np.pi / 8),
                       BoxObstacle([10, 10, 30], [10, 35, 0], alpha=0)],
                      Pose(-25, 30, 10), Pose(25, 30, 10)),
             Scenario([BoxObstacle([10, 30, 20], [0, 35, 0], alpha=np.pi / 2),
                       BoxObstacle([10, 10, 35], [0, 25, 0], alpha=0)],
                      Pose(-25, 30, 10), Pose(25, 30, 10)),
             Scenario([BoxObstacle([20, 20, 20], [-20, 40, 0], alpha=np.pi / 2),
                       BoxObstacle([10, 10, 35], [0, 25, 0], alpha=0)],
                      Pose(-25, 20, 10), Pose(20, 40, 10)),
             Scenario([BoxObstacle([10, 40, 20], [10, 40, 0], alpha=-np.pi / 8),
                       BoxObstacle([10, 10, 35], [-5, 38, 0], alpha=0)],
                      Pose(-25, 40, 10), Pose(20, 20, 10)),
             Scenario([BoxObstacle([10, 10, 40], [5, 30, 0], alpha=0),
                       BoxObstacle([30, 30, 20], [-5, 40, 0], alpha=0)],
                      Pose(-35, 15, 10), Pose(25, 30, 30)),
             Scenario([BoxObstacle([10, 40, 20], [10, 40, 0], alpha=-np.pi / 4),
                       BoxObstacle([10, 40, 20], [-10, 40, 0], alpha=np.pi / 4)],
                      Pose(-35, 15, 10), Pose(25, 30, 20)),
             Scenario([BoxObstacle([10, 40, 20], [10, 35, 0], alpha=0),
                       BoxObstacle([10, 40, 20], [-10, 40, 0], alpha=np.pi / 2)],
                      Pose(-30, 25, 10), Pose(35, 20, 10)),

             Scenario([BoxObstacle([10, 40, 20], [-10, 35, 0], alpha=0),
                       BoxObstacle([10, 40, 20], [15, 40, 0], alpha=np.pi / 2)],
                      Pose(-30, 25, 10), Pose(20, 25, 10)),

             Scenario([BoxObstacle([10, 30, 20], [5, 30, 0], alpha=0),
                       BoxObstacle([10, 30, 20], [25, 40, 0], alpha=np.pi / 2),
                       BoxObstacle([10, 10, 40], [-5, 30, 0], alpha=np.pi / 2)],
                      Pose(-30, 25, 10), Pose(30, 25, 10)),
             Scenario([BoxObstacle([10, 30, 20], [-5, 30, 0], alpha=0),
                       BoxObstacle([10, 30, 20], [-25, 40, 0], alpha=np.pi / 2),
                       BoxObstacle([10, 10, 40], [5, 30, 0], alpha=np.pi / 2)],
                      Pose(-30, 25, 10), Pose(30, 25, 10)),

             Scenario([BoxObstacle([10, 30, 20], [15, 35, 0], alpha=0),
                       BoxObstacle([10, 30, 20], [-15, 35, 0], alpha=np.pi / 2),
                       BoxObstacle([10, 10, 40], [5, 30, 0], alpha=np.pi / 2)],
                      Pose(-30, 15, 10), Pose(30, 25, 10)),

             Scenario([BoxObstacle([10, 40, 10], [0, 30, 0], alpha=0),
                       BoxObstacle([10, 40, 40], [40, 25, 0], alpha=0),
                       BoxObstacle([80, 10, 40], [0, 40, 0], alpha=0),
                       BoxObstacle([10, 40, 40], [-40, 25, 0], alpha=0)],
                      Pose(-25, 20, 10), Pose(25, 20, 10)),

             Scenario([BoxObstacle([10, 40, 20], [0, 35, 0], alpha=np.pi / 4),
                       BoxObstacle([10, 40, 20], [0, 35, 0], alpha=-np.pi / 4),
                       BoxObstacle([10, 10, 40], [0, 35, 0], alpha=-np.pi / 4)
                       ],
                      Pose(-30, 25, 10), Pose(30, 25, 10)),

             Scenario([BoxObstacle([10, 40, 25], [-10, 35, 0], alpha=0),
                       BoxObstacle([10, 40, 20], [15, 35, 0], alpha=-np.pi / 4),
                       ],
                      Pose(-30, 25, 10), Pose(30, 25, 10)),

             Scenario([BoxObstacle([10, 40, 15], [-10, 35, 0], alpha=0),
                            BoxObstacle([10, 40, 20], [20, 35, 0], alpha=-np.pi/4),
                            BoxObstacle([10, 40, 20], [40, 35, 0], alpha=np.pi/4)
                            ],
                           Pose(-20, 25, 30), Pose(30, 25, 10)),

             Scenario([BoxObstacle([10, 40, 15], [10, 35, 0], alpha=0),
                       BoxObstacle([10, 40, 20], [-35, 35, 0], alpha=-np.pi / 4),
                       BoxObstacle([10, 40, 20], [-15, 35, 0], alpha=np.pi / 4)
                       ],
                      Pose(-25, 25, 10), Pose(30, 25, 30))]

if __name__ == '__main__':
    scenarios = [Scenario([BoxObstacle([10, 10, 20], [0, 35, 0], alpha=np.pi / 4),
                           BoxObstacle([10, 10, 20], [0, 35, 0], alpha=np.pi / 4)],
                          Pose(-20, 15, 10), Pose(20, 15, 10)),
                 Scenario([BoxObstacle([20, 20, 20], [0, 35, 0], alpha=np.pi / 4),
                           BoxObstacle([10, 10, 20], [0, 35, 0], alpha=np.pi / 4)],
                          Pose(-20, 15, 10), Pose(20, 15, 10))
                 ]
