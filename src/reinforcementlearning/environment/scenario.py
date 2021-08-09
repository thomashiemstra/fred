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

    def copy(self):
        obstacles_copy = [obstacle.copy() for obstacle in self.obstacles]
        return Scenario(obstacles_copy, self.start_pose, self.target_pose)


scenarios_no_obstacles = [
    Scenario([],
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
             Pose(0, 35, 10), Pose(25, 30, 30))]

scenarios_obstacles = [
    # 0
    Scenario([BoxObstacle([20, 25, 40], [0, 40, 0], alpha=np.pi / 4)],
             Pose(-25, 25, 10), Pose(25, 25, 10)),
    # 1
    Scenario([BoxObstacle([10, 10, 30], [-5, 35, 0], alpha=0),
              BoxObstacle([10, 20, 20], [5, 35, 0], alpha=np.pi / 4)],
             Pose(-25, 20, 10), Pose(30, 30, 10)),
    # 2
    Scenario([BoxObstacle([10, 20, 20], [-10, 36, 0], alpha=-np.pi / 4),
              BoxObstacle([10, 20, 20], [10, 36, 0], alpha=np.pi / 4)],
             Pose(-25, 20, 10), Pose(25, 20, 10)),
    # 3
    Scenario([BoxObstacle([10, 40, 25], [0, 40, 0], alpha=0)],
             Pose(-25, 30, 10), Pose(25, 30, 10)),
    # 4
    Scenario([BoxObstacle([10, 30, 20], [0, 30, 0], alpha=np.pi / 8),
              BoxObstacle([10, 10, 30], [10, 35, 0], alpha=0)],
             Pose(-25, 30, 10), Pose(25, 30, 10)),
    # 5
    Scenario([BoxObstacle([10, 30, 20], [0, 35, 0], alpha=np.pi / 2),
              BoxObstacle([10, 10, 35], [0, 30, 0], alpha=0)],
             Pose(-25, 30, 10), Pose(25, 30, 10)),
    # 6
    Scenario([BoxObstacle([20, 20, 20], [-20, 40, 0], alpha=np.pi / 2),
              BoxObstacle([10, 10, 35], [0, 30, 0], alpha=0)],
             Pose(-25, 20, 10), Pose(20, 40, 10)),
    # 7
    Scenario([BoxObstacle([10, 40, 20], [10, 40, 0], alpha=-np.pi / 8),
              BoxObstacle([10, 10, 35], [-5, 38, 0], alpha=0)],
             Pose(-25, 30, 10), Pose(25, 20, 10)),
    # 8
    Scenario([BoxObstacle([10, 10, 40], [5, 30, 0], alpha=0),
              BoxObstacle([30, 30, 20], [-5, 40, 0], alpha=0)],
             Pose(-35, 15, 10), Pose(25, 30, 30)),
    # 9
    Scenario([BoxObstacle([10, 40, 20], [10, 40, 0], alpha=-np.pi / 4),
              BoxObstacle([10, 40, 20], [-10, 40, 0], alpha=np.pi / 4)],
             Pose(-35, 15, 10), Pose(25, 30, 15)),
    # 10
    Scenario([BoxObstacle([10, 40, 20], [10, 35, 0], alpha=0),
              BoxObstacle([10, 40, 20], [-10, 40, 0], alpha=np.pi / 2)],
             Pose(-30, 25, 10), Pose(35, 20, 10)),
    # 11
    Scenario([BoxObstacle([10, 40, 25], [-10, 35, 0], alpha=0),
              BoxObstacle([10, 40, 25], [15, 40, 0], alpha=np.pi / 2)],
             Pose(-30, 25, 10), Pose(20, 25, 10)),
    # 12
    Scenario([BoxObstacle([10, 30, 20], [5, 30, 0], alpha=0),
              BoxObstacle([10, 30, 20], [25, 40, 0], alpha=np.pi / 2),
              BoxObstacle([10, 10, 40], [-5, 33, 0], alpha=np.pi / 2)],
             Pose(-30, 25, 10), Pose(30, 25, 10)),
    # 13
    Scenario([BoxObstacle([10, 30, 20], [-5, 30, 0], alpha=0),
              BoxObstacle([10, 30, 20], [-25, 40, 0], alpha=np.pi / 2),
              BoxObstacle([10, 10, 40], [5, 30, 0], alpha=np.pi / 2)],
             Pose(-30, 25, 10), Pose(30, 25, 10)),
    # 14
    Scenario([BoxObstacle([10, 30, 20], [15, 35, 0], alpha=0),
              BoxObstacle([10, 30, 20], [-15, 35, 0], alpha=np.pi / 2),
              BoxObstacle([10, 10, 40], [5, 30, 0], alpha=np.pi / 2)],
             Pose(-30, 15, 10), Pose(35, 25, 10)),
    # 15
    Scenario([BoxObstacle([40, 10, 25], [10, 40, 0], alpha=np.pi / 4),
              BoxObstacle([10, 10, 40], [5, 30, 0], alpha=np.pi / 4)],
             Pose(-25, 20, 10), Pose(25, 35, 10)),
    # 16
    Scenario([BoxObstacle([10, 40, 20], [0, 35, 0], alpha=np.pi / 4),
              BoxObstacle([10, 40, 20], [0, 35, 0], alpha=-np.pi / 4),
              BoxObstacle([10, 10, 40], [0, 35, 0], alpha=-np.pi / 4)
              ],
             Pose(-30, 25, 10), Pose(30, 25, 10)),
    # 17
    Scenario([BoxObstacle([15, 40, 25], [-10, 35, 0], alpha=0),
              BoxObstacle([10, 40, 20], [15, 35, 0], alpha=-np.pi / 4),
              ],
             Pose(-30, 25, 10), Pose(30, 25, 10)),
    # 18
    Scenario([BoxObstacle([10, 40, 15], [-10, 35, 0], alpha=0),
              BoxObstacle([10, 40, 25], [20, 35, 0], alpha=-np.pi / 4),
              BoxObstacle([10, 40, 20], [40, 35, 0], alpha=np.pi / 4)],
             Pose(-20, 25, 25), Pose(30, 25, 10)),
    # 19
    Scenario([BoxObstacle([10, 40, 15], [10, 35, 0], alpha=0),
              BoxObstacle([10, 40, 20], [-35, 35, 0], alpha=-np.pi / 4),
              BoxObstacle([10, 40, 20], [-15, 35, 0], alpha=np.pi / 4)
              ],
             Pose(-25, 25, 10), Pose(30, 25, 15)),
    # 20
    Scenario([
        BoxObstacle([10, 40, 20], [10, 40, 0], alpha=-np.pi / 4),
        BoxObstacle([10, 40, 20], [-10, 40, 0], alpha=np.pi / 4),
        BoxObstacle([10, 10, 40], [0, 32, 0], alpha=np.pi / 4),
        BoxObstacle([10, 40, 20], [-20, 30, 0])],
        Pose(-35, 15, 10), Pose(30, 30, 15))
]

start_pose = Pose(-30, 25, 10)
end_pose = Pose(30, 25, 10)

super_easy_scenarios = [
    Scenario([],
             start_pose, end_pose),
]

easy_scenarios = [
    Scenario([BoxObstacle([10, 40, 10], [0, 35, 0])],
             start_pose, end_pose),

    Scenario([BoxObstacle([10, 40, 10], [0, 40, 0], alpha=np.pi / 4)],
             start_pose, end_pose),

    Scenario([BoxObstacle([20, 20, 40], [0, 50, 0])],
             start_pose, end_pose),

    Scenario([BoxObstacle([10, 30, 10], [0, 32, 0]), BoxObstacle([10, 10, 30], [0, 40, 0])],
             start_pose, end_pose),

    Scenario([BoxObstacle([10, 40, 10], [0, 32, 0], alpha=np.pi / 4),
              BoxObstacle([10, 40, 10], [0, 32, 0], alpha=-np.pi / 4),
              BoxObstacle([10, 10, 40], [0, 32, 0], alpha=-np.pi / 4)],
             start_pose, end_pose),

    Scenario([BoxObstacle([10, 30, 10], [-10, 30, 0]), BoxObstacle([10, 10, 40], [0, 30, 0], alpha=np.pi / 4)],
             start_pose, end_pose),

    Scenario([BoxObstacle([10, 10, 40], [0, 25, 0], alpha=np.pi / 4), BoxObstacle([10, 40, 10], [0, 30, 0]),
              BoxObstacle([40, 10, 10], [0, 25, 0])],
             start_pose, end_pose)
]

medium_scenarios = [
    Scenario([BoxObstacle([10, 40, 15], [0, 35, 0])],
             start_pose, end_pose),

    Scenario([BoxObstacle([10, 40, 15], [0, 40, 0], alpha=np.pi / 4)],
             start_pose, end_pose),

    Scenario([BoxObstacle([20, 20, 40], [0, 40, 0])],
             start_pose, end_pose),

    Scenario([BoxObstacle([10, 30, 20], [0, 32, 0]), BoxObstacle([15, 15, 30], [0, 40, 0])],
             start_pose, end_pose),

    Scenario([BoxObstacle([10, 40, 15], [0, 35, 0], alpha=np.pi / 4),
              BoxObstacle([10, 40, 15], [0, 35, 0], alpha=-np.pi / 4),
              BoxObstacle([10, 10, 40], [0, 35, 0], alpha=-np.pi / 4)],
             start_pose, end_pose),

    Scenario([BoxObstacle([10, 30, 15], [-10, 30, 0]), BoxObstacle([10, 10, 40], [0, 30, 0], alpha=np.pi / 4)],
             start_pose, end_pose),

    Scenario([BoxObstacle([10, 10, 40], [0, 32, 0], alpha=np.pi / 4),
              BoxObstacle([10, 40, 15], [0, 37, 0]),
              BoxObstacle([40, 10, 15], [0, 32, 0])],
             start_pose, end_pose)
]

hard_scenarios = [
    Scenario([BoxObstacle([10, 40, 25], [0, 40, 0])],
             start_pose, end_pose),

    Scenario([BoxObstacle([10, 40, 25], [0, 40, 0], alpha=np.pi / 4)],
             start_pose, end_pose),

    Scenario([BoxObstacle([20, 20, 40], [0, 35, 0])],
             start_pose, end_pose),

    Scenario([BoxObstacle([10, 30, 30], [0, 35, 0]), BoxObstacle([10, 10, 40], [0, 45, 0])],
             start_pose, end_pose),

    Scenario([BoxObstacle([10, 40, 20], [0, 34, 0], alpha=np.pi / 4),
              BoxObstacle([10, 40, 20], [0, 34, 0], alpha=-np.pi / 4),
              BoxObstacle([10, 10, 40], [0, 34, 0], alpha=-np.pi / 4)],
             start_pose, end_pose),

    Scenario([BoxObstacle([10, 30, 20], [-5, 35, 0]), BoxObstacle([10, 10, 40], [0, 32, 0], alpha=np.pi / 4)],
             start_pose, end_pose),

    Scenario([BoxObstacle([10, 10, 40], [0, 35, 0], alpha=np.pi / 4),
              BoxObstacle([10, 40, 20], [0, 35, 0]),
              BoxObstacle([40, 10, 20], [0, 30, 0])],
             start_pose, end_pose)
]

sensible_scenarios = [
    Scenario([BoxObstacle([10, 40, 10], [0, 32, 0])],
             start_pose, end_pose),

    Scenario([BoxObstacle([10, 40, 25], [0, 32, 0])],
             start_pose, end_pose),

    Scenario([BoxObstacle([10, 40, 15], [0, 32, 0], alpha=np.pi / 4),
              BoxObstacle([10, 40, 15], [0, 32, 0], alpha=-np.pi / 4),
              BoxObstacle([10, 10, 40], [0, 32, 0], alpha=-np.pi / 4)],
             start_pose, end_pose),

    Scenario([BoxObstacle([10, 40, 20], [0, 32, 0]),
              BoxObstacle([15, 15, 40], [0, 35, 0])],
             start_pose, end_pose),

    Scenario([BoxObstacle([10, 40, 20], [0, 32, 0]),
              BoxObstacle([15, 15, 40], [15, 35, 0])],
             start_pose, end_pose),

    Scenario([BoxObstacle([20, 20, 40], [0, 40, 0])],
             start_pose, end_pose),

    Scenario([BoxObstacle([10, 40, 15], [-5, 32, 0]),
              BoxObstacle([10, 40, 25], [5, 32, 0])],
             start_pose, end_pose),

    Scenario([BoxObstacle([10, 40, 25], [-5, 34, 0]),
              BoxObstacle([10, 10, 40], [5, 40, 0]),
              BoxObstacle([40, 10, 20], [-20, 40, 0])],
             start_pose, end_pose),

    Scenario([BoxObstacle([30, 15, 20], [0, 22, 0]),
              BoxObstacle([10, 10, 40], [0, 32, 0])],
             start_pose, end_pose),

    Scenario([BoxObstacle([20, 10, 20], [0, 20, 0])],
             start_pose, end_pose),

    Scenario([BoxObstacle([10, 30, 20], [0, 30, 0], alpha=np.pi / 4)],
             start_pose, end_pose),

    Scenario([BoxObstacle([10, 40, 25], [10, 35, 0]),
              BoxObstacle([30, 10, 15], [-10, 35, 0])],
             start_pose, end_pose),

    Scenario([BoxObstacle([10, 30, 25], [-10, 35, 0]),
              BoxObstacle([30, 10, 15], [10, 40, 0])],
             start_pose, end_pose),

    Scenario([BoxObstacle([8, 23, 21], [0, 32, 0])],
             start_pose, end_pose),

    Scenario([BoxObstacle([8, 23, 21], [0, 35, 0]),
              BoxObstacle([19, 5, 15], [0, 20, 0])],
             start_pose, end_pose)
]

train_scenarios = [
    Scenario([BoxObstacle([30, 10, 20], [0, 20, 0]),
              BoxObstacle([10, 10, 40], [0, 30, 0])],
             Pose(15, 25, 30), end_pose),
    Scenario([BoxObstacle([10, 40, 25], [0, 32, 0])],
             Pose(15, 25, 30), end_pose),
    Scenario([BoxObstacle([10, 40, 15], [-5, 32, 0]),
              BoxObstacle([10, 40, 25], [5, 32, 0])],
             Pose(20, 25, 30), end_pose),
    Scenario([BoxObstacle([10, 40, 20], [0, 32, 0]),
              BoxObstacle([15, 15, 40], [0, 35, 0])],
             Pose(0, 22, 30), end_pose),
    Scenario([BoxObstacle([10, 40, 25], [-5, 34, 0]),
              BoxObstacle([10, 10, 40], [5, 40, 0]),
              BoxObstacle([40, 10, 20], [-20, 40, 0])],
             Pose(0, 25, 35), end_pose)
]

test = [
    Scenario([BoxObstacle([8, 23, 21], [0, 35, 0]),
              BoxObstacle([19, 5, 15], [0, 20, 0])],
             start_pose, end_pose)
]

if __name__ == '__main__':
    scenarios = [Scenario([BoxObstacle([10, 10, 20], [0, 35, 0], alpha=np.pi / 4),
                           BoxObstacle([10, 10, 20], [0, 35, 0], alpha=np.pi / 4)],
                          Pose(-20, 15, 10), Pose(20, 15, 10)),
                 Scenario([BoxObstacle([20, 20, 20], [0, 35, 0], alpha=np.pi / 4),
                           BoxObstacle([10, 10, 20], [0, 35, 0], alpha=np.pi / 4)],
                          Pose(-20, 15, 10), Pose(20, 15, 10))
                 ]
