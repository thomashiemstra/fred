import numpy as np
from src.kinematics.kinematics_utils import Pose
from src.reinforcementlearning.environment.occupancy_grid_util import create_hilbert_curve_from_obstacles
from src.reinforcementlearning.environment.robot_env import RobotEnv
from src.reinforcementlearning.environment.scenario import Scenario, \
    sensible_scenarios
from src.utils.obstacle import BoxObstacle
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


class RobotEnvWithObstacles(RobotEnv):

    def __init__(self, use_gui=False, raw_obs=False, scenarios=None, is_eval=False, robot_controller=None,
                 angle_control=False):
        if scenarios is None:
            raise ValueError("RobotEnvWithObstacles should be initialized with scenarios, "
                             "otherwise we would default to no obstacle scenarios!")
        super().__init__(use_gui, raw_obs, is_eval=is_eval, scenarios=scenarios, robot_controller=robot_controller,
                         angle_control=angle_control)
        self._hilbert_curve_iteration = 3
        self._grid_len_x = 40
        self._grid_len_y = 40
        self._observation_spec = (array_spec.BoundedArraySpec(
            shape=(20,),
            dtype=np.float32, minimum=-1, maximum=1, name='observation'),
                                  array_spec.BoundedArraySpec(
                                      shape=(2 ** (2 * self._hilbert_curve_iteration),),
                                      dtype=np.float32, minimum=0, maximum=1, name='hilbert_curve'),
        )
        self._max_steps_to_take_before_failure = 100
        self._update_step_size = 0.03
        self._curve = create_hilbert_curve_from_obstacles(self._obstacles, grid_len_x=self._grid_len_x,
                                                          grid_len_y=self._grid_len_y,
                                                          iteration=self._hilbert_curve_iteration)

    def _reset(self):
        super(RobotEnvWithObstacles, self)._reset()
        self._curve = create_hilbert_curve_from_obstacles(self._obstacles, grid_len_x=self._grid_len_x,
                                                          grid_len_y=self._grid_len_y,
                                                          iteration=self._hilbert_curve_iteration)
        observation, _ = self._get_observations()
        return ts.restart(observation)

    def _get_observations(self):
        no_obstacle_obs, total_distance = super()._get_observations()

        curve = self._curve

        total_observation = [np.array(no_obstacle_obs, dtype=np.float32),
                             np.array(np.array(curve.tolist()), dtype=np.float32)]

        return total_observation, total_distance

    def show_occupancy_grid_and_curve(self):
        from src.reinforcementlearning.environment.occupancy_grid_util import create_occupancy_grid_from_obstacles

        grid = create_occupancy_grid_from_obstacles(self._obstacles, grid_len_x=self._grid_len_x,
                                                          grid_len_y=self._grid_len_y, grid_size=4)
        curve = create_hilbert_curve_from_obstacles(self._obstacles, grid_len_x=self._grid_len_x,
                                                          grid_len_y=self._grid_len_y,
                                                          iteration=self._hilbert_curve_iteration)

        import matplotlib.pyplot as plt

        plt.set_cmap('hot')

        fig = plt.figure()
        ax1 = fig.add_subplot(2, 2, 1)

        reshape = 2 ** self._hilbert_curve_iteration
        ax1.imshow(curve.reshape(reshape, reshape))
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(grid)

        plt.show()


if __name__ == '__main__':
    x = Scenario([BoxObstacle([10, 40, 15], [-10, 35, 0]), BoxObstacle([10, 40, 25], [10, 35, 0])],
             Pose(-25, 20, 10), Pose(30, 30, 10)),
    scenario = Scenario([BoxObstacle([10, 10, 30], [-5, 20, 0], alpha=0),
                         BoxObstacle([10, 20, 20], [5, 35, 0], alpha=np.pi / 4)],
                        Pose(-25, 20, 10), Pose(30, 30, 10))
    env = RobotEnvWithObstacles(use_gui=True, scenarios=sensible_scenarios)
    state = env.observation_spec()
    print(state)
    obs = env.reset()
    env.show_occupancy_grid_and_curve()
    print("hoi")

# ░░░░░░░█▐▓▓░████▄▄▄█▀▄▓▓▓▌█ Epic code
# ░░░░░▄█▌▀▄▓▓▄▄▄▄▀▀▀▄▓▓▓▓▓▌█
# ░░░▄█▀▀▄▓█▓▓▓▓▓▓▓▓▓▓▓▓▀░▓▌█
# ░░█▀▄▓▓▓███▓▓▓███▓▓▓▄░░▄▓▐█▌ level is so high
# ░█▌▓▓▓▀▀▓▓▓▓███▓▓▓▓▓▓▓▄▀▓▓▐█
# ▐█▐██▐░▄▓▓▓▓▓▀▄░▀▓▓▓▓▓▓▓▓▓▌█▌
# █▌███▓▓▓▓▓▓▓▓▐░░▄▓▓███▓▓▓▄▀▐█ much quality
# █▐█▓▀░░▀▓▓▓▓▓▓▓▓▓██████▓▓▓▓▐█
# ▌▓▄▌▀░▀░▐▀█▄▓▓██████████▓▓▓▌█▌
# ▌▓▓▓▄▄▀▀▓▓▓▀▓▓▓▓▓▓▓▓█▓█▓█▓▓▌█▌ Wow.
# █▐▓▓▓▓▓▓▄▄▄▓▓▓▓▓▓█▓█▓█▓█▓▓▓▐█
