import unittest

from src.kinematics.kinematics_utils import Pose
from src.reinforcementlearning.environment.robot_env import RobotEnv
from tf_agents.environments import utils
import numpy as np

from src.reinforcementlearning.environment.robot_env_with_obstacles import RobotEnvWithObstacles
from src.reinforcementlearning.environment.scenario import Scenario
from src.utils.obstacle import BoxObstacle

scenario_no_obstacles = Scenario([], Pose(-25, 35, 10), Pose(25, 35, 10))
scenario_obstacles = Scenario([BoxObstacle([10, 40, 10], [0, 35, 0])], Pose(-25, 35, 10), Pose(25, 35, 10))


class TestRobotEnv(unittest.TestCase):

    def test_env_no_obs(self):
        env = RobotEnv(use_gui=False)
        env._externally_set_scenario = scenario_no_obstacles
        utils.validate_py_environment(env, episodes=5)

    def test_env_obstacles(self):
        env = RobotEnvWithObstacles(scenarios=[scenario_obstacles])
        utils.validate_py_environment(env, episodes=5)

    def test_initial_observations_normalized(self):
        env = RobotEnv(use_gui=False)
        initial_obs = env.reset()

        norm_1 = np.linalg.norm(initial_obs.observation[0:3])

        self.assertAlmostEqual(norm_1, 1.0, places=2, msg="The norm of observation vector 1 should be 1")

    def test_subsequent_observation_normalized(self):
        env = RobotEnv(use_gui=False, scenarios=[scenario_no_obstacles])
        env.reset()

        simple_action = np.array([-1, 0, 0, 0, 0], dtype=np.float32)
        time_step = env.step(simple_action)

        norm_1 = np.linalg.norm(time_step.observation[0:3])

        self.assertAlmostEqual(norm_1, 1.0, places=2, msg="The norm of observation vector 1 should be 1")

