import unittest

from src.reinforcementlearning.environment.robot_env import RobotEnv
from tf_agents.environments import utils
import numpy as np


class TestRobotEnv(unittest.TestCase):

    def test_env_no_obs(self):
        env = RobotEnv(use_gui=False)
        env.scenario_id = 0
        utils.validate_py_environment(env, episodes=5)

    def test_env_obs(self):
        env = RobotEnv(use_gui=False, no_obstacles=False)
        env.scenario_id = 0
        utils.validate_py_environment(env, episodes=5)

    def test_initial_observasions_normalized(self):
        env = RobotEnv(use_gui=False)
        initial_obs = env.reset()

        norm_1 = np.linalg.norm(initial_obs.observation[0:3])
        norm_2 = np.linalg.norm(initial_obs.observation[3:6])

        self.assertAlmostEqual(norm_1, 1.0, places=2, msg="The norm of observation vector 1 should be 1")
        self.assertAlmostEqual(norm_2, 1.0, places=2, msg="The norm of observation vector 2 should be 1")

    def test_subsequent_observation_normalized(self):
        env = RobotEnv(use_gui=False)
        env.reset()

        simple_action = np.array([-1, 0, 0, 0, 0, 0], dtype=np.float32)
        timestep = env.step(simple_action)

        norm_1 = np.linalg.norm(timestep.observation[0:3])
        norm_2 = np.linalg.norm(timestep.observation[3:6])

        self.assertAlmostEqual(norm_1, 1.0, places=2, msg="The norm of observation vector 1 should be 1")
        self.assertAlmostEqual(norm_2, 1.0, places=2, msg="The norm of observation vector 2 should be 1")
