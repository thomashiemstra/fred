import unittest

from src.reinforcementlearning.robot_env import RobotEnv
from tf_agents.environments import utils


class TestRobotEnv(unittest.TestCase):

    def test_env(self):
        env = RobotEnv(use_gui=False)
        utils.validate_py_environment(env, episodes=5)
