import inspect
import os

import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import policy_step
from time import sleep

from src.global_constants import sac_network_weights
from src.reinforcementlearning.environment.robot_env import RobotEnv
from src.reinforcementlearning.environment.robot_env_with_obstacles import RobotEnvWithObstacles
from src.reinforcementlearning.softActorCritic.sac_utils import create_agent, \
    initialize_and_restore_train_checkpointer
from src.reinforcementlearning.environment.scenario import easy_scenarios, medium_scenarios, hard_scenarios, \
    super_easy_scenarios
from src.utils.os_utils import get_project_root


robot_env_no_obstacles = False

train_dir = sac_network_weights
tf.compat.v1.enable_v2_behavior()
global_step = tf.compat.v1.train.create_global_step()

print(get_project_root())

# for debugging
# tf.config.experimental_run_functions_eagerly(True)

use_gui = True

if robot_env_no_obstacles:
    eval_py_env = tf_py_environment.TFPyEnvironment(RobotEnv(use_gui=use_gui,  is_eval=True))
else:
    eval_py_env = tf_py_environment.TFPyEnvironment(RobotEnvWithObstacles(use_gui=use_gui, scenarios=medium_scenarios, is_eval=True))

with tf.compat.v2.summary.record_if(False):
    tf_agent = create_agent(eval_py_env, None, robot_env_no_obstacles)
    initialize_and_restore_train_checkpointer(train_dir, tf_agent, global_step)

num_episodes = 100


# eval_py_env.pyenv.envs[0].scenario = scenario.scenarios_no_obstacles[3]
for i in range(1900):
    # eval_py_env.pyenv.envs[0].scenario = scenario.scenarios_no_obstacles[i % 10]
    # eval_py_env.pyenv.envs[0].reverse_scenario = random.choice([True, False])
    reward = 0

    # eval_py_env.pyenv.envs[0].set_scenario(medium_scenarios[i])
    time_step = eval_py_env.reset()
    while not time_step.is_last():
        action_step = tf_agent.policy.action(time_step)

        means, stds, network_state = tf_agent.get_actor_network().call_raw(time_step.observation, time_step.step_type, (), None)

        actions = tf.constant([[0, 1]], dtype=tf.int32)
        action_steps = policy_step.PolicyStep(actions)

        time_step = eval_py_env.step(action_step.action)
        reward += time_step.reward
        sleep(0.05)
    print("reward for episode {} is= {}".format(i, reward))

print("whoop")

# make_video(env_name, tf_agent, video_filename='eval')
