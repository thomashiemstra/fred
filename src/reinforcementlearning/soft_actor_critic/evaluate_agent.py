import os
import random

from tf_agents.environments import tf_py_environment, suite_gym
import tensorflow as tf

from src.reinforcementlearning.environment import scenario
from src.reinforcementlearning.environment.robot_env import RobotEnv
from src.reinforcementlearning.soft_actor_critic.sac_utils import create_agent, make_video, \
    initialize_and_restore_train_checkpointer



root_dir = os.path.expanduser('/home/thomas/PycharmProjects/fred/src/reinforcementlearning/checkpoints/robotenv')
train_dir = os.path.join(root_dir, 'train')
tf.compat.v1.enable_v2_behavior()

global_step = tf.compat.v1.train.create_global_step()
with tf.compat.v2.summary.record_if(False):
    tf_env = tf_py_environment.TFPyEnvironment(RobotEnv())

    tf_agent = create_agent(tf_env, None)
    initialize_and_restore_train_checkpointer(train_dir, tf_agent, global_step)

num_episodes = 100

eval_py_env = tf_py_environment.TFPyEnvironment(RobotEnv(use_gui=True))

eval_py_env.pyenv.envs[0].scenario = scenario.scenarios_no_obstacles[0]
for i in range(num_episodes):
    # eval_py_env.pyenv.envs[0].scenario = scenario.scenarios_no_obstacles[i % 10]
    # eval_py_env.pyenv.envs[0].reverse_scenario = random.choice([True, False])
    time_step = eval_py_env.reset()
    while not time_step.is_last():
        action_step = tf_agent.policy.action(time_step)
        time_step = eval_py_env.step(action_step.action)

# make_video(env_name, tf_agent, video_filename='eval')
