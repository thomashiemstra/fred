import os

from tf_agents.environments import tf_py_environment, suite_gym
import tensorflow as tf

from src.reinforcementlearning.environment.robot_env import RobotEnv
from src.reinforcementlearning.soft_actor_critic.sac_utils import create_agent, make_video, \
    initialize_and_restore_train_checkpointer

env_name = 'BipedalWalker-v2'

root_dir = os.path.expanduser('/home/thomas/PycharmProjects/fred/src/reinforcementlearning/checkpoints/robotenv')
train_dir = os.path.join(root_dir, 'train')
tf.compat.v1.enable_v2_behavior()

global_step = tf.compat.v1.train.create_global_step()
with tf.compat.v2.summary.record_if(False):
    eval_py_env = RobotEnv(use_gui=True)

    tf_agent = create_agent(eval_py_env, None)
    initialize_and_restore_train_checkpointer(train_dir, tf_agent, global_step)

num_episodes = 3

for _ in range(num_episodes):
    time_step = eval_py_env.reset()
    while not time_step.is_last():
        action_step = tf_agent.policy.action(time_step)
        time_step = eval_py_env.step(action_step.action)

# make_video(env_name, tf_agent, video_filename='eval')
