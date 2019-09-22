import os

from tf_agents.environments import tf_py_environment, suite_gym
import tensorflow as tf

from src.reinforcementlearning.soft_actor_critic.sac_utils import create_agent, make_video, \
    initialize_and_restore_train_checkpointer

env_name = 'BipedalWalker-v2'

root_dir = os.path.expanduser('/home/thomas/PycharmProjects/fred/src/reinforcementlearning/soft_actor_critic')
train_dir = os.path.join(root_dir, 'train')
tf.compat.v1.enable_v2_behavior()

global_step = tf.compat.v1.train.get_or_create_global_step()
with tf.compat.v2.summary.record_if(False):
    eval_tf_env = tf_py_environment.TFPyEnvironment(suite_gym.load(env_name))

    tf_agent = create_agent(eval_tf_env, None)
    initialize_and_restore_train_checkpointer(train_dir, tf_agent, global_step)

make_video(env_name, tf_agent, video_filename='eval')
