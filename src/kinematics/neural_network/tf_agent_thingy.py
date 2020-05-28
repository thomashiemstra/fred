import inspect
import os

import tensorflow as tf
from tf_agents.environments import tf_py_environment

from src.reinforcementlearning.environment.robot_env import RobotEnv
from src.reinforcementlearning.soft_actor_critic.sac_utils import generate_agent_and_networks

checkpoint_dir = 'reward_test_3'
tf.config.experimental_run_functions_eagerly(True)


current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

root_dir = os.path.expanduser(current_dir + '/checkpoints/' + checkpoint_dir)

train_dir = os.path.join(root_dir, 'train')
tf.compat.v1.enable_v2_behavior()

global_step = tf.compat.v1.train.create_global_step()
with tf.compat.v2.summary.record_if(False):
    tf_env = tf_py_environment.TFPyEnvironment(RobotEnv())

    tf_agent, actor_net, critic_net = generate_agent_and_networks(tf_env, None)
    actor_net.summary()

    actor_net.compile(optimizer=tf.keras.optimizers.RMSprop(),  # Optimizer
              # Loss function to minimize
              loss=tf.keras.losses.MeanSquaredError()
              )


    # critic_net.summary()


eval_py_env = tf_py_environment.TFPyEnvironment(RobotEnv(use_gui=False))

time_step = eval_py_env.reset()

action_step = tf_agent.policy.action(time_step)
