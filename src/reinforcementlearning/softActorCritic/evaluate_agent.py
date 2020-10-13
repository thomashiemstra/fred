import inspect
import os
import random

from tf_agents.environments import tf_py_environment, suite_gym
import tensorflow as tf
from tf_agents.trajectories import policy_step

from src.reinforcementlearning.environment import scenario
from src.reinforcementlearning.environment.robot_env import RobotEnv
from src.reinforcementlearning.softActorCritic.sac_utils import create_agent, \
    initialize_and_restore_train_checkpointer

checkpoint_dir = 'test'

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
root_dir = os.path.expanduser(current_dir + '/checkpoints/' + checkpoint_dir)

train_dir = os.path.join(root_dir, 'train')
tf.compat.v1.enable_v2_behavior()
global_step = tf.compat.v1.train.create_global_step()

eval_py_env = tf_py_environment.TFPyEnvironment(RobotEnv(use_gui=True))

with tf.compat.v2.summary.record_if(False):
    tf_agent = create_agent(eval_py_env, None, True)
    initialize_and_restore_train_checkpointer(train_dir, tf_agent, global_step)

num_episodes = 100


# eval_py_env.pyenv.envs[0].scenario = scenario.scenarios_no_obstacles[3]
for i in range(num_episodes):
    # eval_py_env.pyenv.envs[0].scenario = scenario.scenarios_no_obstacles[i % 10]
    # eval_py_env.pyenv.envs[0].reverse_scenario = random.choice([True, False])
    reward = 0

    time_step = eval_py_env.reset()
    while not time_step.is_last():
        action_step = tf_agent.policy.action(time_step)

        means, stds, network_state = tf_agent.get_actor_network().call_raw(time_step.observation, time_step.step_type, (), None)

        actions = tf.constant([[0, 1]], dtype=tf.int32)
        action_steps = policy_step.PolicyStep(actions)

        time_step = eval_py_env.step(action_step.action)
        reward += time_step.reward
    print("reward for this episode = {}".format(reward))

# make_video(env_name, tf_agent, video_filename='eval')
