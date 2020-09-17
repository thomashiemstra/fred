from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import imageio
import tensorflow as tf
from tf_agents.environments import suite_gym, parallel_py_environment, tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.networks import value_network
from tf_agents.utils import common

from src.reinforcementlearning.environment.robot_env import RobotEnv
from src.reinforcementlearning.environment.robot_env_with_obstacles import RobotEnvWithObstacles
from src.reinforcementlearning.soft_actor_critic.custom_objects.actor_distribution_network_trainable import \
    ActorDistributionNetworkTrainable
from src.reinforcementlearning.soft_actor_critic.custom_objects.custom_sac_agent import CustomSacAgent
from src.reinforcementlearning.soft_actor_critic.custom_objects.normal_projection_network_trainable import \
    NormalProjectionNetworkTrainable
from src.utils.os_utils import is_linux


def normal_projection_net(action_spec):
    return NormalProjectionNetworkTrainable(
        action_spec,
        mean_transform=None,
        scale_distribution=True)


def create_agent(env,
                 global_step,
                 actor_fc_layers=(32, 16),
                 target_update_tau=0.005,
                 target_update_period=1,
                 actor_learning_rate=3e-4,
                 critic_learning_rate=3e-4,
                 alpha_learning_rate=3e-4,
                 td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
                 gamma=0.99,
                 reward_scale_factor=0.1,
                 gradient_clipping=None,
                 debug_summaries=False,
                 summarize_grads_and_vars=False):
    time_step_spec = env.time_step_spec()
    observation_spec = time_step_spec.observation
    action_spec = env.action_spec()

    actor_net = ActorDistributionNetworkTrainable(
        observation_spec,
        action_spec,
        fc_layer_params=actor_fc_layers,
        continuous_projection_net=normal_projection_net)

    critic_input_spec, critic_preprocessing_layer = get_cirit_input_spec_and_preprocessing_layer(env, observation_spec,
                                                                                                 action_spec)
    critic_net = value_network.ValueNetwork(
        critic_input_spec,
        preprocessing_layers=critic_preprocessing_layer,
        preprocessing_combiner=tf.keras.layers.Concatenate(axis=-1),
        fc_layer_params=(16,),
        kernel_initializer='glorot_uniform'
    )

    agent = CustomSacAgent(
        time_step_spec,
        action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=actor_learning_rate),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=critic_learning_rate),
        alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=alpha_learning_rate),
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        # td_errors_loss_fn=td_errors_loss_fn,
        gamma=gamma,
        # target_entropy=-6,
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=gradient_clipping,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=global_step)
    agent.initialize()
    return agent


def get_actor_preprocessing_layer(env):
    python_env = env.pyenv.envs[0]
    if isinstance(python_env, RobotEnv):
        preprocessing_layer = (
            tf.keras.layers.Dense(16)
        )
        return preprocessing_layer

    if isinstance(python_env, RobotEnvWithObstacles):
        preprocessing_layer = (
            tf.keras.layers.Dense(16),
            tf.keras.layers.Dense(128)
        )
        return preprocessing_layer


def get_cirit_input_spec_and_preprocessing_layer(env, observation_spec, action_spec):
    python_env = env.pyenv.envs[0]
    if isinstance(python_env, RobotEnv):
        input_spec = (observation_spec,) + (action_spec,)
        postprocessing_layer = (
            # 20 observations               5 actions
            tf.keras.layers.Dense(32),      tf.keras.layers.Dense(16)
        )
        return input_spec, postprocessing_layer

    if isinstance(python_env, RobotEnvWithObstacles):
        input_spec = observation_spec + (action_spec,)
        postprocessing_layer = (
            # 20 normal observations   64 hilbert curve observations    5 actions
            tf.keras.layers.Dense(32), tf.keras.layers.Dense(128),      tf.keras.layers.Dense(16)
        )
        return input_spec, postprocessing_layer
    raise ValueError("got an unknown env type", python_env)


def create_envs(robot_env_no_obstacles, num_parallel_environments):
    if not is_linux():  # Windows does not handle multiprocessing well
        if robot_env_no_obstacles:
            tf_env = tf_py_environment.TFPyEnvironment(RobotEnv())
            eval_tf_env = tf_py_environment.TFPyEnvironment(RobotEnv())
        else:
            tf_env = tf_py_environment.TFPyEnvironment(RobotEnvWithObstacles())
            eval_tf_env = tf_py_environment.TFPyEnvironment(RobotEnvWithObstacles())

        return tf_env, eval_tf_env

    if robot_env_no_obstacles:
        tf_env = tf_py_environment.TFPyEnvironment(
            parallel_py_environment.ParallelPyEnvironment(
                [lambda: RobotEnv()] * num_parallel_environments))

        eval_tf_env = tf_py_environment.TFPyEnvironment(RobotEnv())
    else:
        tf_env = tf_py_environment.TFPyEnvironment(
            parallel_py_environment.ParallelPyEnvironment(
                [lambda: RobotEnvWithObstacles()] * num_parallel_environments))

        eval_tf_env = tf_py_environment.TFPyEnvironment(RobotEnvWithObstacles())

    return tf_env, eval_tf_env


def compute_metrics(eval_metrics,
                    eval_tf_env,
                    eval_policy,
                    num_eval_episodes,
                    global_step,
                    eval_summary_writer):
    results = metric_utils.eager_compute(
        eval_metrics,
        eval_tf_env,
        eval_policy,
        num_episodes=num_eval_episodes,
        train_step=global_step,
        summary_writer=eval_summary_writer,
        summary_prefix='Metrics',
    )
    metric_utils.log_metrics(eval_metrics)
    return results


def save_checkpoints(global_step_val,
                     train_checkpoint_interval,
                     policy_checkpoint_interval,
                     rb_checkpoint_interval,
                     train_checkpointer,
                     policy_checkpointer,
                     rb_checkpointer
                     ):
    if global_step_val % train_checkpoint_interval == 0:
        train_checkpointer.save(global_step=global_step_val)

    if global_step_val % policy_checkpoint_interval == 0:
        policy_checkpointer.save(global_step=global_step_val)

    if global_step_val % rb_checkpoint_interval == 0:
        rb_checkpointer.save(global_step=global_step_val)


def make_and_initialze_checkpointers(train_dir,
                                     tf_agent,
                                     global_step,
                                     eval_policy,
                                     replay_buffer,
                                     train_metrics):
    train_checkpointer = common.Checkpointer(
        ckpt_dir=train_dir,
        agent=tf_agent,
        global_step=global_step,
        metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'policy'),
        policy=eval_policy,
        global_step=global_step)
    rb_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
        max_to_keep=1,
        replay_buffer=replay_buffer)

    train_checkpointer.initialize_or_restore()
    rb_checkpointer.initialize_or_restore()

    return train_checkpointer, policy_checkpointer, rb_checkpointer


def initialize_and_restore_train_checkpointer(train_dir, tf_agent, global_step):
    train_checkpointer = common.Checkpointer(
        ckpt_dir=train_dir,
        agent=tf_agent,
        global_step=global_step,
        metrics=metric_utils.MetricsGroup([], 'nothing'))
    train_checkpointer.initialize_or_restore()


def make_video(env_name, tf_agent, video_filename='test'):
    video_filename += '.mp4'
    eval_py_env = suite_gym.load(env_name)
    num_episodes = 3
    with imageio.get_writer(video_filename, fps=60) as video:
        for _ in range(num_episodes):
            time_step = eval_py_env.reset()
            video.append_data(eval_py_env.render())
            while not time_step.is_last():
                action_step = tf_agent.policy.action(time_step)
                time_step = eval_py_env.step(action_step.action)
                video.append_data(eval_py_env.render())


def show_progress(agent, env):
    time_step = env.reset()
    steps = 0
    while not time_step.is_last() and steps < 400:
        action_step = agent.policy.action(time_step)
        time_step = env.step(action_step.action)
        steps += 1


def print_time_progression(time_before_training, global_step_taken, total_train_steps):
    if global_step_taken == 0:
        return
    time_elapsed_so_far = time.time() - time_before_training
    steps_taken_so_far = global_step_taken
    avrg_time_per_step = time_elapsed_so_far / steps_taken_so_far

    steps_left_to_take = total_train_steps - global_step_taken
    remaining_time = steps_left_to_take * avrg_time_per_step
    print("Total time taken so far: {}, time left: {}"
          .format(time.strftime("%H:%M:%S", time.gmtime(time_elapsed_so_far)),
                  time.strftime("%H:%M:%S", time.gmtime(remaining_time))))
