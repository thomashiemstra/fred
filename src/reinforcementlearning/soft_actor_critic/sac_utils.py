from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import imageio
from absl import logging

import tensorflow as tf

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_mujoco, suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import py_metrics
from tf_agents.metrics import tf_metrics
from tf_agents.metrics import tf_py_metric
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common


def normal_projection_net(action_spec,
                          init_action_stddev=0.35,
                          init_means_output_factor=0.1):
    del init_action_stddev
    return normal_projection_network.NormalProjectionNetwork(
        action_spec,
        mean_transform=None,
        state_dependent_std=True,
        init_means_output_factor=init_means_output_factor,
        std_transform=sac_agent.std_clip_transform,
        scale_distribution=True)


def create_agent(env,
                 global_step,
                 actor_fc_layers=(256, 256),
                 critic_obs_fc_layers=None,
                 critic_action_fc_layers=None,
                 critic_joint_fc_layers=(256, 256),
                 target_update_tau=0.005,
                 target_update_period=1,
                 actor_learning_rate=3e-4,
                 critic_learning_rate=3e-4,
                 alpha_learning_rate=3e-4,
                 td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
                 gamma=0.99,
                 reward_scale_factor=1.0,
                 gradient_clipping=None,
                 debug_summaries=False,
                 summarize_grads_and_vars=False):
    time_step_spec = env.time_step_spec()
    observation_spec = time_step_spec.observation
    action_spec = env.action_spec()

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        observation_spec,
        action_spec,
        fc_layer_params=actor_fc_layers,
        continuous_projection_net=normal_projection_net)
    critic_net = critic_network.CriticNetwork(
        (observation_spec, action_spec),
        observation_fc_layer_params=critic_obs_fc_layers,
        action_fc_layer_params=critic_action_fc_layers,
        joint_fc_layer_params=critic_joint_fc_layers)
    agent = sac_agent.SacAgent(
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
        td_errors_loss_fn=td_errors_loss_fn,
        gamma=gamma,
        target_entropy=-12,
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=gradient_clipping,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=global_step)
    agent.initialize()
    return agent


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
        env.render()
        steps += 1
