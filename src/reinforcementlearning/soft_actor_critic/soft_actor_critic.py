from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import os
import sys
import time

import tensorflow as tf
from absl import logging
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import parallel_py_environment, suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import py_metrics
from tf_agents.metrics import tf_metrics
from tf_agents.metrics import tf_py_metric
from tf_agents.policies import greedy_policy, fixed_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
import numpy as np


from absl import app
from absl import flags

from src.reinforcementlearning.behavioral_cloning.behavioral_cloning import fill_replay_buffer_with_gradient_descent
from src.reinforcementlearning.environment.robot_env import RobotEnv
from src.reinforcementlearning.soft_actor_critic.custom_policy import CustomPolicy
from src.reinforcementlearning.soft_actor_critic.sac_utils import create_agent, compute_metrics, save_checkpoints, \
    make_and_initialze_checkpointers, print_time_progression, show_progress

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')

FLAGS = flags.FLAGS


def train_eval(checkpoint_dir,
               total_train_steps=2000000,
               # Params for collect,
               initial_collect_steps=10000,
               collect_steps_per_iteration=150,
               replay_buffer_capacity=1000000,
               # Params for target update,
               # Params for train,
               train_steps_per_iteration=150,
               batch_size=256,
               use_tf_functions=True,
               # Params for eval,
               num_eval_episodes=1,
               eval_interval=2500,
               # Params for summaries and logging,
               train_checkpoint_interval=5000,
               policy_checkpoint_interval=5000,
               rb_checkpoint_interval=50000,
               log_interval=5000,
               summary_interval=1000,
               summaries_flush_secs=10,
               robot_env_no_obstacles=True,
               num_parallel_environments=16):
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

    # tf.config.experimental_run_functions_eagerly(True)

    root_dir = os.path.expanduser(current_dir + '/checkpoints/' + checkpoint_dir)
    train_dir = os.path.join(root_dir, 'train')
    eval_dir = os.path.join(root_dir, 'eval')

    train_summary_writer = tf.compat.v2.summary.create_file_writer(
        train_dir, flush_millis=summaries_flush_secs * 1000)
    train_summary_writer.set_as_default()

    eval_summary_writer = tf.compat.v2.summary.create_file_writer(
        eval_dir, flush_millis=summaries_flush_secs * 1000)
    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
    ]

    global_step = tf.compat.v1.train.get_or_create_global_step()
    with tf.compat.v2.summary.record_if(
            lambda: tf.math.equal(global_step % summary_interval, 0)):
        # tf_env = tf_py_environment.TFPyEnvironment(RobotEnv(no_obstacles=robot_env_no_obstacles))
        # eval_tf_env = tf_py_environment.TFPyEnvironment(RobotEnv(no_obstacles=robot_env_no_obstacles))

        tf_env = tf_py_environment.TFPyEnvironment(
            parallel_py_environment.ParallelPyEnvironment(
                [lambda: RobotEnv(no_obstacles=robot_env_no_obstacles)] * num_parallel_environments))

        eval_tf_env = tf_py_environment.TFPyEnvironment(RobotEnv(no_obstacles=robot_env_no_obstacles))

        eval_py_env = RobotEnv(no_obstacles=robot_env_no_obstacles, use_gui=False)

        tf_agent = create_agent(tf_env, global_step)

        environment_steps_metric = tf_metrics.EnvironmentSteps()
        step_metrics = [
            tf_metrics.NumberOfEpisodes(),
            environment_steps_metric,
        ]

        # Make the replay buffer.
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=tf_agent.collect_data_spec,
            batch_size=tf_env.batch_size,
            max_length=replay_buffer_capacity)
        replay_observer = [replay_buffer.add_batch]

        train_metrics = step_metrics + [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_py_metric.TFPyMetric(py_metrics.AverageReturnMetric(batch_size=tf_env.batch_size)),
            tf_py_metric.TFPyMetric(py_metrics.AverageEpisodeLengthMetric(batch_size=tf_env.batch_size)),
        ]

        eval_policy = greedy_policy.GreedyPolicy(tf_agent.policy)
        initial_collect_policy = random_tf_policy.RandomTFPolicy(
            tf_env.time_step_spec(), tf_env.action_spec())
        collect_policy = tf_agent.collect_policy

        train_checkpointer, policy_checkpointer, rb_checkpointer = make_and_initialze_checkpointers(train_dir,
                                                                                                    tf_agent,
                                                                                                    global_step,
                                                                                                    eval_policy,
                                                                                                    replay_buffer,
                                                                                                    train_metrics)

        # show_progress(tf_agent, eval_py_env)
        # experience, _ = next(iterator)
        initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
            tf_env,
            initial_collect_policy,
            observers=replay_observer,
            num_steps=initial_collect_steps)

        collect_driver = dynamic_step_driver.DynamicStepDriver(
            tf_env,
            collect_policy,
            observers=replay_observer + train_metrics,
            num_steps=collect_steps_per_iteration)

        if use_tf_functions:
            initial_collect_driver.run = common.function(initial_collect_driver.run)
            collect_driver.run = common.function(collect_driver.run)
            tf_agent.train = common.function(tf_agent.train)

        if global_step.numpy() == 0:
            # Collect initial replay data.
            logging.info(
                'Initializing replay buffer by collecting experience for %d steps with '
                'a random policy.', initial_collect_steps)
            # initial_collect_driver.run()
            with tf.device('/CPU:0'):
                fill_replay_buffer_with_gradient_descent(tf_env, initial_collect_steps, replay_buffer)
        else:
            logging.info("skipping initial collect because we already have data")

        # For debugging
        # tf.config.experimental_run_functions_eagerly(False)

        compute_metrics(eval_metrics, eval_tf_env, eval_policy, num_eval_episodes, global_step, eval_summary_writer)

        time_step = None
        policy_state = collect_policy.get_initial_state(tf_env.batch_size)

        timed_at_step = global_step.numpy()
        time_acc = 0

        # Prepare replay buffer as dataset with invalid transitions filtered.
        def _filter_invalid_transition(trajectories, unused_arg1):
            return ~trajectories.is_boundary()[0]
        # Dataset generates trajectories with shape [Bx2x...]
        dataset = replay_buffer.as_dataset(
            sample_batch_size=batch_size,
            num_steps=2).unbatch().filter(
            _filter_invalid_transition).batch(batch_size).prefetch(5)
        iterator = iter(dataset)

        def train_step():
            experience, _ = next(iterator)
            return tf_agent.train(experience=experience)

        if use_tf_functions:
            train_step = common.function(train_step)

        time_before_training = time.time()

        steps_taken_in_prev_round = global_step.numpy()  # From a previous training round
        while global_step.numpy() < total_train_steps:
            global_steps_taken = global_step.numpy()

            start_time = time.time()

            for _ in range(train_steps_per_iteration):
                 train_loss = train_step()

            time_step, policy_state = collect_driver.run(
                time_step=time_step,
                policy_state=policy_state,
            )

            time_acc += time.time() - start_time

            if global_steps_taken % log_interval == 0:
                logging.info('step = %d, loss = %f', global_steps_taken,
                             train_loss.loss)
                steps_per_sec = (global_step.numpy() - timed_at_step) / time_acc
                logging.info('%.3f steps/sec', steps_per_sec)
                tf.compat.v2.summary.scalar(
                    name='global_steps_per_sec', data=steps_per_sec, step=global_step)
                timed_at_step = global_step.numpy()
                time_acc = 0

                print("current step: {}".format(global_steps_taken))
                print_time_progression(time_before_training, global_steps_taken - steps_taken_in_prev_round,
                                       total_train_steps - steps_taken_in_prev_round)

            for train_metric in train_metrics:
                train_metric.tf_summaries(
                    train_step=global_step, step_metrics=train_metrics[:2])

            if global_steps_taken % eval_interval == 0:
                compute_metrics(eval_metrics, eval_tf_env, eval_policy, num_eval_episodes, global_step,
                                eval_summary_writer)

            save_checkpoints(global_steps_taken, train_checkpoint_interval, policy_checkpoint_interval,
                             rb_checkpoint_interval, train_checkpointer, policy_checkpointer, rb_checkpointer)

        time_after_trianing = time.time()

        elapsed_time = time_after_trianing - time_before_training
        print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


def main(_):
    tf.compat.v1.enable_v2_behavior()
    logging.set_verbosity(logging.INFO)

    print("GPU Available: ", tf.test.is_gpu_available())

    print("eager is on: {}".format(tf.executing_eagerly()))

    # if not tf.test.is_gpu_available():
    #     print("no point in training without a gpu, go watch the grass grow instead")
    #     sys.exit()

    train_eval(FLAGS.root_dir)


# PYTHONUNBUFFERED=1;LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64
if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    app.run(main)
