from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import inspect
import multiprocessing as mp
import os
import time

import gin
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
from tf_agents.drivers import dynamic_step_driver
from tf_agents.metrics import py_metrics
from tf_agents.metrics import tf_metrics
from tf_agents.metrics import tf_py_metric
from tf_agents.policies import greedy_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.utils import common
import numpy as np

from src.reinforcementlearning.environment.scenario import easy_scenarios, medium_scenarios, hard_scenarios
from src.reinforcementlearning.softActorCritic.IntervalManager import IntervalManager
from src.reinforcementlearning.softActorCritic.sac_utils import create_agent, compute_metrics, save_checkpoints, \
    make_and_initialze_checkpointers, print_time_progression, initialize_and_restore_train_checkpointer, create_envs

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('behavioral_cloning_checkpoint_dir', None,
                    'Directory in the root dir where the results for the behavioral cloning are saved')

flags.DEFINE_string('difficulty', None,
                    'Difficulty to start at')


flags.DEFINE_float('reward_scaling', None, 'reward scaling')
flags.DEFINE_float('entropy_target', None, 'entropy target')

flags.DEFINE_multi_string('gin_file', None, 'Path to the trainer config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding to pass through.')

FLAGS = flags.FLAGS

NUM_PARALLEL = 1


def train_eval(checkpoint_dir,
               checkpoint_dir_behavioral_cloning=None,
               total_train_steps=1000000,
               # Params for collect,
               initial_collect_steps=10000,
               initial_bc_collect_steps=100000,
               collect_steps_per_iteration=NUM_PARALLEL,
               replay_buffer_capacity=1000000,
               # Params for target update,
               # Params for train,
               train_steps_per_iteration=NUM_PARALLEL,
               batch_size=256,
               use_tf_functions=True,
               # Params for eval,
               num_eval_episodes=len(medium_scenarios),
               eval_interval=2000,
               # Params for summaries and logging,
               train_checkpoint_interval=5000,
               policy_checkpoint_interval=5000,
               rb_checkpoint_interval=5000,
               log_interval=5000,
               summary_interval=1000,
               summaries_flush_secs=10,
               robot_env_no_obstacles=False,
               num_parallel_environments=NUM_PARALLEL,
               reward_scaling=1.0,
               entropy=None,
               difficulty=None):
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

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
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
        tf_metrics.MinReturnMetric(buffer_size=num_eval_episodes)
    ]

    global_step = tf.compat.v1.train.get_or_create_global_step()
    with tf.compat.v2.summary.record_if(
            lambda: tf.math.equal(global_step % summary_interval, 0)):

        # tf_env, eval_tf_env = create_envs(robot_env_no_obstacles, num_parallel_environments, scenarios=easy_scenarios)

        logging.info("difficulty = {}".format(difficulty))
        if difficulty == 'easy':
            tf_env, eval_tf_env = create_envs(robot_env_no_obstacles, num_parallel_environments,
                                              scenarios=easy_scenarios)
        elif difficulty == 'med':
            tf_env, eval_tf_env = create_envs(robot_env_no_obstacles, num_parallel_environments,
                                              scenarios=medium_scenarios)
        elif difficulty == 'hard':
            tf_env, eval_tf_env = create_envs(robot_env_no_obstacles, num_parallel_environments,
                                              scenarios=hard_scenarios)
            total_train_steps += total_train_steps

        tf_agent = create_agent(tf_env, global_step, robot_env_no_obstacles,
                                reward_scale_factor=reward_scaling, entropy=entropy)

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
            tf_metrics.MinReturnMetric(buffer_size=num_eval_episodes),
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

        logging.info("replay buffer size: {}".format(replay_buffer.num_frames().numpy()))

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

        restore_from_behavioral_cloning = checkpoint_dir_behavioral_cloning is not None and global_step.numpy() == 0
        if restore_from_behavioral_cloning:
            logging.info("restoring agent from the behavioral cloning run")
            restore_agent_from_behavioral_cloning(current_dir, checkpoint_dir_behavioral_cloning, tf_agent, global_step)
            logging.info('Initializing replay buffer by collecting experience for %d steps with '
                         'the behavioral cloning policy.', initial_bc_collect_steps)
            for _ in range(int(initial_bc_collect_steps / collect_steps_per_iteration)):
                collect_driver.run()
        elif replay_buffer.num_frames().numpy() == 0:
            if global_step.numpy() == 0:
                # Collect initial replay data.
                logging.info(
                    'Initializing replay buffer by collecting experience for %d steps with '
                    'a random policy.', initial_collect_steps)
                initial_collect_driver.run()
            else:
                logging.info(
                    'Initializing replay buffer by collecting experience with initialized agent for %d steps with '
                    'the trained agent\'s policy.', initial_collect_steps)
                for i in range(int(initial_collect_steps / collect_steps_per_iteration)):
                    collect_driver.run()
        else:
            logging.info("skipping initial collect because we already have data")

        compute_metrics(eval_metrics, eval_tf_env, eval_policy, num_eval_episodes, global_step, eval_summary_writer)

        time_step = None
        policy_state = collect_policy.get_initial_state(tf_env.batch_size)

        timed_at_step = global_step.numpy()
        time_acc = 0

        dataset = replay_buffer.as_dataset(
            sample_batch_size=batch_size,
            num_steps=2).unbatch().batch(batch_size).prefetch(5)
        iterator = iter(dataset)

        def train_step():
            experience, _ = next(iterator)
            return tf_agent.train(experience=experience)

        if use_tf_functions:
            train_step = common.function(train_step)

        time_before_training = time.time()

        steps_taken_in_prev_round = global_step.numpy()  # From a previous training round

        log_interval_manager = IntervalManager(log_interval, steps_taken_in_prev_round)
        eval_interval_keeper = IntervalManager(eval_interval, steps_taken_in_prev_round)

        train_checkpoint_interval_manager = IntervalManager(train_checkpoint_interval, steps_taken_in_prev_round)
        policy_checkpoint_interval_manager = IntervalManager(policy_checkpoint_interval, steps_taken_in_prev_round)
        rb_checkpoint_interval_manager = IntervalManager(rb_checkpoint_interval, steps_taken_in_prev_round)

        logging.info("training")
        while global_step.numpy() <= total_train_steps:
            global_steps_taken = global_step.numpy()

            start_time = time.time()

            for _ in range(train_steps_per_iteration):
                train_loss = train_step()

            time_step, policy_state = collect_driver.run(
                time_step=time_step,
                policy_state=policy_state,
            )

            time_acc += time.time() - start_time

            if log_interval_manager.should_trigger(global_steps_taken):
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

            if eval_interval_keeper.should_trigger(global_steps_taken):
                results = compute_metrics(eval_metrics, eval_tf_env, eval_policy, num_eval_episodes, global_step,
                                eval_summary_writer)
                # results['MinReturn'].numpy()

            save_checkpoints(global_steps_taken,
                             train_checkpoint_interval_manager,
                             policy_checkpoint_interval_manager,
                             rb_checkpoint_interval_manager,
                             train_checkpointer, policy_checkpointer, rb_checkpointer, global_step)

        time_after_trianing = time.time()

        save_checkpoints(global_steps_taken,
                         train_checkpoint_interval_manager,
                         policy_checkpoint_interval_manager,
                         rb_checkpoint_interval_manager,
                         train_checkpointer, policy_checkpointer, rb_checkpointer, global_step)

        elapsed_time = time_after_trianing - time_before_training
        print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


def restore_agent_from_behavioral_cloning(current_dir, checkpoint_dir_behavioral_cloning, tf_agent, global_step):
    root_dir_behavioral_cloning = os.path.expanduser(current_dir + '/checkpoints/' + checkpoint_dir_behavioral_cloning)
    train_dir_behavioral_cloning = os.path.join(root_dir_behavioral_cloning, 'train')
    initialize_and_restore_train_checkpointer(train_dir_behavioral_cloning, tf_agent, global_step)


def main(_):
    tf.compat.v1.enable_v2_behavior()
    logging.set_verbosity(logging.DEBUG)
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)

    print("GPU Available: ", tf.test.is_gpu_available())

    print("eager is on: {}".format(tf.executing_eagerly()))

    print("Cores available: {}".format(mp.cpu_count()))

    # for debugging
    # tf.config.experimental_run_functions_eagerly(True)
    reward_scaling = FLAGS.reward_scaling
    difficulty = FLAGS.difficulty
    if reward_scaling is None:
        reward_scaling = 1.0

    train_eval(FLAGS.root_dir, FLAGS.behavioral_cloning_checkpoint_dir, reward_scaling=reward_scaling,
               entropy=FLAGS.entropy_target, difficulty=difficulty)


# PYTHONUNBUFFERED=1;LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64
if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    multiprocessing.handle_main(functools.partial(app.run, main))
