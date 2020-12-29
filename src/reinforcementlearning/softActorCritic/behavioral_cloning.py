import functools
import inspect
import os
from multiprocessing import Pool

import numpy as np
import tensorflow as tf
from absl import app
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.system import system_multiprocessing
from tf_agents.trajectories import trajectory, policy_step
from tf_agents.utils import common

from src import global_constants
from src.kinematics.kinematics import jacobian_transpose_on_f
from src.reinforcementlearning.environment import robot_env_utils
from src.reinforcementlearning.softActorCritic.IntervalManager import IntervalManager
from src.reinforcementlearning.softActorCritic.sac_utils import create_agent, create_envs
from src.utils.decorators import timer


def get_forces(raw_observation):
    if isinstance(raw_observation, tuple):
        observation = raw_observation[0]
    else:
        observation = raw_observation

    c1_attr = np.zeros(3)
    c2_attr = observation[0:3]
    c3_attr = observation[3:6]

    c1_rep = 3 * observation[6:9]
    c2_rep = 3 * observation[9:12]
    c3_rep = 3 * observation[12:15]

    attractive_forces = np.stack((c1_attr, c2_attr, c3_attr))
    repulsive_forces = np.stack((c1_rep, c2_rep, c3_rep))

    return attractive_forces + repulsive_forces


def handle_observation(raw_observation):
    forces = get_forces(raw_observation)

    current_angles = robot_env_utils.get_de_normalized_current_angles(raw_observation[15:20])

    joint_forces = jacobian_transpose_on_f(forces, np.append([0], current_angles),
                                           global_constants.simulated_robot_config, 11.2)

    absolute_force = np.linalg.norm(joint_forces)

    action_ = (joint_forces / absolute_force)

    return action_[1:6]


def gradient_descent_action(raw_observations, pool, robot_env_no_obstacles):
    if robot_env_no_obstacles:
        observations = raw_observations
    else:
        observations = raw_observations[0]

    total_action = np.array(pool.map(handle_observation, observations))
    tf_action = tf.constant(total_action, shape=(observations.shape[0], 5), dtype=tf.float32)
    return policy_step.PolicyStep(tf_action, (), ())


@timer
def fill_replay_buffer_with_gradient_descent(tf_env, total_collect_steps, replay_buffer, robot_env_no_obstacles,
                                             rb_checkpointer):
    current_time_step = tf_env.reset()

    traj_array = []

    replay_buffer_interval_manager = IntervalManager(1000)

    with Pool(tf_env.batch_size) as pool:
        while replay_buffer.num_frames().numpy() < total_collect_steps:
            action_step = gradient_descent_action(current_time_step.observation, pool, robot_env_no_obstacles)

            next_time_step = tf_env.step(action_step.action)

            traj = trajectory.from_transition(current_time_step, action_step, next_time_step)
            replay_buffer.add_batch(traj)

            current_time_step = next_time_step

            if replay_buffer_interval_manager.should_trigger(replay_buffer.num_frames().numpy()):
                rb_checkpointer.save(replay_buffer.num_frames().numpy())
                print("saved")

    return traj_array


def fill_and_get_replay_buffer(train_dir, collect_data_spec, tf_env, robot_env_no_obstacles, total_collect_steps=10000):
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=total_collect_steps)

    replay_buffer_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
        max_to_keep=1,
        replay_buffer=replay_buffer)

    replay_buffer_checkpointer.initialize_or_restore()

    print("replay buffer size: {}".format(replay_buffer.num_frames().numpy()))

    if replay_buffer.num_frames().numpy() < total_collect_steps:
        fill_replay_buffer_with_gradient_descent(tf_env, total_collect_steps, replay_buffer, robot_env_no_obstacles,
                                                 replay_buffer_checkpointer)
        replay_buffer_checkpointer.save(replay_buffer.num_frames().numpy())
    else:
        print("got a full buffer from the checkpoint")

    print("done filling buffer")
    print(replay_buffer.num_frames().numpy())

    return replay_buffer


def train_agent(tf_agent, replay_buffer, train_steps, batch_size, train_checkpointer, global_step):
    dataset = replay_buffer.as_dataset(
        sample_batch_size=batch_size,
        num_steps=2).unbatch().batch(batch_size).prefetch(5)
    iterator = iter(dataset)

    def train_step():
        experience, _ = next(iterator)
        return tf_agent.train_behavioral_cloning(experience=experience)

    train_step = common.function(train_step)

    print("training")
    for step in range(train_steps):
        train_loss, actor_loss = train_step()
        if step % 50 == 0:
            print("train loss {}, actor_loss {}".format(train_loss, actor_loss))
            train_checkpointer.save(global_step)

    print("done training")


def main(_):
    # tf.config.experimental_run_functions_eagerly(True)
    checkpoint_dir = 'behavioral_cloning_obstacles/'
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    root_dir = os.path.expanduser(current_dir + '/checkpoints/' + checkpoint_dir)
    train_dir = os.path.join(root_dir, 'train/')
    total_collect_steps = 100000
    batch_size = 256
    train_steps = 1500
    robot_env_no_obstacles = False

    global_step = tf.compat.v1.train.get_or_create_global_step()

    tf_env, eval_tf_env = create_envs(robot_env_no_obstacles, 20)
    # tf_env = tf_py_environment.TFPyEnvironment(RobotEnvWithObstacles(use_gui=True))

    tf_agent = create_agent(tf_env, None, robot_env_no_obstacles)

    with tf.device('/CPU:0'):
        replay_buffer = fill_and_get_replay_buffer(train_dir, tf_agent.collect_data_spec, tf_env,
                                                   robot_env_no_obstacles, total_collect_steps=total_collect_steps)

    train_checkpointer = common.Checkpointer(
        ckpt_dir=train_dir,
        agent=tf_agent,
        max_to_keep=5,
        global_step=global_step)

    train_checkpointer.initialize_or_restore()

    train_agent(tf_agent, replay_buffer, train_steps, batch_size, train_checkpointer, global_step)


if __name__ == '__main__':
    system_multiprocessing.handle_main(functools.partial(app.run, main))
