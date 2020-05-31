import inspect
import os

from tf_agents.agents.behavioral_cloning import behavioral_cloning_agent
from tf_agents.environments import tf_py_environment, parallel_py_environment
from tf_agents.eval import metric_utils
from tf_agents.policies import actor_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory, policy_step
from tf_agents.utils import common
import tensorflow as tf
import numpy as np

from src import global_constants
from src.kinematics.kinematics import jacobian_transpose_on_f
from src.reinforcementlearning.environment import robot_env_utils
from src.reinforcementlearning.environment.robot_env import RobotEnv
from src.reinforcementlearning.soft_actor_critic.sac_utils import create_agent
from src.utils.decorators import timer
from multiprocessing import Pool


def get_forces(observation):
    c1_attr = np.zeros(3)
    c2_attr = 2 * observation[0:3]
    c3_attr = observation[3:6]

    c1_rep = observation[6:9]
    c2_rep = observation[9:12]
    c3_rep = observation[12:15]

    attractive_forces = np.stack((c1_attr, c2_attr, c3_attr))
    repulsive_forces = np.stack((c1_rep, c2_rep, c3_rep))

    return attractive_forces + repulsive_forces


def handle_observation(observation):
    forces = get_forces(observation)

    current_angles = robot_env_utils.get_de_normalized_current_angles(observation[15:20])

    joint_forces = jacobian_transpose_on_f(forces, np.append([0], current_angles),
                                           global_constants.simulated_robot_config, 11.2)

    absolute_force = np.linalg.norm(joint_forces)

    action_ = (joint_forces / absolute_force)

    return action_[1:6]


def gradient_descent_action(observations, pool):
    total_action = np.array(pool.map(handle_observation, observations))
    tf_action = tf.constant(total_action, shape=(observations.shape[0], 5), dtype=tf.float32)
    return policy_step.PolicyStep(tf_action, (), ())


@timer
def fill_replay_buffer_with_gradient_descent(tf_env, total_collect_steps, replay_buffer, rb_checkpointer=None, global_step=None):
    current_time_step = tf_env.reset()

    traj_array = []

    with Pool(tf_env.batch_size) as pool:
        while replay_buffer.num_frames().numpy() < total_collect_steps:
            action_step = gradient_descent_action(current_time_step.observation, pool)

            next_time_step = tf_env.step(action_step.action)

            traj = trajectory.from_transition(current_time_step, action_step, next_time_step)
            replay_buffer.add_batch(traj)

            current_time_step = next_time_step

            if rb_checkpointer is not None and replay_buffer.num_frames().numpy() % 1000 == 0:
                rb_checkpointer.save(global_step)
                print("saved")

    return traj_array


def fill_and_get_replay_buffer(global_step, train_dir, collect_data_spec, tf_env, total_collect_steps=10000):
    # tf_env = tf_py_environment.TFPyEnvironment(
    #     parallel_py_environment.ParallelPyEnvironment(
    #         [lambda: RobotEnv(no_obstacles=True)] * 16))

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=1000000)

    replay_buffer_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
        max_to_keep=1,
        replay_buffer=replay_buffer)

    replay_buffer_checkpointer.initialize_or_restore()

    if replay_buffer.num_frames().numpy() < total_collect_steps:
        fill_replay_buffer_with_gradient_descent(tf_env, total_collect_steps, replay_buffer, replay_buffer_checkpointer,
                                                 global_step)
        replay_buffer_checkpointer.save(global_step)

    print("done filling buffer")
    print(replay_buffer.num_frames().numpy())

    return replay_buffer

def train_agent(replay_buffer, train_steps):
    # Prepare replay buffer as dataset with invalid transitions filtered.
    def _filter_invalid_transition(trajectories, unused_arg1):
        return ~trajectories.is_boundary()[0]
    dataset = replay_buffer.as_dataset(
        sample_batch_size=batch_size,
        num_steps=2).unbatch().filter(
        _filter_invalid_transition).batch(batch_size).prefetch(5)
    iterator = iter(dataset)

    def train_step():
        experience, _ = next(iterator)
        return tf_agent.train_behavioral_cloning(experience=experience)

    train_step = common.function(train_step)

    print("training")
    for step in range(train_steps):
        train_loss, actor_loss, critic_loss = train_step()
        if step % 50 == 0:
            print("train loss {}, actor_loss {}, critic_loss {}".format(train_loss, actor_loss, critic_loss))
            train_checkpointer.save(global_step)

    print("done training")


if __name__ == '__main__':
    # tf.config.experimental_run_functions_eagerly(True)
    checkpoint_dir = 'bc/'
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    root_dir = os.path.expanduser(current_dir + '/checkpoints/' + checkpoint_dir)
    train_dir = os.path.join(root_dir, 'train/')
    total_collect_steps = 100000
    batch_size = 256
    train_steps = 300

    global_step = tf.compat.v1.train.get_or_create_global_step()

    tf_env = tf_py_environment.TFPyEnvironment(RobotEnv(no_obstacles=True, use_gui=False))
    tf_agent = create_agent(tf_env, None)

    replay_buffer = fill_and_get_replay_buffer(global_step, train_dir, tf_agent.collect_data_spec, tf_env,
                                               total_collect_steps=total_collect_steps)

    train_checkpointer = common.Checkpointer(
        ckpt_dir=train_dir,
        agent=tf_agent,
        global_step=global_step)

    train_checkpointer.initialize_or_restore()

    train_agent(replay_buffer, train_steps)

