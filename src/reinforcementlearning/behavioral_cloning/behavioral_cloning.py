from tf_agents.agents.behavioral_cloning import behavioral_cloning_agent
from tf_agents.environments import tf_py_environment, parallel_py_environment
from tf_agents.policies import actor_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory, policy_step
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
def fill_replay_buffer_with_gradient_descent(tf_env, total_collect_steps, replay_buffer):
    current_time_step = tf_env.reset()

    traj_array = []

    with Pool(tf_env.batch_size) as pool:
        while replay_buffer.num_frames().numpy() < total_collect_steps:
            action_step = gradient_descent_action(current_time_step.observation, pool)

            next_time_step = tf_env.step(action_step.action)

            traj = trajectory.from_transition(current_time_step, action_step, next_time_step)
            replay_buffer.add_batch(traj)

            current_time_step = next_time_step

    return traj_array


class ActorBCAgent(behavioral_cloning_agent.BehavioralCloningAgent):
  """BehavioralCloningAgent for Actor policies/networks."""

  def _get_policies(self, time_step_spec, action_spec, cloning_network):
    policy = actor_policy.ActorPolicy(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        actor_network=cloning_network,
        clip=True)

    return policy, policy


if __name__ == '__main__':
    total_collect_steps = 1000
    tf.config.experimental_run_functions_eagerly(True)
    # tf_env = tf_py_environment.TFPyEnvironment(
    #     parallel_py_environment.ParallelPyEnvironment(
    #         [lambda: RobotEnv(no_obstacles=True)] * 16))

    tf_env = tf_py_environment.TFPyEnvironment(RobotEnv(no_obstacles=True, use_gui=True))

    tf_agent = create_agent(tf_env, None)

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=1000000)

    fill_replay_buffer_with_gradient_descent(tf_env, total_collect_steps, replay_buffer)

    print("done")
    print(replay_buffer.num_frames().numpy())