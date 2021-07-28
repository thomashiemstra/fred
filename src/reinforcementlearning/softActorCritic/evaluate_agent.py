import inspect
import os

import tensorflow as tf
from kinematics.kinematics_utils import Pose
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import policy_step
from time import sleep

from src.reinforcementlearning.environment.robot_env import RobotEnv
from src.reinforcementlearning.environment.robot_env_with_obstacles import RobotEnvWithObstacles
from src.reinforcementlearning.softActorCritic.sac_utils import create_agent, \
    initialize_and_restore_train_checkpointer
from src.reinforcementlearning.environment.scenario import easy_scenarios, medium_scenarios, hard_scenarios, \
    super_easy_scenarios, sensible_scenarios, Scenario
from utils.obstacle import BoxObstacle


def main():
    checkpoint_dir = 'rs_01_direct_control'
    robot_env_no_obstacles = False

    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    root_dir = os.path.expanduser(current_dir + '/checkpoints/' + checkpoint_dir)

    train_dir = os.path.join(root_dir, 'train')
    tf.compat.v1.enable_v2_behavior()
    global_step = tf.compat.v1.train.create_global_step()

    # for debugging
    # tf.config.experimental_run_functions_eagerly(True)

    use_gui = True

    start_pose = Pose(-30, 25, 10)
    end_pose = Pose(30, 25, 10)
    scenario = [  Scenario([BoxObstacle([30, 10, 20], [0, 20, 0]),
                  BoxObstacle([10, 10, 40], [0, 30, 0])],
                 start_pose, end_pose)]

    env = RobotEnvWithObstacles(use_gui=use_gui, scenarios=sensible_scenarios, is_eval=True)
    env.set_step_size(2, 0.2, 10000)

    eval_py_env = tf_py_environment.TFPyEnvironment(env)

    with tf.compat.v2.summary.record_if(False):
        tf_agent = create_agent(eval_py_env, None, robot_env_no_obstacles)
        initialize_and_restore_train_checkpointer(train_dir, tf_agent, global_step)

    num_episodes = 100


    # eval_py_env.pyenv.envs[0].scenario = scenario.scenarios_no_obstacles[3]
    for i in range(1900):
        # eval_py_env.pyenv.envs[0].scenario = scenario.scenarios_no_obstacles[i % 10]
        # eval_py_env.pyenv.envs[0].reverse_scenario = random.choice([True, False])
        reward = 0

        # eval_py_env.pyenv.envs[0].set_scenario(medium_scenarios[i])
        time_step = eval_py_env.reset()
        while not time_step.is_last():
            action_step = tf_agent.policy.action(time_step)

            means, stds, network_state = tf_agent.get_actor_network().call_raw(time_step.observation, time_step.step_type, (), None)

            actions = tf.constant([[0, 1]], dtype=tf.int32)
            action_steps = policy_step.PolicyStep(actions)

            time_step = eval_py_env.step(action_step.action)
            reward += time_step.reward
            sleep(0.05)
        print("reward for episode {} is= {}".format(i, reward))

    print("whoop")

    # make_video(env_name, tf_agent, video_filename='eval')


if __name__ == '__main__':
    with tf.device('/CPU:0'):
        main()