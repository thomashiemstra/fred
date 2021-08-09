import inspect
import os
from time import sleep

import tensorflow as tf
from src.reinforcementlearning.environment.robot_env_with_obstacles import RobotEnvWithObstacles
from src.reinforcementlearning.environment.scenario import sensible_scenarios, test
from src.reinforcementlearning.softActorCritic.sac_utils import create_agent, \
    initialize_and_restore_train_checkpointer
from tf_agents.environments import tf_py_environment


def main():
    checkpoint_dir = 'rs_01_grid_new_network'
    robot_env_no_obstacles = False

    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    root_dir = os.path.expanduser(current_dir + '/checkpoints/' + checkpoint_dir)

    train_dir = os.path.join(root_dir, 'train')
    tf.compat.v1.enable_v2_behavior()
    global_step = tf.compat.v1.train.create_global_step()

    # for debugging
    # tf.config.experimental_run_functions_eagerly(True)

    use_gui = True

    env = RobotEnvWithObstacles(use_gui=use_gui, scenarios=test, is_eval=True)

    # env.set_step_size(2, 0.2, 10000)

    eval_py_env = tf_py_environment.TFPyEnvironment(env)

    with tf.compat.v2.summary.record_if(False):
        tf_agent = create_agent(eval_py_env, None, robot_env_no_obstacles)
        initialize_and_restore_train_checkpointer(train_dir, tf_agent, global_step)

    for i in range(100):
        reward = 0

        time_step = eval_py_env.reset()
        while not time_step.is_last():
            action_step = tf_agent.policy.action(time_step)
            time_step = eval_py_env.step(action_step.action)
            reward += time_step.reward
            sleep(0.05)
        print("reward for episode {} is= {}".format(i, reward))

    print("whoop")

    # make_video(env_name, tf_agent, video_filename='eval')


if __name__ == '__main__':
    tf.config.experimental_run_functions_eagerly(True)
    with tf.device('/CPU:0'):
        main()
