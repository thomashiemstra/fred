import inspect
import os
from time import sleep

import tensorflow as tf

from src.reinforcementlearning.environment.pose_recorder import PoseRecorder
from src.reinforcementlearning.environment.robot_env_with_obstacles import RobotEnvWithObstacles
from src.reinforcementlearning.environment.scenario import sensible_scenarios, test
from src.reinforcementlearning.softActorCritic.sac_utils import create_agent, \
    initialize_and_restore_train_checkpointer
from tf_agents.environments import tf_py_environment

from src.simulation.simulation_utils import start_simulated_robot
from src.utils.movement import SplineMovement
from src.utils.movement_utils import b_spline_plot
import numpy as np


def pose_to_pose_distance(p1, p2):
    return np.sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2))


def get_usable_poses(poses, target_pose):
    result = []
    current_pose = poses[0]
    result.append(current_pose)

    for pose in poses[:-2]:
        d = pose_to_pose_distance(current_pose, pose)
        if d > 5:
            result.append(pose)
            current_pose = pose

    result.append(target_pose)
    return result


def run_agent(eval_py_env, tf_agent, initial_state=None):
    if initial_state is None:
        time_step = eval_py_env.reset()
    else:
        time_step = initial_state
    while not time_step.is_last():
        action_step = tf_agent.policy.action(time_step)
        time_step = eval_py_env.step(action_step.action)


def main():
    checkpoint_dir = 'rs_01_grid_new_network'
    robot_env_no_obstacles = False

    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    root_dir = os.path.expanduser(current_dir + '/checkpoints/' + checkpoint_dir)

    train_dir = os.path.join(root_dir, 'train')
    tf.compat.v1.enable_v2_behavior()
    global_step = tf.compat.v1.train.create_global_step()

    pose_recorder = PoseRecorder()
    robot_controller = start_simulated_robot(True)

    env = RobotEnvWithObstacles(use_gui=False, scenarios=sensible_scenarios, is_eval=True,
                                draw_debug_lines=True, pose_recorder=pose_recorder)
    eval_py_env = tf_py_environment.TFPyEnvironment(env)

    with tf.compat.v2.summary.record_if(False):
        tf_agent = create_agent(eval_py_env, None, robot_env_no_obstacles)
        initialize_and_restore_train_checkpointer(train_dir, tf_agent, global_step)

    scenario = None
    physics_client = robot_controller.physics_client
    for _ in range(len(sensible_scenarios)):
        run_agent(eval_py_env, tf_agent)

        recoded_poses = pose_recorder.get_recorded_poses()

        if scenario is not None:
            scenario.destroy_scenario(physics_client)

        scenario = env.get_current_scenario()
        usable_poses = get_usable_poses(recoded_poses, scenario.target_pose)

        smoothing_factor = 1000
        # b_spline_plot(usable_poses, s=smoothing_factor)

        spline_move = SplineMovement(usable_poses, 2, s=smoothing_factor)
        scenario.build_scenario(physics_client)
        robot_controller.reset_to_pose(usable_poses[0])

        spline_move.move(robot_controller)


if __name__ == '__main__':
    tf.config.experimental_run_functions_eagerly(True)
    with tf.device('/CPU:0'):
        main()
