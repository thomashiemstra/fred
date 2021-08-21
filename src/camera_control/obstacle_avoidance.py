import threading

from tf_agents.environments import tf_py_environment

from src.camera_control.marker_obstacle_relations import get_obstacle_from_marker
from src.global_constants import sac_network_weights
from src.kinematics.kinematics import jacobian_transpose_on_f
from src.kinematics.kinematics_utils import Pose
from src.reinforcementlearning.environment.pose_recorder import PoseRecorder
from src.reinforcementlearning.environment.robot_env_with_obstacles import RobotEnvWithObstacles
from src.reinforcementlearning.environment.scenario import Scenario
import tensorflow as tf

import numpy as np
from time import sleep

from src.reinforcementlearning.softActorCritic.sac_utils import create_agent, initialize_and_restore_train_checkpointer
from src.reinforcementlearning.softActorCritic.smooth_paths import run_agent, get_usable_poses
from src.utils.decorators import synchronized_with_lock
from src.utils.movement import SplineMovement
from src.utils.movement_exception import MovementException


class ObstacleAvoidance:

    def __init__(self, aruco_image_handler, robot):
        self.aruco_image_handler = aruco_image_handler
        self.robot = robot
        self.lock = threading.RLock()
        self.stop_lock = threading.RLock()
        self.done = False
        self.start_pose = Pose(-25, 25, 10)
        self.target_pose = Pose(25, 25, 10)
        self.env = None
        self.tf_env = None
        self.eval_tf_env = None
        self.control_point_1_position = 11.2
        self.stopped = False
        self.thread = None
        self._initial_state = None
        self.tf_agent = None
        self.pose_recorder = PoseRecorder()

    def start_sac(self):
        tf.compat.v1.enable_v2_behavior()
        global_step = tf.compat.v1.train.create_global_step()
        with tf.compat.v2.summary.record_if(False):
            train_dir = sac_network_weights
            tf_agent = create_agent(self.tf_env, None, False)
            initialize_and_restore_train_checkpointer(train_dir, tf_agent, global_step)
        return tf_agent

    @synchronized_with_lock("stop_lock")
    def is_stopped(self):
        return self.stopped

    @synchronized_with_lock("stop_lock")
    def set_stopped(self, val):
        self.stopped = val

    def _find_obstacles(self):
        detected_markers = self.aruco_image_handler.get_detected_markers()

        obstacles = []
        for marker in detected_markers:
            obstacle = get_obstacle_from_marker(marker)
            if obstacle is not None:
                obstacles.append(obstacle)

        return obstacles

    @synchronized_with_lock("lock")
    def create_and_set_scenario(self):
        if self.tf_env is not None:
            self.tf_env.close()

        self.set_stopped(False)

        obstacles = self._find_obstacles()
        self.env, self.tf_env, self.eval_tf_env = self.get_env(obstacles)
        self._initial_state = self.tf_env.reset()

    def get_env(self, obstacles):
        scenario = Scenario(obstacles, start_pose=self.start_pose, target_pose=self.target_pose)
        env = RobotEnvWithObstacles(scenarios=[scenario],
                                    robot_controller=self.robot, is_eval=True,
                                    use_gui=True)
        eval_env = RobotEnvWithObstacles(scenarios=[scenario],
                                         use_gui=False,
                                         pose_recorder=self.pose_recorder)
        env.set_target_reached_distance(10)
        env.disable_max_steps_to_take_before_failure()
        env.set_xyz_update_step_size(1)

        tf_env = tf_py_environment.TFPyEnvironment(env)
        return env, tf_env, eval_env

    def enable_servos(self):
        self.robot.enable_servos()

    def disable_servos(self):
        self.robot.disable_servos()

    def stop(self):
        if self.thread is None:
            print("already stopped")
            return
        self.set_stopped(True)
        self.thread.join()
        self.thread = None
        self.set_stopped(False)

    @synchronized_with_lock("lock")
    def obstacle_avoidance_sac(self):
        if self.thread is not None:
            print("already have thread running, shut it down first!")
            return

        self.thread = threading.Thread(target=self.__start_sac)
        self.thread.start()
        self.thread.join()
        self.stop()

    def __start_sac(self):
        if self.tf_env is None:
            print("first set a scenario based on the image in order to create an env")
            return

        if self.tf_agent is None:
            self.tf_agent = self.start_sac()

        run_agent(self.eval_tf_env, self.tf_agent, self._initial_state)
        recoded_poses = self.pose_recorder.get_recorded_poses()

        usable_poses = get_usable_poses(recoded_poses, self.target_pose)

        smoothing_factor = 1000
        # b_spline_plot(usable_poses, s=smoothing_factor)

        spline_move = SplineMovement(usable_poses, 5, s=smoothing_factor)

        # self.robot.reset_to_pose(self.start_pose)
        try:
            spline_move.move(self.robot)
        except MovementException:
            print("unable to perform move")

        ans = input("Move back the way you came? y/n")
        if ans == 'y':
            pass
        else:
            return

        try:
            spline_move.move_reversed(self.robot)
        except MovementException:
            print("unable to perform reverse move")

    @synchronized_with_lock("lock")
    def obstacle_avoidance_gradient_descent(self):
        if self.thread is not None:
            print("already have thread running, shut it down first!")
            return

        self.thread = threading.Thread(target=self.__start_gradient_descent)
        self.thread.start()
        self.thread.join()
        self.stop()

    def __start_gradient_descent(self):
        if self.tf_env is None:
            print("first set a scenario based on the image in order to create an env")
            return

        state = self._initial_state
        self.env.set_angle_control(True)

        while not self.is_stopped():
            raw_observation = state.observation
            observation = raw_observation['observation'].numpy()[0]

            c1_attr = np.zeros(3)
            c2_attr = np.zeros(3)
            c3_attr = observation[0:3]

            c1_rep = 4 * observation[3:6]
            c2_rep = 4 * observation[6:9]
            c3_rep = 4 * observation[9:12]

            attractive_forces = np.stack((c1_attr, c2_attr, c3_attr))
            repulsive_forces = np.stack((c1_rep, c2_rep, c3_rep))

            forces = attractive_forces + repulsive_forces

            current_angles = self.tf_env.pyenv.envs[0].current_angles

            robot_controller = self.tf_env.pyenv.envs[0].robot_controller

            joint_forces = jacobian_transpose_on_f(forces, current_angles,
                                                   robot_controller.robot_config, self.control_point_1_position)

            absolute_force = np.linalg.norm(joint_forces)

            action = (joint_forces / absolute_force)
            tensor_action = tf.constant([action[1:6]])

            state = self.tf_env.step(tensor_action)

            if state.step_type == 2:
                if state.reward == 0:
                    print("stuck")
                else:
                    print("goal reached!")
                break
