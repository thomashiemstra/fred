import threading

from src.camera_control.marker_obstacle_relations import get_obstacle_from_marker
from src.kinematics.kinematics import jacobian_transpose_on_f
from src.kinematics.kinematics_utils import Pose
from src.reinforcementlearning.environment.robot_env_with_obstacles import RobotEnvWithObstacles
from src.reinforcementlearning.environment.scenario import Scenario

import numpy as np
from time import sleep

from src.utils.decorators import synchronized_with_lock


class ObstacleAvoidance:

    def __init__(self, aruco_image_handler, robot):
        self.aruco_image_handler = aruco_image_handler
        self.robot = robot
        self.lock = threading.RLock()
        self.done = False
        self.start_pose = Pose(-25, 35, 10)
        self.target_pose = Pose(25, 35, 10)
        self.env = None
        self.control_point_1_position = 11.2
        self.stopped = False
        self.thread = None
        self._initial_state = None

    @synchronized_with_lock("lock")
    def is_stopped(self):
        return self.stopped

    @synchronized_with_lock("lock")
    def set_stopped(self, val):
        self.stopped = val

    def _find_obstacles(self):
        detected_markers = self.aruco_image_handler.get_detected_markers()

        obstacles = []
        for marker in detected_markers:
            obstacle = get_obstacle_from_marker(marker)
            obstacles.append(obstacle)

        return obstacles

    @synchronized_with_lock("lock")
    def create_and_set_scenario(self):
        if self.env is not None:
            self.env.close()

        self.set_stopped(False)

        obstacles = self._find_obstacles()
        self.env = self.get_env(obstacles)
        self._initial_state = self.env.reset()

    def get_env(self, obstacles):
        scenario = Scenario(obstacles, start_pose=self.start_pose, target_pose=self.target_pose)
        env = RobotEnvWithObstacles(scenarios=[scenario], robot_controller=self.robot)
        env._update_step_size = 0.001
        env.set_target_reached_distance(5)
        env.disable_max_steps_to_take_before_failure()
        return env

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
    def obstacle_avoidance_gradient_descent(self):
        if self.thread is not None:
            print("already have thread running, shut it down first!")
            return

        self.thread = threading.Thread(target=self.__start_gradient_descent)
        self.thread.start()

    def __start_gradient_descent(self):
        if self.env is None:
            print("first set a scenario based on the image in order to create an env")
            return

        state = self._initial_state

        while not self.is_stopped():
            raw_observation = state.observation
            observation = raw_observation[0]

            c1_attr = np.zeros(3)
            c2_attr = 3 * observation[0:3]
            c3_attr = observation[3:6]

            c1_rep = 4 * observation[6:9]
            c2_rep = 4 * observation[9:12]
            c3_rep = 4 * observation[12:15]

            attractive_forces = np.stack((c1_attr, c2_attr, c3_attr))
            repulsive_forces = np.stack((c1_rep, c2_rep, c3_rep))

            forces = attractive_forces + repulsive_forces

            current_angles = self.env.current_angles

            robot_controller = self.env.robot_controller

            joint_forces = jacobian_transpose_on_f(forces, current_angles,
                                                   robot_controller.robot_config, self.control_point_1_position)

            absolute_force = np.linalg.norm(joint_forces)

            action = (joint_forces / absolute_force)
            # sleep(0.01)

            state = self.env.step(action[1:6])

            if state.step_type == 2:
                print("goal reached!")
                break
