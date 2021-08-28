import threading
from copy import copy
from functools import lru_cache
from time import sleep

import numpy as np

import src.global_constants
from src import global_constants
from src.global_objects import get_robot
from src.kinematics.kinematics_utils import Pose
from src.utils.decorators import synchronized_with_lock
from src.utils.movement_utils import from_current_angles_to_pose, pose_to_pose


@lru_cache(maxsize=1)
def get_board_to_board_controller(image_hanlder):
    robot = get_robot(src.global_constants.dynamixel_robot_arm_port)
    controller = BoardToBoardRobotController(robot, image_hanlder)
    return controller


def apply_workspace_limits(old_x, old_y, old_z, new_x, new_y, new_z):
    limits = global_constants.WorkSpaceLimits

    x, y, z = new_x, new_y, new_z

    radius = np.linalg.norm([new_x, new_y])
    if radius > limits.radius_max or radius < limits.radius_min:
        x, y = old_x, old_y

    if new_y < limits.y_min:
        y = old_y

    if new_z < limits.z_min:
        z = old_z

    return x, y, z


class BoardToBoardRobotController:

    def __init__(self, robot, board_to_board_image_handler):
        self.robot = robot
        self.board_to_board_image_handler = board_to_board_image_handler
        self.lock = threading.RLock()
        self.should_filter_lock = threading.RLock()
        self.should_filter = True
        self.thread = None
        self.startup_pose = Pose(21, 21.0, 4)
        self.neutral_pose = Pose(0, 30, 10)
        self.current_pose = None
        self.done = False
        self.dt = 1.0 / 100
        self.kg = 0.04

    @synchronized_with_lock("lock")
    def stop(self):
        self.set_done(True)
        self.thread.join()
        self.thread = None

    @synchronized_with_lock("lock")
    def start(self):
        self.current_pose = copy(self.startup_pose)
        if self.thread is None:
            self.thread = threading.Thread(target=self.__start_internal, args=())
            self.thread.start()
            return True
        else:
            return False

    def stop_robot(self):
        from_current_angles_to_pose(self.startup_pose, self.robot, 4)
        self.robot.disable_servos()

    @synchronized_with_lock("lock")
    def set_done(self, val):
        self.done = val

    @synchronized_with_lock("lock")
    def is_done(self):
        return self.done

    @synchronized_with_lock("should_filter_lock")
    def should_filter(self):
        return self.should_filter

    @synchronized_with_lock("should_filter_lock")
    def set_should_filter(self, val):
        self.should_filter = val

    def get_new_filtered_pose(self):
        relative_matrix, translation_vector = self.board_to_board_image_handler.get_revlative_vecs()
        if relative_matrix is None or translation_vector is None:
            return self.current_pose

        # Measured values
        x_m = translation_vector[0]
        y_m = translation_vector[1] + 10
        z_m = translation_vector[2] + 5
        m_orientation = self.get_target_matrix(relative_matrix)

        # current values
        old_orientation = self.current_pose.get_euler_matrix()
        old_x = self.current_pose.x
        old_y = self.current_pose.y
        old_z = self.current_pose.z

        if self.should_filter:
            new_x = old_x + self.kg * (x_m - old_x)
            new_y = old_y + self.kg * (y_m - old_y)
            new_z = old_z + self.kg * (z_m - old_z)
            orientation = old_orientation + self.kg * (m_orientation - old_orientation)

        else:
            new_x = x_m
            new_y = y_m
            new_z = z_m
            orientation = m_orientation

        x, y, z = apply_workspace_limits(old_x, old_y, old_z, new_x, new_y, new_z)

        new_pose = Pose(x, y, z, euler_matrix=orientation)

        return new_pose

    @staticmethod
    def get_target_matrix(m):
        t = np.zeros((3, 3))
        t[0][0] = m[0][2]; t[0][1] = m[0][0]; t[0][2] = m[0][1]
        t[1][0] = m[1][2]; t[1][1] = m[1][0]; t[1][2] = m[1][1]
        t[2][0] = m[2][2]; t[2][1] = m[2][0]; t[2][2] = m[2][1]
        return t

    def __start_internal(self):
        # The robot could be anywhere, first move it from it's current position to the target pose
        from_current_angles_to_pose(self.current_pose, self.robot, 1)

        self.current_pose = copy(self.neutral_pose)
        from_current_angles_to_pose(self.current_pose, self.robot, 2)

        new_pose = self.get_new_filtered_pose()
        pose_to_pose(self.current_pose, new_pose, self.robot, time=2)
        self.current_pose = new_pose

        while True:
            if self.is_done():
                break

            self.current_pose = self.get_new_filtered_pose()
            recommended_time, time_taken = self.robot.move_to_pose(self.current_pose)
            time_to_sleep = np.maximum(np.maximum(recommended_time, self.dt) - time_taken, 0)
            sleep(time_to_sleep)

        self.stop_robot()
