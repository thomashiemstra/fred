import threading
from copy import copy
from functools import lru_cache

import src.global_constants
from src.global_objects import get_robot
from src.kinematics.kinematics_utils import Pose
from src.utils.decorators import synchronized_with_lock
from src.utils.movement_utils import from_current_angles_to_pose, pose_to_pose
import numpy as np
from time import sleep


@lru_cache(maxsize=1)
def get_board_to_board_controller(image_hanlder):
    robot = get_robot(src.global_constants.dynamixel_robot_arm_port)
    controller = BoardToBoardRobotController(robot, image_hanlder)
    return controller


class BoardToBoardRobotController:

    def __init__(self, robot, board_to_board_image_handler):
        self.robot = robot
        self.board_to_board_image_handler = board_to_board_image_handler
        self.lock = threading.RLock()
        self.thread = None
        self.start_pose = Pose(21, 21.0, 4)
        self.current_pose = None
        self.done = False
        self.dt = 0.1

    @synchronized_with_lock("lock")
    def stop(self):
        self.set_done(True)
        self.thread.join()
        self.thread = None

    @synchronized_with_lock("lock")
    def start(self):
        self.current_pose = copy(self.start_pose)
        if self.thread is None:
            self.thread = threading.Thread(target=self.__start_internal, args=())
            self.thread.start()
            return True
        else:
            return False

    def stop_robot(self):
        from_current_angles_to_pose(self.start_pose, self.robot, 4)
        self.robot.disable_servos()

    @synchronized_with_lock("lock")
    def set_done(self, val):
        self.done = val

    @synchronized_with_lock("lock")
    def is_done(self):
        return self.done

    def get_new_pose(self):
        relative_matrix, translation_vector = self.board_to_board_image_handler.get_revlative_vecs()
        x = translation_vector[0]
        y = translation_vector[1] + 10
        z = translation_vector[2] + 5

        target_matrix = self.get_target_matrix(relative_matrix)
        return Pose(x, y, z, euler_matrix=target_matrix)

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

        new_pose = self.get_new_pose()
        pose_to_pose(self.current_pose, new_pose, self.robot, time=2)
        self.current_pose = new_pose

        while True:
            if self.is_done():
                break

            self.current_pose = self.get_new_pose()
            recommended_time, time_taken = self.robot.move_to_pose(self.current_pose)
            time_to_sleep = np.maximum(np.maximum(recommended_time, self.dt) - time_taken, 0)
            sleep(time_to_sleep)

        self.stop_robot()
