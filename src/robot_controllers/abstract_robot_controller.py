from abc import ABC, abstractmethod


class AbstractRobotController(ABC):

    @abstractmethod
    def enable_servos(self):
        pass

    @abstractmethod
    def disable_servos(self):
        pass

    @abstractmethod
    def move_to_pose(self, pose):
        pass

    @abstractmethod
    def move_to_pose_and_give_new_angles(self, pose):
        pass

    @abstractmethod
    def move_servos(self, angles):
        pass

    @abstractmethod
    def get_current_angles(self):
        pass

    @abstractmethod
    def pose_to_angles(self, pose):
        pass

    @abstractmethod
    def set_gripper(self, new_gripper_state):
        pass

    @abstractmethod
    def reset_to_pose(self, pose):
        pass

    @abstractmethod
    def set_servo_1_and_2_low_current(self):
        pass

    @abstractmethod
    def set_servo_1_and_2_full_current(self):
        pass
