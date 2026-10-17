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

    def get_gripper_state(self):
        """returns the actual state of the gripper (0 to 100), or None if it is unknown"""
        return None

    def get_angle_limits(self):
        """returns a mapping of joint number to the (min angle, max angle) the joint can reach, or None if unknown"""
        return None

    def get_overheating_servos(self):
        """returns a mapping of servo id to temperature of the servos that are too warm, or None if the temperatures
        could not be read"""
        return {}

    def get_hardware_errors(self):
        """returns a mapping of servo id to the names of its hardware errors for the servos that shut down, or None
        if the servos could not be read"""
        return {}
