from abc import ABC, abstractmethod


class AbstractRobotController(ABC):

    @abstractmethod
    def move_to_pose(self, pose):
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