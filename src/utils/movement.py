import json
from abc import ABC, abstractmethod
import logging as log
import numpy as np

from src.utils.movement_exception import MovementException
from src.utils.movement_utils import b_spline_curve, pose_to_pose, from_current_angles_to_pose


def convert_center_to_float(center):
    if center is None:
        return None
    return [float(x) for x in center]


class Movement(ABC):

    def __init__(self, poses, time, center=None, workspace_limits=None) -> None:
        if len(poses) < 2:
            raise ValueError("at least 2 poses should be given in order to make a movement")
        self.poses = poses
        self.time = float(time)
        self.center = convert_center_to_float(center)
        self.workspace_limits = workspace_limits

    def go_to_start_of_move(self, servo_controller, time=None):
        if time is None:
            time = 4
        print(self.poses[0])
        from_current_angles_to_pose(self.poses[0], servo_controller, time)

    def move(self, servo_controller):
        if not self.is_robot_at_start_pose(self.poses[0], servo_controller):
            raise MovementException("robot is not at the start pose, not executing move")
        return self._move_internal(self.poses, servo_controller)

    def move_reversed(self, servo_controller):
        if not self.is_robot_at_start_pose(self.poses[-1], servo_controller):
            raise MovementException("robot is not at the start pose, not executing move")
        return self._move_internal(self.poses[::-1], servo_controller)

    @abstractmethod
    def _move_internal(self, poses, servo_controller):
        pass

    @abstractmethod
    def check_workspace_limits(self, servo_controller, workspace_limits):
        pass

    def to_json(self):
        json_poses = [pose.to_json() for pose in self.poses]
        dump_dict = {'poses': json_poses}
        return json.dumps(dump_dict)

    @staticmethod
    def is_robot_at_start_pose(start_pose, servo_controller):
        current_angles = servo_controller.get_current_angles()
        start_pose_angles = servo_controller.pose_to_angles(start_pose)
        return np.allclose(current_angles, start_pose_angles, atol=0.1)


class SplineMovement(Movement):

    def _move_internal(self, poses, servo_controller):
        return b_spline_curve(poses, self.time, servo_controller,
                              workspace_limits=self.workspace_limits, center=self.center)

    def check_workspace_limits(self, servo_controller, workspace_limits):
        try:
            b_spline_curve(self.poses, self.time, servo_controller, workspace_limits=workspace_limits, center=self.center, calculate_only=True)
        except MovementException:
            return False
        return True


class PoseToPoseMovement(Movement):

    def _init__(self, poses, time, center, workspace_limits=None) -> None:
        super().__init__(poses, time, center, workspace_limits)
        if len(self.poses) > 2:
            log.warning("more than 2 poses provided for a pose to pose movement, "
                        "using the first and the last pose given")

    def _move_internal(self, poses, servo_controller):
        return pose_to_pose(poses[0], poses[-1], servo_controller, time=self.time)

    def check_workspace_limits(self, servo_controller, workspace_limits):
        return True
