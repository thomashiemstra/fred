import json
from abc import ABC, abstractmethod
import logging as log
import numpy as np

from src.utils.movement_utils import b_spline_curve, pose_to_pose


class Movement(ABC):

    def __init__(self, servo_controller, poses, time, center=None, workspace_limits=None) -> None:
        if len(poses) < 2:
            raise ValueError("at least 2 poses should be given in order to make a movement")
        self.servo_controller = servo_controller
        self.poses = poses
        self.time = time
        self.center = center
        self.workspace_limits = workspace_limits

    def move(self):
        if not self.is_robot_at_start_pose(self.poses[0]):
            log.warning("robot is not at the start pose, not executing move")
            return self.poses[0]
        return self._move_internal(self.poses)

    def move_reversed(self):
        if not self.is_robot_at_start_pose(self.poses[-1]):
            log.warning("robot is not at the start pose, not executing move")
            return self.poses[-1]
        return self._move_internal(self.poses[::-1])

    @abstractmethod
    def _move_internal(self, poses):
        pass

    def to_json(self):
        json_poses = [pose.to_json() for pose in self.poses]
        dump_dict = {'poses': json_poses}
        return json.dumps(dump_dict)

    def is_robot_at_start_pose(self, start_pose):
        current_angles = self.servo_controller.get_current_angles()
        start_pose_angles = self.servo_controller.pose_to_angles(start_pose)
        return np.allclose(current_angles, start_pose_angles, atol=0.1)


class SplineMovement(Movement):

    def _move_internal(self, poses):
        return b_spline_curve(poses, self.time, self.servo_controller,
                              workspace_limits=self.workspace_limits, center=self.center)


class PoseToPoseMovement(Movement):

    def _init__(self, servo_controller, poses, time, center, workspace_limits=None) -> None:
        super().__init__(servo_controller, poses, time, center, workspace_limits)
        if len(self.poses) > 2:
            log.warning("more than 2 poses provided for a pose to pose movement, "
                        "using the first and the last pose given")

    def _move_internal(self, poses):
        return pose_to_pose(poses[0], poses[-1], self.servo_controller, time=self.time)
