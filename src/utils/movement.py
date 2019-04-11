import json
from abc import ABC, abstractmethod
import logging as log

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

    @abstractmethod
    def move(self):
        pass

    @abstractmethod
    def move_reversed(self):
        pass

    def to_json(self):
        json_poses = [pose.to_json() for pose in self.poses]
        print(json_poses)
        dump_dict = {'poses': json_poses}
        return json.dumps(dump_dict)


class SplineMovement(Movement):

    def move(self):
        b_spline_curve(self.poses, self.time, self.servo_controller,
                       workspace_limits=self.workspace_limits, center=self.center)

    def move_reversed(self):
        b_spline_curve(reversed(self.poses), self.time, self.servo_controller,
                       workspace_limits=self.workspace_limits, center=self.center)


class PoseToPoseMovement(Movement):

    def __init__(self, servo_controller, poses, time, center, workspace_limits=None) -> None:
        super().__init__(servo_controller, poses, time, center, workspace_limits)
        if len(self.poses) > 2:
            log.warning("more than 2 poses provided for a pose to pose movement, "
                        "using the first and the last pose given")

    def move(self):
        pose_to_pose(self.poses[0], self.poses[-1], self.servo_controller, time=self.time)

    def move_reversed(self):
        pose_to_pose(self.poses[-1], self.poses[0], self.servo_controller, time=self.time)
