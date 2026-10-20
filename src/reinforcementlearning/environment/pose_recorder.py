from copy import copy

from src.kinematics.kinematics_utils import Pose


class PoseRecorder:

    def __init__(self) -> None:
        self.poses: list[Pose] = []

    def add_pose(self, pose: Pose) -> None:
        copied_pose = copy(pose)
        self.poses.append(copied_pose)

    def clear_poses(self) -> None:
        self.poses = []

    def get_recorded_poses(self) -> list[Pose]:
        return [copy(pose) for pose in self.poses]


class DummyPoseRecorder(PoseRecorder):
    def add_pose(self, pose: Pose) -> None:
        pass

    def clear_poses(self) -> None:
        pass

    def get_recorded_poses(self) -> list[Pose]:
        return []
