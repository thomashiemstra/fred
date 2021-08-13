from copy import copy


class PoseRecorder:

    def __init__(self):
        self.poses = []

    def add_pose(self, pose):
        copied_pose = copy(pose)
        self.poses.append(copied_pose)

    def clear_poses(self):
        self.poses = []

    def get_recorded_poses(self):
        return [copy(pose) for pose in self.poses]


class DummyPoseRecorder(PoseRecorder):
    def add_pose(self, pose):
        pass

    def clear_poses(self):
        pass

    def get_recorded_poses(self):
        return []
