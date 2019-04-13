import unittest

from src.kinematics.kinematics_utils import Pose
from src.utils.movement import SplineMovement, PoseToPoseMovement
from src.xbox_control.xbox_robot_controller import determine_time, create_move


class UtilsTests(unittest.TestCase):

    def test_determine_time(self):
        pose1 = Pose(0, 0, 0)
        pose2 = Pose(1, 0, 0)
        pose3 = Pose(1, 1, 0)
        poses = [pose1, pose2, pose3]

        dr = determine_time(poses, 1)
        self.assertIsNotNone(dr)
        self.assertEqual(2.0, dr)

    def test_create_move_spline(self):
        pose1 = Pose(0, 0, 0)
        pose2 = Pose(1, 0, 0)
        pose3 = Pose(1, 1, 0)
        poses = [pose1, pose2, pose3]

        move = create_move(None, poses, 1, None, None)
        self.assertIsInstance(move, SplineMovement)

    def test_create_move_pose_to_pose(self):
        pose1 = Pose(0, 0, 0)
        pose2 = Pose(0, 0, 0)
        poses = [pose1, pose2]

        move = create_move(None, poses, 1, None, None)
        self.assertIsInstance(move, PoseToPoseMovement)
