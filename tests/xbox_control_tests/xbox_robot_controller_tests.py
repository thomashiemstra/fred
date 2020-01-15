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

        time = determine_time(poses, 1)
        self.assertIsNotNone(time, "should have gotten a time")
        self.assertEqual(2.0, time, "Should have gotten a time of 2 seconds")

    def test_determine_time_orientation_adjustment(self):
        pose1 = Pose(0, 0, 0, alpha=0)
        pose2 = Pose(1, 0, 0, alpha=1)
        poses = [pose1, pose2]

        time = determine_time(poses, 1)
        self.assertIsNotNone(time, "should have gotten a time")
        self.assertTrue(time > 0, "should have gotten a positive time")

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
