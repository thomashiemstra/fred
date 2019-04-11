import unittest

from src.kinematics.kinematics_utils import Pose
from src.xbox_control.xbox360controller.xbox_pose_updater import XboxPoseUpdater


class ControllerStateMock:

    def __init__(self, l_thumb_x=0, l_thumb_y=0, r_thumb_x=0, r_thumb_y=0, lr_trigger=0):
        self.lr_trigger = lr_trigger
        self.r_thumb_y = r_thumb_y
        self.r_thumb_x = r_thumb_x
        self.l_thumb_y = l_thumb_y
        self.l_thumb_x = l_thumb_x

    def get_left_thumb(self):
        return self.l_thumb_x, self.l_thumb_y

    def get_right_thumb(self):
        return self.r_thumb_x, self.r_thumb_y

    def get_lr_trigger(self):
        return self.lr_trigger

    def reset_buttons(self):
        pass


class XboxPoseUpdaterTests(unittest.TestCase):

    def test_no_change(self):
        state_mock = ControllerStateMock()
        pose_updater = XboxPoseUpdater(state_mock, 10, 1, None)
        old_pose = Pose(0, 0, 0)

        new_pose = pose_updater.get_updated_pose_from_controller(old_pose, False, None)

        self.assertIsNotNone(new_pose)
        self.assertEqual(new_pose, old_pose)

    def test_move_positive_x(self):
        state_mock = ControllerStateMock(l_thumb_x=1)
        pose_updater = XboxPoseUpdater(state_mock, 10, 1, None)
        old_pose = Pose(0, 0, 0)

        new_pose = pose_updater.get_updated_pose_from_controller(old_pose, False, None)

        self.assertIsNotNone(new_pose)
        self.assertTrue(new_pose.x > old_pose.x)

    def test_move_negative_x(self):
        state_mock = ControllerStateMock(l_thumb_x=-1)
        pose_updater = XboxPoseUpdater(state_mock, 10, 1, None)
        old_pose = Pose(0, 0, 0)

        new_pose = pose_updater.get_updated_pose_from_controller(old_pose, False, None)

        self.assertIsNotNone(new_pose)
        self.assertTrue(new_pose.x < old_pose.x)

    def test_move_positive_y(self):
        state_mock = ControllerStateMock(l_thumb_y=1)
        pose_updater = XboxPoseUpdater(state_mock, 10, 1, None)
        old_pose = Pose(0, 0, 0)

        new_pose = pose_updater.get_updated_pose_from_controller(old_pose, False, None)

        self.assertIsNotNone(new_pose)
        self.assertTrue(new_pose.y > old_pose.y)

    def test_move_negative_y(self):
        state_mock = ControllerStateMock(l_thumb_y=-1)
        pose_updater = XboxPoseUpdater(state_mock, 10, 1, None)
        old_pose = Pose(0, 0, 0)

        new_pose = pose_updater.get_updated_pose_from_controller(old_pose, False, None)

        self.assertIsNotNone(new_pose)
        self.assertTrue(new_pose.y < old_pose.y)

    def test_move_positive_z(self):
        state_mock = ControllerStateMock(lr_trigger=-1)
        pose_updater = XboxPoseUpdater(state_mock, 10, 1, None)
        old_pose = Pose(0, 0, 0)

        new_pose = pose_updater.get_updated_pose_from_controller(old_pose, False, None)

        self.assertIsNotNone(new_pose)
        self.assertTrue(new_pose.z > old_pose.z)

    def test_move_negative_z(self):
        state_mock = ControllerStateMock(lr_trigger=1)
        pose_updater = XboxPoseUpdater(state_mock, 10, 1, None)
        old_pose = Pose(0, 0, 0)

        new_pose = pose_updater.get_updated_pose_from_controller(old_pose, False, None)

        self.assertIsNotNone(new_pose)
        self.assertTrue(new_pose.z < old_pose.z)

    def test_move_positive_alpha(self):
        state_mock = ControllerStateMock(r_thumb_x=-1)
        pose_updater = XboxPoseUpdater(state_mock, 10, 1, None)
        old_pose = Pose(0, 0, 0)

        new_pose = pose_updater.get_updated_pose_from_controller(old_pose, False, None)

        self.assertIsNotNone(new_pose)
        self.assertTrue(new_pose.alpha > old_pose.alpha)

    def test_move_negative_alpha(self):
        state_mock = ControllerStateMock(r_thumb_x=1)
        pose_updater = XboxPoseUpdater(state_mock, 10, 1, None)
        old_pose = Pose(0, 0, 0)

        new_pose = pose_updater.get_updated_pose_from_controller(old_pose, False, None)

        self.assertIsNotNone(new_pose)
        self.assertTrue(new_pose.alpha < old_pose.alpha)

    def test_move_positive_gamma(self):
        state_mock = ControllerStateMock(r_thumb_y=-1)
        pose_updater = XboxPoseUpdater(state_mock, 10, 1, None)
        old_pose = Pose(0, 0, 0)

        new_pose = pose_updater.get_updated_pose_from_controller(old_pose, False, None)

        self.assertIsNotNone(new_pose)
        self.assertTrue(new_pose.gamma > old_pose.gamma)

    def test_move_negative_gamma(self):
        state_mock = ControllerStateMock(r_thumb_y=1)
        pose_updater = XboxPoseUpdater(state_mock, 10, 1, None)
        old_pose = Pose(0, 0, 0)

        new_pose = pose_updater.get_updated_pose_from_controller(old_pose, False, None)

        self.assertIsNotNone(new_pose)
        self.assertTrue(new_pose.gamma < old_pose.gamma)

    def test_dont_move_gamma_during_find_center(self):
        state_mock = ControllerStateMock(r_thumb_y=1)
        pose_updater = XboxPoseUpdater(state_mock, 10, 1, None)
        old_pose = Pose(0, 0, 0)

        new_pose = pose_updater.get_updated_pose_from_controller(old_pose, True, None)

        self.assertIsNotNone(new_pose)
        self.assertTrue(new_pose.gamma == old_pose.gamma)

    def test_dont_move_z_during_find_center(self):
        state_mock = ControllerStateMock(lr_trigger=1)
        pose_updater = XboxPoseUpdater(state_mock, 10, 1, None)
        old_pose = Pose(0, 0, 0)

        new_pose = pose_updater.get_updated_pose_from_controller(old_pose, True, None)

        self.assertIsNotNone(new_pose)
        self.assertTrue(new_pose.z == old_pose.z)
