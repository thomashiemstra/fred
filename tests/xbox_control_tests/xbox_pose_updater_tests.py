import unittest

import numpy as np
from numpy import pi

from src.global_constants import WorkSpaceLimits
from src.kinematics.kinematics_utils import Pose
from src.utils.os_utils import is_linux
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

    def test_move_positive_alpha(self):
        state_mock = ControllerStateMock(r_thumb_x=-1)
        pose_updater = XboxPoseUpdater(state_mock, 10, 1, None)
        old_pose = Pose(0, 0, 0)

        new_pose = pose_updater.get_updated_pose_from_controller(old_pose, False, None)

        self.assertIsNotNone(new_pose)
        if is_linux():
            self.assertTrue(new_pose.alpha < old_pose.alpha)
        else:
            self.assertTrue(new_pose.alpha > old_pose.alpha)

    def test_move_negative_alpha(self):
        state_mock = ControllerStateMock(r_thumb_x=1)
        pose_updater = XboxPoseUpdater(state_mock, 10, 1, None)
        old_pose = Pose(0, 0, 0)

        new_pose = pose_updater.get_updated_pose_from_controller(old_pose, False, None)

        self.assertIsNotNone(new_pose)
        if is_linux():
            self.assertTrue(new_pose.alpha > old_pose.alpha)
        else:
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


class XboxPoseUpdaterWorkspaceLimitsTests(unittest.TestCase):

    def test_no_change(self):
        pose_updater = XboxPoseUpdater(ControllerStateMock(), 10, 1, WorkSpaceLimits)
        old_pose = Pose(30, 22, 4)

        new_pose = pose_updater.get_updated_pose_from_controller(old_pose, False, None)

        self.assertEqual((old_pose.x, old_pose.y, old_pose.z), (new_pose.x, new_pose.y, new_pose.z))

    def test_y_stays_above_the_minimum(self):
        for direction in (-1, 1):
            pose_updater = XboxPoseUpdater(ControllerStateMock(l_thumb_y=direction), 10, 1, WorkSpaceLimits)
            pose = Pose(30, WorkSpaceLimits.y_min, 4)
            for _ in range(10):
                pose = pose_updater.get_updated_pose_from_controller(pose, False, None)
                self.assertGreaterEqual(pose.y, WorkSpaceLimits.y_min)

    def test_pose_does_not_move_beyond_the_maximum_radius(self):
        pose_updater = XboxPoseUpdater(ControllerStateMock(l_thumb_y=100), 10, 0.1, SmallWorkSpaceLimits)
        pose = Pose(0, 19.5, 1)
        for _ in range(100):
            pose = pose_updater.get_updated_pose_from_controller(pose, False, None)
            self.assertLessEqual(np.linalg.norm([pose.x, pose.y, pose.z]), SmallWorkSpaceLimits.radius_max)
        # it did move up to the limit
        self.assertGreater(pose.y, 19.8)

    def test_pose_does_not_move_within_the_minimum_radius(self):
        pose_updater = XboxPoseUpdater(ControllerStateMock(l_thumb_y=-100), 10, 0.1, SmallWorkSpaceLimits)
        pose = Pose(0, 10.5, 1)
        for _ in range(100):
            pose = pose_updater.get_updated_pose_from_controller(pose, False, None)
            self.assertGreaterEqual(np.linalg.norm([pose.x, pose.y, pose.z]), SmallWorkSpaceLimits.radius_min)
        self.assertLess(pose.y, 10.2)

    def test_z_does_not_move_below_the_minimum_while_x_still_moves(self):
        pose_updater = XboxPoseUpdater(ControllerStateMock(l_thumb_x=100, lr_trigger=-100), 10, 0.1,
                                       SmallWorkSpaceLimits)
        pose = Pose(0, 15, 0.5)
        for _ in range(100):
            pose = pose_updater.get_updated_pose_from_controller(pose, False, None)
            self.assertGreater(pose.z, SmallWorkSpaceLimits.z_min)
        self.assertLess(pose.z, 0.1)
        self.assertGreater(pose.x, 5)


class SmallWorkSpaceLimits:
    radius_min = 10
    radius_max = 20
    y_min = 1
    z_min = 0


class XboxPoseUpdaterVelocityTests(unittest.TestCase):

    def test_speeds_up_to_the_maximum_speed_and_slows_down_to_a_stop(self):
        state_mock = ControllerStateMock(l_thumb_x=100)
        # 100 steps per second, so 10 steps to speed up to 10 cm/s
        pose_updater = XboxPoseUpdater(state_mock, 10, 0.1, None)
        pose = Pose(0, 0, 0)

        for _ in range(5):
            pose = pose_updater.get_updated_pose_from_controller(pose, False, None)
        self.assertAlmostEqual(5, pose_updater.v_x)
        for _ in range(20):
            pose = pose_updater.get_updated_pose_from_controller(pose, False, None)
        self.assertAlmostEqual(10, pose_updater.v_x)

        state_mock.l_thumb_x = 0
        for _ in range(10):
            pose = pose_updater.get_updated_pose_from_controller(pose, False, None)
        self.assertEqual(0, pose_updater.v_x)
        stopped_x = pose.x
        pose = pose_updater.get_updated_pose_from_controller(pose, False, None)
        self.assertEqual(stopped_x, pose.x)

    def test_partial_stick_input_limits_the_speed(self):
        pose_updater = XboxPoseUpdater(ControllerStateMock(l_thumb_y=-50), 10, 0.1, None)
        pose = Pose(0, 0, 0)

        for _ in range(30):
            pose = pose_updater.get_updated_pose_from_controller(pose, False, None)

        self.assertAlmostEqual(-5, pose_updater.v_y)

    def test_changing_the_maximum_speed_changes_the_speed_and_the_acceleration(self):
        pose_updater = XboxPoseUpdater(ControllerStateMock(l_thumb_x=100), 10, 0.1, None)
        pose_updater.maximum_speed = 20
        pose = Pose(0, 0, 0)

        pose = pose_updater.get_updated_pose_from_controller(pose, False, None)
        self.assertAlmostEqual(2, pose_updater.v_x)
        for _ in range(20):
            pose = pose_updater.get_updated_pose_from_controller(pose, False, None)
        self.assertAlmostEqual(20, pose_updater.v_x)
        self.assertEqual(20, pose_updater.maximum_speed)

    def test_orientation_stays_within_its_limits(self):
        for r_thumb_x, r_thumb_y in ((1000, 1000), (-1000, -1000)):
            pose_updater = XboxPoseUpdater(ControllerStateMock(r_thumb_x=r_thumb_x, r_thumb_y=r_thumb_y), 10, 0.1,
                                           None)
            pose = Pose(0, 20, 0)
            for _ in range(500):
                pose = pose_updater.get_updated_pose_from_controller(pose, False, None)
            np.testing.assert_allclose([pi / 2, pi / 2], [abs(pose.alpha), abs(pose.gamma)])


class XboxPoseUpdaterCenterTests(unittest.TestCase):

    def test_end_effector_points_towards_the_center(self):
        # the right stick is ignored when there is a center
        pose_updater = XboxPoseUpdater(ControllerStateMock(r_thumb_x=1000, r_thumb_y=1000), 10, 0.1, None)

        pose = pose_updater.get_updated_pose_from_controller(Pose(0, 20, 0), False, [0, 30, 10])

        self.assertAlmostEqual(0, pose.alpha)
        self.assertAlmostEqual(pi / 4, pose.gamma)
        self.assertEqual(0, pose.beta)

        pose = pose_updater.get_updated_pose_from_controller(Pose(0, 20, 0), False, [-10, 30, 0])
        self.assertAlmostEqual(pi / 4, pose.alpha)
        self.assertAlmostEqual(0, pose.gamma)

    def test_center_behind_the_end_effector_is_clipped(self):
        pose_updater = XboxPoseUpdater(ControllerStateMock(), 10, 0.1, None)

        pose = pose_updater.get_updated_pose_from_controller(Pose(0, 20, 0), False, [1, 10, 0])

        self.assertAlmostEqual(-pi / 2, pose.alpha)

    def test_flip_is_kept(self):
        pose_updater = XboxPoseUpdater(ControllerStateMock(l_thumb_x=100), 10, 0.1, WorkSpaceLimits)

        pose = pose_updater.get_updated_pose_from_controller(Pose(30, 22, 4, flip=True), False, None)

        self.assertTrue(pose.flip)
