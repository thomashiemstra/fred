import unittest

import numpy as np
from numpy import pi

from src.kinematics.kinematics_utils import Pose
from src.xbox_control.xbox360controller import xbox_pose_updater
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


def updater(maximum_speed=10, ramp_up_time=1, **stick):
    return XboxPoseUpdater(ControllerStateMock(**stick), maximum_speed, ramp_up_time)


class XboxPoseUpdaterTests(unittest.TestCase):

    def test_no_change(self):
        old_pose = Pose(0, 0, 0)
        new_pose = updater().get_updated_pose_from_controller(old_pose, False, None)
        self.assertEqual(new_pose, old_pose)

    def test_the_left_stick_moves_x_and_y(self):
        self.assertGreater(updater(l_thumb_x=1).get_updated_pose_from_controller(Pose(0, 0, 0), False, None).x, 0)
        self.assertLess(updater(l_thumb_x=-1).get_updated_pose_from_controller(Pose(0, 0, 0), False, None).x, 0)
        self.assertGreater(updater(l_thumb_y=1).get_updated_pose_from_controller(Pose(0, 0, 0), False, None).y, 0)
        self.assertLess(updater(l_thumb_y=-1).get_updated_pose_from_controller(Pose(0, 0, 0), False, None).y, 0)

    def test_the_triggers_move_z(self):
        self.assertGreater(updater(lr_trigger=1).get_updated_pose_from_controller(Pose(0, 0, 0), False, None).z, 0)
        self.assertLess(updater(lr_trigger=-1).get_updated_pose_from_controller(Pose(0, 0, 0), False, None).z, 0)

    def test_the_right_stick_turns_alpha_and_gamma(self):
        # the signs of the gamepad are sorted out in Xbox360Controller, here right and up are positive
        self.assertGreater(updater(r_thumb_x=1).get_updated_pose_from_controller(Pose(0, 0, 0), False, None).alpha, 0)
        self.assertLess(updater(r_thumb_x=-1).get_updated_pose_from_controller(Pose(0, 0, 0), False, None).alpha, 0)
        self.assertGreater(updater(r_thumb_y=1).get_updated_pose_from_controller(Pose(0, 0, 0), False, None).gamma, 0)
        self.assertLess(updater(r_thumb_y=-1).get_updated_pose_from_controller(Pose(0, 0, 0), False, None).gamma, 0)

    def test_gamma_and_z_are_held_during_find_center(self):
        old_pose = Pose(0, 0, 0)
        new_pose = updater(r_thumb_y=1, lr_trigger=1, l_thumb_x=1).get_updated_pose_from_controller(old_pose, True, None)
        self.assertEqual((old_pose.gamma, old_pose.z), (new_pose.gamma, new_pose.z))
        self.assertGreater(new_pose.x, old_pose.x)

    def test_flip_is_kept_and_beta_stays_zero(self):
        pose = updater(l_thumb_x=100).get_updated_pose_from_controller(Pose(30, 22, 4, flip=True), False, None)
        self.assertTrue(pose.flip)
        self.assertEqual(0, pose.beta)

    def test_there_is_no_workspace_clipping(self):
        # the control loop checks the workspace with the robot, the updater just integrates the sticks
        pose = Pose(0, 1, 0)
        for _ in range(100):
            pose = updater(10, 0.1, l_thumb_y=-100, lr_trigger=-100).get_updated_pose_from_controller(pose, False, None)
        self.assertLess(pose.y, 0)
        self.assertLess(pose.z, 0)


class XboxPoseUpdaterVelocityTests(unittest.TestCase):

    def test_speeds_up_to_the_maximum_speed_and_slows_down_to_a_stop(self):
        state_mock = ControllerStateMock(l_thumb_x=100)
        # 100 steps per second, so 10 steps to speed up to 10 cm/s
        pose_updater = XboxPoseUpdater(state_mock, 10, 0.1)
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
        pose_updater = updater(10, 0.1, l_thumb_y=-50)
        pose = Pose(0, 0, 0)
        for _ in range(30):
            pose = pose_updater.get_updated_pose_from_controller(pose, False, None)
        self.assertAlmostEqual(-5, pose_updater.v_y)

    def test_easing_off_the_stick_ramps_down_instead_of_dropping(self):
        state_mock = ControllerStateMock(l_thumb_x=100)
        pose_updater = XboxPoseUpdater(state_mock, 10, 0.1)
        pose = Pose(0, 0, 0)
        for _ in range(20):
            pose = pose_updater.get_updated_pose_from_controller(pose, False, None)
        self.assertAlmostEqual(10, pose_updater.v_x)

        state_mock.l_thumb_x = 50
        pose_updater.get_updated_pose_from_controller(pose, False, None)
        self.assertAlmostEqual(9, pose_updater.v_x)
        for _ in range(10):
            pose = pose_updater.get_updated_pose_from_controller(pose, False, None)
        self.assertAlmostEqual(5, pose_updater.v_x)

    def test_changing_the_maximum_speed_changes_the_speed_and_the_acceleration(self):
        pose_updater = updater(10, 0.1, l_thumb_x=100)
        pose_updater.maximum_speed = 20
        pose = Pose(0, 0, 0)

        pose = pose_updater.get_updated_pose_from_controller(pose, False, None)
        self.assertAlmostEqual(2, pose_updater.v_x)
        for _ in range(20):
            pose = pose_updater.get_updated_pose_from_controller(pose, False, None)
        self.assertAlmostEqual(20, pose_updater.v_x)
        self.assertEqual(20, pose_updater.maximum_speed)

    def test_the_angular_speed_is_independent_of_the_linear_speed(self):
        pose_updater = XboxPoseUpdater(ControllerStateMock(r_thumb_x=100), 10, 0.1, maximum_angular_speed=0.5)
        pose_updater.maximum_speed = 50
        pose = Pose(0, 20, 0)
        for _ in range(20):
            pose = pose_updater.get_updated_pose_from_controller(pose, False, None)
        self.assertAlmostEqual(0.5, pose_updater.v_alpha)

    def test_a_step_covers_the_time_that_passed(self):
        pose_updater = updater(10, 0.1, l_thumb_x=100)
        pose = Pose(0, 0, 0)
        for _ in range(20):
            pose = pose_updater.get_updated_pose_from_controller(pose, False, None)
        x = pose.x
        pose = pose_updater.get_updated_pose_from_controller(pose, False, None, dt=0.02)
        self.assertAlmostEqual(x + 0.2, pose.x)
        pose = pose_updater.get_updated_pose_from_controller(pose, False, None, dt=0.01)
        self.assertAlmostEqual(x + 0.3, pose.x)

    def test_a_stall_of_the_loop_does_not_make_the_pose_jump(self):
        pose_updater = updater(10, 0.1, l_thumb_x=100)
        pose = Pose(0, 0, 0)
        for _ in range(20):
            pose = pose_updater.get_updated_pose_from_controller(pose, False, None)
        x = pose.x
        pose = pose_updater.get_updated_pose_from_controller(pose, False, None, dt=5.0)
        self.assertAlmostEqual(x + 10 * xbox_pose_updater.MAX_STEP_TIME, pose.x)

    def test_orientation_stays_within_its_limits(self):
        for r_thumb_x, r_thumb_y in ((1000, 1000), (-1000, -1000)):
            pose_updater = updater(10, 0.1, r_thumb_x=r_thumb_x, r_thumb_y=r_thumb_y)
            pose = Pose(0, 20, 0)
            for _ in range(500):
                pose = pose_updater.get_updated_pose_from_controller(pose, False, None)
            np.testing.assert_allclose([pi / 2, pi / 2], [abs(pose.alpha), abs(pose.gamma)])


class XboxPoseUpdaterCenterTests(unittest.TestCase):

    def test_end_effector_points_towards_the_center(self):
        # the right stick is ignored when there is a center
        pose_updater = updater(10, 0.1, r_thumb_x=1000, r_thumb_y=1000)

        pose = pose_updater.get_updated_pose_from_controller(Pose(0, 20, 0), False, [0, 30, 10])

        self.assertAlmostEqual(0, pose.alpha)
        self.assertAlmostEqual(pi / 4, pose.gamma)
        self.assertEqual(0, pose.beta)

        pose = pose_updater.get_updated_pose_from_controller(Pose(0, 20, 0), False, [-10, 30, 0])
        self.assertAlmostEqual(pi / 4, pose.alpha)
        self.assertAlmostEqual(0, pose.gamma)

    def test_center_behind_the_end_effector_is_clipped(self):
        pose = updater(10, 0.1).get_updated_pose_from_controller(Pose(0, 20, 0), False, [1, 10, 0])
        self.assertAlmostEqual(-pi / 2, pose.alpha)


if __name__ == '__main__':
    unittest.main()
