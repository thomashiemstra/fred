import unittest

import numpy as np
from numpy import pi

from src.kinematics.kinematics import forward_position_kinematics, inverse_kinematics
from src.kinematics.kinematics_utils import Pose, RobotConfig
from src.utils import workspace
from src.utils.movement_exception import MovementException

test_config = RobotConfig(d1=13.92, a2=20, d4=22, d6=12)


class DummyWorkspaceLimits:
    radius_min = 15
    wrist_reach_max = 30
    y_min = 16
    z_min = 5


class WristRadiusLimits:
    """loose limits except the one under test: the wrist centre at least 12.5cm from the base axis"""
    radius_min = 0
    wrist_reach_max = 100
    y_min = -100
    z_min = -100
    wrist_radius_min = 12.5


class WorkspaceViolationTests(unittest.TestCase):
    """the check every robot exposes as check_workspace, on the pose of the angles"""

    def violation(self, x, y, z, gamma=0.0, limits=DummyWorkspaceLimits):
        return workspace.workspace_violation(inverse_kinematics(Pose(x, y, z, gamma=gamma), test_config),
                                             test_config, limits)

    def test_point_within_the_limits(self):
        self.assertIsNone(self.violation(0, 30, 10))

    def test_point_on_a_limit_is_inside(self):
        self.assertIsNone(self.violation(0, 30, 5))

    def test_point_too_far_away(self):
        # the wrist at (0, 33, 10), 33.2cm from the shoulder at (0, 0, 13.92)
        self.assertIn('the wrist would move to [ 0. 33. 10.], 33.2cm from the shoulder, farther than 30',
                      self.violation(0, 45, 10))

    def test_reach_is_measured_from_the_shoulder(self):
        # the tip 41.2cm from the ground under the base, the wrist 28.3cm from the shoulder
        self.assertIsNone(self.violation(0, 40, 10))
        # with the gripper pointing down the tip is closer, but the wrist above it is 31.1cm from the shoulder
        self.assertIn('from the shoulder', self.violation(0, 30, 10, gamma=-pi / 2))

    def test_point_too_close(self):
        self.assertIn('from the base, closer than 15', self.violation(0, 10, 10))

    def test_point_too_low(self):
        self.assertIn('below the minimum height', self.violation(0, 30, 4))

    def test_point_too_close_to_the_base_plane(self):
        self.assertIn('closer than y=', self.violation(8, 15, 10))

    def test_wrist_too_low(self):
        # the gripper points up, the wrist is 12cm below the tip; without wrist_z_min the limits use z_min for it
        self.assertIn('the wrist would move to [ 0. 25. -2.], below the minimum height',
                      self.violation(0, 25, 10, gamma=pi / 2))

    def test_wrist_has_its_own_minimum(self):
        class LowWrist(DummyWorkspaceLimits):
            wrist_z_min = -5

        self.assertIsNone(self.violation(0, 25, 10, gamma=pi / 2, limits=LowWrist))

    def test_check_raises(self):
        angles = inverse_kinematics(Pose(0, 30, 4), test_config)
        with self.assertRaises(MovementException):
            workspace.check_workspace(angles, test_config, DummyWorkspaceLimits)
        workspace.check_workspace(inverse_kinematics(Pose(0, 30, 10), test_config), test_config, DummyWorkspaceLimits)

    def test_without_limits_everything_is_allowed(self):
        self.assertIsNone(workspace.workspace_violation(inverse_kinematics(Pose(0, 0, -100), test_config),
                                                        test_config, None))


class WristRadiusTests(unittest.TestCase):
    """the wrist centre has to stay wrist_radius_min from the vertical axis through the base, at any height"""

    @staticmethod
    def pointing_down(x, y, wrist_z):
        """the joint angles of the pose with the gripper pointing straight down and the wrist centre above (x, y)"""
        return inverse_kinematics(Pose(x, y, wrist_z - test_config.d6, gamma=-pi / 2), test_config)

    @staticmethod
    def wrist_of(angles):
        return forward_position_kinematics(angles, test_config)[3]

    def violation(self, x, y, wrist_z, limits=WristRadiusLimits):
        return workspace.workspace_violation(self.pointing_down(x, y, wrist_z), test_config, limits)

    def test_the_poses_have_the_wrist_where_the_tests_say(self):
        wrist = self.wrist_of(self.pointing_down(3, 4, 20))
        np.testing.assert_allclose([3, 4, 20], wrist, atol=1e-6)

    def test_wrist_closer_to_the_base_axis_is_refused(self):
        self.assertIn('the wrist would move to [ 0. 10. 20.], 10.0cm from the base, closer than 12.5',
                      self.violation(0, 10, 20))

    def test_wrist_just_outside_the_limit_passes(self):
        self.assertIsNone(self.violation(0, 12.6, 20))

    def test_wrist_just_inside_the_limit_is_refused(self):
        self.assertIn('closer than 12.5', self.violation(0, 12.4, 20))

    def test_wrist_exactly_on_the_limit_passes(self):
        self.assertIsNone(self.violation(0, 12.5, 20))
        # on the limit sideways too, where it is the distance to the axis that counts, not y
        self.assertIsNone(self.violation(7.5, 10.0, 20))

    def test_the_distance_to_the_axis_counts_not_y_alone(self):
        # y is 12 and x is 5, so the wrist is 13cm from the axis; with x = 0 the same y is refused
        self.assertIsNone(self.violation(5, 12, 20))
        self.assertIn('closer than 12.5', self.violation(0, 12, 20))
        self.assertIn('closer than 12.5', self.violation(-3, 4, 20))

    def test_the_height_does_not_matter(self):
        for wrist_z in (5, 13.92, 20, 35):
            with self.subTest(wrist_z=wrist_z):
                self.assertIn('closer than 12.5', self.violation(0, 5, wrist_z))
                self.assertIsNone(self.violation(0, 13, wrist_z))

    def test_the_measured_pose_of_26_november_2026_is_refused(self):
        # the wrist was 10.4cm from the axis and 13.2cm above the table with the parts touching
        self.assertIn('10.4cm from the base', self.violation(0, 10.4, 13.2))

    def test_limits_without_the_attribute_check_nothing(self):
        class OldLimits:
            radius_min = 0
            wrist_reach_max = 100
            y_min = -100
            z_min = -100

        self.assertIsNone(self.violation(0, 5, 20, limits=OldLimits))

    def test_check_raises(self):
        with self.assertRaises(MovementException):
            workspace.check_workspace(self.pointing_down(0, 10, 20), test_config, WristRadiusLimits)
        workspace.check_workspace(self.pointing_down(0, 13, 20), test_config, WristRadiusLimits)


if __name__ == '__main__':
    unittest.main()
