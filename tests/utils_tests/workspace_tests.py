import unittest

from numpy import pi

from src.kinematics.kinematics import inverse_kinematics
from src.kinematics.kinematics_utils import Pose, RobotConfig
from src.utils import workspace
from src.utils.movement_exception import MovementException

test_config = RobotConfig(d1=13.92, a2=20, d4=22, d6=12)


class DummyWorkspaceLimits:
    radius_min = 15
    radius_max = 40
    y_min = 16
    z_min = 5


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
        self.assertIn('from the base, outside of', self.violation(0, 40, 10))

    def test_point_too_close(self):
        self.assertIn('from the base, outside of', self.violation(0, 10, 10))

    def test_point_too_low(self):
        self.assertIn('below the minimum height', self.violation(0, 30, 4))

    def test_point_too_close_to_the_base_plane(self):
        self.assertIn('closer than y=', self.violation(8, 15, 10))

    def test_wrist_too_low(self):
        # the gripper points up, the wrist is 12cm below the tip; without wrist_z_min the limits use z_min for it
        self.assertIn('the wrist would move', self.violation(0, 30, 10, gamma=pi / 2))

    def test_wrist_has_its_own_minimum(self):
        class LowWrist(DummyWorkspaceLimits):
            wrist_z_min = -5

        self.assertIsNone(self.violation(0, 30, 10, gamma=pi / 2, limits=LowWrist))

    def test_check_raises(self):
        angles = inverse_kinematics(Pose(0, 30, 4), test_config)
        with self.assertRaises(MovementException):
            workspace.check_workspace(angles, test_config, DummyWorkspaceLimits)
        workspace.check_workspace(inverse_kinematics(Pose(0, 30, 10), test_config), test_config, DummyWorkspaceLimits)

    def test_without_limits_everything_is_allowed(self):
        self.assertIsNone(workspace.workspace_violation(inverse_kinematics(Pose(0, 0, -100), test_config),
                                                        test_config, None))


if __name__ == '__main__':
    unittest.main()
