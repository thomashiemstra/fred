import unittest
from unittest import mock

import numpy as np

from src.kinematics.kinematics import inverse_kinematics
from src.kinematics.kinematics_utils import Pose, RobotConfig
from src.utils import movement_utils
from src.utils.movement import SplineMovement, PoseToPoseMovement, Movement
from src.utils.movement_exception import MovementException
from src.utils.robot_controller_utils import get_recommended_wait_time
from src import global_constants

test_config = RobotConfig(d1=13.92, a2=20, d4=22, d6=12)


class DummyWorkspaceLimits:
    radius_min = 15
    radius_max = 40
    y_min = 16
    z_min = 5


class FakeServoController:
    """records what would be sent to the servos"""

    def __init__(self, current_pose):
        self.robot_config = test_config
        self.current_angles = inverse_kinematics(current_pose, test_config)
        self.sent_angles = []
        self.sent_poses = []

    def move_servos(self, angles):
        self.sent_angles.append(np.array(angles, dtype=np.float64))

    def move_to_pose(self, pose):
        self.sent_poses.append(pose)
        return 0.0, 0.0

    def get_current_angles(self):
        return self.current_angles

    def pose_to_angles(self, pose):
        return inverse_kinematics(pose, self.robot_config)


poses = [Pose(-20, 20, 5), Pose(0, 30, 10), Pose(20, 20, 5)]


class MovementTests(unittest.TestCase):

    def test_needs_at_least_2_poses(self):
        self.assertRaises(ValueError, lambda: SplineMovement([Pose(0, 30, 10)], 2))

    def test_center_is_converted_to_floats(self):
        movement = SplineMovement(poses, 2, center=[0, np.int64(30), 10])
        self.assertEqual([0.0, 30.0, 10.0], movement.center)
        self.assertTrue(all(type(value) is float for value in movement.center))

    def test_robot_close_to_the_start_pose_is_at_the_start_pose(self):
        controller = FakeServoController(poses[0])
        controller.current_angles = controller.current_angles + 0.39
        self.assertTrue(Movement.is_robot_at_start_pose(poses[0], controller))
        controller.current_angles[3] += 0.02
        self.assertFalse(Movement.is_robot_at_start_pose(poses[0], controller))

    def test_move_is_refused_when_the_robot_is_not_at_the_start_pose(self):
        controller = FakeServoController(poses[-1])
        movement = SplineMovement(poses, 2)
        with mock.patch.object(movement_utils, 'sleep'):
            self.assertRaises(MovementException, lambda: movement.move(controller))
        self.assertEqual([], controller.sent_poses)
        self.assertEqual([], controller.sent_angles)

    def test_move_reversed_starts_at_the_last_pose(self):
        movement = SplineMovement(poses, 2)
        with mock.patch.object(movement_utils, 'sleep'):
            self.assertRaises(MovementException, lambda: movement.move_reversed(FakeServoController(poses[0])))

            controller = FakeServoController(poses[-1])
            movement.move_reversed(controller)
        first = controller.sent_poses[0]
        np.testing.assert_allclose([20, 20, 5], [first.x, first.y, first.z], atol=1e-9)

    def test_spline_movement_checks_the_workspace_limits(self):
        controller = FakeServoController(poses[0])
        self.assertTrue(SplineMovement(poses, 2).check_workspace_limits(controller, DummyWorkspaceLimits))
        # the curve bulges out to more than 40cm from the base
        outside = [Pose(-20, 20, 5), Pose(0, 45, 10), Pose(20, 20, 5)]
        self.assertFalse(SplineMovement(outside, 2).check_workspace_limits(controller, DummyWorkspaceLimits))

    def test_pose_to_pose_movement_goes_from_the_first_to_the_last_pose(self):
        controller = FakeServoController(poses[0])
        with mock.patch.object(movement_utils, 'sleep'):
            PoseToPoseMovement(poses, 0.5).move(controller)
        np.testing.assert_allclose(inverse_kinematics(poses[0], test_config), controller.sent_angles[0], atol=1e-9)
        np.testing.assert_allclose(inverse_kinematics(poses[-1], test_config), controller.sent_angles[-1],
                                   atol=1e-9)


class RecommendedWaitTimeTests(unittest.TestCase):

    def test_slowest_joint_at_the_maximum_speed_determines_the_time(self):
        current = np.array([0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0])
        new = np.array([0, 0.7, 0.0, 0.1, 0.0, 0.0, -0.4])
        self.assertAlmostEqual(1.0 / global_constants.recommended_max_servo_speed,
                               get_recommended_wait_time(current, new))

    def test_no_wait_without_movement(self):
        angles = np.array([0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(0, get_recommended_wait_time(angles, angles))


if __name__ == '__main__':
    unittest.main()
