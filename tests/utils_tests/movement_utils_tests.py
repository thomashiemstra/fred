import unittest
from math import ceil
from unittest import mock

from src import global_constants
from src.kinematics.kinematics import inverse_kinematics
from src.kinematics.kinematics_utils import Pose, RobotConfig
from src.utils import linalg_utils, movement_utils
import numpy as np
from numpy import pi
import numpy.testing as test

from src.utils.movement_exception import MovementException
from src.utils.movement_utils import b_spline_curve, b_spline_curve_calculate_only, get_curve_val, \
    check_workspace_limits, get_spline_step_arrays, angles_to_angles, pose_to_pose, from_current_angles_to_pose, \
    get_angles_center, get_adjustments_and_stop_pose

test_config = RobotConfig(d1=13.92, a2=20, d4=22, d6=12)


class FakeServoController:
    """records what would be sent to the servos"""

    def __init__(self, current_angles=None):
        self.robot_config = test_config
        self.current_angles = current_angles
        self.sent_angles = []
        self.sent_poses = []

    def move_servos(self, angles):
        # the caller reuses the array, so copy it
        self.sent_angles.append(np.array(angles, dtype=np.float64))

    def move_to_pose(self, pose):
        self.sent_poses.append(pose)
        return 0.0, 0.0

    def get_current_angles(self):
        return self.current_angles

    def pose_to_angles(self, pose):
        return inverse_kinematics(pose, self.robot_config)


class GetCentreTests(unittest.TestCase):

    def test_right_angle_lines_1(self):
        pose1 = Pose(0, 0, 0,)
        pose2 = Pose(10, 10, 0, alpha=pi/2)

        centre = linalg_utils.get_center(pose1, pose2)
        self.assert_centre_equals([0, 10, 0], centre)

    def test_right_angle_lines_2(self):
        pose1 = Pose(0, 0, 0,)
        pose2 = Pose(10, 10, 0, alpha=-pi/2)

        centre = linalg_utils.get_center(pose1, pose2)
        self.assert_centre_equals([0, 10, 0], centre)

    def test_parallel_1(self):
        pose1 = Pose(0, 0, 0,  alpha=-pi/2)
        pose2 = Pose(10, 10, 0, alpha=-pi/2)

        centre = linalg_utils.get_center(pose1, pose2)
        self.assertIsNone(centre)

    def test_parallel_2(self):
        pose1 = Pose(0, 0, 0,  alpha=pi/2)
        pose2 = Pose(10, 10, 0, alpha=-pi/2)

        centre = linalg_utils.get_center(pose1, pose2)
        self.assertIsNone(centre)

    def test_same_y_for_points_1(self):
        pose1 = Pose(0, 0, 0, alpha=pi / 4)
        pose2 = Pose(10, 0, 0, alpha=-pi / 4)

        centre = linalg_utils.get_center(pose1, pose2)
        self.assert_centre_equals([5, -5, 0], centre)

    def test_same_y_for_points_2(self):
        pose1 = Pose(0, 0, 0, alpha=-pi / 4)
        pose2 = Pose(10, 0, 0, alpha=pi / 4)

        centre = linalg_utils.get_center(pose1, pose2)
        self.assert_centre_equals([5, 5, 0], centre)

    def test_no_points_at_origin_1(self):
        pose1 = Pose(5, 5, 0, alpha=pi / 8)
        pose2 = Pose(10, 10, 0, alpha=-pi / 8)

        centre = linalg_utils.get_center(pose1, pose2)
        self.assert_centre_equals([6.46, 1.46, 0], centre)
        # the centre is where the lines along the direction of both poses cross
        for pose in (pose1, pose2):
            self.assert_on_the_line_of(pose, centre)

    def test_different_heights_have_no_centre(self):
        self.assertIsNone(linalg_utils.get_center(Pose(0, 0, 0), Pose(10, 10, 5, alpha=pi / 2)))

    def test_poses_not_pointing_horizontally_have_no_centre(self):
        self.assertIsNone(linalg_utils.get_center(Pose(0, 0, 0, gamma=0.5), Pose(10, 10, 0, alpha=pi / 2)))

    def assert_on_the_line_of(self, pose, centre):
        # alpha turns the end effector, which points along +y, around the z-axis
        direction = Pose(0, 0, 0, alpha=pose.alpha).get_euler_matrix()[:2, 2]
        to_centre = np.asarray(centre[:2]) - [pose.x, pose.y]
        self.assertAlmostEqual(0.0, np.cross(direction, to_centre))

    def assert_centre_equals(self, expected_centre, actual_centre):
        self.assertAlmostEqual(expected_centre[0], actual_centre[0], places=1, msg='x does not match')
        self.assertAlmostEqual(expected_centre[1], actual_centre[1], places=1, msg='y does not match')
        self.assertAlmostEqual(expected_centre[2], actual_centre[2], places=1, msg='z does not match')


class GetRotationMatrixParamsTests(unittest.TestCase):

    def test_no_rotation(self):
        pose1 = Pose(0, 0, 0)
        pose2 = Pose(1, 0, 0)
        center = [1, 1, 0]

        r, p0_prime, p1_prime, pc_prime = linalg_utils.get_rotation_matrix_params(pose1, pose2, center)
        self.assertIsNotNone(r)

        expected = np.eye(3)
        test.assert_allclose(r, expected)
        test.assert_allclose(p0_prime, [0, 0, 0], err_msg='x_prime mismatch')
        test.assert_allclose(p1_prime, [1, 0, 0], err_msg='y_prime mismatch')
        test.assert_allclose(pc_prime, [1, 1, 0], err_msg='z_prime mismatch')

    def test_pi_rotation(self):
        pose1 = Pose(1, 0, 0)
        pose2 = Pose(0, 0, 0)
        center = [1, 1, 0]

        r, p0_prime, p1_prime, pc_prime = linalg_utils.get_rotation_matrix_params(pose1, pose2, center)
        self.assertIsNotNone(r)

        expected = np.array([
            [-1, 0,  0],
            [0,  1,  0],
            [0,  0, -1]
        ])
        test.assert_allclose(r, expected)
        test.assert_allclose(p0_prime, [0, 0, 0], err_msg='x_prime mismatch')
        test.assert_allclose(p1_prime, [-1, 0, 0], err_msg='y_prime mismatch')
        test.assert_allclose(pc_prime, [0, 1, 0], err_msg='z_prime mismatch')

    def test_center_above_xy(self):
        pose1 = Pose(0, 0, 0)
        pose2 = Pose(1, 0, 0)
        center = [0, 0, 1]

        r, p0_prime, p1_prime, pc_prime = linalg_utils.get_rotation_matrix_params(pose1, pose2, center)
        self.assertIsNotNone(r)

        expected = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        test.assert_allclose(r, expected)
        test.assert_allclose(p0_prime, [0, 0, 0], err_msg='x_prime mismatch')
        test.assert_allclose(p1_prime, [1, 0, 0], err_msg='y_prime mismatch')
        test.assert_allclose(pc_prime, [0, 0, 1], err_msg='z_prime mismatch')

    def test_aligned_points_have_no_plane(self):
        pose1 = Pose(1, 1, 1)
        pose2 = Pose(2, 2, 2)
        center = [-1, -1, -1]

        res = linalg_utils.get_rotation_matrix_params(pose1, pose2, center)
        self.assertIsNone(res)

    def test_3d_points(self):
        pose1 = Pose(1, 1, 1)
        pose2 = Pose(1, 2, 1)
        center = [1, 1, 2]

        r, p0_prime, p1_prime, pc_prime = linalg_utils.get_rotation_matrix_params(pose1, pose2, center)
        self.assertIsNotNone(r)

        expected = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])
        test.assert_allclose(r, expected)
        test.assert_allclose(p0_prime, [0, 0, 0], err_msg='x_prime mismatch')
        test.assert_allclose(p1_prime, [0, 1, 0], err_msg='y_prime mismatch')
        test.assert_allclose(pc_prime, [0, 0, 1], err_msg='z_prime mismatch')


class DummyWorkspaceLimits:
    radius_min = 15
    radius_max = 40
    y_min = 16
    z_min = 5


class CurveValueTests(unittest.TestCase):

    def test_goes_from_0_to_1(self):
        self.assertEqual(0.0, get_curve_val(0.0))
        self.assertEqual(1.0, get_curve_val(1.0))
        self.assertEqual(0.5, get_curve_val(0.5))

    def test_is_symmetric_and_increasing(self):
        t = np.linspace(0, 1, 1001)
        values = get_curve_val(t)
        test.assert_allclose(1 - values[::-1], values, atol=1e-12)
        self.assertTrue(np.all(np.diff(values) > 0))

    def test_starts_and_stops_without_speed_and_acceleration(self):
        h = 1e-4
        for t in (0.0, 1.0):
            speed = (get_curve_val(t + h) - get_curve_val(t - h)) / (2 * h)
            acceleration = (get_curve_val(t + h) - 2 * get_curve_val(t) + get_curve_val(t - h)) / h ** 2
            self.assertAlmostEqual(0.0, speed, places=6)
            self.assertAlmostEqual(0.0, acceleration, places=6)


class CheckWorkspaceLimitsTests(unittest.TestCase):

    def check(self, *point):
        x, y, z = ([value] for value in point)
        return check_workspace_limits(x, y, z, 1, DummyWorkspaceLimits)

    def test_point_within_the_limits(self):
        self.assertTrue(self.check(0, 30, 10))

    def test_point_too_far_away(self):
        self.assertFalse(self.check(0, 40, 10))

    def test_point_too_close(self):
        self.assertFalse(self.check(0, 10, 10))

    def test_point_too_low(self):
        self.assertFalse(self.check(0, 30, 4))

    def test_any_point_outside_is_refused(self):
        x, y, z = [0, 0, 0], [30, 30, 30], [10, 4, 10]
        self.assertFalse(check_workspace_limits(x, y, z, 3, DummyWorkspaceLimits))

    def test_without_limits_everything_is_allowed(self):
        self.assertTrue(check_workspace_limits([0], [0], [-100], 1, None))


class BSplineCurveTests(unittest.TestCase):

    def test_outside_workspace_limits(self):
        pose1 = Pose(0, 0, 0)
        pose2 = Pose(100, 100, 100)
        poses = [pose1, pose2]

        self.assertRaises(MovementException, lambda: b_spline_curve_calculate_only(poses, 2, DummyWorkspaceLimits))

    def test_curve_leaving_the_workspace_between_the_poses_is_refused(self):
        # the start and stop pose are within the limits, the curve bulges out to more than 40cm from the base
        poses = [Pose(-20, 20, 5), Pose(0, 45, 10), Pose(20, 20, 5)]
        self.assertRaises(MovementException, lambda: b_spline_curve_calculate_only(poses, 2, DummyWorkspaceLimits))

    # With these poses the b-spline should start and stop exactly as required
    def test_spline_perfect_fit(self):
        pose1 = Pose(-20, 20, 5)
        pose2 = Pose(0, 30, 10)
        pose3 = Pose(20, 20, 5)

        poses = [pose1, pose2, pose3]

        stop_pose = b_spline_curve_calculate_only(poses, 2, DummyWorkspaceLimits)
        self.assertIsNotNone(stop_pose)
        self.assertEqual(pose3, stop_pose)

    # With these poses the b-spline should not exactly end at the last pose
    def test_spline_non_perfect_fit(self):
        pose1 = Pose(-20, 20, 5)
        pose2 = Pose(-10, 30, 10)
        pose3 = Pose(0, 20, 10)
        pose4 = Pose(10, 30, 10)
        pose5 = Pose(20, 20, 5)

        poses = [pose1, pose2, pose3, pose4, pose5]

        stop_pose = b_spline_curve_calculate_only(poses, 2, DummyWorkspaceLimits)
        self.assertIsNotNone(stop_pose)
        self.assertNotEqual(pose5, stop_pose)
        self.assertAlmostEqual(stop_pose.x, pose5.x, places=1)
        self.assertAlmostEqual(stop_pose.y, pose5.y, places=1)
        self.assertAlmostEqual(stop_pose.z, pose5.z, places=1)

    def test_needs_at_least_2_poses(self):
        self.assertRaises(ValueError, lambda: get_spline_step_arrays([Pose(0, 30, 10)], 2))

    def test_number_of_steps_depends_on_the_time(self):
        poses = [Pose(-20, 20, 5), Pose(0, 30, 10), Pose(20, 20, 5)]
        x_steps, _, _, total_steps, _, _, _ = get_spline_step_arrays(poses, 1.5)
        self.assertEqual(ceil(1.5 * global_constants.steps_per_second), total_steps)
        self.assertEqual(total_steps, len(x_steps))

    def test_curve_outside_the_workspace_does_not_move_the_robot(self):
        servo_controller = FakeServoController()
        poses = [Pose(-20, 20, 5), Pose(0, 45, 10), Pose(20, 20, 5)]
        with mock.patch.object(movement_utils, 'sleep'):
            self.assertRaises(MovementException,
                              lambda: b_spline_curve(poses, 2, servo_controller, workspace_limits=DummyWorkspaceLimits))
        self.assertEqual([], servo_controller.sent_poses)
        self.assertEqual([], servo_controller.sent_angles)

    def test_curve_starts_at_the_start_pose(self):
        # the curve does not go through the poses exactly, so it is shifted to start at the first one
        servo_controller = FakeServoController()
        poses = [Pose(-20, 20, 5, flip=True), Pose(-10, 30, 10), Pose(0, 20, 10), Pose(10, 30, 10),
                 Pose(20, 20, 5, flip=True)]
        with mock.patch.object(movement_utils, 'sleep'):
            stop_pose = b_spline_curve(poses, 2, servo_controller, workspace_limits=DummyWorkspaceLimits)

        sent = servo_controller.sent_poses
        self.assertEqual(2 * global_constants.steps_per_second, len(sent))
        test.assert_allclose([-20, 20, 5], [sent[0].x, sent[0].y, sent[0].z], atol=1e-9)
        test.assert_allclose([stop_pose.x, stop_pose.y, stop_pose.z], [sent[-1].x, sent[-1].y, sent[-1].z],
                             atol=1e-9)
        self.assertTrue(all(pose.flip for pose in sent))


class AnglesToAnglesTests(unittest.TestCase):

    def test_goes_from_the_start_to_the_stop_angles(self):
        servo_controller = FakeServoController()
        start = np.array([0, 0.5, 1.0, 0.0, 0.2, 1.0, -1.0])
        stop = np.array([0, 1.5, 0.5, 0.3, 0.2, 0.5, 1.0])
        with mock.patch.object(movement_utils, 'sleep'):
            angles_to_angles(start, stop, 0.5, servo_controller)

        sent = servo_controller.sent_angles
        self.assertEqual(ceil(0.5 * global_constants.steps_per_second) + 1, len(sent))
        test.assert_allclose(start, sent[0])
        test.assert_allclose(stop, sent[-1])
        test.assert_allclose((start + stop) / 2, sent[len(sent) // 2])
        test.assert_allclose([0, 0.5, 1.0, 0.0, 0.2, 1.0, -1.0], start, err_msg='the start angles should not change')

    def test_pose_to_pose_takes_the_time_of_the_stop_pose(self):
        servo_controller = FakeServoController()
        start_pose = Pose(25, 25, 10)
        stop_pose = Pose(0, 30, 10, time=0.3)
        with mock.patch.object(movement_utils, 'sleep'):
            self.assertIs(stop_pose, pose_to_pose(start_pose, stop_pose, servo_controller))

        sent = servo_controller.sent_angles
        self.assertEqual(ceil(0.3 * global_constants.steps_per_second) + 1, len(sent))
        test.assert_allclose(inverse_kinematics(start_pose, test_config), sent[0], atol=1e-9)
        test.assert_allclose(inverse_kinematics(stop_pose, test_config), sent[-1], atol=1e-9)

    def test_already_at_the_target_does_not_move(self):
        target = Pose(25, 25, 10)
        servo_controller = FakeServoController(inverse_kinematics(target, test_config) + 0.05)
        with mock.patch.object(movement_utils, 'sleep'):
            from_current_angles_to_pose(target, servo_controller, 1)
        self.assertEqual([], servo_controller.sent_angles)

    def test_moves_from_the_current_angles_to_the_pose(self):
        target = Pose(25, 25, 10)
        current_angles = inverse_kinematics(Pose(0, 30, 10), test_config)
        servo_controller = FakeServoController(current_angles)
        with mock.patch.object(movement_utils, 'sleep'):
            from_current_angles_to_pose(target, servo_controller, 1)
        test.assert_allclose(current_angles, servo_controller.sent_angles[0], atol=1e-9)
        test.assert_allclose(inverse_kinematics(target, test_config), servo_controller.sent_angles[-1], atol=1e-9)


class OrientationTests(unittest.TestCase):

    def test_angles_center_point_the_end_effector_at_the_center(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            position = rng.uniform(-30, 30, 3)
            center = rng.uniform(-30, 30, 3)
            alpha, beta, gamma = get_angles_center(*position, center)
            approach = Pose(*position, alpha=alpha, beta=beta, gamma=gamma).get_euler_matrix()[:, 2]
            to_center = center - position
            test.assert_allclose(to_center / np.linalg.norm(to_center), approach, atol=1e-9)

    def test_adjustments_shift_the_curve_to_the_start_pose(self):
        start_pose = Pose(0, 30, 10)
        stop_pose = Pose(10, 30, 10, alpha=0.3)
        dx, dy, dz, actual_stop_pose = get_adjustments_and_stop_pose(start_pose, stop_pose, [1, 5, 11], [29, 30, 31],
                                                                     [10, 10, 12])
        self.assertEqual((1, -1, 0), (dx, dy, dz))
        self.assertEqual((10, 32, 12), (actual_stop_pose.x, actual_stop_pose.y, actual_stop_pose.z))
        self.assertEqual(0.3, actual_stop_pose.alpha)
        self.assertEqual(10, stop_pose.x, 'the stop pose should not change')


if __name__ == '__main__':
    unittest.main()
