"""the joint space moves, the splines and the recorded moves of src/utils/movement.py; the planner of straight
lines and the checks are in movement_planning_tests.py"""
import unittest
from math import ceil
from unittest import mock

import numpy as np
from numpy import pi
import numpy.testing as test

from src import global_constants
from src.kinematics.kinematics import inverse_kinematics
from src.kinematics.kinematics_utils import Pose, RobotConfig
from src.utils import movement, workspace
from src.utils.movement import SplineMovement, PoseToPoseMovement, Movement, plan_spline, spline_poses, \
    plan_move_to_angles, move_between_poses, move_from_current_angles, get_curve_val, get_spline_step_arrays, \
    get_angles_center, get_adjustments_and_stop_pose
from src.utils.movement_exception import MovementException
from src.utils.robot_controller_utils import get_recommended_wait_time

test_config = RobotConfig(d1=13.92, a2=20, d4=22, d6=12)


class DummyWorkspaceLimits:
    radius_min = 15
    radius_max = 40
    y_min = 16
    z_min = 5


class FakeServoController:
    """a robot with real kinematics that records the angles it is asked to move to"""

    def __init__(self, current_pose=None, workspace_limits=None, current_angles=None):
        self.robot_config = test_config
        self.workspace_limits = workspace_limits
        if current_pose is not None:
            current_angles = inverse_kinematics(current_pose, test_config)
        self.current_angles = current_angles
        self.sent_angles = []

    def get_angle_limits(self):
        return None

    def workspace_violation(self, angles):
        return workspace.workspace_violation(angles, self.robot_config, self.workspace_limits)

    def check_workspace(self, angles):
        workspace.check_workspace(angles, self.robot_config, self.workspace_limits)

    def move_servos(self, angles):
        # the caller may reuse the array, so copy it
        self.sent_angles.append(np.array(angles, dtype=np.float64))

    def get_current_angles(self):
        return self.current_angles

    def pose_to_angles(self, pose):
        return inverse_kinematics(pose, self.robot_config)


def limited_controller(current_pose=None):
    """a fake robot with the dummy workspace limits"""
    return FakeServoController(current_pose, DummyWorkspaceLimits)


poses = [Pose(-20, 20, 5), Pose(0, 30, 10), Pose(20, 20, 5)]
# the curve through these bulges out to more than 40cm from the base
outside = [Pose(-20, 20, 5), Pose(0, 45, 10), Pose(20, 20, 5)]


def angles_of(pose):
    return inverse_kinematics(pose, test_config)


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


class JointMoveTests(unittest.TestCase):
    """moves with a given duration: minimum jerk interpolation in joint space, slowed down when too fast"""

    start = np.array([0, 0.5, 1.0, 0.0, 0.2, 1.0, -1.0])
    # at most 0.8rad in 0.5s: a minimum jerk profile peaks at 1.875 times the average, 3rad/s, below the limit of 4
    stop = np.array([0, 1.0, 0.5, 0.3, 0.2, 0.5, -0.2])

    def test_goes_from_the_start_to_the_stop_angles_in_the_duration(self):
        move = plan_move_to_angles(FakeServoController(), self.start, self.stop, duration=0.5)

        self.assertEqual(0.5, move.duration)
        self.assertEqual(ceil(0.5 * global_constants.steps_per_second) + 1, len(move.angles))
        test.assert_allclose(self.start, move.angles[0])
        test.assert_allclose(self.stop, move.angles[-1])
        test.assert_allclose((self.start + self.stop) / 2, move.angles[len(move.angles) // 2])
        test.assert_allclose([0, 0.5, 1.0, 0.0, 0.2, 1.0, -1.0], self.start, err_msg='the start angles should not change')

    def test_a_move_that_is_too_fast_is_slowed_down(self):
        # 0.6rad in 0.1s peaks at 11rad/s, the move is stretched until the fastest joint stays below 4rad/s
        stop = self.start.copy()
        stop[2] += 0.6
        with self.assertLogs(movement.logger, 'INFO'):
            move = plan_move_to_angles(FakeServoController(), self.start, stop, duration=0.1)
        self.assertGreater(move.duration, 0.25)
        self.assertLess(move.duration, 0.4)
        speeds = np.abs(np.diff(move.angles[:, 1:], axis=0)) * global_constants.steps_per_second
        self.assertLessEqual(speeds.max(), global_constants.recommended_max_servo_speed)
        test.assert_allclose(stop, move.angles[-1])

    def test_a_jump_is_refused_rather_than_stretched_without_end(self):
        # 3rad in 0.1s would need 14 times the duration, more than MAX_STRETCH
        stop = self.start.copy()
        stop[2] += 3.0
        with self.assertRaisesRegex(MovementException, 'rad/s'):
            plan_move_to_angles(FakeServoController(), self.start, stop, duration=0.1)

    def test_move_between_poses_takes_the_time_of_the_stop_pose(self):
        robot = FakeServoController()
        start_pose = Pose(25, 25, 10)
        stop_pose = Pose(0, 30, 10, time=0.6)  # the base turns 45 degrees, in 0.6s that peaks at 2.5rad/s
        with mock.patch.object(movement, 'sleep'):
            self.assertIs(stop_pose, move_between_poses(robot, start_pose, stop_pose))

        sent = robot.sent_angles
        self.assertEqual(ceil(0.6 * global_constants.steps_per_second) + 1, len(sent))
        test.assert_allclose(angles_of(start_pose), sent[0], atol=1e-9)
        test.assert_allclose(angles_of(stop_pose), sent[-1], atol=1e-9)

    def test_already_at_the_target_does_not_move(self):
        target = Pose(25, 25, 10)
        robot = FakeServoController(current_angles=angles_of(target) + 0.05)
        with mock.patch.object(movement, 'sleep'):
            move_from_current_angles(robot, target, 1)
        self.assertEqual([], robot.sent_angles)

    def test_moves_from_the_current_angles_to_the_pose(self):
        target = Pose(25, 25, 10)
        current_angles = angles_of(Pose(0, 30, 10))
        robot = FakeServoController(current_angles=current_angles)
        with mock.patch.object(movement, 'sleep'):
            move_from_current_angles(robot, target, 1)
        test.assert_allclose(current_angles, robot.sent_angles[0], atol=1e-9)
        test.assert_allclose(angles_of(target), robot.sent_angles[-1], atol=1e-9)

    def test_moves_outside_the_workspace_are_refused_before_anything_moves(self):
        robot = limited_controller(current_pose=poses[0])
        with mock.patch.object(movement, 'sleep'):
            with self.assertRaises(MovementException):
                move_from_current_angles(robot, Pose(0, 45, 10), 1)
            with self.assertRaises(MovementException):
                move_between_poses(robot, poses[0], Pose(0, 45, 10), 1)
        self.assertEqual([], robot.sent_angles)

    def test_unreadable_current_angles_are_refused(self):
        with self.assertRaisesRegex(MovementException, 'current angles'):
            move_from_current_angles(FakeServoController(), Pose(25, 25, 10), 1)


class SplineTests(unittest.TestCase):

    def test_outside_workspace_limits(self):
        with self.assertRaises(MovementException):
            plan_spline(limited_controller(), [Pose(0, 0, 0), Pose(100, 100, 100)], 2)

    def test_curve_leaving_the_workspace_between_the_poses_is_refused(self):
        # the start and stop pose are within the limits, the curve bulges out to more than 40cm from the base
        with self.assertRaises(MovementException):
            plan_spline(limited_controller(), outside, 2)

    def test_curve_is_not_checked_without_limits(self):
        move, _ = plan_spline(FakeServoController(), outside, 2)
        self.assertIsNotNone(move)

    def test_the_orientation_counts_for_the_wrist(self):
        # the gripper points up along the whole curve, so the wrist is 12cm below the tip: below z_min = 5
        up = [Pose(-20, 20, 10, gamma=pi / 2), Pose(0, 30, 12, gamma=pi / 2), Pose(20, 20, 10, gamma=pi / 2)]
        with self.assertRaisesRegex(MovementException, 'the wrist would move'):
            plan_spline(limited_controller(), up, 2)
        # the same curve with the gripper level keeps the wrist at the height of the tip
        level = [Pose(-20, 20, 10), Pose(0, 30, 12), Pose(20, 20, 10)]
        move, _ = plan_spline(limited_controller(), level, 2)
        self.assertIsNotNone(move)

    def test_spline_perfect_fit(self):
        # with these poses the b-spline starts and stops exactly as required
        move, stop_pose = plan_spline(limited_controller(), poses, 2)
        self.assertEqual(poses[-1], stop_pose)
        self.assertEqual(2.0, move.duration)
        self.assertEqual(2 * global_constants.steps_per_second, len(move.angles))
        test.assert_allclose(angles_of(poses[0]), move.angles[0], atol=1e-9)
        test.assert_allclose(angles_of(poses[-1]), move.angles[-1], atol=1e-9)

    def test_spline_non_perfect_fit(self):
        # with these poses the b-spline does not exactly end at the last pose
        five = [Pose(-20, 20, 5), Pose(-10, 30, 10), Pose(0, 20, 10), Pose(10, 30, 10), Pose(20, 20, 5)]
        move, stop_pose = plan_spline(limited_controller(), five, 2)
        self.assertNotEqual(five[-1], stop_pose)
        self.assertAlmostEqual(stop_pose.x, five[-1].x, places=1)
        self.assertAlmostEqual(stop_pose.y, five[-1].y, places=1)
        self.assertAlmostEqual(stop_pose.z, five[-1].z, places=1)
        test.assert_allclose(angles_of(stop_pose), move.angles[-1], atol=1e-9)

    def test_curve_starts_at_the_start_pose_and_keeps_the_flip(self):
        # the curve does not go through the poses exactly, so it is shifted to start at the first one
        flipped = [Pose(-20, 20, 5, flip=True), Pose(-10, 30, 10), Pose(0, 20, 10), Pose(10, 30, 10),
                   Pose(20, 20, 5, flip=True)]
        steps, stop_pose = spline_poses(flipped, 2)
        self.assertEqual(2 * global_constants.steps_per_second, len(steps))
        test.assert_allclose([-20, 20, 5], [steps[0].x, steps[0].y, steps[0].z], atol=1e-9)
        test.assert_allclose([stop_pose.x, stop_pose.y, stop_pose.z], [steps[-1].x, steps[-1].y, steps[-1].z],
                             atol=1e-9)
        self.assertTrue(all(pose.flip for pose in steps))

    def test_a_curve_that_is_too_fast_is_slowed_down(self):
        with self.assertLogs(movement.logger, 'INFO'):
            move, _ = plan_spline(limited_controller(), poses, 0.3)
        self.assertGreater(move.duration, 0.3)
        speeds = np.abs(np.diff(move.angles[:, 1:], axis=0)) * global_constants.steps_per_second
        self.assertLessEqual(speeds.max(), global_constants.recommended_max_servo_speed)

    def test_pointing_at_a_center_first_turns_towards_it(self):
        center = [0, 60, 10]
        move, stop_pose = plan_spline(FakeServoController(), poses, 3, center=center)
        # the plan starts with a turn from the recorded orientation to pointing at the center, half a second unless
        # the wrist would turn too fast; the curve itself may have been slowed down too
        pointed = Pose(-20, 20, 5)
        pointed.alpha, pointed.beta, pointed.gamma = get_angles_center(-20, 20, 5, center)
        test.assert_allclose(angles_of(poses[0]), move.angles[0], atol=1e-9)
        turn_steps = next(i for i, angles in enumerate(move.angles) if np.allclose(angles, angles_of(pointed), atol=1e-6))
        self.assertGreaterEqual(turn_steps, ceil(movement.TURN_TOWARDS_CENTER * global_constants.steps_per_second))
        self.assertGreaterEqual(move.duration, 3 + movement.TURN_TOWARDS_CENTER)
        self.assertEqual(ceil(move.duration * global_constants.steps_per_second), len(move.angles))
        self.assertEqual(get_angles_center(stop_pose.x, stop_pose.y, stop_pose.z, center)[0], stop_pose.alpha)
        # a start pose that points at the center already needs no turn
        pointed_poses = [pointed] + poses[1:]
        move, _ = plan_spline(FakeServoController(), pointed_poses, 3, center=center)
        self.assertEqual(ceil(move.duration * global_constants.steps_per_second), len(move.angles))

    def test_needs_at_least_2_poses(self):
        self.assertRaises(ValueError, lambda: get_spline_step_arrays([Pose(0, 30, 10)], 2))

    def test_number_of_steps_depends_on_the_time(self):
        x_steps, _, _, total_steps, _, _, _ = get_spline_step_arrays(poses, 1.5)
        self.assertEqual(ceil(1.5 * global_constants.steps_per_second), total_steps)
        self.assertEqual(total_steps, len(x_steps))


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


class MovementTests(unittest.TestCase):
    """the recorded moves of the xbox controller"""

    def test_needs_at_least_2_poses(self):
        self.assertRaises(ValueError, lambda: SplineMovement([Pose(0, 30, 10)], 2))

    def test_center_is_converted_to_floats(self):
        movement_ = SplineMovement(poses, 2, center=[0, np.int64(30), 10])
        self.assertEqual([0.0, 30.0, 10.0], movement_.center)
        self.assertTrue(all(type(value) is float for value in movement_.center))

    def test_robot_close_to_the_start_pose_is_at_the_start_pose(self):
        controller = FakeServoController(poses[0])
        controller.current_angles = controller.current_angles + 0.39
        self.assertTrue(Movement.is_robot_at_start_pose(poses[0], controller))
        controller.current_angles[3] += 0.02
        self.assertFalse(Movement.is_robot_at_start_pose(poses[0], controller))

    def test_move_is_refused_when_the_robot_is_not_at_the_start_pose(self):
        controller = FakeServoController(poses[-1])
        with mock.patch.object(movement, 'sleep'):
            self.assertRaises(MovementException, lambda: SplineMovement(poses, 2).move(controller))
        self.assertEqual([], controller.sent_angles)

    def test_spline_move_sends_the_curve(self):
        controller = FakeServoController(poses[0])
        with mock.patch.object(movement, 'sleep'):
            stop_pose = SplineMovement(poses, 2).move(controller)
        self.assertEqual(poses[-1], stop_pose)
        self.assertEqual(2 * global_constants.steps_per_second, len(controller.sent_angles))
        test.assert_allclose(angles_of(poses[0]), controller.sent_angles[0], atol=1e-9)
        test.assert_allclose(angles_of(poses[-1]), controller.sent_angles[-1], atol=1e-9)

    def test_move_reversed_starts_at_the_last_pose(self):
        movement_ = SplineMovement(poses, 2)
        with mock.patch.object(movement, 'sleep'):
            self.assertRaises(MovementException, lambda: movement_.move_reversed(FakeServoController(poses[0])))

            controller = FakeServoController(poses[-1])
            movement_.move_reversed(controller)
        test.assert_allclose(angles_of(poses[-1]), controller.sent_angles[0], atol=1e-9)
        test.assert_allclose(angles_of(poses[0]), controller.sent_angles[-1], atol=1e-9)

    def test_go_to_the_start_of_the_move(self):
        controller = FakeServoController(poses[-1])
        with mock.patch.object(movement, 'sleep'):
            SplineMovement(poses, 2).go_to_start_of_move(controller, 2)
        self.assertEqual(2 * global_constants.steps_per_second + 1, len(controller.sent_angles))
        test.assert_allclose(angles_of(poses[-1]), controller.sent_angles[0], atol=1e-9)
        test.assert_allclose(angles_of(poses[0]), controller.sent_angles[-1], atol=1e-9)

    def test_spline_movement_is_possible_within_the_workspace_limits_of_the_robot(self):
        controller = limited_controller(poses[0])
        self.assertTrue(SplineMovement(poses, 2).is_possible(controller))
        self.assertFalse(SplineMovement(outside, 2).is_possible(controller))
        self.assertTrue(SplineMovement(outside, 2).is_possible(FakeServoController(poses[0])), 'no limits')

    def test_pose_to_pose_movement_checks_its_poses(self):
        controller = limited_controller(poses[0])
        self.assertTrue(PoseToPoseMovement([poses[0], poses[-1]], 1).is_possible(controller))
        self.assertFalse(PoseToPoseMovement([poses[0], Pose(0, 45, 10)], 1).is_possible(controller))

    def test_a_movement_outside_the_workspace_does_not_move_the_robot(self):
        controller = limited_controller(poses[0])
        with mock.patch.object(movement, 'sleep'):
            self.assertRaises(MovementException, lambda: PoseToPoseMovement([poses[0], Pose(0, 45, 10)], 1).move(controller))
            self.assertRaises(MovementException, lambda: SplineMovement(outside, 2).move(controller))
        self.assertEqual([], controller.sent_angles)

    def test_pose_to_pose_movement_goes_from_the_first_to_the_last_pose(self):
        controller = FakeServoController(poses[0])
        with mock.patch.object(movement, 'sleep'):
            PoseToPoseMovement(poses, 0.5).move(controller)
        test.assert_allclose(angles_of(poses[0]), controller.sent_angles[0], atol=1e-9)
        test.assert_allclose(angles_of(poses[-1]), controller.sent_angles[-1], atol=1e-9)


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
