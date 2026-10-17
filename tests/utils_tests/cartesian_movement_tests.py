import unittest
from unittest import mock

import numpy as np
from numpy import pi

from src import global_constants
from src.kinematics.kinematics import inverse_kinematics, forward_kinematics, forward_position_kinematics
from src.kinematics.kinematics_utils import Pose, RobotConfig
from src.utils import cartesian_movement
from src.utils.cartesian_movement import plan_move_to_pose, plan_straight_line, check_move, execute_move, \
    PlannedMove
from src.utils.movement_exception import MovementException

test_config = RobotConfig(d1=13.92, a2=20, d4=22, d6=12)
angle_limits = {1: (0, pi), 2: (0, pi), 3: (-pi / 2, pi / 2), 4: (-pi, pi), 5: (-pi, pi), 6: (-pi, pi)}
# the angles of the robot resting on the table
resting_angles = np.array([0, 0.183, 0.0, 0.736, -1.437, 1.689, 0.012])
start_pose = Pose(30, 22.0, 4)


class TestWorkspaceLimits:
    radius_min = 10
    radius_max = 57
    y_min = 12.1
    z_min = -5


def end_effector_positions(move):
    return np.array([forward_position_kinematics(angles, test_config)[-1] for angles in move.angles])


def single_step_move(pose):
    """a move of a single step, so the joint speed check has nothing to check"""
    return PlannedMove('single step', inverse_kinematics(pose, test_config)[np.newaxis], 0.0)


def standing_still(angles, steps=2):
    return PlannedMove('standing still', np.tile(np.asarray(angles, dtype=np.float64), (steps, 1)), 1.0)


class PlanMoveToPoseTests(unittest.TestCase):

    def test_starts_at_the_current_angles_and_ends_at_the_target_pose(self):
        move = plan_move_to_pose(resting_angles, start_pose, test_config, angle_limits=angle_limits)

        np.testing.assert_allclose(resting_angles, move.angles[0], atol=1e-9)
        np.testing.assert_allclose(inverse_kinematics(start_pose, test_config), move.angles[-1], atol=1e-9)

    def test_end_effector_moves_in_a_straight_line(self):
        move = plan_move_to_pose(resting_angles, start_pose, test_config, angle_limits=angle_limits)

        positions = end_effector_positions(move)
        direction = positions[-1] - positions[0]
        distance_to_line = np.linalg.norm(np.cross(positions - positions[0], direction), axis=1) / \
            np.linalg.norm(direction)
        self.assertLess(distance_to_line.max(), 1e-6)

    def test_orientation_turns_smoothly_to_the_target_orientation(self):
        target = Pose(25, 25, 10, alpha=0.3, gamma=-0.2)
        move = plan_move_to_pose(resting_angles, target, test_config, angle_limits=angle_limits)

        reached = forward_kinematics(move.angles[-1], test_config)
        np.testing.assert_allclose(target.get_euler_matrix(), reached.get_euler_matrix(), atol=1e-9)

    def test_keeps_the_flip_of_the_wrist(self):
        current_pose = forward_kinematics(resting_angles, test_config)
        current_pose.flip = True
        flipped_angles = inverse_kinematics(current_pose, test_config)

        move = plan_move_to_pose(flipped_angles, start_pose, test_config, angle_limits=angle_limits)

        reached = forward_kinematics(move.angles[-1], test_config)
        self.assertTrue(reached.flip)
        np.testing.assert_allclose([30, 22, 4], [reached.x, reached.y, reached.z], atol=1e-9)

    def test_flip_of_the_target_is_ignored(self):
        move = plan_move_to_pose(resting_angles, Pose(30, 22, 4, flip=True), test_config, angle_limits=angle_limits)

        self.assertTrue(np.all(move.angles[:, 5] > 0), 'the wrist should not flip')
        np.testing.assert_allclose(inverse_kinematics(start_pose, test_config), move.angles[-1], atol=1e-9)

    def test_target_pose_is_not_changed(self):
        target = Pose(30, 22, 4, flip=True)
        plan_move_to_pose(resting_angles, target, test_config, angle_limits=angle_limits)
        self.assertTrue(target.flip)

    def test_duration_depends_on_the_distance_and_rotation(self):
        move = plan_move_to_pose(resting_angles, start_pose, test_config, angle_limits=angle_limits,
                                 linear_speed=1.0, angular_speed=10.0)
        distance = np.linalg.norm(end_effector_positions(move)[-1] - end_effector_positions(move)[0])
        self.assertAlmostEqual(distance, move.duration)

    def test_duration_of_a_rotation_depends_on_the_angular_speed(self):
        current_angles = inverse_kinematics(Pose(25, 25, 10), test_config)
        # turning alpha turns the end effector around the z-axis
        move = plan_move_to_pose(current_angles, Pose(25, 25, 10, alpha=0.4), test_config, angle_limits=angle_limits,
                                 angular_speed=0.1)

        self.assertAlmostEqual(0.4 / 0.1, move.duration)
        self.assertEqual(4 * global_constants.steps_per_second + 1, len(move.angles))
        np.testing.assert_allclose(np.tile([25, 25, 10], (len(move.angles), 1)), end_effector_positions(move),
                                   atol=1e-9)

    def test_move_to_the_current_pose_stands_still_for_the_minimum_duration(self):
        current_angles = inverse_kinematics(Pose(25, 25, 10), test_config)
        move = plan_move_to_pose(current_angles, Pose(25, 25, 10), test_config, angle_limits=angle_limits)

        self.assertEqual(1.0, move.duration)
        self.assertEqual(global_constants.steps_per_second + 1, len(move.angles))
        np.testing.assert_allclose(np.tile(current_angles, (len(move.angles), 1)), move.angles, atol=1e-9)

    def test_end_effector_starts_and_stops_smoothly(self):
        move = plan_move_to_pose(resting_angles, start_pose, test_config, angle_limits=angle_limits)

        positions = end_effector_positions(move)
        step_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        progress = np.linalg.norm(positions - positions[0], axis=1)
        self.assertTrue(np.all(np.diff(progress) >= -1e-9), 'the end effector should never move back')
        # minimum jerk: the speed is 0 at the start and the end and highest halfway
        self.assertLess(step_lengths[0], 0.01 * step_lengths.max())
        self.assertLess(step_lengths[-1], 0.01 * step_lengths.max())
        self.assertAlmostEqual(len(step_lengths) / 2, step_lengths.argmax(), delta=1)

    def test_unreachable_target_is_refused(self):
        with self.assertRaisesRegex(MovementException, 'can not be reached'):
            plan_move_to_pose(resting_angles, Pose(60, 22, 4), test_config, angle_limits=angle_limits)

    def test_target_outside_the_angle_limits_is_refused(self):
        with self.assertRaisesRegex(MovementException, 'below its range'):
            plan_move_to_pose(resting_angles, Pose(30, 22, -10), test_config, angle_limits=angle_limits)

    def test_joint_at_the_limit_of_its_range_is_allowed(self):
        # servo 2 reads below its calibrated range, so its angle is clamped to exactly 0
        angles_at_limit = np.radians([0, 50.3, 0.0, 43.0, -86.3, 96.8, 0.7])
        for _ in range(20):
            angles_at_limit[1] += 0.01
            plan_move_to_pose(angles_at_limit, Pose(30, 22, 10), test_config, angle_limits=angle_limits)

    def test_target_below_the_workspace_is_refused(self):
        with self.assertRaisesRegex(MovementException, 'below the minimum height'):
            plan_move_to_pose(resting_angles, Pose(30, 22, -10), test_config)

    def test_target_too_close_to_the_base_is_refused(self):
        with self.assertRaisesRegex(MovementException, 'closer than y='):
            plan_move_to_pose(resting_angles, Pose(30, 8, 4), test_config)

    def test_moving_through_the_straight_wrist_is_refused(self):
        # angle 5 changes sign halfway, keeping the flip the inverse kinematics turns angle 4 and 6 around in one step
        straight_wrist = inverse_kinematics(Pose(0, 30, 20), test_config)
        current_angles = straight_wrist.copy()
        current_angles[5] = 0.2
        through_the_straight_wrist = straight_wrist.copy()
        through_the_straight_wrist[5] = -0.2
        target = forward_kinematics(through_the_straight_wrist, test_config)

        with self.assertRaisesRegex(MovementException, 'joint 4 would move at'):
            plan_move_to_pose(current_angles, target, test_config, angle_limits=angle_limits)

    def test_current_angles_on_another_inverse_kinematics_branch_are_refused(self):
        # the elbow is bent the other way than the inverse kinematics would, so the first step would jump
        elbow_bent_backwards = np.array([0, pi / 2, 0.3, 1.9, 0.0, 0.5, 0.0])
        with self.assertRaisesRegex(MovementException, 'away from the current angles'):
            plan_move_to_pose(elbow_bent_backwards, Pose(0, 30, 10), test_config)


class CheckMoveTests(unittest.TestCase):

    def test_move_that_does_not_start_at_the_current_angles_is_refused(self):
        move = plan_move_to_pose(resting_angles, start_pose, test_config)
        moved_by_hand = resting_angles.copy()
        moved_by_hand[1] += 0.2
        with self.assertRaisesRegex(MovementException, 'away from the current angles'):
            check_move(move, test_config, current_angles=moved_by_hand)

    def test_small_difference_with_the_current_angles_is_allowed(self):
        current_angles = resting_angles.copy()
        current_angles[1] += 0.8 * cartesian_movement.MAX_START_ANGLE_ERROR
        # index 0 is not a joint
        current_angles[0] = 100
        check_move(standing_still(resting_angles), test_config, current_angles=current_angles)

    def test_joints_that_move_too_fast_are_refused(self):
        angles = np.zeros((2, 7))
        angles[1, 4] = 2 * global_constants.recommended_max_servo_speed / global_constants.steps_per_second
        with self.assertRaisesRegex(MovementException, 'joint 4 would move at'):
            check_move(PlannedMove('too fast', angles, 0.01), test_config)

    def test_joints_just_below_the_maximum_speed_are_allowed(self):
        max_step = global_constants.recommended_max_servo_speed / global_constants.steps_per_second
        angles = np.zeros((3, 7))
        angles[1, 6] = 0.99 * max_step
        angles[2, 6] = angles[1, 6] - 0.99 * max_step
        check_move(PlannedMove('fast', angles, 0.02), test_config)

        angles[2, 6] = angles[1, 6] - 1.01 * max_step
        with self.assertRaisesRegex(MovementException, 'joint 6 would move at .* at step 1,'):
            check_move(PlannedMove('too fast', angles, 0.02), test_config)

    def test_angle_just_past_the_limit_is_allowed_within_the_tolerance(self):
        tolerance = cartesian_movement.ANGLE_LIMIT_TOLERANCE
        for joint, (min_angle, max_angle) in angle_limits.items():
            for angle in (min_angle - 0.5 * tolerance, max_angle + 0.5 * tolerance):
                angles = resting_angles.copy()
                angles[joint] = angle
                check_move(standing_still(angles), test_config, angle_limits=angle_limits)

    def test_angle_past_the_tolerance_is_refused(self):
        tolerance = cartesian_movement.ANGLE_LIMIT_TOLERANCE
        for joint, (min_angle, max_angle) in angle_limits.items():
            for angle, direction in ((min_angle - 10 * tolerance, 'below'), (max_angle + 10 * tolerance, 'above')):
                angles = resting_angles.copy()
                angles[joint] = angle
                with self.assertRaisesRegex(MovementException, 'joint {} would move .* {} its range'
                                            .format(joint, direction)):
                    check_move(standing_still(angles), test_config, angle_limits=angle_limits)

    def test_any_step_past_the_angle_limits_is_refused(self):
        angles = np.tile(resting_angles, (5, 1))
        angles[:, 3] = pi / 2 - 0.01
        angles[2, 3] = pi / 2 + 0.01
        with self.assertRaisesRegex(MovementException, 'joint 3 would move .* above its range'):
            check_move(PlannedMove('one step too far', angles, 0.05), test_config, angle_limits=angle_limits)

    def test_only_joints_with_limits_are_checked(self):
        angles = resting_angles.copy()
        angles[1] = -1.0
        check_move(standing_still(angles), test_config, angle_limits={2: angle_limits[2]})

    def test_end_effector_too_far_from_the_base_is_refused(self):
        # reachable, but 58.3cm from the base
        with self.assertRaisesRegex(MovementException, 'the end effector .* from the base, outside of'):
            check_move(single_step_move(Pose(0, 50, 30)), test_config, TestWorkspaceLimits)

    def test_end_effector_too_close_to_the_base_is_refused(self):
        class LargeMinimumRadius(TestWorkspaceLimits):
            radius_min = 40

        # 37.4cm from the base
        with self.assertRaisesRegex(MovementException, 'the end effector .* from the base, outside of'):
            check_move(single_step_move(start_pose), test_config, LargeMinimumRadius)

    def test_end_effector_below_the_workspace_is_refused(self):
        with self.assertRaisesRegex(MovementException, 'the end effector would move .* below the minimum height'):
            check_move(single_step_move(Pose(30, 22, -6)), test_config, TestWorkspaceLimits)

    def test_end_effector_closer_than_y_min_is_refused(self):
        with self.assertRaisesRegex(MovementException, 'the end effector would move .* closer than y='):
            check_move(single_step_move(Pose(30, 12, 4)), test_config, TestWorkspaceLimits)

    def test_wrist_below_the_workspace_is_refused(self):
        # the end effector points up, so the wrist is d6 = 12cm below it
        with self.assertRaisesRegex(MovementException, 'the wrist would move to .* below the minimum height'):
            check_move(single_step_move(Pose(0, 30, 2, gamma=pi / 2)), test_config, TestWorkspaceLimits)

    def test_pose_within_the_workspace_is_allowed(self):
        check_move(single_step_move(start_pose), test_config, TestWorkspaceLimits)
        # the wrist at z=-4
        check_move(single_step_move(Pose(0, 30, 8, gamma=pi / 2)), test_config, TestWorkspaceLimits)

    def test_workspace_is_not_checked_without_limits(self):
        check_move(single_step_move(Pose(30, 22, -10)), test_config, workspace_limits=None)

    def test_straight_line_between_differently_flipped_poses_is_refused(self):
        with self.assertRaisesRegex(MovementException, 'flipped differently'):
            plan_straight_line(Pose(30, 22, 4), Pose(30, 22, 10, flip=True), test_config)


class FakeClock:

    def __init__(self):
        self.time = 0.0
        self.sleeps = []

    def perf_counter(self):
        return self.time

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        if seconds < 0:
            raise ValueError('sleep length must be non-negative')
        self.time += seconds


class ExecuteMoveTests(unittest.TestCase):

    def test_every_step_is_sent_to_the_robot(self):
        move = plan_move_to_pose(resting_angles, start_pose, test_config)
        robot = mock.Mock()
        with mock.patch.object(cartesian_movement, 'sleep'):
            execute_move(robot, move)

        self.assertEqual(len(move.angles), robot.move_servos.call_count)
        for expected, sent in zip(move.angles, robot.move_servos.call_args_list):
            np.testing.assert_allclose(expected, sent[0][0])

    def test_returns_true_when_the_whole_move_is_executed(self):
        should_stop = mock.Mock(return_value=False)
        with mock.patch.object(cartesian_movement, 'sleep'):
            self.assertTrue(execute_move(mock.Mock(), standing_still(resting_angles, 5), should_stop=should_stop))
        self.assertEqual(5, should_stop.call_count)

    def test_stops_when_should_stop_returns_true(self):
        robot = mock.Mock()
        # checked before every step, this stops before the 4th step
        should_stop = mock.Mock(side_effect=[False, False, False, True])
        with mock.patch.object(cartesian_movement, 'sleep'):
            self.assertFalse(execute_move(robot, standing_still(resting_angles, 10), should_stop=should_stop))
        self.assertEqual(3, robot.move_servos.call_count)

    def run_with_fake_clock(self, move, time_to_send_a_step):
        clock = FakeClock()
        robot = mock.Mock()
        robot.move_servos.side_effect = lambda _: clock.sleep(time_to_send_a_step)
        with mock.patch.object(cartesian_movement, 'perf_counter', clock.perf_counter), \
                mock.patch.object(cartesian_movement, 'sleep', clock.sleep):
            execute_move(robot, move)
        return clock

    def test_steps_are_sent_at_steps_per_second(self):
        # the time it takes to send a step should not add up over the move
        steps = 50
        clock = self.run_with_fake_clock(standing_still(resting_angles, steps), time_to_send_a_step=0.003)
        self.assertAlmostEqual(steps / global_constants.steps_per_second, clock.time)

    def test_slow_robot_does_not_cause_a_negative_sleep(self):
        steps = 50
        dt = 1.0 / global_constants.steps_per_second
        clock = self.run_with_fake_clock(standing_still(resting_angles, steps), time_to_send_a_step=1.5 * dt)
        self.assertAlmostEqual(steps * 1.5 * dt, clock.time)


if __name__ == '__main__':
    unittest.main()
