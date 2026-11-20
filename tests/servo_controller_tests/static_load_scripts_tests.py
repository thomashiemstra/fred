import argparse
import contextlib
import csv
import inspect
import io
import os
import re
import tempfile
import unittest
from unittest import mock

import numpy as np
import yaml
from numpy import pi

from src.robot_controllers.combined_robot.combined_robot import CombinedRobot
from src.global_constants import WorkSpaceLimits, dynamixel_robot_config
from src.kinematics.kinematics import forward_position_kinematics, inverse_kinematics
from src.kinematics.kinematics_utils import RobotConfig
from src.robot_controllers.dynamixel_robot.calibration import record_static_load, fit_static_load, tune_servo_gains
from src.robot_controllers.dynamixel_robot.calibration.tune_servo_gains import parse_pose
from src.utils import workspace
from src.utils.movement import MAX_POSITION_ERROR, plan_move_to_angles
from src.robot_controllers.dynamixel_robot.servo_config import load_servo_config, parse_servo_config
from src.robot_controllers.dynamixel_robot.calibration.load_directions import DOWN, UP
from src.robot_controllers.dynamixel_robot.static_compensation import regressor_row, term_names

TEST_SERVO_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_servo_config.yaml')
test_config = RobotConfig(d1=15.7, a2=20, d4=22, d6=12)
angle_limits = {1: (0, pi), 2: (0, pi), 3: (-pi / 2, pi / 2), 4: (-pi, pi), 5: (-pi, pi), 6: (-pi, pi)}


# the angles of joints 1 to 6 of the arm lying on the table with the torque off (13 November 2026)
RESTING_ANGLES = np.radians([0, 91.8, 19.1, -0.3, 1.4, 76.8, -8.7])


class TestLimits:
    """workspace limits of these tests, independent of global_constants.WorkSpaceLimits"""
    radius_min = 10
    wrist_reach_max = 41.8
    y_min = 12.1
    z_min = -5
    wrist_z_min = -5


class FakeRobot:
    """a robot that sags by a known amount: present = goal + sag(angles), with the servo objects of the test yaml"""

    def __init__(self, angles, sag=None, workspace_limits=TestLimits):
        self.robot_config = test_config
        self.workspace_limits = workspace_limits
        self.servo_config = load_servo_config(TEST_SERVO_CONFIG_PATH)
        self.servo_handler = FakeHandler(self.servo_config)
        self.angles = np.array(angles, dtype=np.float64)
        self.sag = sag if sag is not None else (lambda servo_id, angles: 0)
        self.moves = []
        self.hardware_errors = {}
        self.temperatures = {servo_id: 30 for servo_id in range(1, 7)}
        self.disabled = False
        # added to the present position and the present pwm of every servo, one entry per reading (they cycle)
        self.position_jitter = [0]
        self.pwm_jitter = [0]
        self.position_reads = 0
        self.pwm_reads = 0

    def get_current_angles(self):
        return self.angles.copy()

    def get_angle_limits(self):
        return angle_limits

    def workspace_violation(self, angles):
        return workspace.workspace_violation(angles, self.robot_config, self.workspace_limits)

    def check_workspace(self, angles):
        workspace.check_workspace(angles, self.robot_config, self.workspace_limits)

    def move_servos(self, angles):
        self.angles = np.array(angles, dtype=np.float64)
        self.moves.append(self.angles.copy())
        for servo_id in range(1, 7):
            self.servo_handler.get_servo(servo_id).set_target_position_from_angle(angles[servo_id], angles)

    def get_current_positions(self):
        jitter = self.position_jitter[self.position_reads % len(self.position_jitter)]
        self.position_reads += 1
        positions = np.zeros(7, dtype=np.int_)
        for servo_id in range(1, 7):
            positions[servo_id] = (self.servo_handler.get_servo(servo_id).target_position
                                   + self.sag(servo_id, self.angles) + jitter)
        return positions

    def get_present_currents(self):
        return {servo_id: 10 * servo_id for servo_id in range(1, 7)}

    def get_present_pwms(self):
        """the servo drives with P / 128 * (goal sent - present), and present is the goal sent plus the sag"""
        jitter = self.pwm_jitter[self.pwm_reads % len(self.pwm_jitter)]
        self.pwm_reads += 1
        return {servo_id: int(round(-self.sag(servo_id, self.angles) * self.servo_handler.get_servo(servo_id).p / 128))
                + jitter for servo_id in range(1, 7)}

    def get_temperatures_and_voltages(self):
        return {servo_id: (self.temperatures[servo_id], 11.9) for servo_id in range(1, 7)}

    def get_hardware_errors(self):
        return self.hardware_errors

    def disable_servos(self):
        self.disabled = True


class FakeHandler:
    def __init__(self, servo_config):
        self.servos = servo_config.servos

    def get_servo(self, servo_id):
        return self.servos[servo_id]


def gravity_sag(servo_id, angles):
    """a sag like the real one: joints 2 and 3 sag with the cosines, the error is present - goal"""
    phi = angles[2] + angles[3]
    if servo_id == 2:
        return int(round(-12 * np.cos(angles[2]) - 5 * np.sin(phi)))
    if servo_id == 3:
        return int(round(30 * np.sin(phi)))  # servo 3 is mounted reversed, its error has the other sign
    return 0


class RecordTests(unittest.TestCase):

    def test_sampling_ranges(self):
        ranges = record_static_load.sampling_ranges(angle_limits, (2, 3, 5), margin=0.1)
        self.assertEqual((0.1, pi - 0.1), ranges[2])
        self.assertEqual((-pi / 2 + 0.1, pi / 2 - 0.1), ranges[3])
        self.assertEqual((-pi / 2, pi / 2), ranges[5], 'the wrist bend is narrowed by default')
        ranges = record_static_load.sampling_ranges(angle_limits, (3, 5), 0.1, overrides={5: (-0.5, 2.0), 3: (0.0, 3.0)})
        self.assertEqual((-0.5, pi / 2), ranges[5])
        self.assertEqual((0.0, pi / 2 - 0.1), ranges[3])
        with self.assertRaises(ValueError):
            record_static_load.sampling_ranges(angle_limits, (3,), 0.1, overrides={3: (2.0, 1.0)})

    def test_sampled_poses_stay_inside_the_ranges(self):
        rng = np.random.default_rng(0)
        ranges = record_static_load.sampling_ranges(angle_limits, (2, 3, 5), margin=0.1)
        poses = record_static_load.sample_poses(50, ranges, pi / 2, rng)
        self.assertEqual(50, len(poses))
        for pose in poses:
            self.assertEqual(pi / 2, pose[1])
            self.assertEqual(0.0, pose[4])
            self.assertEqual(0.0, pose[6])
            for joint, (low, high) in ranges.items():
                self.assertGreaterEqual(pose[joint], low)
                self.assertLessEqual(pose[joint], high)

    def test_sampled_poses_are_filtered_and_the_sampling_gives_up(self):
        rng = np.random.default_rng(0)
        ranges = record_static_load.sampling_ranges(angle_limits, (2, 3), margin=0.1)
        poses = record_static_load.sample_poses(10, ranges, pi / 2, rng, accept=lambda pose: pose[2] < 1.0)
        self.assertEqual(10, len(poses))
        self.assertTrue(all(pose[2] < 1.0 for pose in poses))
        self.assertEqual([], record_static_load.sample_poses(10, ranges, pi / 2, rng, accept=lambda pose: False))

    def test_limits_with_floor(self):
        limits = record_static_load.limits_with_floor(3.0, TestLimits)
        self.assertEqual(3.0, limits.z_min)
        self.assertEqual(3.0, limits.wrist_z_min, 'the wrist floor is raised along')
        self.assertEqual(TestLimits.wrist_reach_max, limits.wrist_reach_max)

        class HighWrist(TestLimits):
            wrist_z_min = 6
        self.assertEqual(6, record_static_load.limits_with_floor(3.0, HighWrist).wrist_z_min, 'but never lowered')

    def test_poses_below_the_floor_or_outside_the_workspace_are_not_accepted(self):
        accepted = lambda pose, floor=-5.0: record_static_load.pose_is_accepted(
            np.array(pose), FakeRobot(pose, workspace_limits=record_static_load.limits_with_floor(floor, TestLimits)),
            np.radians(5), (2, 3))
        self.assertTrue(accepted([0, pi / 2, 0.8, 0.0, 0, 0, 0]))
        # the gripper tip ends 3.4 cm above the table and the lower overshoot dips to 0.7 cm: allowed with the
        # floor 5 cm below the table or at 0, not at 3 cm above it
        self.assertTrue(accepted([0, pi / 2, 0.0, 1.2, 0, 0, 0]))
        self.assertTrue(accepted([0, pi / 2, 0.0, 1.2, 0, 0, 0], floor=0.0))
        self.assertFalse(accepted([0, pi / 2, 0.0, 1.2, 0, 0, 0], floor=3.0))
        # the same for a pose 6 cm up whose lower overshoot dips to 1.2 cm
        self.assertTrue(accepted([0, pi / 2, 0.8, 0.0, 0, 0, 0], floor=1.0))
        self.assertFalse(accepted([0, pi / 2, 0.8, 0.0, 0, 0, 0], floor=3.0))
        # a pose whose lower overshoot goes 5.1 cm below the table is refused even with the lowest floor
        self.assertFalse(accepted([0, pi / 2, 0.0, 1.0, 0, 0, 0]))
        # the upper arm folded back puts the gripper behind the base
        self.assertFalse(accepted([0, pi / 2, -0.3, -1.2, 0, 0, 0]))
        # a pose at the end of a joint range is refused for the overshoot that leaves the workspace, not for the
        # overshoot that is clipped to the range (the elbow straightened towards 90 degrees takes the wrist past its
        # reach)
        self.assertFalse(accepted([0, pi / 2, 0.8, pi / 2 - np.radians(2), 0, 0, 0]))

    def test_range_arguments(self):
        ranges = record_static_load.parse_ranges(['5=-45:45', '3=0:90'])
        self.assertAlmostEqual(-pi / 4, ranges[5][0])
        self.assertAlmostEqual(pi / 2, ranges[3][1])
        with self.assertRaises(Exception):
            record_static_load.parse_ranges(['5=45'])

    def test_approach_sequence_overshoots_both_ways_within_the_range(self):
        pose = np.array([0, pi / 2, 0.05, 1.0, 0, 0, 0])
        sequence = record_static_load.approach_sequence(pose, 0.1, (2, 3), angle_limits)
        (over_down, target_down, down), (over_up, target_up, up) = sequence
        self.assertEqual((DOWN, UP), (down, up))
        self.assertAlmostEqual(0.15, over_down[2])
        self.assertAlmostEqual(1.1, over_down[3])
        self.assertAlmostEqual(0.0, over_up[2])  # clipped at the range
        self.assertAlmostEqual(0.9, over_up[3])
        np.testing.assert_array_equal(pose, target_down)
        np.testing.assert_array_equal(pose, target_up)

    def test_current_units(self):
        self.assertEqual((269.0, 'mA'), record_static_load.current_in_units(100, 'XM430-W350'))
        self.assertEqual((100.0, 'load_0.1%'), record_static_load.current_in_units(100, 'XC430-W240'))
        self.assertEqual((None, '?'), record_static_load.current_in_units(100, None))

    def record(self, robot, poses, workspace_limits=TestLimits, **kwargs):
        robot.workspace_limits = workspace_limits
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=record_static_load.FIELDS)
        writer.writeheader()
        done = record_static_load.record(robot, poses, writer, robot.servo_config.models, 45, joints=(2, 3),
                                         delta=np.radians(5), settle=0.0, sleep=lambda seconds: None,
                                         clock=lambda: 1.0, **kwargs)
        return done, list(csv.DictReader(io.StringIO(output.getvalue())))

    def test_records_both_directions_of_every_pose(self):
        robot = FakeRobot([0, pi / 2, 0.4, 0.8, 0, 0, 0], gravity_sag)
        poses = [np.array([0, pi / 2, 0.8, 0.0, 0, 0, 0]), np.array([0, pi / 2, 1.0, -0.5, 0, 0, 0])]
        done, rows = self.record(robot, poses)
        self.assertEqual(2, done)
        self.assertEqual(2 * 2 * 6, len(rows))  # 2 poses, 2 directions, 6 servos
        self.assertEqual({'0', '1'}, {row['pose'] for row in rows})
        self.assertEqual({UP, DOWN}, {row['direction'] for row in rows})
        for row in rows:
            pose = poses[int(row['pose'])]
            self.assertAlmostEqual(float(row['present_position']) - int(row['goal_position']), float(row['error']),
                                   places=2)
            self.assertAlmostEqual(gravity_sag(int(row['servo']), pose), float(row['error']), places=2)
            self.assertAlmostEqual(pose[int(row['servo'])], float(row['angle']), places=6)
            self.assertEqual('30', row['temperature'])
        # the moves approached every pose from above and from below
        targets = [tuple(np.round(move[2:4], 3)) for move in robot.moves]
        self.assertIn(tuple(np.round([0.8 + np.radians(5), 0.0 + np.radians(5)], 3)), targets)
        self.assertIn(tuple(np.round([0.8 - np.radians(5), 0.0 - np.radians(5)], 3)), targets)
        self.assertEqual(tuple(np.round([1.0, -0.5], 3)), targets[-1])
        xm_rows = [row for row in rows if row['servo'] == '2']
        self.assertEqual('mA', xm_rows[0]['current_unit'])
        self.assertEqual('{:.2f}'.format(20 * 2.69), xm_rows[0]['current'])
        self.assertEqual('load_0.1%', [row for row in rows if row['servo'] == '4'][0]['current_unit'])

    def test_the_rows_log_the_gains_the_mode_and_the_measured_pwm(self):
        robot = FakeRobot([0, pi / 2, 0.4, 0.8, 0, 0, 0], gravity_sag)
        pose = np.array([0, pi / 2, 0.8, 0.0, 0, 0, 0])
        _, rows = self.record(robot, [pose])
        self.assertEqual(('mode', 'speed', 'samples', 'p', 'i', 'd', 'pwm'), record_static_load.FIELDS[-7:])
        self.assertEqual(set(record_static_load.FIELDS), set(rows[0]))
        for row in rows:
            servo_id = int(row['servo'])
            servo = robot.servo_handler.get_servo(servo_id)
            self.assertEqual('static', row['mode'])
            self.assertEqual('', row['speed'])
            self.assertEqual('5', row['samples'], 'the default is 5 readings')
            self.assertEqual((str(servo.p), str(servo.i), str(servo.d)), (row['p'], row['i'], row['d']))
            # the servo drives with P / 128 per position of error, against the sag
            self.assertAlmostEqual(-gravity_sag(servo_id, pose) * servo.p / 128, float(row['pwm']), delta=0.5)
        self.assertGreater(len({row['p'] for row in rows}), 1, 'every servo has its own gains')

    def test_a_row_is_the_mean_of_the_readings(self):
        robot = FakeRobot([0, pi / 2, 0.4, 0.8, 0, 0, 0], gravity_sag)
        robot.position_jitter = [0, 1, 1, 1]  # a mean of 0.75
        robot.pwm_jitter = [0, 10, 20, 21]  # a mean of 12.75
        pose = np.array([0, pi / 2, 0.8, 0.0, 0, 0, 0])
        _, rows = self.record(robot, [pose], reading_count=4)
        self.assertEqual(12, len(rows))
        for row in rows:
            servo = robot.servo_handler.get_servo(int(row['servo']))
            sag = gravity_sag(int(row['servo']), pose)
            self.assertEqual('4', row['samples'])
            self.assertEqual('{:.2f}'.format(servo.target_position + sag + 0.75), row['present_position'])
            self.assertEqual('{:.2f}'.format(sag + 0.75), row['error'])
            self.assertEqual('{:.1f}'.format(round(-sag * servo.p / 128) + 12.75), row['pwm'])
        # 4 readings of each of the 2 approaches, every reading asks for the positions and the pwms once
        self.assertEqual(8, robot.position_reads)
        self.assertEqual(8, robot.pwm_reads)

    def test_readings_wait_between_the_readings(self):
        robot = FakeRobot([0, pi / 2, 0.4, 0.8, 0, 0, 0], gravity_sag)
        pauses = []
        rows = record_static_load.readings(robot, {}, 0, UP, np.zeros(7), 1.0, 5, 0.05, pauses.append)
        self.assertEqual([0.05] * 4, pauses)
        self.assertEqual(6, len(rows))
        self.assertEqual({5}, {row['samples'] for row in rows})
        self.assertEqual({''}, {row['current'] for row in rows}, 'no model: no current')
        pauses.clear()
        rows = record_static_load.readings(robot, {}, 0, UP, np.zeros(7), 1.0, 1, 0.05, pauses.append)
        self.assertEqual([], pauses)
        self.assertEqual({1}, {row['samples'] for row in rows})
        with self.assertRaises(ValueError):
            record_static_load.readings(robot, {}, 0, UP, np.zeros(7), 1.0, 0, 0.05, pauses.append)

    def test_the_pause_between_the_readings_goes_through_the_sleep_of_record(self):
        robot = FakeRobot([0, pi / 2, 0.4, 0.8, 0, 0, 0], gravity_sag)
        pauses = []
        robot.workspace_limits = TestLimits
        writer = csv.DictWriter(io.StringIO(), fieldnames=record_static_load.FIELDS)
        record_static_load.record(robot, [np.array([0, pi / 2, 0.8, 0.0, 0, 0, 0])], writer, {}, 45, joints=(2, 3),
                                  delta=np.radians(5), settle=7.0, sleep=pauses.append, clock=lambda: 1.0,
                                  reading_count=3, reading_pause=0.02)
        self.assertEqual([7.0, 0.02, 0.02] * 2, pauses, 'the settle, then 2 pauses, for each of the 2 approaches')

    def test_a_failed_read_gives_no_rows(self):
        for failing in ('get_current_positions', 'get_present_currents', 'get_present_pwms',
                        'get_temperatures_and_voltages'):
            with self.subTest(failing=failing):
                robot = FakeRobot([0, pi / 2, 0.4, 0.8, 0, 0, 0], gravity_sag)
                original = getattr(robot, failing)
                calls = []

                def flaky(original=original, calls=calls):
                    calls.append(1)
                    return None if len(calls) == 3 else original()  # the third of five readings fails
                setattr(robot, failing, flaky)
                self.assertIsNone(record_static_load.readings(robot, {}, 0, UP, np.zeros(7), 1.0, 5, 0.0,
                                                              lambda seconds: None))

                robot = FakeRobot([0, pi / 2, 0.4, 0.8, 0, 0, 0], gravity_sag)
                setattr(robot, failing, lambda: None)
                with self.assertRaises(record_static_load.StopRecording):
                    self.record(robot, [np.array([0, pi / 2, 0.8, 0.0, 0, 0, 0])])

    def test_the_readings_option_is_a_whole_number_of_at_least_one(self):
        self.assertEqual(3, record_static_load.positive_int('3'))
        for value in ('0', '-2', '1.5', 'many'):
            with self.assertRaises(argparse.ArgumentTypeError, msg=value):
                record_static_load.positive_int(value)

    def test_a_pose_the_planner_refuses_is_skipped(self):
        robot = FakeRobot([0, pi / 2, 0.4, 0.8, 0, 0, 0], gravity_sag)
        # the upper arm folded back under the lower arm puts the gripper behind the base, outside the workspace
        poses = [np.array([0, pi / 2, -0.3, -1.2, 0, 0, 0]), np.array([0, pi / 2, 0.8, 0.0, 0, 0, 0])]
        done, rows = self.record(robot, poses)
        self.assertEqual(1, done)
        self.assertEqual({'1'}, {row['pose'] for row in rows})

    def test_stops_when_a_servo_is_too_warm_or_broken(self):
        robot = FakeRobot([0, pi / 2, 0.4, 0.8, 0, 0, 0], gravity_sag)
        robot.temperatures[2] = 46
        with self.assertRaises(record_static_load.StopRecording):
            self.record(robot, [np.array([0, pi / 2, 0.8, 0.0, 0, 0, 0])])
        self.assertEqual([], robot.moves)
        robot.temperatures[2] = 30
        robot.hardware_errors = {3: ['overload']}
        with self.assertRaises(record_static_load.StopRecording):
            self.record(robot, [np.array([0, pi / 2, 0.8, 0.0, 0, 0, 0])])

    def test_servos_are_checked_before_every_approach(self):
        robot = FakeRobot([0, pi / 2, 0.4, 0.8, 0, 0, 0], gravity_sag)
        checks = []
        original = robot.get_hardware_errors

        def warming_up():
            checks.append(len(robot.moves))
            if len(checks) == 2:
                robot.temperatures[3] = 50
            return original()
        robot.get_hardware_errors = warming_up
        with self.assertRaises(record_static_load.StopRecording):
            self.record(robot, [np.array([0, pi / 2, 0.8, 0.0, 0, 0, 0])])
        self.assertEqual(2, len(checks), 'checked before the first and the second approach')

    def test_a_failed_status_read_is_tried_again(self):
        robot = FakeRobot([0, pi / 2, 0.4, 0.8, 0, 0, 0])
        original = robot.get_temperatures_and_voltages
        results = [None, None]  # two corrupt status packets, then the read works

        def flaky():
            return results.pop(0) if results else original()
        robot.get_temperatures_and_voltages = flaky
        record_static_load.check_servos(robot, 45)
        self.assertEqual([], results)

        robot.get_temperatures_and_voltages = lambda: None
        with self.assertRaisesRegex(record_static_load.StopRecording,
                                    'failed to read the temperatures {} times in a row'
                                    .format(record_static_load.READ_ATTEMPTS)):
            record_static_load.check_servos(robot, 45)

    def test_moves_are_checked_against_the_floor(self):
        robot = FakeRobot([0, pi / 2, 0.4, 0.8, 0, 0, 0], gravity_sag)
        low_pose = np.array([0, pi / 2, 0.0, 1.2, 0, 0, 0])  # the gripper tip 3.4 cm up, its lower overshoot 0.7
        done, _ = self.record(robot, [low_pose], workspace_limits=record_static_load.limits_with_floor(3.0, TestLimits))
        self.assertEqual(0, done)
        done, _ = self.record(robot, [low_pose], workspace_limits=record_static_load.limits_with_floor(0.0, TestLimits))
        self.assertEqual(1, done)


class RestRobot(FakeRobot):
    """the fake robot with the link lengths and the workspace limits of the real one, recording what it is told"""

    def __init__(self, angles=RESTING_ANGLES, **kwargs):
        super().__init__(angles, **kwargs)
        self.robot_config = dynamixel_robot_config
        self.workspace_limits = WorkSpaceLimits
        self.events = []

    def move_servos(self, angles):
        super().move_servos(angles)
        self.events.append('move')

    def disable_servos(self):
        super().disable_servos()
        self.events.append('disable')


class LowerToRestTests(unittest.TestCase):

    def setUp(self):
        patcher = mock.patch('src.utils.movement.sleep')  # the planned moves run at their real speed otherwise
        patcher.start()
        self.addCleanup(patcher.stop)
        self.rest_angles = inverse_kinematics(record_static_load.REST_POSE, dynamixel_robot_config)

    def lower(self, robot, prompt=None):
        """lower_and_disable_torque with the prompt answered by prompt (default enter); returns (result, prompts)"""
        prompts = []

        def answer(text):
            prompts.append((text, robot.disabled))  # whether the torque was already off when the question came
            if isinstance(prompt, BaseException):
                raise prompt
            return ''
        with mock.patch('builtins.input', side_effect=answer), mock.patch('builtins.print'):
            result = record_static_load.lower_and_disable_torque(
                robot, sleep=lambda seconds: robot.events.append('settle'))
        return result, prompts

    def assertPrompted(self, result, prompts):
        self.assertFalse(result)
        self.assertEqual(1, len(prompts))
        self.assertFalse(prompts[0][1], 'the torque was off before the prompt was answered')

    def test_the_rest_pose_is_inside_the_limits_and_above_the_table(self):
        robot = RestRobot()
        robot.check_workspace(self.rest_angles)
        _, _, _, wrist, tip = forward_position_kinematics(self.rest_angles, dynamixel_robot_config)
        self.assertGreaterEqual(wrist[2], WorkSpaceLimits.wrist_z_min + 0.5)
        self.assertGreaterEqual(tip[2], WorkSpaceLimits.z_min + 0.5)
        rest = record_static_load.REST_POSE
        np.testing.assert_allclose([rest.x, rest.y, rest.z], tip, atol=MAX_POSITION_ERROR)

    def test_the_planner_accepts_the_move_to_the_rest_pose_from_the_tuning_poses_and_from_rest(self):
        robot = RestRobot()
        starts = {name: parse_pose(name) for name in tune_servo_gains.POSES}
        starts['resting'] = np.radians(np.array([0, 91.8, 19.1, -0.3, 1.4, 76.8, -8.7]))
        for name, start in starts.items():
            with self.subTest(start=name):
                move = plan_move_to_angles(robot, start, self.rest_angles, angular_speed=record_static_load.REST_SPEED)
                np.testing.assert_allclose(self.rest_angles, move.angles[-1], atol=1e-9)

    def test_the_arm_is_lowered_and_only_then_the_torque_goes_off(self):
        robot = RestRobot(parse_pose('elbow_out'))
        result, prompts = self.lower(robot)
        self.assertTrue(result)
        self.assertEqual([], prompts, 'input is not called')
        self.assertTrue(robot.disabled)
        self.assertGreater(len(robot.moves), 1)
        self.assertEqual(['disable'], robot.events[-1:])
        self.assertEqual(['settle'], robot.events[-2:-1], 'it settles before the torque goes off')
        self.assertEqual({'move'}, set(robot.events[:-2]))
        np.testing.assert_allclose(self.rest_angles, robot.moves[-1], atol=1e-6)

    def test_a_servo_with_a_hardware_error_keeps_the_others_where_they_are(self):
        robot = RestRobot()
        robot.hardware_errors = {3: ['overload']}
        result, prompts = self.lower(robot)
        self.assertPrompted(result, prompts)
        self.assertEqual([], robot.moves)
        self.assertEqual(['disable'], robot.events, 'only after the prompt was answered')

    def test_hardware_errors_that_can_not_be_read_do_not_move_the_arm(self):
        robot = RestRobot()
        robot.get_hardware_errors = lambda: None
        result, prompts = self.lower(robot)
        self.assertPrompted(result, prompts)
        self.assertEqual([], robot.moves)

    def test_a_move_the_planner_refuses_ends_at_the_prompt(self):
        robot = RestRobot(parse_pose('elbow_out'))
        robot.workspace_limits = record_static_load.limits_with_floor(10)  # the rest pose is below 10 cm
        result, prompts = self.lower(robot)
        self.assertPrompted(result, prompts)
        self.assertEqual([], robot.moves)

    def test_a_pose_the_arm_can_not_reach_ends_at_the_prompt(self):
        robot = RestRobot(parse_pose('elbow_out'))
        robot.robot_config = RobotConfig(d1=15.7, a2=5, d4=5, d6=14)
        result, prompts = self.lower(robot)
        self.assertPrompted(result, prompts)
        self.assertEqual([], robot.moves)

    def test_angles_that_can_not_be_read_end_at_the_prompt(self):
        robot = RestRobot()
        robot.get_current_angles = lambda: None
        result, prompts = self.lower(robot)
        self.assertPrompted(result, prompts)
        self.assertEqual([], robot.moves)

    def test_ctrl_c_during_the_move_ends_at_the_prompt_with_the_servos_holding(self):
        robot = RestRobot(parse_pose('elbow_out'))
        original = robot.move_servos

        def interrupted(angles):
            if len(robot.moves) == 20:
                raise KeyboardInterrupt
            original(angles)
        robot.move_servos = interrupted
        result, prompts = self.lower(robot)
        self.assertPrompted(result, prompts)
        self.assertEqual(20, len(robot.moves))

    def test_ctrl_c_during_the_settle_ends_at_the_prompt(self):
        robot = RestRobot(parse_pose('elbow_out'))

        def interrupted(seconds):
            raise KeyboardInterrupt
        prompts = []

        def answer(text):
            prompts.append(robot.disabled)
            return ''
        with mock.patch('builtins.input', side_effect=answer), mock.patch('builtins.print'):
            result = record_static_load.lower_and_disable_torque(robot, sleep=interrupted)
        self.assertFalse(result)
        self.assertEqual([False], prompts, 'the torque was on when the prompt came')

    def test_ctrl_c_at_the_prompt_leaves_the_torque_on(self):
        robot = RestRobot()
        robot.hardware_errors = {3: ['overload']}
        result, prompts = self.lower(robot, prompt=KeyboardInterrupt())
        self.assertPrompted(result, prompts)
        self.assertFalse(robot.disabled)
        robot = RestRobot()
        with mock.patch('builtins.input', side_effect=KeyboardInterrupt), mock.patch('builtins.print'):
            record_static_load.disable_torque_on_request(robot)
        self.assertFalse(robot.disabled)


class MainTests(unittest.TestCase):
    """the end of record_static_load.main, with the robot, the poses and the recording replaced"""

    def run_main(self, argv=(), record=None, answer='yes'):
        """returns (result, mocks), the mocks by the name of the function of record_static_load they replace"""
        robot = mock.MagicMock(spec=CombinedRobot)
        robot.workspace_limits = TestLimits
        robot.servo_config = mock.MagicMock()
        robot.get_angle_limits = lambda: angle_limits
        robot.enable_servos = lambda: True
        names = {'get_robot': mock.Mock(return_value=robot),
                 'sample_poses': mock.Mock(return_value=[np.array([0, pi / 2, 0.8, 0.0, 0, 0, 0])]),
                 'record': record if record is not None else mock.Mock(return_value=1),
                 'lower_and_disable_torque': mock.Mock(return_value=True),
                 'disable_torque_on_request': mock.Mock()}
        with tempfile.TemporaryDirectory() as directory, mock.patch('builtins.print'), \
                mock.patch('builtins.input', return_value=answer), \
                mock.patch.multiple(record_static_load, **names):
            result = record_static_load.main(['--output', os.path.join(directory, 'static_load.csv'), *argv])
        names['robot'] = robot
        return result, names

    def test_the_arm_is_lowered_at_the_end(self):
        result, mocks = self.run_main()
        self.assertEqual(0, result)
        mocks['lower_and_disable_torque'].assert_called_once_with(mocks['robot'])
        mocks['disable_torque_on_request'].assert_not_called()

    def test_the_arm_is_lowered_after_ctrl_c_and_after_a_stop(self):
        for exception in (KeyboardInterrupt(), record_static_load.StopRecording('a servo is too warm')):
            with self.subTest(exception=type(exception).__name__):
                result, mocks = self.run_main(record=mock.Mock(side_effect=exception))
                self.assertEqual(0, result)
                mocks['lower_and_disable_torque'].assert_called_once_with(mocks['robot'])
                mocks['disable_torque_on_request'].assert_not_called()

    def test_nothing_is_lowered_when_the_recording_was_not_started(self):
        result, mocks = self.run_main(answer='no')
        self.assertEqual(0, result)
        mocks['record'].assert_not_called()
        mocks['lower_and_disable_torque'].assert_not_called()

    def test_the_readings_option_reaches_the_recording(self):
        _, mocks = self.run_main()
        self.assertEqual(record_static_load.DEFAULT_READINGS, mocks['record'].call_args.kwargs['reading_count'])
        self.assertEqual(5, record_static_load.DEFAULT_READINGS)
        _, mocks = self.run_main(['--readings', '3'])
        self.assertEqual(3, mocks['record'].call_args.kwargs['reading_count'])
        with self.assertRaises(SystemExit), mock.patch('sys.stderr'):
            self.run_main(['--readings', '0'])


class FitTests(unittest.TestCase):

    def setUp(self):
        self.true = {2: {'constant': 1.0, 'cos_q2': -12.0, 'sin_q2': 1.5, 'sin_phi': -5.0, 'cos_phi': 0.8,
                         'wrist_reach': -2.0},
                     3: {'constant': -2.0, 'sin_phi': 30.0, 'cos_phi': -3.0, 'wrist_reach': 6.0},
                     5: {'constant': 0.0, 'wrist_pitch': 8.0}}

    def synthetic_rows(self, count, rng, hysteresis=4.0, noise=0.7, gains=(128.0,), compensation=False,
                       pwm_scale=None, mode='static'):
        """
        rows like record_static_load writes, from the true coefficients (the holding drive in pwm units) plus
        hysteresis and noise. The servo drives with P / 128 * (goal sent - present), so the recorded error is the
        compensation less the drive in positions: error = compensation - drive * 128 / P.
        :param gains: the P gains the poses are recorded at, one is drawn per pose
        :param compensation: record with a compensation of up to 12 positions added to the goal
        :param pwm_scale: the measured pwm is this times the drive (None: the pwm column is blank)
        """
        compensations = np.random.default_rng(99)  # apart from rng, so the same rng gives the same drives
        rows = []
        for pose in range(count):
            angles = np.zeros(7)
            angles[1] = pi / 2
            angles[2], angles[3] = rng.uniform(0.2, 2.9), rng.uniform(-1.3, 1.3)
            angles[4], angles[5] = rng.uniform(-1.0, 1.0), rng.uniform(-1.5, 1.5)
            p = float(rng.choice(gains))
            for direction, sign in ((UP, 1), (DOWN, -1)):
                for servo_id in (2, 3, 5):
                    names = tuple(self.true[servo_id])
                    static = float(regressor_row(servo_id, angles, names) @ np.array(
                        [self.true[servo_id][name] for name in names]))
                    drive = static + sign * hysteresis + rng.normal(0, noise)
                    added = int(compensations.integers(-12, 13)) if compensation else 0
                    error = added - drive * 128 / p
                    rows.append(self.row(pose, direction, servo_id, angles[servo_id], p, error, added, mode=mode,
                                         pwm='' if pwm_scale is None else '{:.3f}'.format(pwm_scale * drive)))
            # the angles of the joints that are not varied are still in the rows of their servos
            for direction in (UP, DOWN):
                for servo_id in (1, 4, 6):
                    rows.append(self.row(pose, direction, servo_id, angles[servo_id], p, 0.0, 0, mode=mode))
        return rows

    @staticmethod
    def row(pose, direction, servo_id, angle, p, error, added, mode='static', pwm=''):
        """a row of record_static_load, with the error as a number or None for a blank one"""
        return {'time': '1.0', 'pose': str(pose), 'direction': direction, 'servo': str(servo_id),
                'angle': '{:.6f}'.format(angle), 'goal_position': '2048', 'compensation': str(added),
                'present_position': '2048' if error is None else '{:.2f}'.format(2048 + error),
                'error': '' if error is None else '{:.4f}'.format(error), 'current_raw': '0', 'current': '',
                'current_unit': 'mA', 'temperature': '30', 'voltage': '11.9', 'mode': mode, 'speed': '',
                'samples': '5', 'p': '' if p is None else '{:g}'.format(p), 'i': '0', 'd': '0', 'pwm': pwm}

    def test_samples_average_the_directions(self):
        rows = self.synthetic_rows(3, np.random.default_rng(0), hysteresis=4.0, noise=0.0)
        samples = fit_static_load.samples_from_rows(rows)
        self.assertEqual(3, len(samples))
        for sample in samples:
            self.assertAlmostEqual(4.0, sample.hysteresis[2], places=3)
            self.assertFalse(hasattr(sample, 'by_direction'), 'the fit only has the mean and the friction')
            self.assertAlmostEqual(pi / 2, sample.angles[1], places=5)
        # a pose without both directions is left out
        samples = fit_static_load.samples_from_rows([row for row in rows if not (row['pose'] == '1' and row['direction'] == UP)])
        self.assertEqual(2, len(samples))

    def test_fit_recovers_the_coefficients(self):
        rng = np.random.default_rng(2)
        samples = fit_static_load.samples_from_rows(self.synthetic_rows(300, rng))
        for joint in (2, 3, 5):
            fit = fit_static_load.fit_joint(joint, samples, holdout=0.2, rng=rng)
            self.assertEqual(tuple(self.true[joint]), fit.names)
            # the coefficients of a 7 term fit on noisy data scatter, what matters is the prediction
            for name, value in self.true[joint].items():
                self.assertAlmostEqual(value, fit.coefficients[name], delta=2.5, msg='joint {} {}'.format(joint, name))
            self.assertLess(fit.rms_train, 1.2)
            self.assertLess(fit.rms_holdout, 1.5)
            self.assertEqual(240, fit.train_count)
            self.assertEqual(60, fit.holdout_count)
            self.assertLess(fit.condition, 300, 'joint {}'.format(joint))

    def test_standard_errors_and_warnings(self):
        rng = np.random.default_rng(8)
        samples = fit_static_load.samples_from_rows(self.synthetic_rows(300, rng))
        fit = fit_static_load.fit_joint(5, samples, rng=rng)
        # the constant of joint 5 is 0 in the data: not determined; wrist_pitch is 8 +- a few hundredths
        self.assertIn('constant', fit.undetermined())
        self.assertNotIn('wrist_pitch', fit.undetermined())
        self.assertLess(fit.standard_errors['wrist_pitch'], 0.2)
        self.assertEqual(['not determined by the data (smaller than twice their standard error): constant'],
                         fit.warnings())
        self.assertIn('<- not determined', fit_static_load.describe(fit))
        # a regressor whose terms can't be told apart is warned about: fit a term twice
        fit = fit_static_load.fit_joint(3, samples, rng=rng, names=('sin_phi', 'cos_phi', 'cos_phi'))
        self.assertTrue(any('condition number' in warning for warning in fit.warnings()))

    def test_fit_uses_all_terms_of_the_joint_by_default(self):
        samples = fit_static_load.samples_from_rows(self.synthetic_rows(60, np.random.default_rng(3)))
        self.assertEqual(term_names(3), fit_static_load.fit_joint(3, samples).names)

    def test_too_few_samples(self):
        samples = fit_static_load.samples_from_rows(self.synthetic_rows(3, np.random.default_rng(5)))
        with self.assertRaises(ValueError):
            fit_static_load.fit_joint(2, samples)

    def test_yaml_block_parses_as_a_servo_config(self):
        rng = np.random.default_rng(6)
        p_old, p_new = 256, 512
        samples = fit_static_load.samples_from_rows(self.synthetic_rows(300, rng, gains=(p_old,)))
        fits = {joint: fit_static_load.fit_joint(joint, samples, rng=rng) for joint in (2, 3, 5)}
        block = fit_static_load.yaml_block(fits, 60)
        self.assertIn('  2:\n    compensation:\n      max_positions: 60\n      terms: {constant: ', block)
        self.assertNotIn('up:', block)
        self.assertNotIn('down:', block)

        # paste the blocks into the test configuration: the compensation of every servo is parsed
        with open(TEST_SERVO_CONFIG_PATH) as config_file:
            data = yaml.safe_load(config_file)
        for servo_id, servo_data in yaml.safe_load(block).items():
            data['servos'][servo_id].update(servo_data)
        config = parse_servo_config(data)
        self.assertEqual(60, config.servos[2].compensation.max_positions)
        # the coefficients are the fitted holding drive as they are, not negated
        self.assertAlmostEqual(fits[3].coefficients['sin_phi'], config.servos[3].compensation.terms['sin_phi'], places=2)
        self.assertAlmostEqual(self.true[3]['sin_phi'], config.servos[3].compensation.terms['sin_phi'], delta=2.5)
        self.assertIsNone(config.servos[4].compensation)

        for sample in samples[:20]:
            for joint in (2, 3, 5):
                compensation = config.servos[joint].compensation
                # the error recorded at P_old: the drive in positions, against the sag
                recorded_error = -sample.static[joint] * 128 / p_old
                at_old = compensation.positions(sample.angles, p_old)
                # the compensation cancels the error it was fitted to: added to the goal, the servo ends up on it
                self.assertAlmostEqual(0.0, recorded_error + at_old, delta=2.0,
                                       msg='joint {} at {}'.format(joint, np.round(sample.angles, 2)))
                # at a higher P the same drive needs fewer positions: scaled by P_old / P_new
                self.assertAlmostEqual(at_old * p_old / p_new, compensation.positions(sample.angles, p_new), delta=1.0)

        # the servos convert with the P they run with: the same drive at another P gives other positions
        sample = samples[0]
        for joint in (2, 3, 5):
            servo = config.servos[joint]
            servo.p = p_old
            servo.set_target_position_from_angle(sample.angles[joint], sample.angles)
            at_old = servo.compensation_positions
            self.assertEqual(servo.compensation.positions(sample.angles, p_old), at_old)
            servo.p = p_new
            servo.set_target_position_from_angle(sample.angles[joint], sample.angles)
            self.assertAlmostEqual(at_old * p_old / p_new, servo.compensation_positions, delta=1.0)

    def test_recorded_error_is_against_the_uncompensated_goal(self):
        from src.robot_controllers.dynamixel_robot.static_compensation import StaticCompensation
        robot = FakeRobot([0, pi / 2, 0.4, 0.8, 0, 0, 0], gravity_sag)
        servo = robot.servo_handler.get_servo(2)
        servo.compensation = StaticCompensation(2, {'cos_q2': 7.0 * servo.p / 128}, 60)  # 7 positions at its P
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=record_static_load.FIELDS)
        writer.writeheader()
        pose = np.array([0, pi / 2, 0.0, 1.2, 0, 0, 0])
        done = record_static_load.record(robot, [pose], writer, robot.servo_config.models, 45, joints=(2, 3),
                                         delta=np.radians(5), settle=0.0, sleep=lambda seconds: None,
                                         clock=lambda: 1.0)
        self.assertEqual(1, done)
        rows = [row for row in csv.DictReader(io.StringIO(output.getvalue())) if row['servo'] == '2']
        self.assertEqual(2, len(rows))
        for row in rows:
            self.assertEqual('7', row['compensation'])  # cos(0) * 7
            self.assertAlmostEqual(float(row['present_position']) - int(row['goal_position']), float(row['error']))
            # the fake servo lands on goal + compensation + sag: against the uncompensated goal that is both
            self.assertAlmostEqual(gravity_sag(2, pose) + 7, float(row['error']))
            # the whole load, which the fit works on, is the error less the compensation
            self.assertAlmostEqual(gravity_sag(2, pose), float(row['error']) - int(row['compensation']))

    def run_main(self, rows, *args):
        """fit_static_load.main on the rows written to a csv file, returns (result, the printed text)"""
        with tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False, newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=record_static_load.FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        printed = []
        try:
            with mock.patch('builtins.print', lambda *args, **kwargs: printed.append(' '.join(str(a) for a in args))):
                result = fit_static_load.main([csv_file.name, '--joints', '2', '3', '5', *args])
        finally:
            os.remove(csv_file.name)
        return result, '\n'.join(printed)

    def test_main_prints_the_block(self):
        rows = self.synthetic_rows(80, np.random.default_rng(7))
        result, output = self.run_main(rows)
        self.assertEqual(0, result)
        self.assertIn('joint 2:', output)
        self.assertIn('held out', output)
        self.assertIn('compensation:', output)
        self.assertNotIn('up: {', output)
        self.assertNotIn('down: {', output)
        self.assertNotIn('measured pwm', output, 'no pwm in the rows: no cross-check')

    def test_a_row_is_converted_to_the_drive_with_its_own_p_and_compensation(self):
        # the servo drives with P / 128 * (goal sent - present) and goal sent - present = -(error - compensation)
        rows = [self.row(0, UP, 2, 1.0, 640, -8.0, 0), self.row(0, DOWN, 2, 1.0, 640, -4.0, 0),
                self.row(1, UP, 2, 1.0, 256, 3.0, 5), self.row(1, DOWN, 2, 1.0, 256, 5.0, 5)]
        first, second = fit_static_load.samples_from_rows(rows)
        # the load pushes towards lower positions: 8 * 640 / 128 = 40 going up and 4 * 640 / 128 = 20 going down
        self.assertAlmostEqual(6 * 5, first.static[2])
        self.assertAlmostEqual(2 * 5, first.hysteresis[2])
        # a compensation bigger than the load: 4 going up and 0 going down
        self.assertAlmostEqual(2.0, second.static[2])
        self.assertAlmostEqual(2.0, second.hysteresis[2])

    def test_a_compensation_that_was_active_in_the_recording_gives_the_same_fit(self):
        plain = fit_static_load.samples_from_rows(self.synthetic_rows(120, np.random.default_rng(10)))
        compensated_rows = self.synthetic_rows(120, np.random.default_rng(10), compensation=True)
        self.assertGreater(len({row['compensation'] for row in compensated_rows}), 3)
        compensated = fit_static_load.samples_from_rows(compensated_rows)
        for joint in (2, 3, 5):
            for a, b in zip(plain, compensated):
                self.assertAlmostEqual(a.static[joint], b.static[joint], places=2)
                self.assertAlmostEqual(a.hysteresis[joint], b.hysteresis[joint], places=2)
            fit_plain = fit_static_load.fit_joint(joint, plain)
            fit_compensated = fit_static_load.fit_joint(joint, compensated)
            for name, value in fit_plain.coefficients.items():
                self.assertAlmostEqual(value, fit_compensated.coefficients[name], places=2)
        # fitting the error alone, as the fit did before the compensation column was used, is a different fit
        ignored = [{**row, 'compensation': '0'} for row in compensated_rows]
        wrong = fit_static_load.fit_joint(2, fit_static_load.samples_from_rows(ignored))
        self.assertGreater(wrong.rms_train, 2 * fit_static_load.fit_joint(2, plain).rms_train)

    def test_rows_with_several_gains_are_fitted_with_their_own_p(self):
        gains = (500.0, 1000.0, 2000.0)
        rows = self.synthetic_rows(200, np.random.default_rng(11), noise=0.0, gains=gains)
        self.assertEqual({'500', '1000', '2000'}, {row['p'] for row in rows if row['servo'] == '2'})
        samples = fit_static_load.samples_from_rows(rows)
        for joint in (2, 3, 5):
            fit = fit_static_load.fit_joint(joint, samples, holdout=0.0)
            for name, value in self.true[joint].items():
                self.assertAlmostEqual(value, fit.coefficients[name], delta=0.05, msg='joint {} {}'.format(joint, name))
            self.assertLess(fit.rms_train, 0.05)
        # converting everything with one of the gains instead would not fit
        flat = [{**row, 'p': '1000'} for row in rows]
        wrong = fit_static_load.fit_joint(2, fit_static_load.samples_from_rows(flat), holdout=0.0)
        self.assertGreater(wrong.rms_train, 1.0)

    def test_the_report_gives_the_errors_in_positions_and_degrees_at_the_p_of_the_joint(self):
        rows = self.synthetic_rows(80, np.random.default_rng(12), gains=(1000.0,))
        result, output = self.run_main(rows)
        self.assertEqual(0, result)
        self.assertRegex(output, r'rms error on the fitted poses [\d.]+ pwm \([\d.]+ positions, [\d.]+ degrees at P 1000\)')
        self.assertRegex(output, r'friction \(half the up/down difference, not compensated\): mean [\d.]+ pwm '
                                 r'\([\d.]+ positions, [\d.]+ degrees at P 1000\), max [\d.]+ pwm \([\d.]+ positions')
        self.assertNotIn('P values seen', output)
        # 7.8125 pwm at P 1000 are 1.0 position
        self.assertEqual(' (1.0 positions, 0.09 degrees at P 1000)', fit_static_load.in_positions(7.8125, 1000))
        self.assertEqual('', fit_static_load.in_positions(7.8125, None))

    def test_the_report_converts_with_the_mean_p_when_there_are_several(self):
        rows = self.synthetic_rows(80, np.random.default_rng(13), gains=(500.0, 1500.0))
        result, output = self.run_main(rows)
        self.assertEqual(0, result)
        self.assertIn('P values seen: 500, 1500, the errors in positions use their mean', output)
        self.assertRegex(output, r'degrees at P 1012\.5\)')

    def test_a_row_without_a_p_is_an_error_naming_the_file(self):
        for p in (None, '', '0', 'x'):
            with self.subTest(p=p):
                rows = self.synthetic_rows(3, np.random.default_rng(14))
                for row in rows:
                    row['file'] = 'old_recording.csv'
                rows[5]['p'] = p
                with self.assertRaisesRegex(ValueError, r'old_recording\.csv.*no usable p.*before the gains were logged'):
                    fit_static_load.samples_from_rows(rows)
        # a csv from before the p column
        rows = self.synthetic_rows(60, np.random.default_rng(14))
        with tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False, newline='') as csv_file:
            fields = [name for name in record_static_load.FIELDS if name != 'p']
            writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        printed = []
        try:
            with mock.patch('builtins.print', lambda *args, **kwargs: printed.append(' '.join(str(a) for a in args))):
                result = fit_static_load.main([csv_file.name])
        finally:
            os.remove(csv_file.name)
        self.assertEqual(1, result)
        self.assertIn(os.path.basename(csv_file.name), printed[0])
        self.assertIn('before the gains were logged', printed[0])

    def test_a_blank_error_gives_only_the_angle_of_the_pose(self):
        rows = self.synthetic_rows(30, np.random.default_rng(15), noise=0.0)
        angle_of_joint_2 = {row['pose']: float(row['angle']) for row in rows if row['servo'] == '2'}
        reference = fit_static_load.samples_from_rows(rows)
        # servo 2 is not measured in pose 3 (no error, no p), and in pose 4 only in one direction
        blanked = []
        for row in rows:
            row = dict(row)
            if row['servo'] == '2' and (row['pose'] == '3' or (row['pose'] == '4' and row['direction'] == UP)):
                row['error'], row['p'], row['present_position'] = '', '', ''
            blanked.append(row)
        samples = fit_static_load.samples_from_rows(blanked)
        self.assertEqual(len(reference), len(samples), 'the poses stay, they have the other joints')
        self.assertEqual(2, sum(2 not in sample.static for sample in samples), 'joint 2 has no sample at 2 poses')
        self.assertEqual(2, sum(2 not in sample.hysteresis for sample in samples))
        self.assertTrue(all(5 in sample.static and 3 in sample.static for sample in samples))
        # the angle of the blank rows is still in the pose
        self.assertAlmostEqual(angle_of_joint_2['3'], samples[3].angles[2], places=5)
        self.assertAlmostEqual(angle_of_joint_2['4'], samples[4].angles[2], places=5)
        # the fit of the other joints is unchanged, the fit of joint 2 uses the 28 poses it has
        fit_3 = fit_static_load.fit_joint(3, samples, holdout=0.0)
        self.assertAlmostEqual(fit_static_load.fit_joint(3, reference, holdout=0.0).coefficients['sin_phi'],
                               fit_3.coefficients['sin_phi'], places=6)
        self.assertEqual(28, fit_static_load.fit_joint(2, samples, holdout=0.0).train_count)
        # a pose in which no servo has an error in both directions gives no sample
        nothing = [dict(row, error='', p='') for row in rows if row['pose'] == '0']
        self.assertEqual([], fit_static_load.samples_from_rows(nothing))

    def test_the_measured_pwm_is_compared_to_the_drive_from_the_position_error(self):
        rows = self.synthetic_rows(60, np.random.default_rng(16), gains=(800.0,), pwm_scale=1.03)
        samples = fit_static_load.samples_from_rows(rows)
        for joint in (2, 3, 5):
            self.assertAlmostEqual(1.03, fit_static_load.pwm_ratio(joint, samples), places=2)
        result, output = self.run_main(rows)
        self.assertEqual(0, result)
        self.assertEqual(3, output.count('measured pwm is 1.03 times the drive computed from the position error '
                                         '(1.0 expected)'))
        # without pwm in the rows, or without any drive, there is no ratio
        without = fit_static_load.samples_from_rows(self.synthetic_rows(10, np.random.default_rng(16)))
        self.assertIsNone(fit_static_load.pwm_ratio(2, without))
        self.assertIsNone(fit_static_load.pwm_ratio(2, []))
        zero = fit_static_load.samples_from_rows(
            [self.row(0, direction, 2, 1.0, 1000, 0.0, 0, pwm='5') for direction in (UP, DOWN)])
        self.assertIsNone(fit_static_load.pwm_ratio(2, zero))

    def test_a_pwm_that_is_only_in_one_direction_is_not_used(self):
        rows = self.synthetic_rows(10, np.random.default_rng(17), pwm_scale=1.0)
        for row in rows:
            if row['direction'] == UP:
                row['pwm'] = ''
        samples = fit_static_load.samples_from_rows(rows)
        self.assertIsNone(fit_static_load.pwm_ratio(2, samples))

    def test_the_fit_has_no_per_direction_blocks_whatever_the_recording(self):
        for mode in ('static', 'sweep'):
            rows = self.synthetic_rows(80, np.random.default_rng(18), mode=mode)
            result, output = self.run_main(rows)
            self.assertEqual(0, result)
            self.assertNotIn('up: {', output, mode)
            self.assertNotIn('down: {', output, mode)
            self.assertNotIn('overcompensate', output, mode)
            self.assertIn('friction (half the up/down difference, not compensated)', output, mode)
        for name in ('split_warning', 'SWEEP'):
            self.assertFalse(hasattr(fit_static_load, name), name)

    def test_split_is_not_an_option_any_more(self):
        rows = self.synthetic_rows(20, np.random.default_rng(18))
        with contextlib.redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit) as raised:
            self.run_main(rows, '--split')
        self.assertEqual(2, raised.exception.code)
        self.assertIn('--split', errors.getvalue())
        self.assertNotIn('direction', inspect.signature(fit_static_load.fit_joint).parameters)
        self.assertNotIn('directional', inspect.signature(fit_static_load.yaml_block).parameters)

    def test_the_directions_are_the_labels_of_the_csv_and_not_in_the_runtime_module(self):
        from src.robot_controllers.dynamixel_robot import static_compensation
        from src.robot_controllers.dynamixel_robot.calibration import load_directions
        self.assertEqual(('up', 'down'), (load_directions.UP, load_directions.DOWN))
        self.assertIs(load_directions.UP, record_static_load.UP)
        self.assertFalse(hasattr(static_compensation, 'UP'))
        # the fit is offline: it must not import the recorders, which pull in the robot and the hardware libraries
        self.assertIsNone(re.search(r'^(from|import) .*record_', inspect.getsource(fit_static_load), re.MULTILINE))

    def test_the_yaml_block_has_the_coefficients_as_fitted(self):
        rng = np.random.default_rng(19)
        samples = fit_static_load.samples_from_rows(self.synthetic_rows(100, rng, noise=0.0))
        fit = fit_static_load.fit_joint(5, samples, holdout=0.0)
        self.assertAlmostEqual(8.0, fit.coefficients['wrist_pitch'], places=2, msg='the drive of the true model')
        block = yaml.safe_load(fit_static_load.yaml_block({5: fit}, 40))
        self.assertEqual(40, block[5]['compensation']['max_positions'])
        self.assertAlmostEqual(fit.coefficients['wrist_pitch'], block[5]['compensation']['terms']['wrist_pitch'], places=3)
        self.assertGreater(block[5]['compensation']['terms']['wrist_pitch'], 0)
        self.assertEqual('{constant: 1.000, b: -2.500}', fit_static_load._terms_line({'constant': 1.0, 'b': -2.5}))


if __name__ == '__main__':
    unittest.main()
