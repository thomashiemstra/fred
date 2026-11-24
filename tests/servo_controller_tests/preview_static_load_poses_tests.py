import contextlib
import io
import os
import re
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from numpy import pi

from src import global_constants
from src.global_constants import WorkSpaceLimits, dynamixel_robot_config
from src.kinematics.kinematics import forward_position_kinematics
from src.robot_controllers.combined_robot.combined_robot import CombinedRobot
from src.robot_controllers.dynamixel_robot import dynamixel_utils
from src.robot_controllers.dynamixel_robot.calibration import record_static_load
from src.robot_controllers.dynamixel_robot.calibration.offline import preview_static_load_poses as preview
from src.robot_controllers.dynamixel_robot.calibration.offline.preview_static_load_poses import (
    NoHardwareError, OfflineRobot, PoseHeights)
from src.robot_controllers.dynamixel_robot.dynamixel_robot_controller import DynamixelRobotController
from src.robot_controllers.dynamixel_robot.servo_config import load_servo_config
from src.utils import workspace
from src.utils.movement import PlannedMove, check_move
from src.utils.movement_exception import MovementException
from tests.servo_controller_tests.static_load_scripts_tests import RestRobot, FakeHandler

# the command line of a recording on the real robot on 21 November 2026
RECORDING = ['--poses', '30', '--joints', '2', '3', '4', '5', '--range', '2=45:115', '3=-60:60', '4=-60:60',
             '5=-60:60', '--speed', '0.3', '--seed', '1']
D1, A2, D4, D6 = (dynamixel_robot_config.d1, dynamixel_robot_config.a2, dynamixel_robot_config.d4,
                  dynamixel_robot_config.d6)


def run_preview(argv):
    """returns (return code, what it printed)"""
    output = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
        result = preview.main(argv)
    return result, output.getvalue()


def previewed_poses(argv):
    """the poses the tool lists for the options, as the 7 element angle arrays"""
    with mock.patch.object(preview, 'print_report') as report, contextlib.redirect_stdout(io.StringIO()):
        result = preview.main(argv)
    assert result == 0, result
    return report.call_args.args[2]


def angles(q2, q3=0.0, q4=0.0, q5=0.0, q1=pi / 2):
    """the 7 element angle array, in radians"""
    return np.array([0, q1, q2, q3, q4, q5, 0.0])


class OfflineRobotTests(unittest.TestCase):

    def setUp(self):
        self.robot = OfflineRobot()

    def test_it_has_the_link_lengths_the_workspace_limits_and_the_joint_ranges_of_the_real_robot(self):
        self.assertIs(dynamixel_robot_config, self.robot.robot_config)
        self.assertIs(WorkSpaceLimits, self.robot.workspace_limits)
        servo_config = load_servo_config()
        self.assertEqual(set(range(1, 7)), set(self.robot.get_angle_limits()))
        for joint, (low, high) in self.robot.get_angle_limits().items():
            servo = servo_config.servos[joint]
            self.assertEqual((min(servo.min_angle, servo.max_angle), max(servo.min_angle, servo.max_angle)),
                             (low, high))

    def test_the_joint_ranges_are_those_of_the_real_robots_own_method(self):
        # the controller of the real robot computes them from its servo handler; the same servos give the same ranges
        servo_config = load_servo_config()
        real = DynamixelRobotController.get_angle_limits(SimpleNamespace(servo_handler=FakeHandler(servo_config)))
        self.assertEqual(real, OfflineRobot(servo_config=servo_config).get_angle_limits())

    def test_it_refuses_a_pose_below_the_floor_like_the_fake_robot_with_the_real_limits(self):
        # the forearm hangs down from the elbow at the height of the shoulder: the tip is 19 cm below the table
        low = angles(0.0)
        fake = RestRobot()
        self.assertIsNotNone(self.robot.workspace_violation(low))
        self.assertEqual(fake.workspace_violation(low), self.robot.workspace_violation(low))
        for robot in (self.robot, fake):
            with self.assertRaisesRegex(MovementException, 'below the minimum height'):
                check_move(PlannedMove('pose', low[np.newaxis], 0.0), robot)

    def test_it_accepts_what_the_fake_robot_with_the_real_limits_accepts(self):
        fake = RestRobot()
        rng = np.random.default_rng(3)
        accepted = []
        for _ in range(300):
            pose = np.zeros(7)
            pose[1] = pi / 2
            pose[2:6] = rng.uniform([0.3, -1.2, -1.5, -1.5], [3.0, 1.2, 1.5, 1.5])
            accepted.append(record_static_load.pose_is_accepted(pose, self.robot, np.radians(5), (2, 3, 4, 5)))
            self.assertEqual(record_static_load.pose_is_accepted(pose, fake, np.radians(5), (2, 3, 4, 5)),
                             accepted[-1])
        # a test with only accepted or only refused poses would prove nothing
        self.assertTrue(any(accepted) and not all(accepted))

    def test_it_refuses_a_joint_out_of_its_range(self):
        low, high = self.robot.get_angle_limits()[3]
        for angle in (high + 0.5, low - 0.5):
            pose = angles(pi / 2, angle)
            with self.assertRaisesRegex(MovementException, 'joint 3'):
                check_move(PlannedMove('pose', pose[np.newaxis], 0.0), self.robot)

    def test_it_refuses_the_workspace_by_the_limits_it_is_given(self):
        pose = angles(pi / 2, 0.0)  # the arm stretched forward, the forearm hanging down, the tip about 30 cm up
        self.assertIsNone(self.robot.workspace_violation(pose))
        self.robot.workspace_limits = record_static_load.limits_with_floor(100.0)
        self.assertIn('below the minimum height', self.robot.workspace_violation(pose))

    def test_whatever_needs_the_hardware_raises(self):
        for call in (lambda: self.robot.move_servos(angles(1.0)), self.robot.enable_servos,
                     self.robot.get_current_angles, lambda: self.robot.set_gripper(0.0)):
            with self.assertRaises(NoHardwareError):
                call()


class HeightTests(unittest.TestCase):

    def heights(self, pose, delta=np.radians(5), joints=(2,), limits=None):
        limits = limits if limits is not None else {2: (0.0, pi), 3: (-pi / 2, pi / 2), 4: (-pi, pi), 5: (-pi, pi)}
        return preview.pose_heights(7, pose, delta, joints, limits)

    def test_the_heights_of_a_hand_checked_pose(self):
        # joint 3 bent 40 degrees from the upper arm 50 degrees above the horizontal, the gripper straight on the
        # forearm: the elbow is a2 sin(q2) above the shoulder, the wrist d4 along the forearm, which stands at q2 + q3
        # from the vertical, below the elbow, and the tip d6 further on
        q2, q3 = np.radians(50), np.radians(40)
        report = self.heights(angles(q2, q3))
        self.assertEqual(7, report.number)
        self.assertAlmostEqual(D1 + A2 * np.sin(q2), report.elbow[0])
        self.assertAlmostEqual(D1 + A2 * np.sin(q2) - D4 * np.cos(q2 + q3), report.wrist[0])
        self.assertAlmostEqual(D1 + A2 * np.sin(q2) - (D4 + D6) * np.cos(q2 + q3), report.tip[0])

    def test_the_arm_hanging_straight_down_from_the_shoulder_height(self):
        report = self.heights(angles(0.0), joints=())
        self.assertAlmostEqual(D1, report.elbow[0])
        self.assertAlmostEqual(D1 - D4, report.wrist[0])
        self.assertAlmostEqual(D1 - D4 - D6, report.tip[0])

    def test_the_stops_are_the_pose_the_overshoot_above_and_the_overshoot_below(self):
        delta = np.radians(5)
        q2, q3 = np.radians(30), np.radians(20)
        report = self.heights(angles(q2, q3), delta, joints=(2, 3))
        for stop, sign in enumerate((0, 1, -1)):
            tip = D1 + A2 * np.sin(q2 + sign * delta) - (D4 + D6) * np.cos(q2 + q3 + 2 * sign * delta)
            self.assertAlmostEqual(tip, report.tip[stop], msg=preview.STOPS[stop])

    def test_the_overshoots_are_clipped_to_the_joint_range_like_the_recorders(self):
        delta = np.radians(5)
        pose = angles(np.radians(2), np.radians(88))
        limits = {2: (0.0, pi), 3: (-pi / 2, pi / 2), 4: (-pi, pi), 5: (-pi, pi)}
        report = self.heights(pose, delta, joints=(2, 3), limits=limits)
        # joint 2 can not go below 0 and joint 3 not above 90 degrees: the stops are what approach_sequence says
        (above, _, _), (below, _, _) = record_static_load.approach_sequence(pose, delta, (2, 3), limits)
        self.assertAlmostEqual(pi / 2, above[3])
        self.assertAlmostEqual(0.0, below[2])
        for stop, overshoot in enumerate((pose, above, below)):
            self.assertAlmostEqual(float(forward_position_kinematics(overshoot, dynamixel_robot_config)[-1][2]),
                                   report.tip[stop])
        self.assertAlmostEqual(
            D1 + A2 * np.sin(0.0) - (D4 + D6) * np.cos(0.0 + np.radians(88 - 5)), report.tip[2])

    def test_only_the_varied_joints_get_an_overshoot(self):
        pose = angles(np.radians(60), np.radians(10))
        report = self.heights(pose, joints=(3,))
        self.assertAlmostEqual(report.elbow[0], report.elbow[1])  # joint 2 stays
        self.assertAlmostEqual(report.elbow[0], report.elbow[2])
        self.assertNotAlmostEqual(report.tip[0], report.tip[1])

    def test_the_lowest_of_the_planners_heights_is_the_wrist_or_the_tip_not_the_elbow(self):
        report = PoseHeights(1, angles(1.0), elbow=(0.5, 9, 9), wrist=(5.0, 4.0, 6.0), tip=(7.0, 3.0, 8.0))
        self.assertEqual(3.0, report.lowest)

    def test_the_lowest_height_names_its_pose_and_stop(self):
        reports = [PoseHeights(1, angles(1.0), (9, 9, 9), (5.0, 6.0, 7.0), (8.0, 9.0, 10.0)),
                   PoseHeights(2, angles(1.0), (9, 9, 9), (4.0, 6.0, 7.0), (8.0, 1.5, 10.0)),
                   PoseHeights(3, angles(1.0), (9, 9, 9), (5.0, 3.5, 7.0), (8.0, 9.0, 10.0))]
        self.assertEqual((1.5, 2, preview.STOPS[1]), preview.lowest(reports, 'tip'))
        self.assertEqual((3.5, 3, preview.STOPS[1]), preview.lowest(reports, 'wrist'))

    def test_a_pose_is_marked_when_the_wrist_or_the_tip_is_under_the_warning_height_at_any_stop(self):
        ok = PoseHeights(1, angles(1.0), (9, 9, 9), (5.0, 6.0, 7.0), (8.0, 9.0, 10.0))
        low_tip = PoseHeights(2, angles(1.0), (9, 9, 9), (5.0, 6.0, 7.0), (8.0, 9.0, 1.9))
        low_wrist = PoseHeights(3, angles(1.0), (9, 9, 9), (5.0, 1.9, 7.0), (8.0, 9.0, 10.0))
        low_elbow = PoseHeights(4, angles(1.0), (1.0, 1.0, 1.0), (5.0, 6.0, 7.0), (8.0, 9.0, 10.0))
        exactly = PoseHeights(5, angles(1.0), (9, 9, 9), (2.0, 6.0, 7.0), (8.0, 9.0, 10.0))
        self.assertNotIn('LOW', preview.format_pose(ok, 2.0))
        self.assertIn('LOW', preview.format_pose(low_tip, 2.0))
        self.assertIn('LOW', preview.format_pose(low_wrist, 2.0))
        self.assertNotIn('LOW', preview.format_pose(low_elbow, 2.0))
        self.assertNotIn('LOW', preview.format_pose(exactly, 2.0), 'the warning is for heights under --warn')
        self.assertIn('LOW', preview.format_pose(exactly, 2.5))


class PreviewTests(unittest.TestCase):

    def test_it_lists_the_poses_of_a_recording_with_the_same_options_and_seed(self):
        poses = previewed_poses(RECORDING)
        self.assertEqual(30, len(poses))
        # the recorder's own main, with its robot replaced by the same limits and its recording by a capture
        offline = OfflineRobot()
        robot = mock.MagicMock(spec=CombinedRobot)
        robot.robot_config = offline.robot_config
        robot.workspace_limits = offline.workspace_limits
        robot.get_angle_limits = offline.get_angle_limits
        robot.check_workspace = lambda a: workspace.check_workspace(a, robot.robot_config, robot.workspace_limits)
        robot.enable_servos = mock.Mock(return_value=True)
        robot.servo_config = mock.MagicMock()
        record = mock.Mock(return_value=0)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.multiple(record_static_load, get_robot=mock.Mock(return_value=robot), record=record,
                                    lower_and_disable_torque=mock.Mock(return_value=True)):
            record_static_load.main(['--yes', '--output', os.path.join(directory, 'out.csv'), *RECORDING])
        recorded = record.call_args.args[1]
        self.assertEqual(len(recorded), len(poses))
        for expected, actual in zip(recorded, poses):
            np.testing.assert_array_equal(expected, actual)

    def test_the_poses_of_the_real_recording_are_listed_with_their_heights(self):
        # poses 1, 3 and 6 in degrees as the tool lists them with the floor at 2 cm. The recordings of 21 to 24 November
        # 2026 had the floor at -2 cm and visited other poses (a pose 3 with the tip at 0.2 cm among them): the poses
        # the floor refuses now are skipped, and the ones that followed move up
        poses = previewed_poses(RECORDING)
        for number, expected in ((1, [66.8, -9.2, 39.3, -10.9]), (3, [97.5, -26.4, -1.8, 57.7]),
                                 (6, [103.8, 1.1, 1.3, 30.4])):
            np.testing.assert_allclose(expected, np.degrees(poses[number - 1][2:6]), atol=0.06)
            self.assertEqual(90.0, np.degrees(poses[number - 1][1]))
        result, output = run_preview(RECORDING)
        self.assertEqual(0, result)
        pose_line = re.compile(r'^ *(\d+) {2,}\d.*\|.*\|.*$', re.M)  # not the summary lines that start with a number
        lines = {int(match.group(1)): match.group(0) for match in pose_line.finditer(output)}
        self.assertEqual(list(range(1, 31)), sorted(lines))
        # pose 24 has the lowest tip of the recording: at the pose, above and below, 4.0 cm is above the 3 cm of --warn
        self.assertRegex(lines[24], r'\|\s+7\.8\s+12\.5\s+4\.0$')
        self.assertNotIn('<-- LOW', output)

    def test_without_a_seed_it_says_that_a_recording_will_differ(self):
        _, output = run_preview(['--poses', '3'])
        self.assertIn('no --seed', output)
        _, output = run_preview(['--poses', '3', '--seed', '4'])
        self.assertNotIn('no --seed', output)
        np.testing.assert_array_equal(previewed_poses(['--poses', '3', '--seed', '4'])[0],
                                      previewed_poses(['--poses', '3', '--seed', '4'])[0])

    def test_a_different_seed_gives_different_poses(self):
        self.assertFalse(np.array_equal(previewed_poses(['--poses', '3', '--seed', '1'])[0],
                                        previewed_poses(['--poses', '3', '--seed', '2'])[0]))

    def test_the_output_has_the_options_the_limits_and_the_summary(self):
        result, output = run_preview(RECORDING)
        self.assertEqual(0, result)
        self.assertIn('joints [2, 3, 4, 5]', output)
        self.assertIn('2=45:115', output)
        self.assertIn('seed 1', output)
        self.assertIn('floor {} cm (wrist {} cm)'.format(WorkSpaceLimits.z_min, WorkSpaceLimits.wrist_z_min), output)
        self.assertIn('joint 2: 45 to 115 degrees', output)
        self.assertIn('30 poses of 30 asked', output)
        self.assertRegex(output, r'lowest gripper tip: 4\.0 cm, pose 24, at the overshoot below')
        self.assertRegex(output, r'lowest wrist centre: 14\.6 cm, pose 26, at the overshoot below')
        self.assertRegex(output, r'lowest elbow: \d+\.\d cm, pose \d+, ')
        self.assertIn('0 poses with the tip or the wrist under 3 cm', output)
        self.assertIn('does not check the gripper fingers or the elbow', output)
        self.assertIn('WorkSpaceLimits', output)
        self.assertIn('the closed fingers reach 1 cm beyond it', output)
        self.assertNotIn('just below the table', output)

    def test_the_default_warning_height_is_1_cm_above_the_limit_of_the_tip(self):
        self.assertEqual(WorkSpaceLimits.z_min + 1.0, preview.DEFAULT_WARN)
        self.assertEqual(preview.DEFAULT_WARN, preview.build_parser().parse_args([]).warn)

    def test_the_summary_of_hand_made_poses(self):
        # joint 2 at 10 degrees with the forearm down: the tip is low, and lowest at the overshoot below
        low = angles(np.radians(10), 0.0)
        high = angles(np.radians(90), np.radians(90))
        with mock.patch.object(record_static_load, 'select_poses',
                               return_value=([high, low], {2: (0.1, 3.0)})):
            result, output = run_preview(['--poses', '5', '--joints', '2', '--warn', '5'])
        self.assertEqual(0, result)
        self.assertIn('2 poses of 5 asked', output)
        below = D1 + A2 * np.sin(np.radians(5)) - (D4 + D6) * np.cos(np.radians(5))
        self.assertIn('lowest gripper tip: {:.1f} cm, pose 2, at the overshoot below (-delta)'.format(below), output)
        wrist = D1 + A2 * np.sin(np.radians(5)) - D4 * np.cos(np.radians(5))
        self.assertIn('lowest wrist centre: {:.1f} cm, pose 2, at the overshoot below (-delta)'.format(wrist), output)
        self.assertIn('lowest elbow: {:.1f} cm, pose 2'.format(D1 + A2 * np.sin(np.radians(5))), output)
        self.assertIn('1 poses with the tip or the wrist under 5 cm', output)
        self.assertEqual(1, output.count('<-- LOW'))
        self.assertRegex(output, r'(?m)^   2 .*<-- LOW$')

    def test_the_warning_height_option_changes_the_marks(self):
        _, default = run_preview(RECORDING)
        _, strict = run_preview(RECORDING + ['--warn', '4.5'])
        _, loose = run_preview(RECORDING + ['--warn', '10'])
        _, tight = run_preview(RECORDING + ['--warn', '0.5'])
        self.assertEqual(0, default.count('<-- LOW'), 'the lowest tip of this recording is 4.0 cm')
        self.assertEqual(0, tight.count('<-- LOW'))
        self.assertEqual(3, strict.count('<-- LOW'))  # poses 12, 21 and 24, their tips at 4.0 to 4.1 cm
        self.assertGreater(loose.count('<-- LOW'), 3)

    def test_floor_changes_the_limits_and_the_accepted_poses(self):
        default = previewed_poses(RECORDING)
        high = previewed_poses(RECORDING + ['--floor', '8'])
        self.assertNotEqual([p.tolist() for p in default], [p.tolist() for p in high])
        # the same limits as the recorder puts on its robot
        _, output = run_preview(RECORDING + ['--floor', '8'])
        limits = record_static_load.limits_with_floor(8.0)
        self.assertIn('floor {} cm (wrist {} cm)'.format(limits.z_min, limits.wrist_z_min), output)
        self.assertIn('set by --floor', output)
        self.assertNotIn('set by --floor', run_preview(RECORDING)[1])
        # every stop of every pose is above the floor now, which was not so for the default floor
        robot = OfflineRobot()
        delta = np.radians(5)
        lowest = [preview.pose_heights(n, pose, delta, (2, 3, 4, 5), robot.get_angle_limits()).lowest
                  for n, pose in enumerate(high, 1)]
        self.assertGreaterEqual(min(lowest), 8.0 - 1e-6)
        lowest = [preview.pose_heights(n, pose, delta, (2, 3, 4, 5), robot.get_angle_limits()).lowest
                  for n, pose in enumerate(default, 1)]
        self.assertLess(min(lowest), 8.0)

    def test_every_listed_pose_is_accepted_by_the_planner_with_its_overshoots(self):
        robot = OfflineRobot()
        for pose in previewed_poses(RECORDING):
            self.assertTrue(record_static_load.pose_is_accepted(pose, robot, np.radians(5), (2, 3, 4, 5)))

    def test_the_options_that_only_matter_while_recording_are_accepted_and_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            output_file = os.path.join(directory, 'static_load.csv')
            result, output = run_preview(RECORDING + ['--settle', '1', '--readings', '2', '--speed', '0.1',
                                                      '--output', output_file, '--yes'])
            self.assertFalse(os.path.exists(output_file))
        self.assertEqual(0, result)
        self.assertEqual(run_preview(RECORDING)[1], output)

    def test_it_returns_1_when_no_pose_is_accepted(self):
        result, output = run_preview(['--poses', '2', '--floor', '1000', '--seed', '1'])
        self.assertEqual(1, result)
        self.assertIn('no pose the planner accepts', output)
        self.assertNotIn('Traceback', output)

    def test_it_returns_1_with_a_message_for_wrong_options(self):
        for argv, message in ((['--joints', '1'], 'only joints 2-6'), (['--joints', '7'], 'only joints 2-6'),
                              (['--range', '5=45'], 'range entries look like'),
                              (['--range', '2=100:100'], 'joint 2 has no range left')):
            with self.subTest(argv=argv):
                result, output = run_preview(argv)
                self.assertEqual(1, result)
                self.assertIn(message, output)
        for argv in (['--poses', 'many'], ['--no-such-option'], ['--warn', 'low'], ['--readings', '0']):
            with self.subTest(argv=argv):
                self.assertEqual(1, run_preview(argv)[0])

    def test_help_returns_0(self):
        self.assertEqual(0, run_preview(['--help'])[0])

    def test_it_accepts_the_same_options_as_the_recorder_and_the_warning_height(self):
        recorder = {action.dest for action in record_static_load.build_parser()._actions}
        tool = {action.dest for action in preview.build_parser()._actions}
        self.assertEqual(recorder | {'warn'}, tool)
        self.assertIn('poses', tool)
        self.assertIn('yes', tool)


class NoPortTests(unittest.TestCase):

    def forbid_hardware(self, stack):
        """patch everything that would build the real robot or open a serial port to raise"""
        def forbidden(*args, **kwargs):
            raise AssertionError('the preview tried to reach the hardware')
        for target in ('src.global_objects.get_robot', 'src.robot_controllers.dynamixel_robot.calibration.'
                       'record_static_load.get_robot', 'serial.Serial',
                       'src.robot_controllers.dynamixel_robot.dynamixel_utils.get_port_handler',
                       'src.robot_controllers.dynamixel_robot.dynamixel_utils.setup_dynamixel_handlers',
                       'src.robot_controllers.dynamixel_robot.dynamixel_utils.find_dynamixel_port',
                       'src.robot_controllers.dynamixel_robot.dynamixel_robot_controller.'
                       'DynamixelRobotController.__init__',
                       'src.simulation.simulation_utils.start_simulated_robot'):
            stack.enter_context(mock.patch(target, side_effect=forbidden))

    def test_it_never_builds_the_robot_or_opens_a_port_whatever_use_simulation_is(self):
        self.assertTrue(hasattr(dynamixel_utils, 'get_port_handler'))
        for use_simulation in (False, True):
            with self.subTest(use_simulation=use_simulation), contextlib.ExitStack() as stack:
                self.forbid_hardware(stack)
                stack.enter_context(mock.patch.object(global_constants, 'use_simulation', use_simulation))
                result, output = run_preview(['--poses', '3', '--seed', '1'])
                self.assertEqual(0, result, output)
                self.assertIn('3 poses of 3 asked', output)

    def test_the_guard_of_these_tests_catches_a_robot_that_is_built(self):
        with contextlib.ExitStack() as stack:
            self.forbid_hardware(stack)
            with self.assertRaises(AssertionError):
                record_static_load.get_robot()

    def test_it_runs_without_importing_the_simulation(self):
        code = ('import sys\n'
                'from src.robot_controllers.dynamixel_robot.calibration.offline import preview_static_load_poses as p\n'
                'result = p.main(["--poses", "2", "--seed", "1"])\n'
                'print("RESULT", result, "pybullet" in sys.modules)\n')
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        completed = subprocess.run([sys.executable, '-c', code], cwd=root, capture_output=True, text=True,
                                   timeout=120)
        self.assertIn('RESULT 0 False', completed.stdout, completed.stderr)


if __name__ == '__main__':
    unittest.main()
