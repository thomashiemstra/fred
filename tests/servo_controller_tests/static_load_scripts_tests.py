import csv
import io
import os
import tempfile
import unittest

import numpy as np
import yaml
from numpy import pi

from src.kinematics.kinematics_utils import RobotConfig
from src.robot_controllers.dynamixel_robot.calibration import record_static_load, fit_static_load
from src.utils import workspace
from src.robot_controllers.dynamixel_robot.servo_config import load_servo_config, parse_servo_config
from src.robot_controllers.dynamixel_robot.static_compensation import regressor_row, term_names, SpringGeometry, \
    UP, DOWN, SPRING_TERMS

TEST_SERVO_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_servo_config.yaml')
test_config = RobotConfig(d1=15.7, a2=20, d4=22, d6=12)
angle_limits = {1: (0, pi), 2: (0, pi), 3: (-pi / 2, pi / 2), 4: (-pi, pi), 5: (-pi, pi), 6: (-pi, pi)}


class TestLimits:
    """workspace limits of these tests, independent of global_constants.WorkSpaceLimits"""
    radius_min = 10
    radius_max = 57
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
        positions = np.zeros(7, dtype=np.int_)
        for servo_id in range(1, 7):
            positions[servo_id] = self.servo_handler.get_servo(servo_id).target_position + self.sag(servo_id, self.angles)
        return positions

    def get_present_currents(self):
        return {servo_id: 10 * servo_id for servo_id in range(1, 7)}

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
        self.assertEqual(TestLimits.radius_max, limits.radius_max)

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
        # overshoot that is clipped to the range (the arm stretched out towards 90 degrees is outside the radius)
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
            self.assertEqual(int(row['present_position']) - int(row['goal_position']), int(row['error']))
            self.assertEqual(gravity_sag(int(row['servo']), pose), int(row['error']))
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

    def test_moves_are_checked_against_the_floor(self):
        robot = FakeRobot([0, pi / 2, 0.4, 0.8, 0, 0, 0], gravity_sag)
        low_pose = np.array([0, pi / 2, 0.0, 1.2, 0, 0, 0])  # the gripper tip 3.4 cm up, its lower overshoot 0.7
        done, _ = self.record(robot, [low_pose], workspace_limits=record_static_load.limits_with_floor(3.0, TestLimits))
        self.assertEqual(0, done)
        done, _ = self.record(robot, [low_pose], workspace_limits=record_static_load.limits_with_floor(0.0, TestLimits))
        self.assertEqual(1, done)


class FitTests(unittest.TestCase):

    def setUp(self):
        self.spring = SpringGeometry((-4.0, -6.0), (2.0, 3.0), 20.0)
        self.true = {2: {'constant': 1.0, 'cos_q2': -12.0, 'sin_q2': 1.5, 'sin_phi': -5.0, 'cos_phi': 0.8,
                         'wrist_reach': -2.0, 'spring_dl': 4.0},
                     3: {'constant': -2.0, 'sin_phi': 30.0, 'cos_phi': -3.0, 'wrist_reach': 6.0, 'spring_dl': -2.0},
                     5: {'constant': 0.0, 'wrist_pitch': 8.0}}

    def synthetic_rows(self, count, rng, hysteresis=4.0, noise=0.7):
        """rows like record_static_load writes, from the true coefficients plus hysteresis and noise"""
        rows = []
        for pose in range(count):
            angles = np.zeros(7)
            angles[1] = pi / 2
            angles[2], angles[3] = rng.uniform(0.2, 2.9), rng.uniform(-1.3, 1.3)
            angles[4], angles[5] = rng.uniform(-1.0, 1.0), rng.uniform(-1.5, 1.5)
            for direction, sign in ((UP, 1), (DOWN, -1)):
                for servo_id in (2, 3, 5):
                    names = tuple(self.true[servo_id])
                    static = float(regressor_row(servo_id, angles, self.spring, names) @ np.array(
                        [self.true[servo_id][name] for name in names]))
                    error = static + sign * hysteresis + rng.normal(0, noise)
                    rows.append({'time': '1.0', 'pose': str(pose), 'direction': direction, 'servo': str(servo_id),
                                 'angle': '{:.6f}'.format(angles[servo_id]), 'goal_position': '2048',
                                 'compensation': '0', 'present_position': str(2048 + int(round(error))),
                                 'error': '{:.1f}'.format(error), 'current_raw': '0', 'current': '',
                                 'current_unit': 'mA', 'temperature': '30', 'voltage': '11.9'})
            # the angles of the joints that are not varied are still in the rows of their servos
            for direction in (UP, DOWN):
                for servo_id in (1, 4, 6):
                    rows.append({'time': '1.0', 'pose': str(pose), 'direction': direction, 'servo': str(servo_id),
                                 'angle': '{:.6f}'.format(angles[servo_id]), 'goal_position': '2048',
                                 'compensation': '0', 'present_position': '2048', 'error': '0', 'current_raw': '0',
                                 'current': '', 'current_unit': 'mA', 'temperature': '30', 'voltage': '11.9'})
        return rows

    def test_samples_average_the_directions(self):
        rows = self.synthetic_rows(3, np.random.default_rng(0), hysteresis=4.0, noise=0.0)
        samples = fit_static_load.samples_from_rows(rows)
        self.assertEqual(3, len(samples))
        for sample in samples:
            self.assertAlmostEqual(4.0, sample.hysteresis[2])
            self.assertEqual({UP, DOWN}, set(sample.by_direction))
            self.assertAlmostEqual(pi / 2, sample.angles[1], places=5)
        # a pose without both directions is left out
        samples = fit_static_load.samples_from_rows([row for row in rows if not (row['pose'] == '1' and row['direction'] == UP)])
        self.assertEqual(2, len(samples))

    def test_fit_recovers_the_coefficients(self):
        rng = np.random.default_rng(2)
        samples = fit_static_load.samples_from_rows(self.synthetic_rows(300, rng))
        for joint in (2, 3, 5):
            fit = fit_static_load.fit_joint(joint, samples, self.spring, holdout=0.2, rng=rng)
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
        fit = fit_static_load.fit_joint(5, samples, self.spring, rng=rng)
        # the constant of joint 5 is 0 in the data: not determined; wrist_pitch is 8 +- a few hundredths
        self.assertIn('constant', fit.undetermined())
        self.assertNotIn('wrist_pitch', fit.undetermined())
        self.assertLess(fit.standard_errors['wrist_pitch'], 0.2)
        self.assertEqual(['not determined by the data (smaller than twice their standard error): constant'],
                         fit.warnings())
        self.assertIn('<- not determined', fit_static_load.describe(fit))
        # a regressor whose terms can't be told apart is warned about: fit the spring term twice
        fit = fit_static_load.fit_joint(3, samples, self.spring, rng=rng, names=('sin_phi', 'spring_dl', 'spring_dl'))
        self.assertTrue(any('condition number' in warning for warning in fit.warnings()))

    def test_fit_without_spring_leaves_the_spring_terms_out(self):
        samples = fit_static_load.samples_from_rows(self.synthetic_rows(60, np.random.default_rng(3)))
        fit = fit_static_load.fit_joint(3, samples, None)
        self.assertEqual(tuple(name for name in term_names(3) if name not in SPRING_TERMS), fit.names)

    def test_directional_fit(self):
        rng = np.random.default_rng(4)
        samples = fit_static_load.samples_from_rows(self.synthetic_rows(200, rng, hysteresis=4.0, noise=0.0))
        up = fit_static_load.fit_joint(5, samples, self.spring, holdout=0.0, direction=UP)
        down = fit_static_load.fit_joint(5, samples, self.spring, holdout=0.0, direction=DOWN)
        # the friction hysteresis is a constant per direction: +4 going up, -4 going down (the angles in the csv
        # are rounded to 6 decimals, so the recovery is not exact)
        self.assertAlmostEqual(4.0, up.coefficients['constant'], places=2)
        self.assertAlmostEqual(-4.0, down.coefficients['constant'], places=2)
        self.assertAlmostEqual(8.0, up.coefficients['wrist_pitch'], places=2)
        self.assertLess(up.rms_train, 0.1)  # the errors in the csv are rounded to one decimal
        self.assertIsNone(up.rms_holdout)
        self.assertEqual(200, up.train_count)

    def test_too_few_samples(self):
        samples = fit_static_load.samples_from_rows(self.synthetic_rows(3, np.random.default_rng(5)))
        with self.assertRaises(ValueError):
            fit_static_load.fit_joint(2, samples, self.spring)

    def test_yaml_block_parses_as_a_servo_config(self):
        rng = np.random.default_rng(6)
        samples = fit_static_load.samples_from_rows(self.synthetic_rows(300, rng))
        fits = {joint: fit_static_load.fit_joint(joint, samples, self.spring, rng=rng) for joint in (2, 3, 5)}
        directional = {5: {direction: fit_static_load.fit_joint(5, samples, self.spring, direction=direction)
                           for direction in (UP, DOWN)}}
        block = fit_static_load.yaml_block(fits, 60, directional)
        self.assertIn('  2:\n    compensation:\n      max_positions: 60\n      terms: {constant: ', block)
        self.assertIn('      up: {constant: ', block)

        # paste the blocks into the test configuration: the compensation of every servo is parsed
        with open(TEST_SERVO_CONFIG_PATH) as config_file:
            data = yaml.safe_load(config_file)
        data['spring'] = {'base_anchor': [-4.0, -6.0], 'arm_anchor': [2.0, 3.0]}
        for servo_id, servo_data in yaml.safe_load(block).items():
            data['servos'][servo_id].update(servo_data)
        config = parse_servo_config(data, a2=20.0)
        self.assertEqual(self.spring, config.spring)
        self.assertEqual(60, config.servos[2].compensation.max_positions)
        self.assertAlmostEqual(-fits[3].coefficients['sin_phi'], config.servos[3].compensation.terms['sin_phi'], places=2)
        self.assertEqual({UP, DOWN}, set(config.servos[5].compensation.directional))
        self.assertIsNone(config.servos[4].compensation)

        # the compensation cancels the error it was fitted to: added to the goal, the servo ends up at the goal
        for sample in samples[:20]:
            for joint in (2, 3, 5):
                compensation = config.servos[joint].compensation.positions(sample.angles)
                self.assertAlmostEqual(0.0, sample.static[joint] + compensation, delta=3.0,
                                       msg='joint {} at {}'.format(joint, np.round(sample.angles, 2)))
            up = config.servos[5].compensation.positions(sample.angles, UP)
            self.assertAlmostEqual(0.0, sample.by_direction[UP][5] + up, delta=3.0)

    def test_recorded_error_is_against_the_uncompensated_goal(self):
        from src.robot_controllers.dynamixel_robot.static_compensation import StaticCompensation
        robot = FakeRobot([0, pi / 2, 0.4, 0.8, 0, 0, 0], gravity_sag)
        servo = robot.servo_handler.get_servo(2)
        servo.compensation = StaticCompensation(2, {'cos_q2': 7.0}, 60)
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
            self.assertEqual(int(row['present_position']) - int(row['goal_position']), int(row['error']))
            # the fake servo lands on goal + compensation + sag: against the uncompensated goal that is both
            self.assertEqual(gravity_sag(2, pose) + 7, int(row['error']))

    def test_main_prints_the_block(self):
        rows = self.synthetic_rows(80, np.random.default_rng(7))
        with tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False, newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=record_static_load.FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        printed = []
        try:
            from unittest import mock
            with mock.patch('builtins.print', lambda *args, **kwargs: printed.append(' '.join(str(a) for a in args))):
                result = fit_static_load.main([csv_file.name, '--no-spring', '--split', '--joints', '2', '3', '5'])
        finally:
            os.remove(csv_file.name)
        self.assertEqual(0, result)
        output = '\n'.join(printed)
        self.assertIn('joint 2:', output)
        self.assertIn('held out', output)
        self.assertIn('compensation:', output)
        self.assertIn('up: {', output)


if __name__ == '__main__':
    unittest.main()
