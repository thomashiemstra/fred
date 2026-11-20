import argparse
import contextlib
import csv
import io
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from src.global_constants import WorkSpaceLimits, dynamixel_robot_config
from src.robot_controllers.dynamixel_robot.calibration import fit_static_load, gain_tuning, record_load_sweep, \
    record_static_load, tune_servo_gains
from src.robot_controllers.dynamixel_robot.calibration.record_load_sweep import (
    DEFAULT_SWEEPS, Options, SweepResult, plan_poses, plan_sweep, pose_span, record_sweeps)
from src.robot_controllers.dynamixel_robot.calibration.tune_servo_gains import StopFlag, parse_pose, position_sign
from src.robot_controllers.dynamixel_robot.servo_config import load_servo_config
from src.robot_controllers.dynamixel_robot.servo_handler import ServoState
from src.robot_controllers.dynamixel_robot.calibration.load_directions import DOWN, UP
from src.robot_controllers.dynamixel_robot.static_compensation import StaticCompensation, term_values
from tests.servo_controller_tests.static_load_scripts_tests import FakeHandler, FakeRobot, RestRobot

FIELDS = record_static_load.FIELDS
ARM = (1, 2, 3, 4, 5, 6)
# the poses of the fake robot: the lower arm vertical and the upper arm horizontal, the elbow out
ELBOW_OUT = (90, 90, 0, 0, 0, 0)
LOWER = (90, 60, 0, 0, 0, 0)
START_OF_ABORT = 20.0  # simulated seconds: the first sweep of the tests below is over and the second one runs


class FakeClock:
    """the time of the simulation: it only moves when the code sleeps or reads a servo"""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(seconds, 0.0)


class SweepHandler:
    def __init__(self, robot):
        self.robot = robot

    def get_servo(self, servo_id):
        return self.robot.servo_config.servos[servo_id]

    def read_state(self, servo_ids):
        return self.robot.read_state(servo_ids)


class SweepRobot(FakeRobot):
    """
    a robot whose swept servo trails its trajectory by what a joint with a load and friction does, in positions:
    present = trajectory + sag(commanded angles) - direction * (friction + lag * speed) with the direction in
    positions (the one of the angle times the position sign of the servo, servo 3 of the test yaml is reversed).
    The error against the uncompensated goal is then known: compensation + sag - direction * trail.
    """
    READ_TIME = 0.003  # seconds a read of the servo takes

    def __init__(self, joint, pose_degrees, clock, sag=None, friction=0.0, lag=0.0, up_extra_lag=0.0):
        angles = np.zeros(7)
        angles[1:] = np.radians(pose_degrees)
        super().__init__(angles)
        self.joint = joint
        self.clock = clock
        self.sag = sag if sag is not None else (lambda commanded: 0.0)
        self.friction = friction
        self.lag = lag  # positions per degree per second
        self.up_extra_lag = up_extra_lag  # the same, only going up: the friction differs between the directions
        self.servo_handler = SweepHandler(self)
        for servo in self.servo_config.servos.values():
            servo.move_time = 15
        self.enabled = False
        self.writes = []  # (time, angle of the joint in degrees)
        self.direction = 1  # of the angle
        self.speed = 0.0  # degrees per second of the commanded angle
        self.reads = 0
        self.rng = np.random.default_rng(0)
        self.failing = False
        self.before_read = None

    def enable_servos(self):
        self.enabled = True
        return True

    def move_servos(self, angles):
        super().move_servos(angles)
        degrees = float(np.degrees(angles[self.joint]))
        if self.writes:
            seconds, previous = self.writes[-1]
            if self.clock.now > seconds:
                self.speed = abs(degrees - previous) / (self.clock.now - seconds)
                if degrees != previous:
                    self.direction = 1 if degrees > previous else -1
        self.writes.append((self.clock.now, degrees))

    def read_state(self, servo_ids):
        self.clock.sleep(self.READ_TIME)
        self.reads += 1
        if self.before_read is not None:
            self.before_read(self)
        if self.failing:
            return None
        servo = self.servo_config.servos[self.joint]
        trajectory = servo.target_position
        trail = self.friction + (self.lag + (self.up_extra_lag if self.direction > 0 else 0.0)) * self.speed \
            if self.speed > 0 else 0.0
        # the real position jitters by a count or two, so the mean of many reads is not quantized: round at random
        exact = trajectory + self.sag(self.angles) - position_sign(servo) * self.direction * trail
        position = int(np.floor(exact + self.rng.uniform()))
        return {self.joint: ServoState(
            tick=int(1000 * self.clock.now) % 32768, moving=0, moving_status=1,
            pwm=int(round(servo.p / 128.0 * (trajectory - position))), current=10, velocity=0, position=position,
            velocity_trajectory=0, position_trajectory=trajectory, voltage=11.9,
            temperature=self.temperatures[self.joint])}


def rows_of(text):
    return list(csv.DictReader(io.StringIO(text)))


def groups_of(rows, joint):
    """the up and the down row of the swept servo of every bin: {pose number: {direction: row}}"""
    groups = {}
    for row in rows:
        if int(row['servo']) == joint:
            groups.setdefault(int(row['pose']), {})[row['direction']] = row
    return groups


class SweepTestCase(unittest.TestCase):
    """the fake robot with a clock that the planner shares, so the moves run without waiting"""

    def setUp(self):
        self.clock = FakeClock()
        for name, replacement in (('sleep', self.clock.sleep), ('perf_counter', self.clock)):
            patcher = mock.patch('src.utils.movement.' + name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def robot(self, joint=3, pose=ELBOW_OUT, **kwargs):
        return SweepRobot(joint, pose, self.clock, **kwargs)

    def spans(self, robot, joint, *spans, pose=ELBOW_OUT):
        return [pose_span(parse_pose(','.join(str(d) for d in pose)), joint, low, high) for low, high in spans]

    def record(self, robot, joint, spans, speeds=(5.0, 10.0), options=None, stop=None, flushes=None):
        """plan and record the sweeps; returns (recording, rows, printed text)"""
        poses = plan_poses(robot, joint, spans, speeds)
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            recording = record_sweeps(robot, joint, poses, writer, options or Options(), stop, self.clock,
                                      self.clock.sleep, wall_clock=lambda: 1000.0,
                                      flush=(lambda: flushes.append(1)) if flushes is not None else None)
        return recording, rows_of(output.getvalue()), printed.getvalue()


class RowTests(SweepTestCase):

    def test_the_rows_of_the_bins(self):
        robot = self.robot(3, sag=lambda q: 4.0 * np.sin(q[3]), friction=3.0, lag=0.5)
        flushes = []
        recording, rows, _ = self.record(robot, 3, self.spans(robot, 3, (-30, 30)), flushes=flushes)
        self.assertIsNone(recording.stopped)
        self.assertEqual(2, recording.sweeps)
        self.assertEqual([1, 1], flushes, 'the file is flushed after every sweep')
        self.assertGreater(recording.bins, 8)
        self.assertEqual(12 * recording.bins, len(rows), '2 directions and 6 servos per bin')
        self.assertEqual(list(FIELDS), list(rows[0]))
        self.assertEqual(set(range(recording.bins)), {int(row['pose']) for row in rows}, 'one counter for the file')
        poses = {}
        for row in rows:
            poses.setdefault(row['pose'], []).append(row)
        for number, bin_rows in poses.items():
            self.assertEqual([UP] * 6 + [DOWN] * 6, [row['direction'] for row in bin_rows])
            self.assertEqual(list(ARM) * 2, [int(row['servo']) for row in bin_rows])
            self.assertEqual(1, len({row['speed'] for row in bin_rows}))
            self.assertEqual({'sweep'}, {row['mode'] for row in bin_rows})
            self.assertEqual({'1000.00'}, {row['time'] for row in bin_rows})
            # the angle of the swept joint is the centre of a bin of the grid, the same in both directions
            centre = np.degrees(float(bin_rows[2]['angle']))
            self.assertAlmostEqual(0.5, (centre / 5.0) % 1.0, places=4)
            self.assertEqual(bin_rows[2]['angle'], bin_rows[8]['angle'])
            for row in bin_rows:
                servo = robot.servo_config.servos[int(row['servo'])]
                self.assertEqual((str(servo.p), str(servo.i), str(servo.d)), (row['p'], row['i'], row['d']))
                if int(row['servo']) == 3:
                    self.assertGreaterEqual(int(row['samples']), gain_tuning.MIN_BIN_SAMPLES)
                    self.assertEqual('30', row['temperature'])
                    self.assertEqual('11.9', row['voltage'])
                    self.assertEqual('10.0', row['current_raw'])
                    self.assertEqual(('26.90', 'mA'), (row['current'], row['current_unit']), 'XM430: 2.69 mA a unit')
                    self.assertAlmostEqual(float(row['present_position']) - float(row['error']),
                                           float(row['goal_position']), delta=0.011)
                    self.assertEqual('0.00', row['compensation'])
                    self.assertNotEqual('', row['pwm'])
                else:
                    # the other joints stood still: their angle is all they have
                    for name in ('error', 'compensation', 'pwm', 'goal_position', 'present_position', 'current_raw',
                                 'current', 'current_unit', 'samples', 'temperature', 'voltage'):
                        self.assertEqual('', row[name], name)
                    self.assertEqual('{:.6f}'.format(robot.angles[int(row['servo'])]), row['angle'])
        # the fit reads the file: every bin is a pose with both directions
        samples = fit_static_load.samples_from_rows(rows)
        self.assertEqual(recording.bins, len(samples))
        self.assertTrue(all(set(sample.static) == {3} for sample in samples))

    def test_the_angle_of_a_bin_is_the_one_the_servo_is_at_even_beyond_its_calibrated_range(self):
        servo = self.robot(3).servo_config.servos[3]  # positions 3072 to 1024 for -90 to 90 degrees
        streamed = []
        for direction in (1, -1):
            for _ in range(gain_tuning.MIN_BIN_SAMPLES):
                # a joint pushed past its calibrated end by its load: position 900 is 100.9 degrees, not the 90
                # that get_angle_from_position clamps it to
                sample = gain_tuning.Sample(time=0.0, tick=0, goal=900, position=900, position_trajectory=905,
                                            velocity=0, velocity_trajectory=0, pwm=0, current=0, moving_status=0,
                                            voltage=11.9, temperature=30)
                step = gain_tuning.SweepStep(0.0, 0.0, direction, True, 0)
                streamed.append(tune_servo_gains.StreamedSample(sample, step, 905, 0))
        ((found, up, down),) = record_load_sweep.bin_cruise_samples(servo, streamed, 5.0)
        self.assertEqual(102.5, found.angle)
        self.assertEqual((-5.0, -5.0), (found.up.error, found.down.error))
        self.assertEqual((30, 30), (found.up.samples, found.down.samples))

    def test_the_bins_of_two_speeds_line_up(self):
        robot = self.robot(3, sag=lambda q: 4.0 * np.sin(q[3]), friction=3.0, lag=0.5)
        _, rows, _ = self.record(robot, 3, self.spans(robot, 3, (-30, 30)))
        centres = {}
        for group in groups_of(rows, 3).values():
            centres.setdefault(group[UP]['speed'], []).append(round(np.degrees(float(group[UP]['angle'])), 3))
        self.assertEqual({'5', '10'}, set(centres))
        for speed, found in centres.items():
            self.assertEqual(sorted(found), found, 'sorted by angle')
            self.assertTrue(all((c / 5.0) % 1.0 == 0.5 for c in found), speed)
        self.assertTrue(set(centres['10']) <= set(centres['5']) or set(centres['5']) <= set(centres['10']))
        interior = {-22.5, -17.5, -12.5, -7.5, -2.5, 2.5, 7.5, 12.5, 17.5, 22.5}
        self.assertTrue(interior <= set(centres['5']) and interior <= set(centres['10']))
        self.assertLessEqual(max(centres['5']), 27.5)
        self.assertGreaterEqual(min(centres['5']), -27.5)

    def test_the_writes_follow_the_clock_of_the_steps(self):
        robot = self.robot(3)
        poses = plan_poses(robot, 3, self.spans(robot, 3, (-30, 30)), (10.0,))
        plan = poses[0].sweeps[0]
        with contextlib.redirect_stdout(io.StringIO()):
            record_sweeps(robot, 3, poses, csv.DictWriter(io.StringIO(), fieldnames=FIELDS), clock=self.clock,
                          sleep=self.clock.sleep)
        written = np.array([seconds for seconds, _ in robot.writes[-len(plan.steps):]])
        np.testing.assert_allclose(written - written[0], [step.time for step in plan.steps], atol=0.003)

    def test_the_loop_of_the_gain_tuning_is_the_one_that_streams(self):
        robot = self.robot(3)
        with mock.patch.object(record_load_sweep, 'stream_sweep', wraps=tune_servo_gains.stream_sweep) as shared:
            recording, _, _ = self.record(robot, 3, self.spans(robot, 3, (-30, 30)), speeds=(10.0,))
        self.assertEqual(1, shared.call_count)
        self.assertEqual(1, recording.sweeps)


class RecoveryTests(SweepTestCase):
    """the mean of the two directions is the sag, half the difference the friction and the lag"""
    FRICTION = 3.0
    LAG = 0.5  # positions per degree per second

    def check(self, joint, spans, pose=ELBOW_OUT, compensation=None):
        robot = self.robot(joint, pose, sag=lambda q: 6.0 * np.sin(q[joint]) + 2.0, friction=self.FRICTION,
                           lag=self.LAG)
        servo = robot.servo_config.servos[joint]
        servo.compensation = compensation
        recording, rows, _ = self.record(robot, joint, self.spans(robot, joint, *spans, pose=pose))
        self.assertIsNone(recording.stopped)
        groups = groups_of(rows, joint)
        self.assertGreater(len(groups), 15)
        sign = position_sign(servo)
        for group in groups.values():
            up, down = group[UP], group[DOWN]
            centre = np.degrees(float(up['angle']))
            speed = float(up['speed'])
            sag = 6.0 * np.sin(np.radians(centre)) + 2.0
            mean = (float(up['error']) + float(down['error'])) / 2.0
            half = (float(up['error']) - float(down['error'])) / 2.0
            comp = float(up['compensation'])
            # the whole load is the error less the compensation that was added to the goal
            self.assertAlmostEqual(sag, mean - comp, delta=0.6, msg='sag at {:.1f}'.format(centre))
            # going up the angle the position trails against its own direction: below the trajectory for a servo
            # whose positions rise with the angle, above it for a reversed one
            self.assertAlmostEqual(-sign * (self.FRICTION + self.LAG * speed), half, delta=0.6,
                                   msg='half the difference at {:.1f}, {} degrees/s'.format(centre, speed))
            self.assertEqual(sign * (float(up['error']) - float(down['error'])) < 0, True)
        return groups

    def test_a_normal_servo(self):
        servo_5 = self.robot(5).servo_config.servos[5]
        self.assertEqual(1, position_sign(servo_5))
        self.check(5, [(-30, 30)])

    def test_a_reversed_servo(self):
        servo_3 = self.robot(3).servo_config.servos[3]
        self.assertEqual(-1, position_sign(servo_3), 'positions: [3072, 1024] for angles: [-90, 90]')
        groups = self.check(3, [(-30, 30)])
        # the angle rises where the up rows are: their goal positions fall
        ordered = [groups[number] for number in sorted(groups) if groups[number][UP]['speed'] == '5']
        goals = [float(group[UP]['goal_position']) for group in ordered]
        self.assertEqual(sorted(goals, reverse=True), goals)
        self.assertGreater(float(ordered[0][UP]['angle']), -1)  # the first bin of the sweep is at -27.5 degrees
        self.assertLess(float(ordered[0][UP]['angle']), 0)

    def test_a_compensation_is_part_of_the_error_and_given_as_its_own_column(self):
        # 300 pwm units * sin(phi) at P 1500 are up to 25.6 positions added to the goal
        compensation = StaticCompensation(3, {'sin_phi': 300.0}, 60)
        groups = self.check(3, [(-30, 30)], compensation=compensation)
        values = {round(float(group[UP]['compensation'])) for group in groups.values()}
        self.assertGreater(len(values), 3, 'the compensation changes over the sweep')
        for group in groups.values():
            for row in group.values():
                # the error is against the uncompensated goal, which the compensation is not part of
                self.assertAlmostEqual(float(row['present_position']) - float(row['error']),
                                       float(row['goal_position']), delta=0.011)

    def test_the_error_and_the_compensation_without_any_sag(self):
        robot = self.robot(3)
        robot.servo_config.servos[3].compensation = StaticCompensation(3, {'constant': 150.0}, 60)  # 12.8 positions
        recording, rows, _ = self.record(robot, 3, self.spans(robot, 3, (-30, 30)), speeds=(10.0,))
        for group in groups_of(rows, 3).values():
            for row in group.values():
                self.assertEqual('13.00', row['compensation'], 'the positions the servo added to its goal')
                self.assertAlmostEqual(13.0, float(row['error']), delta=0.01, msg='the fake lands on goal + 0 sag')
                self.assertAlmostEqual(0.0, float(row['error']) - float(row['compensation']), delta=0.01)


class FitTests(SweepTestCase):
    """rows recorded from a joint that sags as a * sin(phi) + b * cos(phi) positions give the fit its coefficients"""

    def fit(self, joint, names, sag, poses_and_spans):
        robot = self.robot(joint, poses_and_spans[0][0], sag=sag, friction=2.0, lag=0.3)
        rows = []
        for pose, span in poses_and_spans:
            robot.angles = np.radians([0.0] + list(pose))
            _, found, _ = self.record(robot, joint, self.spans(robot, joint, span, pose=pose), speeds=(5.0,))
            offset = 1000 * len(rows)  # the pose numbers only count within a file
            rows.extend({**row, 'pose': str(int(row['pose']) + offset)} for row in found)
        samples = fit_static_load.samples_from_rows(rows)
        self.assertGreater(len(samples), 30)
        return robot, fit_static_load.fit_joint(joint, samples, holdout=0.0, names=names)

    def test_joint_3_the_reversed_servo(self):
        a, b = 9.0, -4.0
        robot, fit = self.fit(3, ('sin_phi', 'cos_phi'), lambda q: a * np.sin(q[2] + q[3]) + b * np.cos(q[2] + q[3]),
                              [(ELBOW_OUT, (-30, 30)), (LOWER, (-30, 30)), ((90, 75, 0, 0, 0, 0), (-30, 30))])
        p = robot.servo_config.servos[3].p
        # the drive is minus the error in positions times P / 128, whichever way the servo is mounted
        self.assertAlmostEqual(-a * p / 128.0, fit.coefficients['sin_phi'], delta=2.5)
        self.assertAlmostEqual(-b * p / 128.0, fit.coefficients['cos_phi'], delta=2.5)
        self.assertLess(fit.rms_train, 5.0)

    def test_joint_5_a_normal_servo(self):
        k = 7.0

        def sag(q):
            return k * term_values(5, q)['wrist_pitch']
        robot, fit = self.fit(5, ('wrist_pitch',), sag, [(ELBOW_OUT, (-30, 30)), (LOWER, (-30, 30)),
                                                         ((90, 75, 0, 0, 0, 0), (-30, 30))])
        p = robot.servo_config.servos[5].p
        self.assertAlmostEqual(-k * p / 128.0, fit.coefficients['wrist_pitch'], delta=2.5)


class RefusalTests(SweepTestCase):

    def test_a_refused_sweep_is_skipped_and_nothing_moves_for_it(self):
        robot = self.robot(3)
        spans = self.spans(robot, 3, (-100, 100), (-30, 30))  # the first leaves the +-90 degrees of the joint
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            poses = plan_poses(robot, 3, spans, (5.0, 10.0))
        self.assertEqual(1, len(poses))
        self.assertEqual((-30, 30), (poses[0].start.low, poses[0].start.high))
        self.assertEqual([5.0, 10.0], [plan.speed for plan in poses[0].sweeps])
        self.assertEqual(2, printed.getvalue().count('skipping the sweep'))
        self.assertIn('would move', printed.getvalue())
        self.assertIn('-100 to 100', printed.getvalue().replace('from -100 to 100', '-100 to 100'))
        self.assertEqual([], robot.moves, 'planning moves nothing')

    def test_a_sweep_without_a_cruise_part_in_both_directions_is_refused(self):
        robot = self.robot(3)
        plan = plan_sweep(self.spans(robot, 3, (-0.5, 0.5))[0], 3, 5.0)
        self.assertIn('no cruise part', record_load_sweep.sweep_refusal(robot, plan))
        wide = plan_sweep(self.spans(robot, 3, (-30, 30))[0], 3, 5.0)
        self.assertIsNone(record_load_sweep.sweep_refusal(robot, wide))

    def test_one_speed_can_be_refused_alone(self):
        robot = self.robot(3)
        # at 5 degrees/s a span of 4 degrees has cruise steps, at 30 degrees/s its legs are triangles
        spans = self.spans(robot, 3, (-2, 2))
        with contextlib.redirect_stdout(io.StringIO()):
            poses = plan_poses(robot, 3, spans, (5.0, 30.0))
        self.assertEqual([5.0], [plan.speed for plan in poses[0].sweeps])

    def test_the_planner_checks_every_step_against_the_workspace(self):
        robot = self.robot(3)
        spans = self.spans(robot, 3, (-30, 30), pose=(90, 30, 0, 0, 0, 0))  # the gripper tip gets below the floor
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            poses = plan_poses(robot, 3, spans, (5.0,))
        self.assertEqual([], poses)
        self.assertIn('below the minimum height', printed.getvalue())


class StopTests(SweepTestCase):
    """what ends a recording early, and what is kept"""

    def sweep_rows(self, rows):
        return {row['speed'] for row in rows}

    def test_an_abort_rule_ends_the_recording_and_keeps_the_finished_sweeps(self):
        robot = self.robot(3, sag=lambda q: 4.0 * np.sin(q[3]), friction=3.0, lag=0.5)
        limit = Options().limits.max_temperature
        robot.before_read = lambda r: r.temperatures.update({3: int(limit)}) if r.clock.now > START_OF_ABORT else None
        recording, rows, printed = self.record(robot, 3, self.spans(robot, 3, (-30, 30)), speeds=(10.0, 5.0))
        self.assertEqual(1, recording.sweeps)
        self.assertIn('aborted: temperature {}C'.format(int(limit)), recording.stopped)
        self.assertEqual({'10'}, self.sweep_rows(rows))
        self.assertEqual(12 * recording.bins, len(rows))
        self.assertGreater(recording.bins, 8)

    def test_a_failed_read_ends_the_recording(self):
        robot = self.robot(3, friction=3.0, lag=0.5)
        failed = []

        def hook(r):
            r.failing = r.clock.now > START_OF_ABORT
            failed.append(r.failing)
        robot.before_read = hook
        recording, rows, _ = self.record(robot, 3, self.spans(robot, 3, (-30, 30)), speeds=(10.0, 5.0))
        self.assertEqual(1, recording.sweeps)
        self.assertEqual('failed to read servo 3 3 times in a row', recording.stopped)
        self.assertEqual(3, recording.read_failures, 'three attempts at the read that gave up')
        self.assertEqual({'10'}, self.sweep_rows(rows))

    def test_one_failed_read_is_tried_again(self):
        robot = self.robot(3, friction=3.0, lag=0.5)
        robot.before_read = lambda r: setattr(r, 'failing', r.reads % 50 == 0)
        recording, rows, _ = self.record(robot, 3, self.spans(robot, 3, (-30, 30)), speeds=(10.0,))
        self.assertIsNone(recording.stopped)
        self.assertGreater(recording.read_failures, 10)
        self.assertEqual(1, recording.sweeps)

    def test_a_stop_request_ends_the_recording_at_a_sample(self):
        robot = self.robot(3, friction=3.0, lag=0.5)
        stop = StopFlag()
        robot.before_read = lambda r: setattr(stop, 'requested', r.clock.now > START_OF_ABORT)
        recording, rows, _ = self.record(robot, 3, self.spans(robot, 3, (-30, 30)), speeds=(10.0, 5.0), stop=stop)
        self.assertEqual('interrupted', recording.stopped)
        self.assertEqual(1, recording.sweeps)
        self.assertEqual({'10'}, self.sweep_rows(rows))

    def test_ctrl_c_ends_the_recording(self):
        robot = self.robot(3, friction=3.0, lag=0.5)

        def interrupt(r):
            if r.clock.now > START_OF_ABORT:
                raise KeyboardInterrupt
        robot.before_read = interrupt
        recording, rows, _ = self.record(robot, 3, self.spans(robot, 3, (-30, 30)), speeds=(10.0, 5.0))
        self.assertEqual('killed', recording.stopped)
        self.assertEqual({'10'}, self.sweep_rows(rows))

    def test_a_warm_servo_or_a_hardware_error_stops_before_the_pose_is_visited(self):
        for what, expected in (('temperature', 'warmer than'), ('error', 'hardware error')):
            with self.subTest(what):
                robot = self.robot(3)
                if what == 'temperature':
                    robot.temperatures[2] = 46  # the warning of the test yaml is 45
                else:
                    robot.hardware_errors = {3: ['overload']}
                recording, rows, _ = self.record(robot, 3, self.spans(robot, 3, (-30, 30)))
                self.assertIn(expected, recording.stopped)
                self.assertEqual(0, recording.sweeps)
                self.assertEqual([], rows)
                self.assertEqual([], robot.moves)

    def test_servos_are_checked_at_every_pose(self):
        robot = self.robot(3)
        checks = []
        original = robot.get_hardware_errors

        def counting():
            checks.append(1)
            return original()
        robot.get_hardware_errors = counting
        spans = self.spans(robot, 3, (-30, 30))
        spans += self.spans(robot, 3, (-20, 20), pose=LOWER)
        self.record(robot, 3, spans, speeds=(10.0,))
        self.assertEqual(2, len(checks))

    def test_a_move_the_planner_refuses_ends_the_recording(self):
        robot = self.robot(3)
        with mock.patch.object(record_static_load, 'move_to', return_value=False):
            recording, rows, _ = self.record(robot, 3, self.spans(robot, 3, (-30, 30)))
        self.assertIn('the planner refuses the move', recording.stopped)
        self.assertEqual(0, recording.sweeps)


class SpeedTests(SweepTestCase):

    def bins(self, speed, sags):
        side = gain_tuning.BinSide
        return SweepResult(speed, [gain_tuning.SweepBin(2.5 + 5.0 * k, side(sag + 1.0, 0.0, 40),
                                                        side(sag - 1.0, 0.0, 40)) for k, sag in enumerate(sags)])

    def test_the_disagreement_is_the_largest_difference_of_the_mean_error_between_two_speeds(self):
        slow, fast, faster = self.bins(5.0, [0.0, 1.0, 2.0]), self.bins(10.0, [0.2, 1.0, 3.5]), \
            self.bins(15.0, [0.0, -0.3, 2.2])
        found = record_load_sweep.speed_disagreement([slow, fast, faster])
        self.assertEqual(12.5, found.angle)
        self.assertAlmostEqual(1.5, found.difference)
        self.assertEqual(5.0, found.slowest)
        # only the bins every sweep has count
        partial = SweepResult(10.0, fast.bins[:2])
        found = record_load_sweep.speed_disagreement([slow, partial])
        self.assertEqual(2.5, found.angle)
        self.assertAlmostEqual(0.2, found.difference)
        self.assertEqual(5.0, found.slowest)
        self.assertIsNone(record_load_sweep.speed_disagreement([slow]))
        self.assertIsNone(record_load_sweep.speed_disagreement([]))
        self.assertIsNone(record_load_sweep.speed_disagreement([slow, SweepResult(10.0, [])]))

    def test_the_warning_is_above_a_position(self):
        warning = record_load_sweep.disagreement_warning
        slow, fast = self.bins(5.0, [0.0]), self.bins(10.0, [1.0])
        self.assertIsNone(warning(record_load_sweep.speed_disagreement([slow, fast]), [5.0, 10.0]), 'exactly 1.0')
        self.assertIsNone(warning(None, [5.0]))
        text = warning(record_load_sweep.speed_disagreement([slow, self.bins(10.0, [1.5])]), [5.0, 10.0])
        self.assertIn('differs by 1.5 positions', text)
        self.assertIn('not the same in both directions', text)
        self.assertIn('trust the sweep at 5 degrees/s', text)

    def test_a_friction_that_differs_between_the_directions_gives_the_warning(self):
        # going up the joint trails by 0.8 positions more per degree per second: the mean of the two directions
        # moves by 0.4 positions per degree per second, 2 positions between 5 and 10 degrees per second
        robot = self.robot(3, sag=lambda q: 4.0 * np.sin(q[3]), friction=3.0, lag=0.5, up_extra_lag=0.8)
        _, rows, printed = self.record(robot, 3, self.spans(robot, 3, (-30, 30)))
        self.assertIn('warning: at', printed)
        self.assertIn('differs by', printed)
        self.assertIn('trust the sweep at 5 degrees/s', printed)

    def test_a_symmetric_joint_gives_no_warning_and_a_summary_per_sweep(self):
        robot = self.robot(3, sag=lambda q: 4.0 * np.sin(q[3]), friction=3.0, lag=0.5)
        recording, _, printed = self.record(robot, 3, self.spans(robot, 3, (-30, 30)))
        self.assertNotIn('warning', printed)
        self.assertEqual(2, printed.count('speed '))
        self.assertIn('speed 5 degrees/s: ', printed)
        self.assertRegex(printed, r'speed 10 degrees/s: \d+ bins, gravity sag -?[\d.]+ to -?[\d.]+ positions, measured '
                                  r'gravity pwm -?\d+ to -?\d+, half the up/down difference [\d.]+ positions '
                                  r'\(friction plus the lag of the speed\)')
        # one speed: nothing to compare
        robot = self.robot(3, friction=3.0, lag=0.5, up_extra_lag=0.8)
        _, _, printed = self.record(robot, 3, self.spans(robot, 3, (-30, 30)), speeds=(10.0,))
        self.assertNotIn('warning', printed)

    def test_a_sweep_without_a_bin_is_reported(self):
        self.assertIn('no bin has', record_load_sweep.describe_sweep(30.0, []))
        robot = self.robot(3, friction=3.0)
        # a bin of 0.2 degrees at 10 degrees/s is passed in 0.02 s, a handful of samples
        recording, rows, printed = self.record(robot, 3, self.spans(robot, 3, (-6, 6)), speeds=(10.0,),
                                               options=Options(bin_degrees=0.2))
        self.assertEqual(1, recording.sweeps)
        self.assertEqual(0, recording.bins)
        self.assertEqual([], rows)
        self.assertIn('no bin has', printed)


class OptionTests(unittest.TestCase):

    def test_the_span(self):
        self.assertEqual((-60.0, 60.0), record_load_sweep.parse_span('-60:60'))
        self.assertEqual((5.5, 70.0), record_load_sweep.parse_span('5.5:70'))
        for bad in ('60:-60', '5', 'a:b', '1:1', '1:2:3', '', 'nan:5'):
            with self.assertRaises(argparse.ArgumentTypeError, msg=bad):
                record_load_sweep.parse_span(bad)

    def test_the_arguments(self):
        parser = record_load_sweep.build_parser()
        args = parser.parse_args(['--joint', '3'])
        self.assertEqual((3, None, None, [5.0, 10.0], 5.0, None, False), (
            args.joint, args.pose, args.span, args.speeds, args.bin, args.floor, args.yes))
        self.assertEqual(record_static_load.REST_SPEED, args.move_speed)
        self.assertIsNone(args.output)
        args = record_load_sweep.parse_arguments(
            ['--joint', '3', '--pose', 'elbow_out', '90,45,0,0,0,0', '--span', '-20:30', '--speeds', '2', '4.5',
             '--bin', '2', '--floor', '3', '--move-speed', '0.2', '--output', 'x.csv', '--yes'])
        self.assertEqual((-20.0, 30.0), record_load_sweep.parse_arguments(['--joint', '3', '--span=-20:30']).span)
        self.assertEqual(2, len(args.pose))
        np.testing.assert_allclose(parse_pose('elbow_out'), args.pose[0])
        np.testing.assert_allclose(np.radians([0, 90, 45, 0, 0, 0, 0]), args.pose[1])
        self.assertEqual((-20.0, 30.0), args.span)
        self.assertEqual([2.0, 4.5], args.speeds)
        self.assertEqual((2.0, 3.0, 0.2, 'x.csv', True), (args.bin, args.floor, args.move_speed, args.output, args.yes))
        for bad in (['--joint', '3', '--pose', 'nowhere'], ['--joint', '3', '--pose', '1,2,3'],
                    ['--joint', '3', '--speeds', '0'], ['--joint', '3', '--speeds', '-5'],
                    ['--joint', '3', '--bin', '0'], ['--joint', '3', '--span', '5:1'], []):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit, msg=bad):
                parser.parse_args(bad)

    def test_the_default_sweeps_and_the_span_that_replaces_them(self):
        poses = record_load_sweep.default_poses(3)
        self.assertEqual(len(DEFAULT_SWEEPS[3]), len(poses))
        for pose, (degrees, (low, high)) in zip(poses, DEFAULT_SWEEPS[3]):
            self.assertEqual((low, high), (pose.low, pose.high))
            expected = list(degrees)
            expected[2] = (low + high) / 2.0  # joint 3 starts at the middle of its span
            np.testing.assert_allclose(np.radians(expected), pose.pose[1:], atol=1e-12)
        replaced = record_load_sweep.default_poses(3, (-10.0, 20.0))
        self.assertEqual([(-10.0, 20.0)] * len(poses), [(p.low, p.high) for p in replaced])
        np.testing.assert_allclose(np.radians(5.0), replaced[0].pose[3])
        # the other joints keep the angles of the pose
        np.testing.assert_allclose(poses[0].pose[[1, 2, 4, 5, 6]], replaced[0].pose[[1, 2, 4, 5, 6]])

    def test_the_angle_of_the_swept_joint_in_the_pose_is_ignored(self):
        pose = parse_pose('90,90,40,0,0,0')
        for joint in (3, 4):
            span = pose_span(pose, joint, 10.0, 30.0)
            self.assertAlmostEqual(np.radians(20.0), span.pose[joint])
            self.assertEqual((10.0, 30.0), (span.low, span.high))
            np.testing.assert_allclose(np.delete(pose, joint), np.delete(span.pose, joint))
        self.assertAlmostEqual(np.radians(40.0), pose[3], msg='the argument is not changed')

    def test_the_top_speeds(self):
        self.assertEqual(10.0, record_load_sweep.top_speed(2))
        for joint in (3, 4, 5):
            self.assertEqual(30.0, record_load_sweep.top_speed(joint))


class DurationTests(SweepTestCase):

    def test_the_estimate_counts_the_sweeps_the_rests_and_the_moves_between_the_poses(self):
        robot = self.robot(3)
        spans = self.spans(robot, 3, (-30, 30)) + self.spans(robot, 3, (-20, 20), pose=LOWER)
        poses = plan_poses(robot, 3, spans, (10.0,))
        options = Options(move_speed=0.3)
        first = gain_tuning.ff_sweep(30, 10.0, 0.01, round_trips=1)[-1].time
        second = gain_tuning.ff_sweep(20, 10.0, 0.01, round_trips=1)[-1].time
        move = float(np.radians(30)) / 0.3  # joint 2 turns most, 30 degrees
        expected = 2 * (record_load_sweep.SETTLE + record_load_sweep.SWEEP_REST) + first + second + move
        self.assertAlmostEqual(expected, record_load_sweep.estimated_seconds(poses, options), places=6)


class RealGeometryRobot(RestRobot):
    """the real link lengths, workspace limits and servo objects of servo_config.yaml, no hardware"""

    def __init__(self):
        super().__init__(parse_pose('elbow_out'))
        self.servo_config = load_servo_config()
        self.servo_handler = FakeHandler(self.servo_config)

    def get_angle_limits(self):
        limits = {}
        for servo_id in ARM:
            servo = self.servo_handler.get_servo(servo_id)
            limits[servo_id] = (min(servo.min_angle, servo.max_angle), max(servo.min_angle, servo.max_angle))
        return limits


class DefaultSweepTests(unittest.TestCase):

    def test_the_default_sweeps_pass_the_planner_with_the_real_limits(self):
        robot = RealGeometryRobot()
        self.assertEqual(dynamixel_robot_config, robot.robot_config)
        self.assertIs(WorkSpaceLimits, robot.workspace_limits)
        self.assertEqual({2, 3, 4, 5}, set(DEFAULT_SWEEPS))
        for joint, sweeps in DEFAULT_SWEEPS.items():
            self.assertGreaterEqual(len(sweeps), 2, 'joint {} has at least two poses'.format(joint))
            self.assertIsNone(record_load_sweep.time_based_refusal(robot, joint))
            speeds = sorted({*record_load_sweep.DEFAULT_SPEEDS, record_load_sweep.top_speed(joint)})
            for pose, (low, high) in sweeps:
                with self.subTest(joint=joint, pose=pose, span=(low, high)):
                    self.assertEqual(6, len(pose))
                    self.assertLess(low, high)
                    span = pose_span(parse_pose(','.join(str(d) for d in pose)), joint, low, high)
                    for speed in speeds:
                        plan = plan_sweep(span, joint, speed)
                        self.assertIsNone(record_load_sweep.sweep_refusal(robot, plan),
                                          'joint {} at {} degrees/s'.format(joint, speed))
                    # and the joint stays inside its own range with the offsets of the sweep
                    servo = robot.servo_handler.get_servo(joint)
                    angles = np.degrees(plan.angles[:, joint])
                    self.assertGreaterEqual(angles.min(), min(servo.min_angle, servo.max_angle) * 180 / np.pi)
                    self.assertAlmostEqual(low, angles.min(), delta=0.05)
                    self.assertAlmostEqual(high, angles.max(), delta=0.05)

    def test_the_default_speeds_are_below_the_top_speed_of_every_joint(self):
        for joint in DEFAULT_SWEEPS:
            self.assertLessEqual(max(record_load_sweep.DEFAULT_SPEEDS), record_load_sweep.top_speed(joint))

    def test_the_default_sweeps_have_cruise_steps_in_both_directions(self):
        robot = RealGeometryRobot()
        for joint, sweeps in DEFAULT_SWEEPS.items():
            for pose, (low, high) in sweeps:
                plan = plan_sweep(pose_span(parse_pose(','.join(str(d) for d in pose)), joint, low, high), joint, 5.0)
                self.assertEqual({1, -1}, {step.direction for step in plan.steps if step.cruise})


class MainTests(SweepTestCase):
    """main with the robot replaced by the fake one, the prompt answered and the lowering mocked"""

    def run_main(self, robot, argv, answer='yes', input_error=None, record=None, clock=True):
        """returns (result, printed text, the mock of lower_and_disable_torque, the rows of the csv or None)"""
        printed = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'sweep.csv')
            get_robot = mock.Mock(return_value=robot)
            patches = [mock.patch.object(record_load_sweep, 'get_robot', get_robot),
                       mock.patch.object(record_load_sweep, 'CombinedRobot', SweepRobot),
                       mock.patch.object(record_static_load, 'lower_and_disable_torque'),
                       mock.patch('builtins.input', side_effect=input_error, return_value=answer)]
            if record is not None:
                patches.append(mock.patch.object(record_load_sweep, 'record_sweeps', record))
            with contextlib.ExitStack() as stack:
                mocks = [stack.enter_context(patch) for patch in patches]
                stack.enter_context(contextlib.redirect_stdout(printed))
                result = record_load_sweep.main(['--output', path, *argv], self.clock, self.clock.sleep)
            rows = None
            if os.path.exists(path):
                with open(path, newline='') as written:
                    rows = rows_of(written.read())
        self.get_robot, self.input = get_robot, mocks[3]
        return result, printed.getvalue(), mocks[2], rows

    def span_arguments(self, joint=3, *extra):
        return ['--joint', str(joint), '--pose', ','.join(str(d) for d in ELBOW_OUT), '--span', '-30:30', *extra]

    def test_a_full_run(self):
        robot = self.robot(3, sag=lambda q: 4.0 * np.sin(q[3]), friction=3.0, lag=0.5)
        result, printed, lower, rows = self.run_main(robot, self.span_arguments(3, '--speeds', '10'))
        self.assertEqual(0, result)
        self.assertTrue(robot.enabled)
        lower.assert_called_once_with(robot)
        self.assertGreater(len(rows), 80)
        self.assertEqual(list(FIELDS), list(rows[0]))
        self.assertIn('type yes', self.input.call_args.args[0])
        self.assertIn('1 sweeps, ', printed)
        self.assertIn('about ', printed)
        self.assertIn('joint 3 (yaml p=1500 d=0)', printed)
        self.assertIn('sweep.csv', printed)

    def test_the_prompt_is_not_asked_with_yes_and_a_no_does_nothing(self):
        robot = self.robot(3)
        result, printed, lower, rows = self.run_main(robot, self.span_arguments(3, '--speeds', '10', '--yes'),
                                                     input_error=AssertionError('no prompt expected'))
        self.assertEqual(0, result)
        self.input.assert_not_called()
        robot = self.robot(3)
        result, printed, lower, rows = self.run_main(robot, self.span_arguments(3), answer='no')
        self.assertEqual(0, result)
        self.assertFalse(robot.enabled)
        lower.assert_not_called()
        self.assertIsNone(rows)
        self.assertEqual([], robot.moves)

    def test_without_a_usable_stdin_the_run_needs_yes(self):
        robot = self.robot(3)
        result, printed, lower, rows = self.run_main(robot, self.span_arguments(3), input_error=EOFError)
        self.assertEqual(1, result)
        self.assertIn('--yes', printed)
        self.assertIn('unattended', printed)
        self.assertFalse(robot.enabled)
        lower.assert_not_called()
        self.assertEqual([], robot.moves)
        self.assertIsNone(rows, 'no file either')

    def test_a_servo_without_the_time_based_profile_is_refused_before_anything_moves(self):
        robot = self.robot(3)
        robot.servo_config.servos[3].move_time = None
        result, printed, lower, rows = self.run_main(robot, self.span_arguments(3))
        self.assertEqual(1, result)
        self.assertIn('servo 3 has no move_time in servo_config.yaml, the sweep needs the time-based profile',
                      printed)
        self.assertFalse(robot.enabled)
        self.assertEqual([], robot.moves)
        self.input.assert_not_called()
        lower.assert_not_called()
        # a joint whose servo is fine still runs
        other = self.robot(3)
        other.servo_config.servos[4].move_time = None
        self.assertEqual(0, self.run_main(other, self.span_arguments(3, '--speeds', '10', '--yes'))[0])

    def test_a_speed_above_the_top_speed_is_refused_first(self):
        for joint, speeds, refused in ((2, ['5', '12'], True), (2, ['10'], False), (3, ['35'], True),
                                       (4, ['30'], False), (5, ['31'], True)):
            with self.subTest(joint=joint, speeds=speeds):
                robot = self.robot(joint, (90, 70, 0, 0, 0, 0))
                argv = ['--joint', str(joint), '--pose', '90,70,0,0,0,0', '--span', '-5:5' if joint != 2 else '60:80',
                        '--speeds', *speeds, '--yes']
                result, printed, lower, rows = self.run_main(robot, argv)
                if refused:
                    self.assertEqual(1, result)
                    self.assertIn('joint {} can not be swept faster than {:g} degrees/s'.format(
                        joint, 10 if joint == 2 else 30), printed)
                    self.get_robot.assert_not_called()
                    self.assertFalse(robot.enabled)
                    self.assertEqual([], robot.moves)
                else:
                    self.get_robot.assert_called_once()

    def test_only_the_joints_with_a_static_load_can_be_swept(self):
        for joint in (0, 1, 6, 7):
            robot = self.robot(3)
            result, printed, _, _ = self.run_main(robot, self.span_arguments(joint))
            self.assertEqual(1, result)
            self.assertIn('only joints 2, 3, 4, 5', printed)
            self.get_robot.assert_not_called()

    def test_poses_need_a_span(self):
        robot = self.robot(3)
        result, printed, _, _ = self.run_main(robot, ['--joint', '3', '--pose', 'elbow_out'])
        self.assertEqual(1, result)
        self.assertIn('--pose needs --span', printed)
        self.assertFalse(robot.enabled)

    def test_the_span_and_the_poses_reach_the_recording(self):
        seen = []

        def record(robot, joint, poses, writer, options, stop, clock, sleep, **kwargs):
            seen.append((joint, poses, options))
            return record_load_sweep.Recording()
        robot = self.robot(3)
        argv = ['--joint', '3', '--pose', 'elbow_out', 'low', '--span', '-20:25', '--speeds', '5', '--bin', '2.5',
                '--move-speed', '0.2', '--yes']
        result, printed, lower, _ = self.run_main(robot, argv, record=record)
        self.assertEqual(1, result, 'nothing was recorded')
        (joint, poses, options), = seen
        self.assertEqual(3, joint)
        self.assertEqual([(-20.0, 25.0)] * 2, [(p.start.low, p.start.high) for p in poses])
        self.assertEqual([[5.0]] * 2, [[plan.speed for plan in p.sweeps] for p in poses])
        np.testing.assert_allclose(np.radians([0, 90, 90, 2.5, 0, 0, 0]), poses[0].start.pose)
        np.testing.assert_allclose(np.radians([0, 90, 60, 2.5, 0, 0, 0]), poses[1].start.pose)
        self.assertEqual((2.5, 0.2), (options.bin_degrees, options.move_speed))
        self.assertEqual(robot.servo_config.temperature_limit, options.limits.max_temperature)
        self.assertTrue(robot.enabled)
        lower.assert_called_once_with(robot)

    def test_with_every_sweep_refused_nothing_is_enabled(self):
        robot = self.robot(3)
        argv = ['--joint', '3', '--pose', 'elbow_out', '--span', '-100:100', '--yes']
        result, printed, lower, rows = self.run_main(robot, argv)
        self.assertEqual(1, result)
        self.assertIn('skipping the sweep', printed)
        self.assertIn('no sweep the planner accepts', printed)
        self.assertFalse(robot.enabled)
        self.assertEqual([], robot.moves)
        lower.assert_not_called()
        self.assertIsNone(rows)

    def test_the_arm_is_lowered_whatever_ended_the_recording(self):
        def temperature(robot):
            robot.temperatures[3] = 50 if robot.clock.now > START_OF_ABORT else 30

        def failing(robot):
            robot.failing = robot.clock.now > START_OF_ABORT

        def interrupt(robot):
            if robot.clock.now > START_OF_ABORT:
                raise KeyboardInterrupt
        for name, hook, expected in (('an abort rule', temperature, 'temperature'),
                                     ('a failed read', failing, 'failed to read servo 3'),
                                     ('Ctrl+C', interrupt, 'killed')):
            with self.subTest(name):
                self.clock.now = 0.0
                robot = self.robot(3, sag=lambda q: 4.0 * np.sin(q[3]), friction=3.0, lag=0.5)
                robot.before_read = hook
                result, printed, lower, rows = self.run_main(robot, self.span_arguments(3, '--speeds', '10', '5',
                                                                                        '--yes'))
                self.assertEqual(0, result, 'one sweep was recorded')
                lower.assert_called_once_with(robot)
                self.assertIn('stopped: ', printed)
                self.assertIn(expected, printed)
                self.assertTrue(rows)
                self.assertEqual({'10'}, {row['speed'] for row in rows}, 'the sweep that finished is in the file')
                self.assertEqual(0, len(rows) % 12)
                self.assertIn('1 sweeps, ', printed)

    def test_a_recording_that_stops_before_the_first_sweep_ends_with_1_and_still_lowers(self):
        robot = self.robot(3)
        robot.hardware_errors = {3: ['overload']}
        result, printed, lower, rows = self.run_main(robot, self.span_arguments(3, '--yes'))
        self.assertEqual(1, result)
        self.assertTrue(robot.enabled)
        lower.assert_called_once_with(robot)
        self.assertEqual([], rows)
        self.assertIn('0 sweeps, 0 bins', printed)

    def test_the_lowering_is_the_last_thing_even_when_the_recording_raises(self):
        robot = self.robot(3)
        result = None
        with self.assertRaises(RuntimeError):
            result, printed, lower, rows = self.run_main(robot, self.span_arguments(3, '--yes'),
                                                         record=mock.Mock(side_effect=RuntimeError('bug')))
        self.assertIsNone(result)

    def test_the_servos_are_not_enabled_when_the_file_can_not_be_written(self):
        robot = self.robot(3)
        with mock.patch('builtins.open', side_effect=OSError('read only')), \
                mock.patch.object(record_load_sweep, 'get_robot', return_value=robot), \
                mock.patch.object(record_load_sweep, 'CombinedRobot', SweepRobot), \
                mock.patch.object(record_static_load, 'lower_and_disable_torque') as lower, \
                contextlib.redirect_stdout(io.StringIO()) as printed:
            result = record_load_sweep.main(self.span_arguments(3, '--yes', '--output', 'x.csv'))
        self.assertEqual(1, result)
        self.assertIn('can not write x.csv', printed.getvalue())
        self.assertFalse(robot.enabled)
        lower.assert_not_called()

    def test_only_the_real_robot_is_swept(self):
        printed = io.StringIO()
        with mock.patch.object(record_load_sweep, 'get_robot', return_value=object()), \
                contextlib.redirect_stdout(printed):
            result = record_load_sweep.main(self.span_arguments(3, '--yes'))
        self.assertEqual(1, result)
        self.assertIn('real robot', printed.getvalue())

    def test_the_floor_replaces_the_workspace_floor(self):
        robot = self.robot(3)
        # a floor of 37 cm is above the wrist and the tip of the elbow-out pose, so the sweeps are refused
        result, printed, _, _ = self.run_main(robot, self.span_arguments(3, '--floor', '37', '--speeds', '10', '--yes'))
        self.assertEqual(37.0, robot.workspace_limits.z_min)
        self.assertEqual(37.0, robot.workspace_limits.wrist_z_min)
        self.assertEqual(1, result)
        self.assertIn('below the minimum height of 37', printed)
        self.assertFalse(robot.enabled)


if __name__ == '__main__':
    unittest.main()
