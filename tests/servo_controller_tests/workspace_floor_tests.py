"""
Do the tuning and calibration scripts and the xbox control still run under the floor of WorkSpaceLimits? Everything
they make the arm do is planned and checked against the real limits and the real link lengths, so a floor that is
raised too far makes a script refuse its own poses. These tests plan what the scripts plan, without hardware, with the
planner's own functions: a pose that comes lower than the floor allows, or a floor above the poses, fails here.
"""
import unittest
from copy import copy

import numpy as np

from src.global_constants import WorkSpaceLimits, dynamixel_robot_config
from src.kinematics.kinematics import forward_kinematics, forward_position_kinematics, inverse_kinematics
from src.robot_controllers.dynamixel_robot.calibration import record_load_sweep, record_static_load, tune_servo_gains
from src.robot_controllers.dynamixel_robot.calibration.offline.preview_static_load_poses import OfflineRobot
from src.robot_controllers.dynamixel_robot.calibration.tune_servo_gains import (
    DEFAULT_POSES, POSES, Settings, parse_pose)
from src.robot_controllers.dynamixel_robot.servo_config import load_servo_config
from src.utils.movement import plan_move_to_angles, plan_move_to_pose
from src.utils.movement_exception import MovementException
from src.xbox_control.xbox_robot_controller import XboxRobotController
from tests.servo_controller_tests.static_load_scripts_tests import RESTING_ANGLES, FakeHandler, RestRobot

# the floors of the tip and the wrist centre in cm and the distance of the wrist centre from the base axis, written out
# here on purpose: the tests below have to fail when WorkSpaceLimits goes higher than the poses of the scripts, not
# follow it. The tip is the end of the closed fingertips (d6 15 cm), the same 1 cm floor the fingertips had with the
# 14 cm gripper and a floor of 2 cm for its tip
TIP_FLOOR = 1.0
WRIST_FLOOR = 1.0
WRIST_RADIUS_MIN = 12.5
# the command line of the recordings of 21 to 24 November 2026
RECORDING = ['--poses', '30', '--joints', '2', '3', '4', '5', '--range', '2=45:115', '3=-60:60', '4=-60:60',
             '5=-60:60']
ARM = (1, 2, 3, 4, 5, 6)
DEFAULT_SWEEPS_JOINTS = tuple(record_load_sweep.DEFAULT_SWEEPS)
STEP_TEST_SPEEDS = (record_static_load.REST_SPEED, Settings().speed)  # rad/s of the moves of the scripts


class RealRobot(RestRobot):
    """the real link lengths, workspace limits and the servo objects of servo_config.yaml, no hardware"""

    def __init__(self, angles=RESTING_ANGLES):
        super().__init__(angles)
        self.servo_config = load_servo_config()
        self.servo_handler = FakeHandler(self.servo_config)

    def get_angle_limits(self):
        limits = {}
        for servo_id in ARM:
            servo = self.servo_handler.get_servo(servo_id)
            limits[servo_id] = (min(servo.min_angle, servo.max_angle), max(servo.min_angle, servo.max_angle))
        return limits


def heights(angles):
    """(wrist centre, gripper tip) in cm above the table at the angles"""
    _, _, _, wrist, tip = forward_position_kinematics(angles, dynamixel_robot_config)
    return float(wrist[2]), float(tip[2])


class TheLimitsTests(unittest.TestCase):

    def test_the_floors_are_where_the_fingertips_and_the_wrist_servo_stay_off_the_table(self):
        self.assertEqual(TIP_FLOOR, WorkSpaceLimits.z_min, 'the tip is the end of the closed fingertips')
        self.assertEqual(WRIST_FLOOR, WorkSpaceLimits.wrist_z_min)

    def test_the_wrist_centre_stays_off_the_base_column(self):
        self.assertEqual(WRIST_RADIUS_MIN, WorkSpaceLimits.wrist_radius_min)
        self.assertEqual(dynamixel_robot_config.d6 + 0.1, WorkSpaceLimits.y_min)

    def test_the_arm_rests_above_both_floors_so_a_move_can_start_from_rest(self):
        wrist, tip = heights(RESTING_ANGLES)
        # 3.7 cm was read with the 14 cm gripper on 24 November 2026, the 15 cm gripper puts its tip 0.1 cm higher
        self.assertAlmostEqual(3.8, tip, delta=0.2, msg='the tip at rest')
        self.assertAlmostEqual(2.3, wrist, delta=0.3, msg='the wrist centre at rest')
        self.assertGreater(tip, WorkSpaceLimits.z_min)
        self.assertGreater(wrist, WorkSpaceLimits.wrist_z_min)
        RealRobot().check_workspace(RESTING_ANGLES)

    def test_the_real_limits_and_link_lengths_are_the_ones_the_tests_plan_with(self):
        robot = RealRobot()
        self.assertIs(WorkSpaceLimits, robot.workspace_limits)
        self.assertEqual(dynamixel_robot_config, robot.robot_config)
        self.assertEqual(15, dynamixel_robot_config.d6, 'the gripper length stays 15 cm')

    def test_the_planner_refuses_a_tip_under_the_floor_and_accepts_one_over_it_with_the_gripper_pointing_down(self):
        robot = RealRobot()
        down = forward_kinematics(np.radians([0, 90, 90, -12, 0, -90, 0]), dynamixel_robot_config)
        for z, accepted in ((0.9, False), (1.1, True), (4.0, True)):
            pose = copy(down)
            pose.z = z
            angles = inverse_kinematics(pose, dynamixel_robot_config)
            wrist, tip = heights(angles)
            with self.subTest(tip=z):
                self.assertAlmostEqual(z, tip, places=6)
                self.assertGreater(wrist - tip, 13, 'the gripper points about straight down')
                if accepted:
                    robot.check_workspace(angles)
                else:
                    with self.assertRaises(MovementException):
                        robot.check_workspace(angles)


class MovesFromRestTests(unittest.TestCase):
    """the moves to the tuning poses from the resting arm and from them to the rest pose (lower_to_rest)"""

    def setUp(self):
        self.robot = RealRobot()
        self.rest_pose_angles = inverse_kinematics(record_static_load.REST_POSE, dynamixel_robot_config)

    def test_the_rest_pose_is_inside_the_limits_with_the_tip_above_the_floor(self):
        self.robot.check_workspace(self.rest_pose_angles)
        wrist, tip = heights(self.rest_pose_angles)
        self.assertAlmostEqual(record_static_load.REST_POSE.z, tip, delta=0.01)
        self.assertGreater(tip, TIP_FLOOR)
        self.assertGreater(wrist, WRIST_FLOOR)

    def test_the_planner_accepts_the_move_from_rest_to_every_tuning_pose(self):
        for name in POSES:
            for speed in STEP_TEST_SPEEDS:
                with self.subTest(pose=name, speed=speed):
                    target = parse_pose(name)
                    move = plan_move_to_angles(self.robot, RESTING_ANGLES, target, angular_speed=speed)
                    np.testing.assert_allclose(target, move.angles[-1], atol=1e-9)
                    self.assertGreaterEqual(min(heights(angles)[1] for angles in move.angles), TIP_FLOOR - 1e-6)

    def test_the_planner_accepts_the_move_from_every_tuning_pose_to_the_rest_pose(self):
        for name in POSES:
            with self.subTest(pose=name):
                move = plan_move_to_angles(self.robot, parse_pose(name), self.rest_pose_angles,
                                           angular_speed=record_static_load.REST_SPEED)
                np.testing.assert_allclose(self.rest_pose_angles, move.angles[-1], atol=1e-9)

    def test_the_planner_accepts_the_move_from_rest_to_the_rest_pose_and_back(self):
        for start, target in ((RESTING_ANGLES, self.rest_pose_angles), (self.rest_pose_angles, RESTING_ANGLES)):
            plan_move_to_angles(self.robot, start, target, angular_speed=record_static_load.REST_SPEED)


class TuneServoGainsTests(unittest.TestCase):
    """what every stage of tune_servo_gains does at the default poses of every joint, by the script's own checks"""

    @classmethod
    def setUpClass(cls):
        cls.robot = RealRobot()
        cls.settings = Settings()

    def test_every_joint_has_default_poses_and_they_are_the_named_ones(self):
        self.assertEqual(set(ARM), set(DEFAULT_POSES))
        for joint, names in DEFAULT_POSES.items():
            self.assertTrue(names)
            self.assertTrue(set(names) <= set(POSES), joint)

    def test_the_excursions_of_the_step_tests_pass_the_planner_at_every_default_pose(self):
        # main() checks the pose and the pose moved by this many degrees either way, before anything moves
        excursion = np.radians(max(self.settings.move_degrees, 6.0))
        for joint in ARM:
            for name in DEFAULT_POSES[joint]:
                with self.subTest(joint=joint, pose=name):
                    self.assertIsNone(tune_servo_gains.pose_is_accepted(parse_pose(name), self.robot, joint,
                                                                       excursion))

    def test_the_largest_step_of_a_trial_is_within_the_excursion_that_is_checked(self):
        # a step is in positions of the servo, 4096 to a turn; the check above covers at least 6 degrees
        for joint in ARM:
            largest = tune_servo_gains.MAX_STEP.get(joint, tune_servo_gains.DEFAULT_MAX_STEP)
            self.assertLessEqual(largest * 360 / 4096, 6.0, joint)

    def test_the_streamed_moves_of_the_trials_pass_the_planner_at_every_default_pose(self):
        # move_test plans the pose to the pose moved by move_degrees and back, in joint space, at --speed
        excursion = np.radians(self.settings.move_degrees)
        for joint in ARM:
            for name in DEFAULT_POSES[joint]:
                with self.subTest(joint=joint, pose=name):
                    pose = parse_pose(name)
                    target = pose.copy()
                    target[joint] += excursion
                    plan_move_to_angles(self.robot, pose, target, angular_speed=self.settings.speed)
                    plan_move_to_angles(self.robot, target, pose, angular_speed=self.settings.speed)

    def test_the_feedforward_stage_runs_at_every_default_pose_with_the_default_degrees_and_speeds(self):
        for joint in ARM:
            for name in DEFAULT_POSES[joint]:
                with self.subTest(joint=joint, pose=name):
                    self.assertIsNone(tune_servo_gains.feedforward_refusal(self.robot, joint, parse_pose(name),
                                                                          self.settings))

    def test_the_swing_stage_runs_at_every_default_pose(self):
        for joint in ARM:
            for name in DEFAULT_POSES[joint]:
                with self.subTest(joint=joint, pose=name):
                    self.assertIsNone(tune_servo_gains.swing_refusal(self.robot, joint, parse_pose(name),
                                                                    self.settings))

    def test_the_lowest_point_of_the_sweeps_is_above_the_floor(self):
        # what the refusals above accept, measured: the lowest tip and wrist of every sweep of every stage
        lowest_tip, lowest_wrist = float('inf'), float('inf')
        for joint in ARM:
            for name in DEFAULT_POSES[joint]:
                pose = parse_pose(name)
                sweeps = [tune_servo_gains.sweep_angles(pose, joint, speed, self.settings)[1]
                          for speed in self.settings.ff_speeds_of(joint)]
                sweeps.append(tune_servo_gains.swing_sweep_angles(pose, joint, self.settings)[1])
                for angles in np.concatenate(sweeps):
                    wrist, tip = heights(angles)
                    lowest_tip, lowest_wrist = min(lowest_tip, tip), min(lowest_wrist, wrist)
        self.assertGreaterEqual(lowest_tip, TIP_FLOOR - 1e-6)
        self.assertGreaterEqual(lowest_wrist, WRIST_FLOOR - 1e-6)


class RecordLoadSweepTests(unittest.TestCase):
    """every default sweep of record_load_sweep at the default speeds, and the moves to and from it"""

    @classmethod
    def setUpClass(cls):
        cls.robot = RealRobot()
        cls.rest_pose_angles = inverse_kinematics(record_static_load.REST_POSE, dynamixel_robot_config)

    def test_every_default_sweep_passes_the_planner_at_the_default_speeds(self):
        for joint in DEFAULT_SWEEPS_JOINTS:
            spans = record_load_sweep.default_poses(joint)
            self.assertEqual(len(record_load_sweep.DEFAULT_SWEEPS[joint]), len(spans))
            for span in spans:
                for speed in record_load_sweep.DEFAULT_SPEEDS:
                    with self.subTest(joint=joint, pose=span.pose, speed=speed):
                        plan = record_load_sweep.plan_sweep(span, joint, speed)
                        self.assertIsNone(record_load_sweep.sweep_refusal(self.robot, plan))
            planned = record_load_sweep.plan_poses(self.robot, joint, spans, record_load_sweep.DEFAULT_SPEEDS)
            self.assertEqual(len(spans), len(planned), 'joint {}: a pose lost all its sweeps'.format(joint))
            for pose in planned:
                self.assertEqual(len(record_load_sweep.DEFAULT_SPEEDS), len(pose.sweeps))

    def test_the_lowest_point_of_every_default_sweep_is_above_the_floor(self):
        for joint in DEFAULT_SWEEPS_JOINTS:
            for span in record_load_sweep.default_poses(joint):
                for speed in record_load_sweep.DEFAULT_SPEEDS:
                    with self.subTest(joint=joint, pose=span.pose, speed=speed):
                        for angles in record_load_sweep.plan_sweep(span, joint, speed).angles:
                            wrist, tip = heights(angles)
                            self.assertGreaterEqual(tip, TIP_FLOOR - 1e-6)
                            self.assertGreaterEqual(wrist, WRIST_FLOOR - 1e-6)

    def test_the_moves_from_rest_to_the_start_of_every_pose_and_back_to_the_rest_pose_are_accepted(self):
        speed = record_load_sweep.Options().move_speed
        for joint in DEFAULT_SWEEPS_JOINTS:
            for span in record_load_sweep.default_poses(joint):
                with self.subTest(joint=joint, pose=span.pose):
                    plan_move_to_angles(self.robot, RESTING_ANGLES, span.pose, angular_speed=speed)
                    plan_move_to_angles(self.robot, span.pose, self.rest_pose_angles, angular_speed=speed)

    def test_the_moves_between_the_poses_of_a_joint_are_accepted(self):
        speed = record_load_sweep.Options().move_speed
        for joint in DEFAULT_SWEEPS_JOINTS:
            spans = record_load_sweep.default_poses(joint)
            for before, after in zip(spans, spans[1:]):
                with self.subTest(joint=joint, before=before.pose, after=after.pose):
                    plan_move_to_angles(self.robot, before.pose, after.pose, angular_speed=speed)



class RecordStaticLoadTests(unittest.TestCase):
    """the recordings of 21 to 24 November 2026: the poses the recorder picks and the moves it makes"""

    @classmethod
    def setUpClass(cls):
        cls.robot = OfflineRobot()
        cls.rest_pose_angles = inverse_kinematics(record_static_load.REST_POSE, dynamixel_robot_config)

    def select(self, seed):
        args = record_static_load.build_parser().parse_args(RECORDING + ['--seed', str(seed)])
        poses, _ = record_static_load.select_poses(args, self.robot)
        return args, poses

    def test_the_offline_robot_has_the_real_limits(self):
        self.assertIs(WorkSpaceLimits, self.robot.workspace_limits)

    def test_the_recorder_still_finds_30_poses_for_the_seeds_of_the_recordings(self):
        for seed in (1, 6, 12):
            with self.subTest(seed=seed):
                _, poses = self.select(seed)
                self.assertEqual(30, len(poses))

    def test_every_pose_and_its_overshoots_have_the_tip_and_the_wrist_above_the_floors(self):
        for seed in (1, 6, 12):
            args, poses = self.select(seed)
            delta = np.radians(args.delta)
            for number, pose in enumerate(poses, 1):
                for overshoot, target, direction in record_static_load.approach_sequence(
                        pose, delta, args.joints, self.robot.get_angle_limits()):
                    for name, angles in (('overshoot', overshoot), ('pose', target)):
                        wrist, tip = heights(angles)
                        with self.subTest(seed=seed, pose=number, direction=direction, stop=name):
                            self.assertGreaterEqual(tip, TIP_FLOOR - 1e-6)
                            self.assertGreaterEqual(wrist, WRIST_FLOOR - 1e-6)

    def test_the_move_from_rest_to_the_first_pose_and_from_the_last_pose_to_the_rest_pose_are_accepted(self):
        for seed in (1, 6, 12):
            args, poses = self.select(seed)
            with self.subTest(seed=seed):
                plan_move_to_angles(self.robot, RESTING_ANGLES, poses[0], angular_speed=record_static_load.REST_SPEED)
                plan_move_to_angles(self.robot, poses[-1], self.rest_pose_angles,
                                    angular_speed=record_static_load.REST_SPEED)

    def test_the_moves_of_the_recording_between_the_stops_are_accepted(self):
        # record(): from the current angles to the overshoot, to the pose, for the two directions of every pose, then
        # lowered to the rest pose
        for seed in (1, 6, 12):
            args, poses = self.select(seed)
            delta = np.radians(args.delta)
            current = RESTING_ANGLES
            for number, pose in enumerate(poses, 1):
                for overshoot, target, direction in record_static_load.approach_sequence(
                        pose, delta, args.joints, self.robot.get_angle_limits()):
                    for stop in (overshoot, target):
                        with self.subTest(seed=seed, pose=number, direction=direction):
                            plan_move_to_angles(self.robot, current, stop, angular_speed=record_static_load.REST_SPEED)
                        current = stop
            plan_move_to_angles(self.robot, current, self.rest_pose_angles,
                                angular_speed=record_static_load.REST_SPEED)


class XboxStartTests(unittest.TestCase):

    def test_the_start_pose_is_above_the_floors(self):
        angles = inverse_kinematics(XboxRobotController.start_pose, dynamixel_robot_config)
        wrist, tip = heights(angles)
        self.assertGreater(tip, TIP_FLOOR)
        self.assertGreater(wrist, WRIST_FLOOR)

    def test_the_planner_accepts_the_move_from_rest_to_the_start_pose(self):
        # start() plans the straight line from the current pose of the robot to the start pose before it enables the
        # servos, and refuses to start when it can't
        robot = RealRobot()
        move = plan_move_to_pose(robot, RESTING_ANGLES, XboxRobotController.start_pose)
        self.assertGreaterEqual(min(heights(angles)[1] for angles in move.angles), TIP_FLOOR - 1e-6)
        self.assertGreaterEqual(min(heights(angles)[0] for angles in move.angles), WRIST_FLOOR - 1e-6)


if __name__ == '__main__':
    unittest.main()
