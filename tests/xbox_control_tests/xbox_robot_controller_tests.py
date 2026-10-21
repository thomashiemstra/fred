import threading
import time
import unittest
from copy import copy
from unittest import mock

import numpy as np
from numpy import pi

from src import global_constants
from src.kinematics.kinematics import inverse_kinematics, forward_kinematics, forward_position_kinematics
from src.kinematics.kinematics_utils import Pose, RobotConfig
from src.utils import movement, workspace
from src.utils.movement import SplineMovement, PoseToPoseMovement
from src.utils.movement_exception import MovementException
from src.xbox_control import xbox_robot_controller
from src.xbox_control.xbox360controller.Xbox360Controller import Buttons
from src.xbox_control.xbox_robot_controller import determine_time, create_move, XboxRobotController

test_config = RobotConfig(d1=13.92, a2=20, d4=22, d6=12)
angle_limits = {1: (0, pi), 2: (0, pi), 3: (-pi / 2, pi / 2), 4: (-pi, pi), 5: (-pi, pi), 6: (-pi, pi)}
# the angles of the robot resting on the table
resting_angles = [0, 0.183, 0.0, 0.736, -1.437, 1.689, 0.012]


class UtilsTests(unittest.TestCase):

    def test_determine_time(self):
        pose1 = Pose(0, 0, 0)
        pose2 = Pose(1, 0, 0)
        pose3 = Pose(1, 1, 0)
        poses = [pose1, pose2, pose3]

        time = determine_time(poses, 1)
        self.assertIsNotNone(time, "should have gotten a time")
        self.assertEqual(2.0, time, "Should have gotten a time of 2 seconds")

    def test_determine_time_orientation_adjustment(self):
        pose1 = Pose(0, 0, 0, alpha=0)
        pose2 = Pose(1, 0, 0, alpha=1)
        poses = [pose1, pose2]

        time = determine_time(poses, 1)
        self.assertIsNotNone(time, "should have gotten a time")
        self.assertTrue(time > 0, "should have gotten a positive time")

    class MockWorkSpaceLimits:
        radius_min = 15
        radius_max = 40
        y_min = 16
        z_min = 5

    def robot_with_limits(self, limits):
        robot = FakeRobot()
        robot.workspace_limits = limits
        return robot

    def test_create_move_spline(self):
        poses = [Pose(-20, 20, 5), Pose(0, 30, 10), Pose(20, 20, 5)]
        move = create_move(self.robot_with_limits(None), poses, 1, None)
        self.assertIsInstance(move, SplineMovement)

    def test_not_in_workspace_limits(self):
        # the curve bulges out to more than 40cm from the base
        poses = [Pose(-20, 20, 5), Pose(0, 45, 10), Pose(20, 20, 5)]
        move = create_move(self.robot_with_limits(self.MockWorkSpaceLimits), poses, 1, None)
        self.assertIsNone(move)

    def test_in_workspace_limits(self):
        poses = [Pose(-20, 20, 5), Pose(0, 30, 10), Pose(20, 20, 5)]
        move = create_move(self.robot_with_limits(self.MockWorkSpaceLimits), poses, 1, None)
        self.assertIsInstance(move, SplineMovement)

    def test_create_move_pose_to_pose(self):
        poses = [Pose(-20, 20, 5), Pose(0, 30, 10)]
        move = create_move(self.robot_with_limits(self.MockWorkSpaceLimits), poses, 1, None)
        self.assertIsInstance(move, PoseToPoseMovement)

    def test_pose_to_pose_outside_the_workspace_is_refused(self):
        poses = [Pose(-20, 20, 5), Pose(0, 45, 10)]
        self.assertIsNone(create_move(self.robot_with_limits(self.MockWorkSpaceLimits), poses, 1, None))


class TestWorkspaceLimits:
    """the limits of these tests: the resting angles above put the wrist of the test config below the table"""
    radius_min = 10
    radius_max = 57
    y_min = 12.1
    z_min = -5
    wrist_z_min = -5


class FakeRobot:
    """a robot with real kinematics that records what it is asked to do"""

    workspace_limits = TestWorkspaceLimits

    def __init__(self, angles=resting_angles):
        self.robot_config = test_config
        self.angles = np.array(angles, dtype=np.float64)
        self.enabled = False
        self.enable_result = True
        self.moved_by_hand_after_enabling = False
        self.hardware_errors = {}
        self.overheating = {}
        self.gripper_calls = []
        self.blocked_gripper_state = None
        self.moving_threads = set()
        self.calls = []
        self.poses = []  # the poses of move_to_pose
        self.sent_angles = []  # the angles of move_servos
        self.angle_limits = angle_limits
        self.physics_client = 3
        # optional function called before every step of a straight line movement
        self.on_move_servos = None

    def get_current_angles(self):
        return None if self.angles is None else self.angles.copy()

    def get_angle_limits(self):
        return self.angle_limits

    def pose_to_angles(self, pose):
        return inverse_kinematics(pose, self.robot_config)

    def workspace_violation(self, angles):
        return workspace.workspace_violation(angles, self.robot_config, self.workspace_limits)

    def check_workspace(self, angles):
        workspace.check_workspace(angles, self.robot_config, self.workspace_limits)

    def enable_servos(self):
        self.calls.append('enable_servos')
        self.enabled = self.enable_result
        if self.moved_by_hand_after_enabling:
            self.angles[1] += 0.3
        return self.enable_result

    def disable_servos(self):
        self.calls.append('disable_servos')
        self.enabled = False

    def move_servos(self, angles):
        self.moving_threads.add(threading.current_thread().name)
        if self.on_move_servos is not None:
            self.on_move_servos()
        self.angles = np.array(angles, dtype=np.float64)
        self.sent_angles.append(self.angles.copy())

    def move_to_pose(self, pose):
        self.moving_threads.add(threading.current_thread().name)
        self.poses.append(copy(pose))
        return 0, 0

    def set_gripper(self, new_gripper_state):
        self.gripper_calls.append(new_gripper_state)

    def get_gripper_state(self):
        # an object blocks the gripper at blocked_gripper_state
        if self.blocked_gripper_state is not None and self.gripper_calls:
            return min(self.gripper_calls[-1], self.blocked_gripper_state)
        return self.gripper_calls[-1] if self.gripper_calls else 0.0

    def get_hardware_errors(self):
        return self.hardware_errors

    def get_overheating_servos(self):
        return self.overheating


def make_controller(robot, buttons=None):
    pose_updater = mock.Mock()
    pose_updater.dt = 0.01
    pose_updater.controller.wait_until_connected.return_value = True
    pose_updater.controller.get_buttons.side_effect = lambda: buttons() if buttons else Buttons()
    pose_updater.get_updated_pose_from_controller.side_effect = lambda pose, find_center_mode, center, dt=None: pose
    return XboxRobotController(test_config, robot, pose_updater)


def wait_for(condition, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError('timed out')
        time.sleep(0.01)


class StartTests(unittest.TestCase):

    def setUp(self):
        # don't wait between the steps of the straight line movements
        patcher = mock.patch.object(movement, 'sleep')
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_robot_does_not_start_without_a_gamepad(self):
        robot = FakeRobot()
        controller = make_controller(robot)
        controller.pose_updater.controller.wait_until_connected.return_value = False

        self.assertFalse(controller.start())
        self.assertEqual([], robot.calls)
        self.assertFalse(controller.is_running())

    def test_robot_does_not_start_when_the_start_pose_can_not_be_reached(self):
        robot = FakeRobot()
        controller = make_controller(robot)
        controller.start_pose = Pose(60, 22, 4)

        self.assertFalse(controller.start())
        # the servos are never enabled
        self.assertEqual([], robot.calls)

    def test_robot_does_not_start_when_the_angles_can_not_be_read(self):
        robot = FakeRobot()
        robot.angles = None
        controller = make_controller(robot)

        self.assertFalse(controller.start())
        self.assertEqual([], robot.calls)

    def test_robot_does_not_start_when_the_servos_can_not_be_enabled(self):
        robot = FakeRobot()
        robot.enable_result = False
        controller = make_controller(robot)

        self.assertFalse(controller.start())
        self.assertFalse(controller.is_running())

    def test_robot_does_not_start_when_it_was_moved_before_enabling(self):
        robot = FakeRobot()
        robot.moved_by_hand_after_enabling = True
        controller = make_controller(robot)

        self.assertFalse(controller.start())
        self.assertEqual(['enable_servos', 'disable_servos'], robot.calls)

    def test_start_moves_in_a_straight_line_to_the_start_pose_and_stop_moves_back(self):
        robot = FakeRobot()
        controller = make_controller(robot)

        self.assertTrue(controller.start())
        self.assertTrue(controller.is_running())
        self.assertFalse(controller.start())

        wait_for(lambda: robot.gripper_calls)  # the gripper opens after the start move
        reached = forward_kinematics(robot.angles, test_config)
        np.testing.assert_allclose([30, 22, 4], [reached.x, reached.y, reached.z], atol=1e-6)
        np.testing.assert_allclose([30, 22, 4], [controller.current_pose.x, controller.current_pose.y,
                                                 controller.current_pose.z], atol=1e-6)

        robot.angles = inverse_kinematics(Pose(25, 25, 10), test_config)
        self.assertTrue(controller.stop())
        reached = forward_kinematics(robot.angles, test_config)
        np.testing.assert_allclose([30, 22, 4], [reached.x, reached.y, reached.z], atol=1e-6)
        self.assertEqual(['enable_servos', 'disable_servos'], robot.calls)
        self.assertFalse(controller.is_running())
        self.assertFalse(controller.stop())

    def test_stop_disables_the_servos_when_the_control_loop_crashed(self):
        robot = FakeRobot()
        robot.hardware_errors = {5: ['overload']}
        controller = make_controller(robot)

        with mock.patch.object(xbox_robot_controller, 'HARDWARE_CHECK_INTERVAL', 0.0):
            self.assertTrue(controller.start())
            wait_for(lambda: not controller.thread.is_alive())
        # the servos keep holding their position until stop
        self.assertTrue(controller.is_running())
        self.assertEqual(['enable_servos'], robot.calls)

        self.assertTrue(controller.stop())
        self.assertEqual(['enable_servos', 'disable_servos'], robot.calls)


class CommandTests(unittest.TestCase):

    def setUp(self):
        patcher = mock.patch.object(movement, 'sleep')
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_find_center_mode_is_not_possible_when_the_robot_is_not_running(self):
        controller = make_controller(FakeRobot())
        self.assertFalse(controller.request_find_center_mode())

    def test_find_center_mode_moves_the_robot_in_the_control_loop(self):
        robot = FakeRobot()
        controller = make_controller(robot)
        reset_threads = []

        def fake_reset_orientation(pose, robot_config, reset_robot):
            reset_threads.append(threading.current_thread())
            return pose

        with mock.patch.object(xbox_robot_controller, 'reset_orientation', fake_reset_orientation):
            self.assertTrue(controller.start())
            self.assertTrue(controller.request_find_center_mode())
            wait_for(lambda: reset_threads)
            self.assertEqual([controller.thread], reset_threads)
            self.assertTrue(controller.find_center_mode)
            controller.stop()

        self.assertEqual(1, len(robot.moving_threads))
        self.assertNotIn(threading.current_thread().name, robot.moving_threads)


class GripperTests(unittest.TestCase):

    def test_bumpers_move_the_gripper(self):
        robot = FakeRobot()
        pressed = []
        controller = make_controller(robot, buttons=lambda: pressed.pop(0) if pressed else Buttons())

        pressed.extend([Buttons(rb=True), Buttons(rb=True), Buttons(lb=True)])
        for _ in range(3):
            controller.handle_buttons()

        self.assertEqual([1, 2, 1], robot.gripper_calls)

    def test_blocked_gripper_squeezes_without_running_away(self):
        robot = FakeRobot()
        robot.blocked_gripper_state = 20
        controller = make_controller(robot, buttons=lambda: Buttons(rb=True))

        for _ in range(50):
            controller.handle_buttons()

        # the target stays GRIPPER_MAX_LEAD beyond the object, so the gripper keeps squeezing it
        self.assertEqual(20 + xbox_robot_controller.GRIPPER_MAX_LEAD, controller.gripper_state)

        # and opening the gripper releases the object within GRIPPER_MAX_LEAD steps, instead of first undoing
        # everything the target would have run ahead
        controller.pose_updater.controller.get_buttons.side_effect = lambda: Buttons(lb=True)
        for _ in range(xbox_robot_controller.GRIPPER_MAX_LEAD + 1):
            controller.handle_buttons()
        self.assertLess(robot.gripper_calls[-1], 20)

    def test_gripper_follows_the_target_when_its_state_can_not_be_read(self):
        robot = FakeRobot()
        robot.get_gripper_state = lambda: None
        controller = make_controller(robot, buttons=lambda: Buttons(rb=True))

        controller.handle_buttons()

        self.assertEqual([1], robot.gripper_calls)

    def test_gripper_is_not_written_beyond_its_range(self):
        robot = FakeRobot()
        controller = make_controller(robot, buttons=lambda: Buttons(lb=True))

        controller.handle_buttons()

        self.assertEqual([], robot.gripper_calls)
        self.assertEqual(0, controller.gripper_state)


class ButtonTests(unittest.TestCase):

    def test_every_pressed_button_is_handled(self):
        robot = FakeRobot()
        controller = make_controller(robot, buttons=lambda: Buttons(rb=True, a=True, y=True))
        controller.current_pose = Pose(30, 22, 4)
        controller.pose_updater.maximum_speed = 20

        controller.handle_buttons()

        self.assertEqual([1], robot.gripper_calls)
        self.assertTrue(controller.current_pose.flip)
        self.assertEqual(1, len(controller.recorded_positions))

    def test_buttons_are_handled_in_the_order_of_the_actions(self):
        controller = make_controller(FakeRobot(), buttons=lambda: Buttons(pad_down=True, a=True, rb=True, start=True))
        order = []
        for button in controller._button_actions:
            controller._button_actions[button] = lambda button=button: order.append(button)

        controller.handle_buttons()

        self.assertEqual(['start', 'rb', 'a', 'pad_down'], order)

    def test_presses_are_not_lost_while_a_bumper_is_held(self):
        # like the gamepad: the bumper stays pressed, a momentary button is reported once
        robot = FakeRobot()
        pressed = [Buttons(rb=True), Buttons(rb=True, a=True), Buttons(rb=True), Buttons(rb=True, pad_up=True)]
        controller = make_controller(robot, buttons=lambda: pressed.pop(0))
        controller.current_pose = Pose(30, 22, 4)
        controller.pose_updater.maximum_speed = 20

        for _ in range(4):
            controller.handle_buttons()

        self.assertEqual([1, 2, 3, 4], robot.gripper_calls)  # the held bumper closes the gripper every read
        self.assertTrue(controller.current_pose.flip)
        self.assertEqual(25, controller.pose_updater.maximum_speed)

    def test_buttons_are_not_handled_after_the_robot_is_stopped(self):
        controller = make_controller(FakeRobot(), buttons=lambda: Buttons(start=True, b=True, a=True))
        controller.current_pose = Pose(30, 22, 4)
        controller.recorded_moves = [mock.Mock()]

        with mock.patch.object(controller, 'playback_recorded_moves',
                               side_effect=lambda moves: controller._stop_requested.set() or controller.current_pose),                 mock.patch.object(xbox_robot_controller, 'reset_orientation') as reset:
            controller.handle_buttons()

        reset.assert_not_called()
        self.assertFalse(controller.current_pose.flip)

    def test_a_toggles_the_flip(self):
        controller = make_controller(FakeRobot(), buttons=lambda: Buttons(a=True))
        controller.current_pose = Pose(30, 22, 4)

        controller.handle_buttons()

        self.assertTrue(controller.current_pose.flip)

    def test_d_pad_changes_the_maximum_speed_within_its_range(self):
        pressed = []
        controller = make_controller(FakeRobot(), buttons=lambda: pressed.pop(0))
        controller.pose_updater.maximum_speed = 45

        pressed.extend([Buttons(pad_up=True), Buttons(pad_up=True)])
        controller.handle_buttons()
        controller.handle_buttons()
        self.assertEqual(xbox_robot_controller.MAXIMUM_SPEED, controller.pose_updater.maximum_speed)

        controller.pose_updater.maximum_speed = 10
        pressed.extend([Buttons(pad_down=True), Buttons(pad_down=True)])
        controller.handle_buttons()
        controller.handle_buttons()
        self.assertEqual(xbox_robot_controller.MINIMUM_SPEED, controller.pose_updater.maximum_speed)

    def test_start_without_recorded_moves_does_nothing(self):
        controller = make_controller(FakeRobot(), buttons=lambda: Buttons(start=True))
        controller.current_pose = Pose(30, 22, 4)

        with mock.patch.object(controller, 'playback_recorded_moves') as playback:
            controller.handle_buttons()
        playback.assert_not_called()


class PlaybackTests(unittest.TestCase):

    def test_playback_stops_between_moves_when_the_robot_is_stopped(self):
        controller = make_controller(FakeRobot())
        controller.current_pose = Pose(30, 22, 4)
        moves = [mock.Mock(), mock.Mock(), mock.Mock()]
        moves[0].move.side_effect = lambda robot: controller._stop_requested.set()

        self.assertEqual(controller.current_pose, controller.playback_recorded_moves(moves))
        moves[0].move.assert_called_once()
        moves[1].move.assert_not_called()


class HardwareErrorTests(unittest.TestCase):

    def test_servo_with_a_hardware_error_stops_the_control_loop(self):
        robot = FakeRobot()
        robot.hardware_errors = {5: ['overload']}
        controller = make_controller(robot)

        with self.assertRaisesRegex(RuntimeError, r'servos \[5\] shut down'):
            controller.check_hardware_errors()

    def test_too_warm_servo_stops_the_robot_normally(self):
        robot = FakeRobot()
        controller = make_controller(robot)

        with mock.patch.object(movement, 'sleep'), \
                mock.patch.object(xbox_robot_controller, 'HARDWARE_CHECK_INTERVAL', 0.0):
            self.assertTrue(controller.start())
            wait_for(lambda: robot.gripper_calls)
            robot.angles = inverse_kinematics(Pose(25, 25, 10), test_config)
            robot.overheating = {2: 51}
            wait_for(lambda: not controller.thread.is_alive())

        # moved back to the start pose and disabled the servos, while they all still worked
        reached = forward_kinematics(robot.angles, test_config)
        np.testing.assert_allclose([30, 22, 4], [reached.x, reached.y, reached.z], atol=1e-6)
        self.assertEqual(['enable_servos', 'disable_servos'], robot.calls)
        self.assertFalse(controller._crashed)
        self.assertTrue(controller.stop())
        self.assertEqual(['enable_servos', 'disable_servos'], robot.calls)

    def test_failed_temperature_read_does_not_stop_the_control_loop(self):
        robot = FakeRobot()
        robot.overheating = None
        controller = make_controller(robot)

        controller.check_hardware_errors()
        self.assertFalse(controller.is_done())

    def test_failed_hardware_error_read_does_not_stop_the_control_loop(self):
        robot = FakeRobot()
        robot.hardware_errors = None
        controller = make_controller(robot)

        controller.check_hardware_errors()
        self.assertFalse(controller.is_done())


def assert_at_start_pose(robot):
    reached = forward_kinematics(robot.angles, test_config)
    np.testing.assert_allclose([30, 22, 4], [reached.x, reached.y, reached.z], atol=1e-6)


class ConcurrencyTests(unittest.TestCase):

    def setUp(self):
        patcher = mock.patch.object(movement, 'sleep')
        patcher.start()
        self.addCleanup(patcher.stop)

    def block_start_move_until_stopped(self, robot, controller):
        """the first step of the start move waits until the robot is stopped, returns an event set at that step"""
        moving = threading.Event()

        def block():
            moving.set()
            controller._stop_requested.wait(timeout=10)

        robot.on_move_servos = block
        return moving

    def test_two_starts_at_the_same_time_start_one_control_loop(self):
        robot = FakeRobot()
        controller = make_controller(robot)
        barrier = threading.Barrier(2)
        results = []

        def start():
            barrier.wait(timeout=5)
            results.append(controller.start())

        threads = [threading.Thread(target=start) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        self.assertEqual([False, True], sorted(results))
        self.assertEqual(['enable_servos'], robot.calls)
        self.assertTrue(controller.stop())
        self.assertEqual(['enable_servos', 'disable_servos'], robot.calls)

    def test_stop_during_the_start_move_stops_it_and_moves_back(self):
        robot = FakeRobot()
        controller = make_controller(robot)
        moving = self.block_start_move_until_stopped(robot, controller)

        self.assertTrue(controller.start())
        self.assertTrue(moving.wait(timeout=10))
        self.assertTrue(controller.stop())

        # the start move didn't finish, so the gripper never opened and the control loop never ran
        self.assertEqual([], robot.gripper_calls)
        controller.pose_updater.get_updated_pose_from_controller.assert_not_called()
        assert_at_start_pose(robot)
        self.assertEqual(['enable_servos', 'disable_servos'], robot.calls)
        self.assertFalse(controller.is_running())

    def test_commands_of_a_previous_run_are_not_handled_after_a_restart(self):
        robot = FakeRobot()
        controller = make_controller(robot)
        reset_calls = []

        def fake_reset_orientation(pose, robot_config, reset_robot):
            reset_calls.append(pose)
            return pose

        with mock.patch.object(xbox_robot_controller, 'reset_orientation', fake_reset_orientation):
            moving = self.block_start_move_until_stopped(robot, controller)
            self.assertTrue(controller.start())
            self.assertTrue(moving.wait(timeout=10))
            # queued while the control loop is still busy with the start move, it is never handled in this run
            self.assertTrue(controller.request_find_center_mode())
            self.assertTrue(controller.stop())

            robot.on_move_servos = None
            self.assertTrue(controller.start())
            updater = controller.pose_updater.get_updated_pose_from_controller
            wait_for(lambda: updater.call_count >= 5)
            controller.stop()

        self.assertEqual([], reset_calls)
        self.assertFalse(controller.find_center_mode)

    def test_stop_during_playback_stops_between_moves_and_moves_back(self):
        robot = FakeRobot()
        pressed = [Buttons(start=True)]
        controller = make_controller(robot, buttons=lambda: pressed.pop(0) if pressed else Buttons())
        playing = threading.Event()

        def play(move_robot):
            playing.set()
            controller._stop_requested.wait(timeout=10)

        moves = [mock.Mock(), mock.Mock()]
        moves[0].move.side_effect = play
        controller.recorded_moves = moves

        self.assertTrue(controller.start())
        self.assertTrue(playing.wait(timeout=10))
        self.assertTrue(controller.stop())

        moves[0].go_to_start_of_move.assert_called_once_with(robot)
        moves[1].move.assert_not_called()
        self.assertFalse(controller._crashed)
        assert_at_start_pose(robot)
        self.assertEqual(['enable_servos', 'disable_servos'], robot.calls)

    def test_crash_in_the_control_loop_keeps_the_servos_enabled_until_stop(self):
        robot = FakeRobot()
        controller = make_controller(robot)
        controller.pose_updater.get_updated_pose_from_controller.side_effect = ValueError('broken')

        self.assertTrue(controller.start())
        wait_for(lambda: not controller.thread.is_alive())
        self.assertTrue(controller._crashed)
        self.assertEqual(['enable_servos'], robot.calls)

        self.assertTrue(controller.stop())
        self.assertEqual(['enable_servos', 'disable_servos'], robot.calls)
        # a new run starts without the crash of the previous one
        controller.pose_updater.get_updated_pose_from_controller.side_effect = \
            lambda pose, find_center_mode, center, dt=None: pose
        self.assertTrue(controller.start())
        self.assertFalse(controller._crashed)
        controller.stop()


class MoreButtonTests(unittest.TestCase):

    def setUp(self):
        patcher = mock.patch.object(movement, 'sleep')
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_b_resets_the_orientation(self):
        robot = FakeRobot()
        controller = make_controller(robot, buttons=lambda: Buttons(b=True))
        controller.current_pose = Pose(30, 22, 4, alpha=0.5)
        reset_pose = Pose(30, 22, 4)

        with mock.patch.object(xbox_robot_controller, 'reset_orientation', return_value=reset_pose) as reset:
            controller.handle_buttons()

        reset.assert_called_once_with(Pose(30, 22, 4, alpha=0.5), test_config, robot)
        self.assertIs(reset_pose, controller.current_pose)

    def test_y_records_the_position(self):
        controller = make_controller(FakeRobot(), buttons=lambda: Buttons(y=True))
        controller.current_pose = Pose(30, 22, 4)

        controller.handle_buttons()

        self.assertEqual([Pose(30, 22, 4)], controller.recorded_positions)
        self.assertIsNone(controller.center)

    def test_a_pose_with_another_flip_than_the_first_is_not_recorded(self):
        controller = make_controller(FakeRobot(), buttons=lambda: Buttons(y=True))
        controller.current_pose = Pose(30, 22, 4)
        controller.handle_buttons()

        controller.current_pose = Pose(25, 20, 4, flip=True)
        with self.assertLogs(xbox_robot_controller.logger, 'WARNING'):
            controller.handle_buttons()
        self.assertEqual([Pose(30, 22, 4)], controller.recorded_positions)

        controller.current_pose = Pose(25, 20, 4)
        controller.handle_buttons()
        self.assertEqual([Pose(30, 22, 4), Pose(25, 20, 4)], controller.recorded_positions)

    def test_the_flip_of_the_first_recorded_move_holds_for_the_next_moves(self):
        controller = make_controller(FakeRobot(), buttons=lambda: Buttons(y=True))
        controller.recorded_moves = [PoseToPoseMovement([Pose(30, 22, 4, flip=True), Pose(25, 20, 4, flip=True)], 2)]

        controller.current_pose = Pose(20, 20, 4)
        with self.assertLogs(xbox_robot_controller.logger, 'WARNING'):
            controller.handle_buttons()
        self.assertEqual([], controller.recorded_positions)

        controller.current_pose = Pose(20, 20, 4, flip=True)
        controller.handle_buttons()
        self.assertEqual([Pose(20, 20, 4, flip=True)], controller.recorded_positions)

    def test_after_clearing_the_recording_any_flip_can_start_a_new_one(self):
        controller = make_controller(FakeRobot(), buttons=lambda: Buttons(y=True))
        controller.current_pose = Pose(30, 22, 4)
        controller.handle_buttons()
        controller.clear_recorded_moves_and_positions()

        controller.current_pose = Pose(25, 20, 4, flip=True)
        controller.handle_buttons()
        self.assertEqual([Pose(25, 20, 4, flip=True)], controller.recorded_positions)

    def test_the_poses_of_find_center_mode_may_have_any_flip(self):
        controller = make_controller(FakeRobot(), buttons=lambda: Buttons(y=True))
        controller.recorded_moves = [PoseToPoseMovement([Pose(30, 22, 4), Pose(25, 20, 4)], 2)]
        controller.find_center_mode = True

        controller.current_pose = Pose(-10, 20, 4, flip=True)
        controller.handle_buttons()
        self.assertEqual([Pose(-10, 20, 4, flip=True)], controller.recorded_positions)

    def test_y_twice_in_find_center_mode_sets_the_center(self):
        controller = make_controller(FakeRobot(), buttons=lambda: Buttons(y=True))
        controller.find_center_mode = True

        controller.current_pose = Pose(-10, 20, 4)
        controller.handle_buttons()
        self.assertTrue(controller.find_center_mode)
        controller.current_pose = Pose(10, 20, 4, alpha=0.5)
        controller.handle_buttons()

        self.assertFalse(controller.find_center_mode)
        self.assertIsNotNone(controller.center)
        self.assertEqual([], controller.recorded_positions)

    def test_poses_that_define_no_center_are_dropped_and_find_center_mode_continues(self):
        controller = make_controller(FakeRobot(), buttons=lambda: Buttons(y=True))
        controller.find_center_mode = True

        controller.current_pose = Pose(-10, 20, 4)
        controller.handle_buttons()
        # a different height, get_center can't intersect the two
        controller.current_pose = Pose(10, 20, 8, alpha=0.5)
        with self.assertLogs(xbox_robot_controller.logger, 'WARNING'):
            controller.handle_buttons()

        self.assertTrue(controller.find_center_mode)
        self.assertIsNone(controller.center)
        self.assertEqual([], controller.recorded_positions)

    def test_x_without_enough_positions_does_not_create_a_move(self):
        controller = make_controller(FakeRobot(), buttons=lambda: Buttons(x=True))
        controller.recorded_positions = [Pose(30, 22, 4)]

        with mock.patch.object(xbox_robot_controller, 'create_move') as create:
            controller.handle_buttons()

        create.assert_not_called()
        self.assertEqual([], controller.recorded_moves)

    def test_x_stores_the_move_and_continues_from_its_last_position(self):
        controller = make_controller(FakeRobot(), buttons=lambda: Buttons(x=True))
        controller.recorded_positions = [Pose(25, 25, 10), Pose(30, 22, 4)]
        move = mock.Mock()

        with mock.patch.object(xbox_robot_controller, 'create_move', return_value=move):
            controller.handle_buttons()

        self.assertEqual([move], controller.recorded_moves)
        self.assertEqual([Pose(30, 22, 4)], controller.recorded_positions)

    def test_x_with_an_impossible_move_goes_back_to_the_first_position(self):
        robot = FakeRobot(angles=inverse_kinematics(Pose(30, 22, 4), test_config))
        controller = make_controller(robot, buttons=lambda: Buttons(x=True))
        controller.current_pose = Pose(30, 22, 4)
        controller.recorded_positions = [Pose(25, 25, 10), Pose(30, 22, 4)]

        with mock.patch.object(xbox_robot_controller, 'create_move', return_value=None):
            controller.handle_buttons()

        self.assertEqual([], controller.recorded_moves)
        self.assertEqual([Pose(25, 25, 10)], controller.recorded_positions)
        reached = forward_kinematics(robot.angles, test_config)
        np.testing.assert_allclose([25, 25, 10], [reached.x, reached.y, reached.z], atol=1e-6)
        np.testing.assert_allclose([25, 25, 10], [controller.current_pose.x, controller.current_pose.y,
                                                  controller.current_pose.z], atol=1e-6)

    def test_start_plays_back_the_recorded_moves(self):
        controller = make_controller(FakeRobot(), buttons=lambda: Buttons(start=True))
        controller.current_pose = Pose(30, 22, 4)
        controller.recorded_moves = [mock.Mock()]
        end_pose = Pose(25, 25, 10)

        with mock.patch.object(controller, 'playback_recorded_moves', return_value=end_pose) as playback:
            controller.handle_buttons()

        playback.assert_called_once_with(controller.recorded_moves)
        self.assertIs(end_pose, controller.current_pose)

    def test_d_pad_changes_the_maximum_speed_in_steps(self):
        pressed = [Buttons(pad_up=True), Buttons(pad_down=True), Buttons(pad_down=True)]
        controller = make_controller(FakeRobot(), buttons=lambda: pressed.pop(0))
        controller.pose_updater.maximum_speed = 20

        controller.handle_buttons()
        self.assertEqual(20 + xbox_robot_controller.MAXIMUM_SPEED_STEP, controller.pose_updater.maximum_speed)
        controller.handle_buttons()
        controller.handle_buttons()
        self.assertEqual(20 - xbox_robot_controller.MAXIMUM_SPEED_STEP, controller.pose_updater.maximum_speed)

    def test_nothing_happens_without_pressed_buttons(self):
        robot = FakeRobot()
        controller = make_controller(robot)
        controller.current_pose = Pose(30, 22, 4)
        controller.pose_updater.maximum_speed = 20

        controller.handle_buttons()

        self.assertEqual([], robot.gripper_calls)
        self.assertEqual([], controller.recorded_positions)
        self.assertEqual(20, controller.pose_updater.maximum_speed)
        self.assertEqual(Pose(30, 22, 4), controller.current_pose)


class MoreGripperTests(unittest.TestCase):

    def test_gripper_is_not_written_beyond_fully_closed(self):
        robot = FakeRobot()
        robot.gripper_calls = [100.0]
        controller = make_controller(robot, buttons=lambda: Buttons(rb=True))
        controller.gripper_state = 100.0

        controller.handle_buttons()

        self.assertEqual([100.0], robot.gripper_calls)
        self.assertEqual(100, controller.gripper_state)

    def test_gripper_is_not_written_beyond_its_range_when_its_state_can_not_be_read(self):
        robot = FakeRobot()
        robot.get_gripper_state = lambda: None
        pressed = [Buttons(rb=True), Buttons(lb=True)]
        controller = make_controller(robot, buttons=lambda: pressed.pop(0))

        controller.gripper_state = 100.0
        controller.handle_buttons()
        controller.gripper_state = 0.0
        controller.handle_buttons()

        self.assertEqual([], robot.gripper_calls)

    def test_stuck_gripper_target_does_not_open_further_than_the_lead(self):
        robot = FakeRobot()
        # the gripper doesn't open beyond 60, e.g. because it is stuck
        robot.get_gripper_state = lambda: 60.0
        controller = make_controller(robot, buttons=lambda: Buttons(lb=True))
        controller.gripper_state = 60.0

        for _ in range(2 * xbox_robot_controller.GRIPPER_MAX_LEAD):
            controller.handle_buttons()

        self.assertEqual(60 - xbox_robot_controller.GRIPPER_MAX_LEAD, controller.gripper_state)
        self.assertEqual(60 - xbox_robot_controller.GRIPPER_MAX_LEAD, robot.gripper_calls[-1])
        self.assertEqual(xbox_robot_controller.GRIPPER_MAX_LEAD, len(robot.gripper_calls))


class MorePlaybackTests(unittest.TestCase):

    def setUp(self):
        patcher = mock.patch.object(movement, 'sleep')
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_playback_returns_the_last_pose_of_the_last_move(self):
        controller = make_controller(FakeRobot())
        moves = [mock.Mock(), mock.Mock()]
        moves[1].poses = [Pose(20, 20, 10), Pose(25, 25, 10)]

        self.assertEqual(Pose(25, 25, 10), controller.playback_recorded_moves(moves))
        moves[0].move.assert_called_once()
        moves[1].move.assert_called_once()

    def test_failed_playback_moves_back_to_the_start_pose(self):
        robot = FakeRobot(angles=inverse_kinematics(Pose(25, 25, 10), test_config))
        controller = make_controller(robot)
        moves = [mock.Mock(), mock.Mock()]
        moves[0].move.side_effect = MovementException('can not reach')

        reached = controller.playback_recorded_moves(moves)

        moves[1].move.assert_not_called()
        assert_at_start_pose(robot)
        np.testing.assert_allclose([30, 22, 4], [reached.x, reached.y, reached.z], atol=1e-6)

    def test_failed_playback_that_can_not_move_back_stops_the_control_loop(self):
        robot = FakeRobot(angles=inverse_kinematics(Pose(25, 25, 10), test_config))
        controller = make_controller(robot)
        controller.start_pose = Pose(60, 22, 4)
        moves = [mock.Mock()]
        moves[0].move.side_effect = MovementException('can not reach')

        with self.assertRaisesRegex(RuntimeError, 'playback failed'):
            controller.playback_recorded_moves(moves)


def pose_of_angles(angles, flip=None):
    """the pose of the angles, with the flip the gamepad keeps (the one of the pose it started from)"""
    pose = forward_kinematics(np.asarray(angles), test_config)
    if flip is not None:
        pose.flip = flip
    return pose


def angles_pointing(t, eps):
    """the angles with the gripper pointing along (eps, t, 1) in the wrist, at eps 0 the path over t goes through the
    straight wrist; joints 4 and 6 stay around +-pi/2, away from the ends of their ranges"""
    direction = np.array([eps, t, 1.0]) / np.linalg.norm([eps, t, 1.0])
    angles = wrist_angles(np.arccos(direction[2]))
    angles[4] = np.arctan2(direction[1], direction[0])
    angles[6] = -angles[4]
    return angles


def tip_of(angles):
    return forward_position_kinematics(angles, test_config)[-1]


def largest_change(angles, other_angles):
    return np.abs(np.asarray(angles)[1:] - np.asarray(other_angles)[1:]).max()


def wrist_angles(q5):
    """angles with the wrist at q5 radians from straight, the rest of the arm is the same"""
    angles = inverse_kinematics(Pose(25, 25, 10), test_config)
    angles[5] = q5
    return angles


class StickStepTests(unittest.TestCase):
    """every step of the sticks is checked before it is taken: wrist flip, reachability, ranges and joint speed"""

    def controller_at(self, pose, robot=None):
        robot = robot or FakeRobot()
        controller = make_controller(robot)
        controller.current_pose = pose
        return controller

    def test_a_safe_step_is_taken(self):
        controller = self.controller_at(Pose(30, 22, 4))

        controller._take_step(Pose(30.05, 22, 4), 0.01)

        self.assertEqual(30.05, controller.current_pose.x)
        self.assertFalse(controller.current_pose.flip)

    def test_passing_a_straight_wrist_keeps_the_flip_and_turns_the_wrist_over_slowly(self):
        controller = self.controller_at(pose_of_angles(wrist_angles(0.03)))
        self.assertFalse(controller.current_pose.flip)
        sent_angles = [controller._step_angles()]

        # the sticks move the wrist from 0.03 to -0.03 rad, then hold there
        targets = [pose_of_angles(wrist_angles(q5), flip=False) for q5 in (0.02, 0.01, 0.005, -0.005, -0.01, -0.02)]
        targets += [pose_of_angles(wrist_angles(-0.03), flip=False)] * 150
        for target in targets:
            controller._take_step(copy(target), 0.01)
            sent_angles.append(controller._step_angles())
            self.assertFalse(controller.current_pose.flip, 'only the a button changes the flip')
            self.assertFalse(controller._step_skipped)

        for angles, next_angles in zip(sent_angles, sent_angles[1:]):
            self.assertLessEqual(largest_change(angles, next_angles),
                                 global_constants.recommended_max_servo_speed * 0.01)
        # with the same flip the wrist turned over: joints 4 and 6 by pi, at the joint speed limit
        expected = inverse_kinematics(pose_of_angles(wrist_angles(-0.03), flip=False), test_config)
        np.testing.assert_allclose(expected, sent_angles[-1], atol=1e-6)
        self.assertAlmostEqual(pi, abs(sent_angles[-1][4] - sent_angles[0][4]), places=2)

    def test_keeping_the_flip_would_spin_joints_4_and_6(self):
        # the situation of the test above: the same pose with the flip kept turns them by pi
        before = wrist_angles(0.005)
        after = inverse_kinematics(pose_of_angles(wrist_angles(-0.005), flip=False), test_config)
        self.assertGreater(abs(after[4] - before[4]), 3)
        self.assertGreater(abs(after[6] - before[6]), 3)

    def test_the_a_button_turns_the_wrist_over_at_the_joint_speed_limit(self):
        # high enough above the table for the gripper to swing through the straight wrist
        controller = self.controller_at(Pose(30, 22, 15))
        before = controller._step_angles()

        controller._toggle_flip()
        sent_angles = [before]
        for _ in range(150):
            # the sticks are released, the gamepad asks for the current pose
            controller._take_step(copy(controller.current_pose), 0.01)
            sent_angles.append(controller._step_angles())
            self.assertTrue(controller.current_pose.flip)
            self.assertFalse(controller._step_skipped)

        for angles, next_angles in zip(sent_angles, sent_angles[1:]):
            self.assertLessEqual(largest_change(angles, next_angles),
                                 global_constants.recommended_max_servo_speed * 0.01)
        # the wrist took many steps to turn over, and ended on the flipped solution of the same pose
        self.assertGreater(sum(largest_change(a, b) > 1e-9 for a, b in zip(sent_angles, sent_angles[1:])), 50)
        np.testing.assert_allclose(inverse_kinematics(Pose(30, 22, 15, flip=True), test_config), sent_angles[-1],
                                   atol=1e-6)

    def test_a_flip_that_would_swing_below_the_floor_holds_and_the_a_button_turns_it_back(self):
        # 4cm above the table the gripper would swing through the floor of the workspace on the way
        controller = self.controller_at(Pose(30, 22, 4))
        before = controller._step_angles().copy()

        controller._toggle_flip()
        with self.assertLogs(xbox_robot_controller.logger, 'WARNING') as logs:
            for _ in range(150):
                controller._take_step(copy(controller.current_pose), 0.01)
        self.assertTrue(controller._step_skipped)
        self.assertIn('minimum height', logs.output[0])
        self.assertGreater(largest_change(before, controller._step_angles()), 0.5)

        controller._toggle_flip()
        for _ in range(150):
            controller._take_step(copy(controller.current_pose), 0.01)
        self.assertFalse(controller._step_skipped)
        np.testing.assert_allclose(before, controller._step_angles(), atol=1e-6)

    def test_a_wrist_joint_at_the_end_of_its_range_holds_instead_of_turning_all_the_way_round(self):
        # joint 6 just inside -pi, the sticks ask for a little further: the inverse kinematics gives almost +pi
        angles = wrist_angles(0.3)
        angles[6] = -pi + 0.01
        controller = self.controller_at(pose_of_angles(angles))
        further = angles.copy()
        further[6] = -pi - 0.01

        for _ in range(5):
            with self.assertLogs(xbox_robot_controller.logger, 'DEBUG'):
                controller._take_step(pose_of_angles(further, flip=False), 0.01)
            self.assertTrue(controller._step_skipped)
            np.testing.assert_allclose(angles, controller._step_angles(), atol=1e-9)

    def test_the_orientation_of_the_new_pose_is_kept(self):
        controller = self.controller_at(Pose(30, 22, 4, alpha=0.1, gamma=0.2))

        controller._take_step(Pose(30.05, 22, 4, alpha=0.11, gamma=0.2), 0.01)

        self.assertEqual(0.11, controller.current_pose.alpha)
        self.assertEqual(0.2, controller.current_pose.gamma)

    def test_an_unreachable_position_is_not_taken(self):
        robot = FakeRobot()
        # nothing else may refuse the step
        robot.workspace_limits = None
        robot.angle_limits = None
        controller = self.controller_at(Pose(0, 52.9, 4), robot)
        pose = controller.current_pose

        # the inverse kinematics gives the stretched arm for this, which ends short of the target
        with self.assertLogs(xbox_robot_controller.logger, 'WARNING') as logs:
            controller._take_step(Pose(0, 54, 4), 0.1)

        self.assertIs(pose, controller.current_pose)
        self.assertIn('can not be reached', logs.output[0])

        # a position that is reachable is taken
        controller._take_step(Pose(0, 52.5, 4), 0.1)
        self.assertEqual(52.5, controller.current_pose.y)

    def test_a_step_outside_the_workspace_limits_is_not_taken(self):
        robot = FakeRobot()
        robot.workspace_limits = type('Limits', (), dict(radius_min=10, radius_max=37.45, y_min=12, z_min=-5,
                                                         wrist_z_min=-5))
        controller = self.controller_at(Pose(30, 22, 4), robot)
        pose = controller.current_pose

        # the limit is 37.45cm from the base, the new poses are 37.48 and 37.54cm away
        with self.assertLogs(xbox_robot_controller.logger, 'WARNING') as logs:
            controller._take_step(Pose(30.05, 22.05, 4), 0.01)
            controller._take_step(Pose(30.1, 22.1, 4), 0.01)

        self.assertIs(pose, controller.current_pose)
        self.assertIn('base', logs.output[0])

        # moving back inside is possible
        controller._take_step(Pose(29.95, 21.95, 4), 0.01)
        self.assertEqual(29.95, controller.current_pose.x)

    def test_a_step_outside_the_range_of_a_servo_is_not_taken(self):
        robot = FakeRobot()
        controller = self.controller_at(Pose(30, 22, 4), robot)
        new_pose = Pose(30, 22.1, 4)
        old_q2 = robot.pose_to_angles(controller.current_pose)[2]
        new_q2 = robot.pose_to_angles(new_pose)[2]
        middle = (old_q2 + new_q2) / 2
        robot.angle_limits = {**angle_limits, 2: (-1.0, middle) if new_q2 > old_q2 else (middle, 3.0)}
        pose = controller.current_pose

        with self.assertLogs(xbox_robot_controller.logger, 'WARNING') as logs:
            controller._take_step(new_pose, 0.1)

        self.assertIs(pose, controller.current_pose)
        self.assertIn('joint 2', logs.output[0])

        # the same step is fine when the range of the robot is not known
        robot.angle_limits = None
        controller._take_step(new_pose, 0.1)
        self.assertEqual(22.1, controller.current_pose.y)

    def test_a_too_fast_step_is_slowed_down_to_the_maximum_joint_speed(self):
        controller = self.controller_at(Pose(30, 22, 4))
        before = controller._step_angles()

        # 4cm in a single step of 10ms is far too fast for the joints: the pose is commanded, the arm goes part of the
        # way there
        controller._take_step(Pose(34, 22, 4), 0.01)

        self.assertFalse(controller._step_skipped)
        self.assertEqual(34, controller.current_pose.x)
        after = controller._step_angles()
        self.assertLessEqual(largest_change(before, after), global_constants.recommended_max_servo_speed * 0.01)
        self.assertGreater(tip_of(after)[0], 30)
        self.assertLess(tip_of(after)[0], 34)

    def test_the_gamepad_waits_until_the_arm_reached_the_commanded_pose(self):
        controller = self.controller_at(Pose(30, 22, 4))
        controller._take_step(Pose(34, 22, 4), 0.01)

        # the gamepad asks for something else, the arm first goes on to 34
        for _ in range(100):
            controller._take_step(Pose(30, 25, 4), 0.01)
            if np.allclose(tip_of(controller._step_angles()), [34, 22, 4], atol=1e-6):
                break
        self.assertEqual(34, controller.current_pose.x)
        np.testing.assert_allclose([34, 22, 4], tip_of(controller._step_angles()), atol=1e-6)

        # and then follows the gamepad again
        controller._take_step(Pose(34.05, 22, 4), 0.01)
        self.assertEqual(34.05, controller.current_pose.x)

    def test_the_slow_down_uses_the_time_of_the_step(self):
        quick = self.controller_at(Pose(30, 22, 4))
        quick._take_step(Pose(31, 22, 4), 0.01)
        self.assertGreater(tip_of(quick._step_angles())[0], 30)
        self.assertLess(tip_of(quick._step_angles())[0], 31)

        # the same step is slow enough when the loop took longer
        longer = self.controller_at(Pose(30, 22, 4))
        longer._take_step(Pose(31, 22, 4), 0.1)
        np.testing.assert_allclose([31, 22, 4], tip_of(longer._step_angles()), atol=1e-6)

    def test_the_arm_moves_slowly_through_the_singularity_instead_of_stopping(self):
        # the gripper points along (eps, t, 1) in the wrist; the path passes eps from the straight wrist, where joints
        # 4 and 6 have to turn by pi for a small turn of the gripper
        for eps in (0.0, 0.02):
            with self.subTest(eps=eps):
                path = [pose_of_angles(angles_pointing(t, eps)) for t in np.linspace(-0.3, 0.3, 601)]
                orientations = np.array([pose.get_euler_matrix().ravel() for pose in path])
                controller = self.controller_at(path[0])

                def nearest(pose):
                    return np.linalg.norm(orientations - pose.get_euler_matrix().ravel(), axis=1).argmin()

                for _ in range(500):
                    if nearest(forward_kinematics(controller._step_angles(), test_config)) == len(path) - 1:
                        break
                    # like the gamepad at full speed, 0.5 rad/s: a step ahead of the commanded pose
                    target = copy(path[min(nearest(controller.current_pose) + 5, len(path) - 1)])
                    target.flip = controller.current_pose.flip
                    before = controller._step_angles()
                    controller._take_step(target, 0.01)
                    self.assertFalse(controller._step_skipped)
                    self.assertFalse(controller.current_pose.flip, 'only the a button changes the flip')
                    self.assertLessEqual(largest_change(before, controller._step_angles()),
                                         global_constants.recommended_max_servo_speed * 0.01)
                arrived = nearest(forward_kinematics(controller._step_angles(), test_config))
                self.assertEqual(len(path) - 1, arrived, 'the arm did not get through the singularity')

    def test_a_skipped_step_warns_once_until_a_step_is_taken(self):
        controller = self.controller_at(Pose(30, 22, 4))
        unreachable = Pose(100, 22, 4)

        with self.assertLogs(xbox_robot_controller.logger, 'WARNING') as logs:
            for _ in range(5):
                controller._take_step(unreachable, 0.01)
        self.assertEqual(1, len(logs.records))

        controller._take_step(Pose(30.05, 22, 4), 0.01)
        self.assertFalse(controller._step_skipped)
        with self.assertLogs(xbox_robot_controller.logger, 'WARNING') as logs:
            controller._take_step(unreachable, 0.01)
        self.assertEqual(1, len(logs.records))

    def test_the_control_loop_only_sends_safe_poses(self):
        robot = FakeRobot()
        controller = make_controller(robot)
        calls = []

        def stick(pose, find_center_mode, center, dt=None):
            calls.append(pose)
            # every other call asks for a jump of 10cm, the rest for steps of 0.01cm
            if len(calls) % 2 == 0:
                return Pose(pose.x + 10, pose.y, pose.z, flip=pose.flip)
            return Pose(pose.x + 0.01, pose.y, pose.z, flip=pose.flip)

        controller.pose_updater.get_updated_pose_from_controller.side_effect = stick
        with mock.patch.object(movement, 'sleep'), self.assertLogs(xbox_robot_controller.logger, 'WARNING'):
            self.assertTrue(controller.start())
            wait_for(lambda: len(calls) > 20)
            controller.stop()

        # what the control loop sent from the start pose on
        positions = [forward_position_kinematics(angles, test_config)[-1] for angles in robot.sent_angles]
        start = next(i for i, position in enumerate(positions) if np.allclose(position, [30, 22, 4], atol=1e-3))
        positions = positions[start:]
        self.assertGreater(len(positions), 20)
        self.assertLess(max(position[0] for position in positions), 31.5)
        for position, next_position in zip(positions, positions[1:]):
            self.assertLess(abs(next_position[0] - position[0]), 0.5)


class ScenarioTests(unittest.TestCase):

    def controller_with_scenarios(self, count=3):
        controller = make_controller(FakeRobot())
        controller.scenarios = [mock.Mock() for _ in range(count)]
        return controller

    def test_a_scenario_is_built_and_the_previous_one_destroyed(self):
        controller = self.controller_with_scenarios()

        self.assertTrue(controller.set_scenario(2))
        controller.scenarios[2].build_scenario.assert_called_once_with(3)
        self.assertEqual(2, controller.current_scenario_id)

        self.assertTrue(controller.set_scenario(0))
        controller.scenarios[2].destroy_scenario.assert_called_once_with(3)
        controller.scenarios[0].build_scenario.assert_called_once_with(3)
        self.assertEqual(0, controller.current_scenario_id)

    def test_ids_outside_of_the_scenarios_are_rejected_and_change_nothing(self):
        controller = self.controller_with_scenarios()
        self.assertTrue(controller.set_scenario(1))

        for scenario_id in (-1, 3, 4, 1000):
            self.assertFalse(controller.set_scenario(scenario_id), scenario_id)

        self.assertEqual(1, controller.current_scenario_id)
        for scenario in controller.scenarios:
            scenario.destroy_scenario.assert_not_called()
        # a valid id still works afterwards
        self.assertTrue(controller.set_scenario(2))

    def test_the_real_scenarios_reject_their_count(self):
        controller = make_controller(FakeRobot())
        self.assertFalse(controller.set_scenario(len(controller.scenarios)))
        self.assertIsNone(controller.current_scenario_id)

    def test_a_scenario_that_fails_to_build_is_not_the_current_one(self):
        controller = self.controller_with_scenarios()
        controller.scenarios[1].build_scenario.side_effect = RuntimeError('pybullet')

        with self.assertRaises(RuntimeError):
            controller.set_scenario(1)

        self.assertIsNone(controller.current_scenario_id)
        # the next call does not try to destroy the scenario that was never built
        self.assertTrue(controller.set_scenario(0))
        controller.scenarios[1].destroy_scenario.assert_not_called()


class RestartTests(unittest.TestCase):

    def test_the_gripper_opens_again_after_a_restart(self):
        robot = FakeRobot()
        controller = make_controller(robot)
        with mock.patch.object(movement, 'sleep'):
            self.assertTrue(controller.start())
            wait_for(lambda: robot.gripper_calls)
            controller.gripper_state = 60  # closed during the run
            self.assertTrue(controller.stop())

            robot.gripper_calls.clear()
            self.assertTrue(controller.start())
            wait_for(lambda: robot.gripper_calls)
            controller.stop()

        self.assertEqual([0], robot.gripper_calls)
        self.assertEqual(0, controller.gripper_state)
