import threading
import time
import unittest
from unittest import mock

import numpy as np
from numpy import pi

from src.kinematics.kinematics import inverse_kinematics, forward_kinematics
from src.kinematics.kinematics_utils import Pose, RobotConfig
from src.utils import cartesian_movement
from src.utils.movement import SplineMovement, PoseToPoseMovement
from src.utils.movement_exception import MovementException
from src.xbox_control import xbox_robot_controller
from src.xbox_control.xbox360controller.XboxController import Buttons
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

    def test_create_move_spline(self):
        pose1 = Pose(0, 0, 0)
        pose2 = Pose(1, 0, 0)
        pose3 = Pose(1, 1, 0)
        poses = [pose1, pose2, pose3]

        move = create_move(None, poses, 1, None, workspace_limits=None)
        self.assertIsInstance(move, SplineMovement)

    def test_not_in_workspace_limits(self):
        class MockWorkSpaceLimits:
            radius_min = 2
            radius_max = 4
            y_min = 2
            z_min = 2

        pose1 = Pose(0, 0, 0)
        pose2 = Pose(1, 0, 0)
        pose3 = Pose(1, 1, 0)
        poses = [pose1, pose2, pose3]

        move = create_move(None, poses, 1, None, workspace_limits=MockWorkSpaceLimits)
        self.assertIsNone(move)

    def test_in_workspace_limits(self):
        class MockWorkSpaceLimits:
            radius_min = 0
            radius_max = 10
            y_min = 0
            z_min = 0

        pose1 = Pose(1, 1, 1)
        pose2 = Pose(2, 1, 1)
        pose3 = Pose(2, 2, 1)
        poses = [pose1, pose2, pose3]

        move = create_move(None, poses, 1, None, workspace_limits=MockWorkSpaceLimits)
        self.assertIsInstance(move, SplineMovement)

    def test_create_move_pose_to_pose(self):
        pose1 = Pose(0, 0, 0)
        pose2 = Pose(0, 0, 0)
        poses = [pose1, pose2]

        move = create_move(None, poses, 1, None, None)
        self.assertIsInstance(move, PoseToPoseMovement)


class FakeRobot:
    """a robot with real kinematics that records what it is asked to do"""

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
        # optional function called before every step of a straight line movement
        self.on_move_servos = None

    def get_current_angles(self):
        return None if self.angles is None else self.angles.copy()

    def get_angle_limits(self):
        return angle_limits

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

    def move_to_pose(self, pose):
        self.moving_threads.add(threading.current_thread().name)
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
    pose_updater.get_updated_pose_from_controller.side_effect = lambda pose, find_center_mode, center: pose
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
        patcher = mock.patch.object(cartesian_movement, 'sleep')
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
        patcher = mock.patch.object(cartesian_movement, 'sleep')
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

    def test_only_the_first_pressed_button_is_handled(self):
        robot = FakeRobot()
        controller = make_controller(robot, buttons=lambda: Buttons(rb=True, a=True))
        controller.current_pose = Pose(30, 22, 4)

        controller.handle_buttons()

        self.assertEqual([1], robot.gripper_calls)
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

        with mock.patch.object(cartesian_movement, 'sleep'), \
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
        patcher = mock.patch.object(cartesian_movement, 'sleep')
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
            lambda pose, find_center_mode, center: pose
        self.assertTrue(controller.start())
        self.assertFalse(controller._crashed)
        controller.stop()


class MoreButtonTests(unittest.TestCase):

    def setUp(self):
        patcher = mock.patch.object(cartesian_movement, 'sleep')
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
        patcher = mock.patch.object(cartesian_movement, 'sleep')
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
