import csv
import math
import os
import tempfile
import unittest
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

from src.block_stacking import calibrate_board
from src.block_stacking.calibrate_board import ABORT, ACCEPT, BACK, Command, Context, CornersSession, FORWARD, HIGHER, \
    KeyReader, LEFT, LOWER, RIGHT, SKIP, STEP, SinkSession, board_pose_lines, decode_console_key, estimate_board, \
    fit_line, limits_with_tip_floor, make_key_reader, parse_typed_command, replace_board_pose, run_session, \
    write_board_pose, BoardEstimate, FINGERTIP_REACH, LOWER_ARM_COMMAND, TYPED_HELP
from src.block_stacking.stacking_config import StackingConfig, StackingConfigError, load_stacking_config
from src.block_stacking.stacking_plan import park_pose
from src.kinematics.kinematics import forward_kinematics, forward_position_kinematics, inverse_kinematics
from src.kinematics.kinematics_utils import Pose, RobotConfig
from src.robot_controllers.abstract_robot_controller import AbstractRobotController
from src.utils.movement import PlannedMove
from src.utils.movement_exception import MovementException
from tests.block_stacking_tests.frozen_config import FROZEN_STACKING_CONFIG_PATH, load_frozen_config

config = load_frozen_config()
DATE = datetime(2026, 11, 26, 14, 5)
D6S = (14.0, 15.0)
ANGLE_LIMITS = {1: (0.0, math.pi), 2: (0.0, math.pi), 3: (-math.pi / 2, math.pi / 2), 4: (-math.pi, math.pi),
                5: (-math.pi, math.pi), 6: (-math.pi, math.pi)}
# the angles of the arm lying on the table, measured, in degrees for joints 1 to 6
MEASURED_REST_DEGREES = [90.5, 9.7, 30.1, -0.3, 58.1, -5.2]
# the corners of the frozen board (position -12, 12, no rotation, 15 by 9 cm)
TOP_RIGHT = (-12.0, 21.0)
TOP_LEFT = (-27.0, 21.0)


class Limits:
    """the standard workspace limits with fixed values: the tests do not depend on d6 in robot_config.yaml"""
    radius_min = 10
    wrist_radius_min = 12.5
    wrist_reach_max = 41.8
    y_min = 15.1
    z_min = 1
    wrist_z_min = 1


class FakeRobot(AbstractRobotController):
    """a robot with real kinematics for a chosen d6 and the real joint ranges"""
    workspace_limits = Limits  # type: ignore[assignment]  # a fixed copy of the standard limits

    def __init__(self, d6: float) -> None:
        self.robot_config = RobotConfig(15.7, 20, 21, d6)
        self.angles: np.ndarray | None = np.radians([0.0] + MEASURED_REST_DEGREES)
        self.events: list[str] = []
        self.enable_result: bool | None = True
        self.enabled = False
        self.gripper_commands: list[float] = []

    def enable_servos(self) -> bool | None:
        self.events.append('enable')
        self.enabled = self.enable_result is not False
        return self.enable_result

    def disable_servos(self) -> bool | None:
        self.events.append('disable')
        return True

    def get_current_angles(self) -> np.ndarray | None:
        return None if self.angles is None else self.angles.copy()

    def pose_to_angles(self, pose: Any) -> np.ndarray:
        return inverse_kinematics(pose, self.robot_config)

    def get_angle_limits(self) -> dict[int, tuple[float, float]] | None:
        return ANGLE_LIMITS

    def set_gripper(self, new_gripper_state: float) -> bool | None:
        self.gripper_commands.append(new_gripper_state)
        self.events.append('gripper:{:g}'.format(new_gripper_state))
        return True

    def move_to_pose(self, pose: Any) -> float | None:
        raise AssertionError('the script may only execute planned moves')

    def move_to_pose_and_give_new_angles(self, pose: Any) -> np.ndarray:
        raise AssertionError('the script may only execute planned moves')

    def move_servos(self, angles: np.ndarray, velocities: np.ndarray | None = None) -> float | None:
        raise AssertionError('the script may only execute planned moves')

    def reset_to_pose(self, pose: Any) -> None:
        raise AssertionError('the script may only execute planned moves')

    def tip(self) -> np.ndarray:
        assert self.angles is not None
        return np.asarray(forward_position_kinematics(self.angles, self.robot_config)[-1])

    def fingertip_height(self) -> float:
        return float(self.tip()[2]) - (FINGERTIP_REACH - self.robot_config.d6)


class FakeExecutor:
    """executes at once: the arm of the fake robot goes to the end of the move, plus a sag the encoders read"""

    def __init__(self, robot: FakeRobot) -> None:
        self.robot = robot
        self.moves: list[PlannedMove] = []
        self.sag = 0.0  # radians that joint 2 reads lower than commanded after every move

    def __call__(self, robot: FakeRobot, move: PlannedMove, should_stop: Callable[[], bool] | None = None) -> bool:
        assert should_stop is None, 'a move of the script can not be stopped half way'
        self.moves.append(move)
        self.robot.events.append('move:{}'.format(len(self.moves) - 1))
        self.robot.angles = move.angles[-1].copy()
        self.robot.angles[2] += self.sag
        return True

    def is_line(self, index: int) -> bool:
        return self.moves[index].description.startswith('move in a straight line')


def tip_of(angles: np.ndarray, d6: float) -> np.ndarray:
    return np.asarray(forward_position_kinematics(angles, RobotConfig(15.7, 20, 21, d6))[-1])


class ScriptedKeys:
    """the commands of the user: typed words (see parse_typed_command), Commands, exceptions to raise, or functions that
    are called when the script asks for the key (to look at the arm at that moment) and give the next command"""

    def __init__(self, keys: Iterable[Any]) -> None:
        self.keys = list(keys)
        self.taken = 0

    def __call__(self) -> Command | None:
        if not self.keys:
            raise AssertionError('the script asked for more keys than the test has')
        key = self.keys.pop(0)
        self.taken += 1
        if callable(key):
            key = key()
        if isinstance(key, BaseException):
            raise key
        if isinstance(key, Command):
            return key
        command = parse_typed_command(key)
        assert command is not None, 'the test has a key the script does not understand: {!r}'.format(key)
        return command


class Rig:
    """the fake robot, executor, keys, clock and output of one run, and a lower_to_rest that records what it saw"""

    def __init__(self, d6: float, keys: Iterable[Any] = (), answers: Iterable[str] = (),
                 own_limits: bool = False) -> None:
        self.d6 = d6
        self.robot = FakeRobot(d6)
        self.original_limits: Any = Limits
        if own_limits:
            self.original_limits = type('OwnLimits', (Limits,), {})
            self.robot.workspace_limits = self.original_limits
        self.executor = FakeExecutor(self.robot)
        self.keys = ScriptedKeys(keys)
        self.out: list[str] = []
        self.sleeps: list[float] = []
        self.answers = list(answers)
        self.questions: list[str] = []
        self.lay_down_result: str | None = None
        self.lay_downs: list[dict[str, Any]] = []

    def ask(self, question: str) -> str:
        self.questions.append(question)
        if not self.answers:
            raise AssertionError('the script asked {!r}, which the test did not expect'.format(question))
        return self.answers.pop(0)

    def context(self, key_reader: Callable[[], Command | None] | None = None) -> Context:
        return Context(self.robot, config, key_reader or self.keys, self.executor, self.sleeps.append,
                       self.out.append, self.ask, lambda: DATE)

    def lower_to_rest(self, robot: Any, *args: Any, **kwargs: Any) -> str | None:
        self.lay_downs.append({'angles': None if robot.angles is None else robot.angles.copy(),
                               'limits': robot.workspace_limits, 'moves': len(self.executor.moves)})
        self.robot.events.append('lay down')
        return self.lay_down_result

    def run(self, session: Any, key_reader: Callable[[], Command | None] | None = None) -> int:
        with mock.patch.object(calibrate_board.record_static_load, 'lower_to_rest', side_effect=self.lower_to_rest):
            return run_session(self.context(key_reader), session)

    def text(self) -> str:
        return '\n'.join(str(line) for line in self.out)

    def park_angles(self) -> np.ndarray:
        return inverse_kinematics(park_pose(config, True), self.robot.robot_config)

    def assert_laid_down_from_park(self, test: unittest.TestCase) -> None:
        """the arm was raised and parked, the limits were back, and only then was it laid down, once"""
        test.assertEqual(1, len(self.lay_downs), 'lower_to_rest has to be called once')
        seen = self.lay_downs[0]
        test.assertTrue(np.allclose(seen['angles'][1:], self.park_angles()[1:],
                                     atol=1e-6 + abs(self.executor.sag)), 'not at the park pose')
        test.assertIs(self.original_limits, seen['limits'], 'the limits are not back when the arm is laid down')
        test.assertEqual(len(self.executor.moves), seen['moves'], 'the arm moved after the park move')


def corners_session(path: str = FROZEN_STACKING_CONFIG_PATH, height: float = 1.0, write: bool | None = False,
                    loaded: StackingConfig | None = None) -> CornersSession:
    return CornersSession(loaded or config, path, height, write)


def sink_session(gripper: str = 'down', x: float = 0.0, distances: Iterable[float] | None = None,
                 output: str | None = None) -> SinkSession:
    chosen = tuple(distances) if distances is not None else calibrate_board.DEFAULT_DISTANCES[gripper]
    return SinkSession(config, gripper, x, chosen, output)


class TemporaryFiles(unittest.TestCase):

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def frozen_text(self) -> str:
        return Path(FROZEN_STACKING_CONFIG_PATH).read_bytes().decode('utf-8').replace('\r\n', '\n')

    def write_yaml(self, text: str | None = None, newline: str = '\n', name: str = 'stacking.yaml') -> Path:
        path = self.directory / name
        path.write_bytes((self.frozen_text() if text is None else text).replace('\n', newline).encode('utf-8'))
        return path


class CornersTests(TemporaryFiles):

    def run_corners(self, d6: float, keys: Iterable[Any], **session_options: Any) -> tuple[Rig, CornersSession, int]:
        rig = Rig(d6, keys)
        session = corners_session(**session_options)
        code = rig.run(session)
        return rig, session, code

    def test_without_a_jog_the_result_is_the_yaml(self) -> None:
        for d6 in D6S:
            with self.subTest(d6=d6):
                rig, session, code = self.run_corners(d6, ['ok', 'ok'])
                self.assertEqual(0, code)
                self.assertEqual({'top-right': TOP_RIGHT, 'top-left': TOP_LEFT}, {
                    name: (round(x, 9), round(y, 9)) for name, (x, y) in session.measured.items()})
                estimate = estimate_board(config.board, session.measured)
                assert estimate is not None
                self.assertAlmostEqual(-12.0, estimate.x, places=9)
                self.assertAlmostEqual(12.0, estimate.y, places=9)
                self.assertAlmostEqual(0.0, estimate.rotation, places=9)
                self.assertIn('15.00 cm apart, the board is 15.00 cm wide: difference +0.00 cm', rig.text())
                self.assertNotIn('WARNING', rig.text())
                self.assertIn('  position: [-12.0, 12.0]  # jogged to the corners on 2026-11-26', rig.out)
                self.assertIn('  rotation: 0.0', rig.out)
                rig.assert_laid_down_from_park(self)

    def test_the_same_jog_at_both_corners_shifts_the_position(self) -> None:
        for d6 in D6S:
            with self.subTest(d6=d6):
                rig, session, code = self.run_corners(d6, ['r', 'r', 'f', 'ok', 'r', 'r', 'f', 'ok'])
                self.assertEqual(0, code)
                estimate = estimate_board(config.board, session.measured)
                assert estimate is not None
                self.assertAlmostEqual(-11.8, estimate.x, places=6)
                self.assertAlmostEqual(12.1, estimate.y, places=6)
                self.assertAlmostEqual(0.0, estimate.rotation, places=6)
                self.assertIn('x +2.0 mm, y +1.0 mm, fingertips 1.00 cm', rig.text())
                self.assertIn('shift x +2.0 mm, y +1.0 mm', rig.text())
                self.assertIn('  position: [-11.8, 12.1]  # jogged to the corners on 2026-11-26', rig.out)

    def test_a_jog_at_one_corner_turns_the_board(self) -> None:
        for d6 in D6S:
            with self.subTest(d6=d6):
                # the top-left corner 3 mm further away from the base: the way from it to the top-right corner now
                # slopes down by 3 mm over 15 cm
                rig, session, code = self.run_corners(d6, ['ok', 'f 3', 'ok'])
                self.assertEqual(0, code)
                estimate = estimate_board(config.board, session.measured)
                assert estimate is not None
                self.assertAlmostEqual(math.degrees(math.atan2(-0.3, 15.0)), estimate.rotation, places=6)
                # the new board puts its top-right corner where the arm pointed and its top-left one in the direction
                # of the measured corner
                from src.camera.block_detection import board_to_robot
                new = type(config.board)(**{**config.board.__dict__, 'x': estimate.x, 'y': estimate.y,
                                            'rotation': estimate.rotation})
                self.assertTrue(np.allclose(board_to_robot([0.0, 9.0], new), TOP_RIGHT, atol=1e-6))
                left = board_to_robot([-15.0, 9.0], new)
                self.assertTrue(np.allclose(left, [-27.0, 21.3], atol=0.01))
                self.assertIn('15.00 cm apart', rig.text())  # 15.003

    def test_a_rotated_board_comes_back_the_same(self) -> None:
        text = self.frozen_text().replace('  rotation: 0.0\nblocks:', '  rotation: 20.0\nblocks:')
        self.assertIn('rotation: 20.0', text)
        path = self.write_yaml(text)
        rotated = load_stacking_config(path)
        for d6 in D6S:
            with self.subTest(d6=d6):
                rig = Rig(d6, ['ok', 'ok'])
                session = CornersSession(rotated, str(path), 1.0, False)
                rig.run(session)
                estimate = estimate_board(rotated.board, session.measured)
                assert estimate is not None
                self.assertAlmostEqual(-12.0, estimate.x, places=6)
                self.assertAlmostEqual(12.0, estimate.y, places=6)
                self.assertAlmostEqual(20.0, estimate.rotation, places=6)

    def test_a_rotation_stays_near_the_old_one(self) -> None:
        """a board turned to 359 degrees in the yaml does not come back as -1"""
        board = type(config.board)(**{**config.board.__dict__, 'rotation': 359.0})
        points = calibrate_board.corner_robot_points(board)
        estimate = estimate_board(board, points)
        assert estimate is not None
        self.assertAlmostEqual(359.0, estimate.rotation, places=6)

    def test_one_corner_skipped_keeps_the_rotation(self) -> None:
        for d6 in D6S:
            with self.subTest(d6=d6):
                rig, session, _ = self.run_corners(d6, ['s', 'r 2', 'ok'])  # the top-right corner is skipped
                self.assertEqual(['top-left'], list(session.measured))
                estimate = estimate_board(config.board, session.measured)
                assert estimate is not None
                self.assertAlmostEqual(-11.8, estimate.x, places=6)
                self.assertAlmostEqual(12.0, estimate.y, places=6)
                self.assertAlmostEqual(0.0, estimate.rotation, places=6)
                self.assertIn('top-right corner: not measured', rig.text())
                self.assertIn('only one corner was measured: the rotation stays 0.0', rig.text())
                self.assertNotIn('apart', rig.text())
                rig.assert_laid_down_from_park(self)

                rig, session, _ = self.run_corners(d6, ['f', 'ok', 's'])  # the top-left one is skipped
                self.assertEqual(['top-right'], list(session.measured))
                estimate = estimate_board(config.board, session.measured)
                assert estimate is not None
                self.assertAlmostEqual(-12.0, estimate.x, places=6)
                self.assertAlmostEqual(12.1, estimate.y, places=6)

    def test_both_corners_skipped_writes_nothing(self) -> None:
        path = self.write_yaml()
        before = path.read_bytes()
        rig, session, code = self.run_corners(15.0, ['s', 's'], path=str(path), write=True)
        self.assertEqual(0, code)
        self.assertEqual({}, session.measured)
        self.assertIn('nothing to write', rig.text())
        self.assertEqual(before, path.read_bytes())
        self.assertEqual([], rig.questions)

    def test_the_corners_start_at_the_height_of_the_fingertips(self) -> None:
        for d6, height in ((14.0, 1.0), (15.0, 1.0), (14.0, 2.5), (15.0, 0.4)):
            with self.subTest(d6=d6, height=height):
                seen: list[tuple[np.ndarray, float]] = []
                rig = Rig(d6)
                rig.keys = ScriptedKeys([lambda: self.look(rig, seen, 'ok'), lambda: self.look(rig, seen, 'ok')])
                rig.run(corners_session(height=height))
                self.assertEqual(2, len(seen))
                for (tip, fingertips), expected in zip(seen, (TOP_RIGHT, TOP_LEFT)):
                    self.assertTrue(np.allclose(tip[:2], expected, atol=1e-6))
                    self.assertAlmostEqual(height, fingertips, places=6)
                    # the tip is higher than the fingertips by what d6 is short of the reach of the fingers
                    self.assertAlmostEqual(height + 15.0 - d6, tip[2], places=6)

    @staticmethod
    def look(rig: Rig, seen: list[tuple[np.ndarray, float]], key: str) -> str:
        seen.append((rig.robot.tip(), rig.robot.fingertip_height()))
        return key

    def test_the_arm_goes_over_the_corner_at_the_hover_height_and_straight_down(self) -> None:
        for d6 in D6S:
            with self.subTest(d6=d6):
                rig, _, _ = self.run_corners(d6, ['ok', 'ok'])
                moves = rig.executor.moves
                # park, over the top-right corner, down, up, over the top-left corner, down, up, park
                self.assertEqual([False, False, True, True, False, True, True, False],
                                 [rig.executor.is_line(i) for i in range(len(moves))])
                hover = config.motion.hover_height
                over = tip_of(moves[1].angles[-1], d6)
                self.assertAlmostEqual(hover + 15.0 - d6, over[2], places=6)
                self.assertTrue(np.allclose(over[:2], TOP_RIGHT, atol=1e-6))
                self.assertAlmostEqual(hover + 15.0 - d6, tip_of(moves[3].angles[-1], d6)[2], places=6)
                self.assertEqual(['enable', 'gripper:100'], rig.robot.events[:2])
                self.assertEqual([1.0], rig.sleeps[:1])
                self.assertEqual([100.0], rig.robot.gripper_commands)

    def test_the_step_sizes(self) -> None:
        for d6 in D6S:
            with self.subTest(d6=d6):
                rig, session, _ = self.run_corners(d6, ['1', 'r', '2', 'r', '3', 'r', 'ok', '3', 'f', '1', 'f', 'ok'])
                self.assertAlmostEqual(TOP_RIGHT[0] + 0.65, session.measured['top-right'][0], places=9)
                self.assertAlmostEqual(TOP_LEFT[1] + 0.55, session.measured['top-left'][1], places=9)
                self.assertIn('step 0.5 mm', rig.text())
                self.assertIn('step 5 mm', rig.text())
                self.assertIn('x +0.5 mm', rig.text())
                self.assertIn('x +1.5 mm', rig.text())
                self.assertIn('x +6.5 mm', rig.text())

    def test_every_direction(self) -> None:
        rig, session, _ = self.run_corners(15.0, ['l 3', 'r', 'f 4', 'b', 'u 5', 'd 2', 'ok', 's'])
        x, y = session.measured['top-right']
        self.assertAlmostEqual(TOP_RIGHT[0] - 0.2, x, places=9)
        self.assertAlmostEqual(TOP_RIGHT[1] + 0.3, y, places=9)
        self.assertIn('fingertips 1.30 cm', rig.text())  # 5 steps up and 2 down from 1.0

    def test_a_refused_jog_leaves_the_arm_and_the_run_goes_on(self) -> None:
        for d6 in D6S:
            with self.subTest(d6=d6):
                rig = Rig(d6)
                held: list[tuple[int, np.ndarray]] = []
                rig.keys = ScriptedKeys([
                    lambda: self.remember(rig, held, 'd 30'),  # to -2 cm, below the floor of the fingertips
                    lambda: self.remember(rig, held, 'r'),  # the arm did not move: this key moves it 1 mm
                    'ok', 's'])
                rig.run(corners_session())
                self.assertIn('refused, the arm stays', rig.text())
                self.assertIn('below the minimum height', rig.text())
                self.assertEqual(1, rig.text().count('refused'))
                self.assertEqual(held[0][0], held[1][0], 'the refused jog was executed')
                self.assertTrue(np.array_equal(held[0][1], held[1][1]))
                # the run went on with the next key
                self.assertEqual(1, len([line for line in rig.out if line.startswith('  offset x +1.0 mm')]))

    @staticmethod
    def remember(rig: Rig, held: list[tuple[int, np.ndarray]], key: str) -> str:
        assert rig.robot.angles is not None
        held.append((len(rig.executor.moves), rig.robot.angles.copy()))
        return key

    def test_the_floor_is_minus_one_centimetre_for_the_fingertips(self) -> None:
        for d6 in D6S:
            with self.subTest(d6=d6):
                rig = Rig(d6)
                limits: list[Any] = []
                low: list[float] = []
                rig.keys = ScriptedKeys([
                    lambda: self.note_limits(rig, limits, 'd 20'),  # to -1.0 exactly: allowed
                    lambda: self.low_point(rig, low, 'd'),  # below it: refused
                    lambda: self.low_point(rig, low, 'ok'), 's'])
                rig.run(corners_session())
                self.assertAlmostEqual(-1.0, low[0], places=6)
                self.assertAlmostEqual(-1.0, low[1], places=6)
                self.assertIn('refused', rig.text())
                # while it ran the robot had limits with the floor of the tip lowered and everything else copied
                during = limits[0]
                self.assertAlmostEqual(-1.0 + 15.0 - d6, during.z_min, places=9)
                self.assertEqual(Limits.wrist_reach_max, during.wrist_reach_max)
                self.assertEqual(Limits.y_min, during.y_min)
                self.assertEqual(Limits.wrist_z_min, during.wrist_z_min)
                self.assertEqual(Limits.radius_min, during.radius_min)
                self.assertEqual(Limits.wrist_radius_min, during.wrist_radius_min)
                self.assertEqual(1, Limits.z_min, 'the standard limits are not touched')

    @staticmethod
    def note_limits(rig: Rig, limits: list[Any], key: str) -> str:
        limits.append(rig.robot.workspace_limits)
        return key

    @staticmethod
    def low_point(rig: Rig, low: list[float], key: str) -> str:
        low.append(rig.robot.fingertip_height())
        return key

    def test_the_floor_is_the_standard_one_without_the_run(self) -> None:
        robot = FakeRobot(15.0)
        lowered = limits_with_tip_floor(robot.workspace_limits, -1.0)
        self.assertEqual(-1.0, lowered.z_min)
        self.assertEqual(1, robot.workspace_limits.z_min)
        self.assertEqual(41.8, lowered.wrist_reach_max)

    def test_the_lowered_floor_carries_every_attribute_of_the_limits(self) -> None:
        class MoreLimits(Limits):
            some_later_limit = 7.5  # a limit added to the robot after this script was written

            @staticmethod
            def a_method() -> None:
                pass

        lowered = limits_with_tip_floor(MoreLimits, -1.0)
        for name in ('radius_min', 'wrist_radius_min', 'wrist_reach_max', 'y_min', 'wrist_z_min', 'some_later_limit'):
            self.assertEqual(getattr(MoreLimits, name), getattr(lowered, name), name)
        self.assertEqual(-1.0, lowered.z_min)
        self.assertFalse(hasattr(lowered, 'a_method'))

    def test_the_wrist_limit_is_still_enforced_while_the_floor_is_lowered(self) -> None:
        for d6 in D6S:
            with self.subTest(d6=d6):
                rig = Rig(d6)
                # the gripper level 15 cm beyond a wrist that is only 11 cm from the base axis, high above the floor
                too_close = rig.robot.pose_to_angles(Pose(0.0, 11.0 + d6, 8.0))
                seen: list[str] = []

                def during_the_run() -> str:
                    # the limits of the run have the lowered floor, and still refuse the wrist that close to the base
                    self.assertLess(rig.robot.workspace_limits.z_min, Limits.z_min)
                    with self.assertRaises(MovementException) as refused:
                        rig.robot.check_workspace(too_close)
                    seen.append(str(refused.exception))
                    return 'ok'

                rig.keys = ScriptedKeys([during_the_run, 'ok'])
                rig.run(corners_session())
                self.assertEqual(1, len(seen))
                self.assertIn('from the base, closer than 12.5', seen[0])
                # the same pose is fine for the limits of a robot without the wrist limit
                rig.robot.workspace_limits = type('WithoutWristLimit', (), {
                    name: getattr(Limits, name) for name in ('radius_min', 'wrist_reach_max', 'y_min', 'z_min',
                                                             'wrist_z_min')})
                rig.robot.check_workspace(too_close)

    def test_the_limits_are_back_after_every_end(self) -> None:
        for own in (False, True):
            for name, keys, expected in (('end', ['ok', 'ok'], 0), ('abort', ['q'], 1),
                                         ('error', [RuntimeError('boom')], None),
                                         ('interrupt', [KeyboardInterrupt()], 1)):
                with self.subTest(end=name, own=own):
                    rig = Rig(15.0, keys, own_limits=own)
                    if expected is None:
                        with self.assertRaises(RuntimeError):
                            rig.run(corners_session())
                    else:
                        self.assertEqual(expected, rig.run(corners_session()))
                    self.assertIs(rig.original_limits, rig.robot.workspace_limits)
                    self.assertEqual(own, 'workspace_limits' in vars(rig.robot))
                    self.assertEqual(1, Limits.z_min)

    def test_every_end_lays_the_arm_down_after_raising_and_parking(self) -> None:
        ends: list[tuple[str, list[Any], Any]] = [
            ('accepted', ['ok', 'ok'], None), ('abort with q', ['q'], None),
            ('abort after a jog', ['d 5', 'q'], None), ('abort at the second corner', ['ok', 'd 4', 'q'], None),
            ('an error', ['d 5', RuntimeError('boom')], RuntimeError),
            ('ctrl+c', ['d 5', KeyboardInterrupt()], None)]
        for d6 in D6S:
            for name, keys, error in ends:
                with self.subTest(d6=d6, end=name):
                    rig = Rig(d6, keys)
                    if error is None:
                        rig.run(corners_session())
                    else:
                        with self.assertRaises(error):
                            rig.run(corners_session())
                    rig.assert_laid_down_from_park(self)
                    # the last two moves: straight up to the hover height, joint move to the park pose
                    moves = rig.executor.moves
                    self.assertTrue(rig.executor.is_line(len(moves) - 2))
                    self.assertFalse(rig.executor.is_line(len(moves) - 1))
                    self.assertAlmostEqual(config.motion.hover_height + 15.0 - d6,
                                           tip_of(moves[-2].angles[-1], d6)[2], places=6)
                    self.assertEqual('lay down', rig.robot.events[-1])
                    self.assertNotIn('disable', rig.robot.events)  # the torque is lower_to_rest's business

    def test_esc_and_q_abort_from_the_console(self) -> None:
        for chars in (['\x1b'], ['q'], ['Q']):
            with self.subTest(chars=chars):
                rig = Rig(15.0)
                reader = make_key_reader(rig.out.append, console=True, getwch=iter(chars).__next__)
                code = rig.run(corners_session(), reader)
                self.assertEqual(1, code)
                self.assertIn('aborted, nothing is written', rig.text())
                rig.assert_laid_down_from_park(self)

    def test_the_end_does_not_raise_when_the_arm_was_moved_nowhere(self) -> None:
        rig = Rig(15.0)
        rig.robot.angles = None  # the angles can not be read
        with self.assertRaises(RuntimeError):
            rig.run(corners_session())
        self.assertEqual([], rig.executor.moves)
        self.assertEqual(1, len(rig.lay_downs))  # the servos were enabled, so the arm is laid down
        self.assertIn('can not be read', rig.text())

    def test_a_refused_move_ends_the_run_and_lays_the_arm_down(self) -> None:
        class WristFloorTooHigh(Limits):
            wrist_z_min = 100  # no pose is allowed, the park move is refused at once

        rig = Rig(15.0, ['ok', 'ok'])
        rig.robot.workspace_limits = WristFloorTooHigh
        self.assertEqual(1, rig.run(corners_session()))
        self.assertIn('stopped, a move is refused', rig.text())
        self.assertIn('failed, nothing is written', rig.text())
        self.assertEqual(1, len(rig.lay_downs))
        self.assertIs(WristFloorTooHigh, rig.robot.workspace_limits)

    def test_servos_that_do_not_enable_move_nothing(self) -> None:
        rig = Rig(15.0)
        rig.robot.enable_result = False
        self.assertEqual(1, rig.run(corners_session()))
        self.assertEqual([], rig.executor.moves)
        self.assertEqual([], rig.lay_downs)
        self.assertEqual([], rig.robot.gripper_commands)
        self.assertIs(Limits, rig.robot.workspace_limits)
        self.assertIn('nothing was moved', rig.text())

    def test_a_lay_down_that_does_not_work_prints_the_hint(self) -> None:
        rig = Rig(15.0, ['ok', 'ok'])
        rig.lay_down_result = 'servos [3] have a hardware error and are limp'
        rig.run(corners_session())
        self.assertIn('the arm is not laid down: servos [3] have a hardware error and are limp', rig.text())
        self.assertIn('the servos still hold the arm; {} lowers it'.format(LOWER_ARM_COMMAND), rig.text())
        rig = Rig(15.0, ['ok', 'ok'])
        rig.run(corners_session())
        self.assertNotIn('lower_arm', rig.text())

    def test_a_corner_that_can_not_be_reached_is_skipped(self) -> None:
        far = type(config.board)(**{**config.board.__dict__, 'x': -45.0, 'y': 40.0})
        loaded = StackingConfig(far, config.blocks, config.tower, config.gripper, config.motion, config.scan,
                                config.click_mode)
        rig = Rig(15.0)
        session = CornersSession(loaded, FROZEN_STACKING_CONFIG_PATH, 1.0, False)
        self.assertEqual(0, rig.run(session))
        self.assertEqual({}, session.measured)
        self.assertEqual(2, rig.text().count('can not go to the'))
        rig.assert_laid_down_from_park(self)

    def test_a_press_is_one_short_move(self) -> None:
        rig, _, _ = self.run_corners(15.0, ['r', 'r 3', 'ok', 's'])
        jogs = [move for move in rig.executor.moves if move.duration < 1.0]
        self.assertEqual(2, len(jogs))
        self.assertAlmostEqual(0.25, jogs[0].duration, places=6)
        self.assertEqual(26, len(jogs[0].angles))


class ConsoleKeyTests(unittest.TestCase):

    @staticmethod
    def decode(chars: str) -> Command | None:
        queue = list(chars)
        command = decode_console_key(lambda: queue.pop(0))
        assert not queue, 'characters left over'
        return command

    def test_arrows_and_page_keys(self) -> None:
        for prefix in ('\x00', '\xe0'):
            for letter, action in (('H', FORWARD), ('P', BACK), ('K', LEFT), ('M', RIGHT), ('I', HIGHER),
                                   ('Q', LOWER)):
                self.assertEqual(Command(action), self.decode(prefix + letter))
        self.assertIsNone(self.decode('\xe0Z'))
        self.assertIsNone(self.decode('\x00;'))  # a function key

    def test_the_other_keys(self) -> None:
        self.assertEqual(Command(ACCEPT), self.decode('\r'))
        self.assertEqual(Command(ACCEPT), self.decode('\n'))
        self.assertEqual(Command(ABORT), self.decode('\x1b'))
        self.assertEqual(Command(ABORT), self.decode('q'))
        self.assertEqual(Command(ABORT), self.decode('Q'))
        self.assertEqual(Command(SKIP), self.decode('s'))
        self.assertEqual(Command(SKIP), self.decode('S'))
        for digit in '123':
            self.assertEqual(Command(STEP, int(digit)), self.decode(digit))
        self.assertIsNone(self.decode('x'))
        self.assertIsNone(self.decode('4'))
        self.assertIsNone(self.decode(' '))

    def test_ctrl_c_is_an_interrupt(self) -> None:
        with self.assertRaises(KeyboardInterrupt):
            self.decode('\x03')

    def test_the_reader_in_a_console(self) -> None:
        chars = iter(['\xe0', 'M', '\r'])
        reader = make_key_reader(console=True, getwch=chars.__next__)
        self.assertEqual(Command(RIGHT), reader())
        self.assertEqual(Command(ACCEPT), reader())
        self.assertIn('Page Up', reader.help)


class TypedCommandTests(unittest.TestCase):

    def test_directions_with_and_without_a_number(self) -> None:
        for letter, action in (('l', LEFT), ('r', RIGHT), ('f', FORWARD), ('b', BACK), ('u', HIGHER), ('d', LOWER)):
            self.assertEqual(Command(action), parse_typed_command(letter))
            self.assertEqual(Command(action, 3), parse_typed_command(letter + ' 3'))
        self.assertEqual(Command(RIGHT, 3), parse_typed_command('r 3'))
        self.assertEqual(Command(RIGHT, 12), parse_typed_command('  R12 '))

    def test_the_others(self) -> None:
        self.assertEqual(Command(ACCEPT), parse_typed_command('ok'))
        self.assertEqual(Command(ACCEPT), parse_typed_command(' OK '))
        self.assertEqual(Command(SKIP), parse_typed_command('s'))
        self.assertEqual(Command(ABORT), parse_typed_command('q'))
        for digit in '123':
            self.assertEqual(Command(STEP, int(digit)), parse_typed_command(digit))
        for nonsense in ('', 'x', 'r 0', 'r -1', 'rr', '4', 'ok 2', 'r 3 4'):
            self.assertIsNone(parse_typed_command(nonsense), nonsense)

    def test_the_typed_reader(self) -> None:
        lines = ['r 3', 'nonsense', 'ok']
        said: list[str] = []

        def typed(prompt: str) -> str:
            return lines.pop(0)

        reader = make_key_reader(said.append, console=False, input_fn=typed)
        self.assertEqual(Command(RIGHT, 3), reader())
        self.assertIsNone(reader())
        self.assertEqual(["not understood: 'nonsense'"], said)
        self.assertEqual(Command(ACCEPT), reader())
        self.assertEqual(TYPED_HELP, reader.help)

    def test_no_more_input_aborts(self) -> None:
        said: list[str] = []

        def closed(prompt: str) -> str:
            raise EOFError

        reader = make_key_reader(said.append, console=False, input_fn=closed)
        self.assertEqual(Command(ABORT), reader())
        self.assertEqual(1, len(said))

    def test_stdin_that_is_not_a_console_falls_back_to_typing(self) -> None:
        class NotATty:
            def isatty(self) -> bool:
                return False

        class ATty:
            def isatty(self) -> bool:
                return True

        with mock.patch('sys.stdin', NotATty()):
            self.assertEqual(TYPED_HELP, make_key_reader().help)
        with mock.patch('sys.stdin', ATty()):
            self.assertEqual(calibrate_board.CONSOLE_HELP, make_key_reader(getwch=lambda: 'q').help)
        with mock.patch('sys.stdin', None):
            self.assertEqual(TYPED_HELP, make_key_reader().help)


class YamlTests(TemporaryFiles):

    estimate = BoardEstimate(-11.8, 12.1, 1.5)

    def test_only_the_two_board_lines_change(self) -> None:
        old = self.frozen_text()
        new = replace_board_pose(old, self.estimate, '2026-11-26')
        old_lines, new_lines = old.split('\n'), new.split('\n')
        self.assertEqual(len(old_lines), len(new_lines))
        changed = [(a, b) for a, b in zip(old_lines, new_lines) if a != b]
        self.assertEqual([('  position: [-12.0, 12.0]',
                           '  position: [-11.8, 12.1]  # jogged to the corners on 2026-11-26'),
                          ('  rotation: 0.0', '  rotation: 1.5')], changed)
        # the same names in the tower section and the documentation are as they were
        self.assertIn('\n  position: [22.0, 18.0]\n  rotation: 0.0\n  clear_radius: 6.0\n', new)
        self.assertIn('#     position: [x, y]       of the centre of the tower', new)
        self.assertIn('  # NOT MEASURED YET: measure it after the board is taped down', new)

    def test_the_comments_of_the_lines_are_kept(self) -> None:
        old = self.frozen_text().replace('position: [-12.0, 12.0]', 'position: [-12.0, 12.0]   # measured by hand') \
            .replace('rotation: 0.0\nblocks', 'rotation:   0.0   # to be checked\nblocks')
        new = replace_board_pose(old, self.estimate, '2026-11-26')
        self.assertIn('  position: [-11.8, 12.1]   # measured by hand; jogged to the corners on 2026-11-26\n', new)
        self.assertIn('  rotation:   1.5   # to be checked\n', new)
        # a second run replaces the note of the first one
        again = replace_board_pose(new, BoardEstimate(-11.0, 12.0, 0.0), '2026-12-01')
        self.assertIn('  position: [-11.0, 12.0]   # measured by hand; jogged to the corners on 2026-12-01\n', again)
        self.assertNotIn('2026-11-26', again)

    def test_numbers_are_short_and_never_negative_zero(self) -> None:
        text = replace_board_pose(self.frozen_text(), BoardEstimate(-0.0000001, 12.3456789, -0.0000001), '2026-11-26')
        self.assertIn('position: [0.0, 12.35]', text)
        self.assertIn('  rotation: 0.0\n', text)

    def test_the_file_is_written_and_loads(self) -> None:
        path = self.write_yaml()
        write_board_pose(path, self.estimate, '2026-11-26')
        board = load_stacking_config(path).board
        self.assertEqual((-11.8, 12.1, 1.5), (board.x, board.y, board.rotation))
        self.assertEqual((22.0, 18.0, 0.0), (lambda t: (t.x, t.y, t.rotation))(load_stacking_config(path).tower))

    def test_line_endings_are_kept(self) -> None:
        for newline in ('\n', '\r\n'):
            with self.subTest(newline=repr(newline)):
                path = self.write_yaml(newline=newline)
                write_board_pose(path, self.estimate, '2026-11-26')
                data = path.read_bytes()
                if newline == '\n':
                    self.assertNotIn(b'\r', data)
                else:
                    self.assertEqual(data.count(b'\n'), data.count(b'\r\n'))
                self.assertEqual(self.frozen_text().count('\n'), data.count(b'\n'))
                self.assertEqual(-11.8, load_stacking_config(path).board.x)

    def test_an_invalid_result_puts_the_old_content_back(self) -> None:
        path = self.write_yaml()
        before = path.read_bytes()
        with self.assertRaises(StackingConfigError):
            write_board_pose(path, BoardEstimate(float('nan'), 12.0, 0.0), '2026-11-26')
        self.assertEqual(before, path.read_bytes())

    def test_a_yaml_without_the_lines_is_refused(self) -> None:
        path = self.write_yaml(self.frozen_text().replace('  position: [-12.0, 12.0]\n', ''))
        before = path.read_bytes()
        with self.assertRaises(ValueError):
            write_board_pose(path, self.estimate, '2026-11-26')
        self.assertEqual(before, path.read_bytes())

    def test_the_lines_to_put_under_board(self) -> None:
        self.assertEqual(['  position: [-11.8, 12.1]  # jogged to the corners on 2026-11-26', '  rotation: 1.5'],
                         board_pose_lines(self.estimate, '2026-11-26'))


class WritingFromTheRunTests(TemporaryFiles):

    def run_with(self, write: bool | None, answers: Iterable[str], keys: Iterable[Any] = ('r', 'ok', 'r', 'ok')
                 ) -> tuple[Rig, Path, bytes]:
        path = self.write_yaml()
        before = path.read_bytes()
        rig = Rig(15.0, keys, answers)
        rig.run(corners_session(str(path), write=write))
        return rig, path, before

    def test_no_write_never_writes_or_asks(self) -> None:
        rig, path, before = self.run_with(False, [])
        self.assertEqual(before, path.read_bytes())
        self.assertEqual([], rig.questions)
        self.assertIn('nothing is written', rig.text())

    def test_an_answer_that_is_not_yes_writes_nothing(self) -> None:
        for answer in ('no', '', 'y'):
            with self.subTest(answer=answer):
                rig, path, before = self.run_with(None, [answer])
                self.assertEqual(before, path.read_bytes())
                self.assertEqual(1, len(rig.questions))

    def test_yes_writes(self) -> None:
        rig, path, before = self.run_with(None, ['yes'])
        self.assertNotEqual(before, path.read_bytes())
        self.assertAlmostEqual(-11.9, load_stacking_config(path).board.x)
        self.assertIn('written to', rig.text())

    def test_write_writes_without_asking(self) -> None:
        rig, path, before = self.run_with(True, [])
        self.assertEqual([], rig.questions)
        self.assertAlmostEqual(-11.9, load_stacking_config(path).board.x)

    def test_no_input_writes_nothing(self) -> None:
        path = self.write_yaml()
        before = path.read_bytes()
        rig = Rig(15.0, ['ok', 'ok'])

        def closed(question: str) -> str:
            raise EOFError

        rig.ask = closed  # type: ignore[method-assign]
        rig.run(corners_session(str(path), write=None))
        self.assertEqual(before, path.read_bytes())
        self.assertIn('use --write', rig.text())

    def test_a_failed_write_is_reported(self) -> None:
        path = self.write_yaml(self.frozen_text().replace('  position: [-12.0, 12.0]\n', ''))
        before = path.read_bytes()
        rig = Rig(15.0, ['ok', 'ok'])
        code = rig.run(corners_session(str(path), write=True))
        self.assertEqual(1, code)
        self.assertEqual(before, path.read_bytes())
        self.assertIn('NOT written', rig.text())


class SinkTests(TemporaryFiles):

    def run_sink(self, d6: float, keys: Iterable[Any], gripper: str = 'down', sag: float = 0.002,
                 **options: Any) -> tuple[Rig, SinkSession, int, Path]:
        output = self.directory / 'sink.csv'
        rig = Rig(d6, keys)
        rig.executor.sag = sag
        session = sink_session(gripper, output=str(output), **options)
        code = rig.run(session)
        return rig, session, code, output

    def read_csv(self, path: Path) -> list[dict[str, str]]:
        with open(path, newline='') as csv_file:
            return list(csv.DictReader(csv_file))

    def test_down_spots_and_start_heights(self) -> None:
        for d6 in D6S:
            with self.subTest(d6=d6):
                rig = Rig(d6)
                seen: list[tuple[np.ndarray, float]] = []
                rig.keys = ScriptedKeys([lambda: CornersTests.look(rig, seen, 'ok')] * 4)
                rig.run(sink_session(output=str(self.directory / 'a.csv')))
                self.assertEqual([18.0, 25.0, 32.0, 39.0], [round(float(tip[1]), 6) for tip, _ in seen])
                self.assertEqual([0.0] * 4, [round(float(tip[0]), 6) for tip, _ in seen])
                for _, fingertips in seen:
                    self.assertAlmostEqual(3.0, fingertips, places=6)
                moves = rig.executor.moves
                # park, then per spot: over it at the hover height, down; and up before the next spot
                self.assertEqual([False, False, True, True, False, True, True, False, True, True, False, True, True,
                                  False],
                                 [rig.executor.is_line(i) for i in range(len(moves))])
                rig.assert_laid_down_from_park(self)

    def test_lowering_seven_steps_records_two_point_three(self) -> None:
        for d6 in D6S:
            with self.subTest(d6=d6):
                keys = ['d 7', 'ok'] + ['d', 'd', 'd', 'd', 'd', 'd', 'd', 'ok'] + ['d 5', 'ok', 's']
                rig, session, code, output = self.run_sink(d6, keys)
                self.assertEqual(0, code)
                self.assertEqual(3, len(session.rows))
                self.assertAlmostEqual(2.3, session.rows[0].commanded, places=9)
                self.assertAlmostEqual(2.3, session.rows[1].commanded, places=9)
                self.assertAlmostEqual(2.5, session.rows[2].commanded, places=9)
                self.assertEqual([18.0, 25.0, 32.0], [row.y for row in session.rows])  # the last one was skipped
                self.assertIn('recorded: commanded 2.30 cm', rig.text())
                self.assertIn('y 39 cm is skipped', rig.text())

    def test_the_encoder_height_comes_from_the_angles_of_the_robot(self) -> None:
        for d6 in D6S:
            with self.subTest(d6=d6):
                rig = Rig(d6)
                rig.executor.sag = 0.01
                at_accept: list[float] = []
                rig.keys = ScriptedKeys(['d 7', lambda: self.note_height(rig, at_accept, 'ok'), 's', 's', 's'])
                session = sink_session(output=str(self.directory / 'b.csv'))
                rig.run(session)
                row = session.rows[0]
                self.assertAlmostEqual(at_accept[0], row.encoder or math.nan, places=9)
                self.assertAlmostEqual(2.3, row.commanded, places=9)
                self.assertGreater(abs((row.encoder or math.nan) - row.commanded), 0.05)  # the sag of joint 2 shows

    @staticmethod
    def note_height(rig: Rig, heights: list[float], key: str) -> str:
        heights.append(rig.robot.fingertip_height())
        return key

    def test_unreadable_encoders_are_recorded_as_empty(self) -> None:
        rig = Rig(15.0)

        def unreadable() -> str:
            rig.robot.angles = None
            return 'ok'

        rig.keys = ScriptedKeys([unreadable, RuntimeError('stop here')])
        session = sink_session(output=str(self.directory / 'c.csv'))
        with self.assertRaises(RuntimeError):
            rig.run(session)
        self.assertEqual(1, len(session.rows))
        self.assertIsNone(session.rows[0].encoder)

    def test_the_table_the_line_and_the_csv(self) -> None:
        for d6 in D6S:
            with self.subTest(d6=d6):
                keys = ['d 5', 'ok', 'd 7', 'ok', 'd 9', 'ok', 'd 11', 'ok']
                rig, session, code, output = self.run_sink(d6, keys)
                self.assertEqual(0, code)
                text = rig.text()
                self.assertIn('distance cm  commanded cm   encoders cm', text)
                for distance, commanded in ((18, 2.5), (25, 2.3), (32, 2.1), (39, 1.9)):
                    row = next(row for row in session.rows if row.y == distance)
                    line = next(line for line in rig.out if line.strip().startswith('{:.1f}'.format(distance)))
                    self.assertIn('{:.2f}'.format(commanded), line)
                    self.assertIn('{:.2f}'.format(row.encoder or math.nan), line)
                self.assertIn('the touch height changes by -0.29 cm per 10 cm of distance', text)
                rows = self.read_csv(output)
                self.assertEqual(['gripper', 'x', 'y', 'commanded_height', 'encoder_height', 'd6', 'date'],
                                 list(rows[0]))
                self.assertEqual(4, len(rows))
                self.assertEqual({'gripper': 'down', 'x': '0', 'y': '18', 'commanded_height': '2.5000',
                                  'encoder_height': '{:.4f}'.format(session.rows[0].encoder or math.nan),
                                  'd6': '{:g}'.format(d6), 'date': '2026-11-26 14:05'}, rows[0])
                self.assertEqual(['2.3000', '2.1000', '1.9000'], [row['commanded_height'] for row in rows[1:]])
                self.assertIn('the rows are in {}'.format(output), text)

    def test_the_default_csv_is_in_the_tuning_folder(self) -> None:
        path = calibrate_board.default_output('forward', DATE)
        self.assertEqual('sink_forward_20261126_1405.csv', os.path.basename(path))
        expected = os.path.join('src', 'robot_controllers', 'dynamixel_robot', 'calibration', 'tuning')
        self.assertIn(expected, os.path.dirname(path))
        self.assertTrue(os.path.isdir(os.path.dirname(path)))

    def test_a_line_needs_two_distances(self) -> None:
        rig, session, _, output = self.run_sink(15.0, ['d 3', 'ok'], distances=[25.0])
        self.assertIn('one distance only, no line to fit', rig.text())
        self.assertEqual(1, len(self.read_csv(output)))
        self.assertEqual(None, fit_line([25.0], [2.0]))
        self.assertEqual(None, fit_line([25.0, 25.0], [2.0, 3.0]))
        intercept, slope = fit_line([0.0, 10.0, 20.0], [1.0, 2.0, 3.0]) or (math.nan, math.nan)
        self.assertAlmostEqual(1.0, intercept)
        self.assertAlmostEqual(0.1, slope)

    def test_x_moves_the_line_and_the_distance_is_from_the_base(self) -> None:
        rig = Rig(15.0)
        seen: list[tuple[np.ndarray, float]] = []
        rig.keys = ScriptedKeys([lambda: CornersTests.look(rig, seen, 'ok')] * 2)
        session = sink_session(x=12.0, distances=[16.0, 25.0], output=str(self.directory / 'd.csv'))
        rig.run(session)
        self.assertTrue(np.allclose([tip[:2] for tip, _ in seen], [[12.0, 16.0], [12.0, 25.0]], atol=1e-6))
        self.assertIn('{:>10.1f}'.format(20.0), rig.text())  # hypot(12, 16)
        self.assertEqual('12', self.read_csv(self.directory / 'd.csv')[0]['x'])

    def test_an_abort_reports_what_was_measured_and_lays_the_arm_down(self) -> None:
        rig, session, code, output = self.run_sink(15.0, ['d 7', 'ok', 'q'])
        self.assertEqual(1, code)
        self.assertEqual(1, len(session.rows))
        self.assertIn('aborted, the spots measured so far are reported', rig.text())
        self.assertEqual(1, len(self.read_csv(output)))
        rig.assert_laid_down_from_park(self)

    def test_the_down_run_has_the_lowered_floor(self) -> None:
        rig = Rig(15.0)
        limits: list[Any] = []
        rig.keys = ScriptedKeys([lambda: CornersTests.note_limits(rig, limits, 'q')])
        rig.run(sink_session(output=str(self.directory / 'e.csv')))
        self.assertEqual(-1.0, limits[0].z_min)
        self.assertIs(Limits, rig.robot.workspace_limits)

    # the gripper that points forward

    def test_forward_spots_and_start_heights(self) -> None:
        for d6 in D6S:
            with self.subTest(d6=d6):
                rig = Rig(d6)
                seen: list[tuple[np.ndarray, float]] = []
                limits: list[Any] = []
                session = sink_session('forward', output=str(self.directory / 'f.csv'))
                rig.keys = ScriptedKeys([lambda: self.look_both(rig, seen, limits, 'ok')] * 4)
                rig.run(session)
                self.assertEqual([30.0, 37.0, 44.0, 51.0], [round(float(tip[1]), 6) for tip, _ in seen])
                for tip, _ in seen:
                    self.assertAlmostEqual(6.0, tip[2], places=6)  # the tip z, whatever d6 is
                    self.assertAlmostEqual(0.0, tip[0], places=6)
                # the floor is the standard one
                self.assertTrue(all(found is Limits for found in limits))
                self.assertEqual(4, len(session.rows))

    @staticmethod
    def look_both(rig: Rig, seen: list[tuple[np.ndarray, float]], limits: list[Any], key: str) -> str:
        limits.append(rig.robot.workspace_limits)
        assert rig.robot.angles is not None
        from src.kinematics.kinematics import forward_orientation_kinematics
        direction = forward_orientation_kinematics(rig.robot.angles)[:, 2]
        assert np.allclose(direction, [0.0, 1.0, 0.0], atol=1e-6), 'the gripper does not point forward'
        return CornersTests.look(rig, seen, key)

    def test_forward_moves_between_the_spots_in_straight_lines_at_the_travel_height(self) -> None:
        for d6 in D6S:
            with self.subTest(d6=d6):
                rig, session, code, _ = self.run_sink(d6, ['d 7', 'ok', 'd 7', 'ok', 's', 's'], gripper='forward',
                                                      distances=[30.0, 37.0, 44.0, 51.0])
                self.assertEqual(0, code)
                # the jogs (0.25 s) are left out: park, joint move over the first spot at z 8, down; up, along at z 8,
                # down; up, on, down; up, on, down; up, park (the last two spots are skipped)
                indices = [i for i, move in enumerate(rig.executor.moves) if move.duration >= 1.0]
                moves = [rig.executor.moves[i] for i in indices]
                self.assertEqual([False, False] + [True] * 11 + [False],
                                 [rig.executor.is_line(i) for i in indices])
                self.assertAlmostEqual(8.0, tip_of(moves[1].angles[-1], d6)[2], places=6)
                self.assertAlmostEqual(6.0, tip_of(moves[2].angles[-1], d6)[2], places=6)
                self.assertAlmostEqual(8.0, tip_of(moves[3].angles[-1], d6)[2], places=6)  # up from 5.3
                along = tip_of(moves[4].angles[-1], d6)
                self.assertAlmostEqual(8.0, along[2], places=6)
                self.assertAlmostEqual(37.0, along[1], places=6)
                self.assertAlmostEqual(6.0, tip_of(moves[5].angles[-1], d6)[2], places=6)
                self.assertAlmostEqual(8.0, tip_of(moves[-2].angles[-1], d6)[2], places=6)
                rig.assert_laid_down_from_park(self)

    def test_forward_lowering_seven_steps_records_five_point_three(self) -> None:
        for d6 in D6S:
            with self.subTest(d6=d6):
                rig, session, code, output = self.run_sink(
                    d6, ['d 7', 'ok', 'd 7', 'ok'], gripper='forward', distances=[30.0, 51.0])
                self.assertEqual(0, code)
                self.assertEqual([5.3, 5.3], [round(row.commanded, 9) for row in session.rows])
                rows = self.read_csv(output)
                self.assertEqual(['forward', 'forward'], [row['gripper'] for row in rows])
                # the encoders read the tip z, and the sag put it lower
                for row in session.rows:
                    self.assertGreater(abs((row.encoder or math.nan) - 5.3), 0.01)
                self.assertIn('tip z', rig.text())


class MainTests(TemporaryFiles):

    def make_main(self, argv: list[str], keys: Iterable[Any] = (), answers: Iterable[str] = (), d6: float = 15.0
                  ) -> tuple[Rig, int, list[int]]:
        rig = Rig(d6, keys, answers)
        built: list[int] = []

        def make_robot() -> FakeRobot:
            built.append(1)
            return rig.robot

        with mock.patch.object(calibrate_board.record_static_load, 'lower_to_rest', side_effect=rig.lower_to_rest):
            code = calibrate_board.main(argv, make_robot=make_robot, read_key=KeyReader(rig.keys, 'keys'),
                                        execute=rig.executor, sleep=rig.sleeps.append, say=rig.out.append,
                                        ask=rig.ask, now=lambda: DATE)
        return rig, code, built

    def yaml_argument(self) -> list[str]:
        return ['--config', str(self.write_yaml())]

    def test_it_asks_before_it_moves(self) -> None:
        rig, code, built = self.make_main(['corners', '--no-write'] + self.yaml_argument(), ['ok', 'ok'], ['yes'],
                                          d6=14.0)
        self.assertEqual(1, len(rig.questions))
        self.assertEqual(0, code)
        self.assertEqual([1], built)
        self.assertGreater(len(rig.executor.moves), 5)
        self.assertEqual(1, len(rig.lay_downs))
        self.assertIn('The arm moves', rig.text())

    def test_a_no_moves_nothing_and_does_not_even_start_the_robot(self) -> None:
        for answer in ('no', '', 'y'):
            with self.subTest(answer=answer):
                rig, code, built = self.make_main(['corners'] + self.yaml_argument(), [], [answer])
                self.assertEqual(0, code)
                self.assertEqual([], built)
                self.assertEqual([], rig.executor.moves)
                self.assertEqual([], rig.robot.events)
                self.assertEqual([], rig.lay_downs)
                self.assertIn('nothing was moved', rig.text())

    def test_yes_option_skips_the_question(self) -> None:
        rig, code, built = self.make_main(['--yes', '--no-write'] + self.yaml_argument(), ['ok', 'ok'], [])
        self.assertEqual([], rig.questions)
        self.assertEqual(0, code)
        self.assertEqual([1], built)

    def test_corners_is_the_default_and_the_options_reach_the_session(self) -> None:
        seen: list[tuple[np.ndarray, float]] = []
        rig = Rig(15.0)
        rig.keys = ScriptedKeys([lambda: CornersTests.look(rig, seen, 'q')])
        path = self.write_yaml()
        before = path.read_bytes()
        with mock.patch.object(calibrate_board.record_static_load, 'lower_to_rest', side_effect=rig.lower_to_rest):
            code = calibrate_board.main(['--yes', '--height', '2.5', '--config', str(path)],
                                        make_robot=lambda: rig.robot, read_key=KeyReader(rig.keys, ''),
                                        execute=rig.executor, sleep=rig.sleeps.append, say=rig.out.append,
                                        ask=rig.ask, now=lambda: DATE)
        self.assertEqual(1, code)
        self.assertAlmostEqual(2.5, seen[0][1], places=6)
        self.assertEqual(before, path.read_bytes())

    def test_write_writes_without_asking(self) -> None:
        path = self.write_yaml()
        rig, code, _ = self.make_main(['--yes', '--write', '--config', str(path)], ['r 5', 'ok', 'r 5', 'ok'])
        self.assertEqual(0, code)
        self.assertEqual([], rig.questions)
        self.assertAlmostEqual(-11.5, load_stacking_config(path).board.x)

    def test_sink_through_main(self) -> None:
        output = self.directory / 'main_sink.csv'
        rig, code, _ = self.make_main(['sink', '--yes', '--gripper', 'forward', '--distances', '30', '40',
                                       '--output', str(output)] + self.yaml_argument(),
                                      ['d 7', 'ok', 'd 7', 'ok'])
        self.assertEqual(0, code)
        self.assertTrue(output.exists())

    def test_options_of_the_other_mode_are_refused(self) -> None:
        for argv in (['corners', '--gripper', 'down'], ['--distances', '20'], ['sink', '--height', '2'],
                     ['sink', '--write'], ['sink', '--distances', '0'], ['--height', '-3'], ['--height', '50'],
                     ['--write', '--no-write'], ['nonsense']):
            with self.subTest(argv=argv):
                rig = Rig(15.0)
                with self.assertRaises(SystemExit), mock.patch('sys.stderr'):
                    calibrate_board.main(argv, make_robot=lambda: rig.robot, ask=rig.ask)
                self.assertEqual([], rig.robot.events)

    def test_no_input_stops_without_moving(self) -> None:
        rig = Rig(15.0)

        def closed(question: str) -> str:
            raise EOFError

        built: list[int] = []

        def make_robot() -> FakeRobot:
            built.append(1)
            return rig.robot

        code = calibrate_board.main(['corners'], make_robot=make_robot, ask=closed, say=rig.out.append)
        self.assertEqual(1, code)
        self.assertEqual([], built)
        self.assertIn('use --yes', rig.text())

    def test_a_broken_yaml_stops_before_anything(self) -> None:
        path = self.write_yaml('board: 3\n')
        rig = Rig(15.0)
        code = calibrate_board.main(['--yes', '--config', str(path)], make_robot=lambda: rig.robot, say=rig.out.append)
        self.assertEqual(1, code)
        self.assertEqual([], rig.robot.events)

    def test_ctrl_c_through_main_returns_one(self) -> None:
        rig, code, _ = self.make_main(['--yes', '--no-write'] + self.yaml_argument(), [KeyboardInterrupt()])
        self.assertEqual(1, code)
        self.assertEqual(1, len(rig.lay_downs))


class FrozenConfigTests(unittest.TestCase):

    def test_the_tests_use_the_frozen_board(self) -> None:
        board = config.board
        self.assertEqual((-12.0, 12.0, 0.0, 5, 3, 3.0), (board.x, board.y, board.rotation, board.squares_x,
                                                         board.squares_y, board.square_length))
        self.assertEqual({'top-right': TOP_RIGHT, 'top-left': TOP_LEFT}, {
            name: (round(x, 9), round(y, 9)) for name, (x, y) in calibrate_board.corner_robot_points(board).items()})


if __name__ == '__main__':
    unittest.main()
