import json
import math
import unittest
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np

from src.block_stacking import block_stacker
from src.block_stacking.block_stacker import BlockStacker, TOWER_TOLERANCE, LOWER_ARM_COMMAND, CLICK_PROMPT, \
    CLICK_HEALTH_INTERVAL
from tests.block_stacking_tests.frozen_config import load_frozen_config
from src.block_stacking.stacking_plan import plan_stack, park_pose, hover_pose, pick_pose, lift_pose, \
    above_tower_pose, place_pose, min_tip_height, travel_height, plan_one, BlockStep
from src.camera.block_detection import DetectedBlock, ScanError
from src.global_constants import dynamixel_robot_config
from src.kinematics.kinematics import inverse_kinematics, forward_position_kinematics
from src.robot_controllers.abstract_robot_controller import AbstractRobotController
from src.robot_controllers.dynamixel_robot.calibration.record_static_load import REST_POSE
from src.utils.movement import PlannedMove
from src.utils.movement_exception import MovementException

config = load_frozen_config()
LOGGER_NAME = 'src.block_stacking.block_stacker'
# the angles of the arm lying on the table, measured, in degrees for joints 1 to 6
MEASURED_REST_DEGREES = [90.5, 9.7, 30.1, -0.3, 58.1, -5.2]
ANGLE_LIMITS = {1: (0, math.pi), 2: (0, math.pi), 3: (-math.pi / 2, math.pi / 2), 4: (-math.pi, math.pi),
                5: (-math.pi, math.pi), 6: (-math.pi, math.pi)}
POSITIONS_PER_STATE = 9.5
# the blocks of the full run: nearest to the tower (22, 18) first is 1, then 3, then 2
THREE_BLOCKS = {1: (10.0, 25.0, 0.0), 2: (-5.0, 30.0, 0.0), 3: (15.0, 32.0, 0.0)}
EXPECTED_ORDER = [1, 3, 2]


class FakeRobot(AbstractRobotController):
    """a robot with real kinematics, a gripper that stops at a block when one is scripted, and counters"""

    def __init__(self) -> None:
        self.robot_config = dynamixel_robot_config
        self.rest_angles = inverse_kinematics(REST_POSE, self.robot_config)
        self.angles: np.ndarray | None = np.radians([0.0] + MEASURED_REST_DEGREES)
        self.events: list[str] = []
        self.enable_result: bool | None = True
        self.enable_calls = 0
        self.disable_calls = 0
        self.enabled = False
        self.hardware_errors: dict[int, list[str]] | None = {}
        self.overheating: dict[int, int] | None = {}
        self.refuse_rest = False
        # the gripper: 0 open to 100 closed
        self.positions_per_state: float | None = POSITIONS_PER_STATE
        self.gripper_state = 0.0
        self.gripper_commands: list[float] = []
        # where each next closing stops (a block is in the gripper), None: it closes to its target on nothing
        self.closing_stops: deque[float | None] = deque()
        self.holding_stop: float | None = None
        self.gripper_read_failures = 0
        self.gripper_read_count = 0
        self.gripper_opens = True
        self.state_at_disable: float | None = None
        self.angles_at_disable: np.ndarray | None = None

    def lose_block(self) -> None:
        """the block falls out: the gripper closes all the way"""
        self.gripper_state = 100.0
        self.holding_stop = None

    def check_workspace(self, angles: np.ndarray) -> None:
        if self.refuse_rest and np.allclose(angles[1:], self.rest_angles[1:], atol=1e-6):
            raise MovementException('the fake refuses the rest pose')
        super().check_workspace(angles)

    def enable_servos(self) -> bool | None:
        self.enable_calls += 1
        self.events.append('enable')
        if self.enable_result is not False:
            self.enabled = True
        return self.enable_result

    def disable_servos(self) -> bool | None:
        self.disable_calls += 1
        self.events.append('disable')
        self.enabled = False
        self.state_at_disable = self.gripper_state
        self.angles_at_disable = None if self.angles is None else self.angles.copy()
        return True

    def get_current_angles(self) -> np.ndarray | None:
        return None if self.angles is None else self.angles.copy()

    def pose_to_angles(self, pose: Any) -> np.ndarray:
        return inverse_kinematics(pose, self.robot_config)

    def get_angle_limits(self) -> dict[int, tuple[float, float]] | None:
        return ANGLE_LIMITS

    def get_hardware_errors(self) -> dict[int, list[str]] | None:
        return self.hardware_errors

    def get_overheating_servos(self) -> dict[int, int] | None:
        return self.overheating

    def gripper_positions_per_state(self) -> float | None:
        return self.positions_per_state

    def set_gripper(self, new_gripper_state: float) -> bool | None:
        self.gripper_commands.append(new_gripper_state)
        self.events.append('gripper:{:.2f}'.format(new_gripper_state))
        if not self.gripper_opens and new_gripper_state < self.gripper_state:
            return True
        if new_gripper_state > self.gripper_state:
            if self.holding_stop is None and self.gripper_state < 1:
                stop = self.closing_stops.popleft() if self.closing_stops else None
                self.holding_stop = stop
            if self.holding_stop is not None:
                self.gripper_state = min(new_gripper_state, self.holding_stop)
            else:
                self.gripper_state = new_gripper_state
        else:
            self.gripper_state = new_gripper_state
            if self.holding_stop is not None and new_gripper_state < self.holding_stop - 1:
                self.holding_stop = None
        return True

    def get_gripper_state(self) -> float | None:
        self.gripper_read_count += 1
        if self.gripper_read_failures > 0:
            self.gripper_read_failures -= 1
            return None
        return self.gripper_state

    def move_to_pose(self, pose: Any) -> float | None:
        raise AssertionError('the stacker may only execute planned moves')

    def move_to_pose_and_give_new_angles(self, pose: Any) -> np.ndarray:
        raise AssertionError('the stacker may only execute planned moves')

    def move_servos(self, angles: np.ndarray, velocities: np.ndarray | None = None) -> float | None:
        raise AssertionError('the stacker may only execute planned moves')

    def reset_to_pose(self, pose: Any) -> None:
        raise AssertionError('the stacker may only execute planned moves')


class Record:
    """a move the stacker executed"""

    def __init__(self, move: PlannedMove, stoppable: bool) -> None:
        self.move = move
        self.start = move.angles[0].copy()
        self.end = move.angles[0].copy()
        self.stoppable = stoppable
        self.completed = False


class FakeExecutor:
    """executes at once: the arm of the fake robot goes through the middle of the move to its end, when not stopped"""

    def __init__(self, robot: FakeRobot) -> None:
        self.robot = robot
        self.records: list[Record] = []
        self.before: Callable[[int, Record], None] | None = None
        self.midway: Callable[[int, Record], None] | None = None
        self.after: Callable[[int, Record], None] | None = None

    def __call__(self, robot: FakeRobot, move: PlannedMove, should_stop: Callable[[], bool] | None = None) -> bool:
        record = Record(move, should_stop is not None)
        index = len(self.records)
        self.records.append(record)
        self.robot.events.append('move:{}'.format(index))
        if self.before is not None:
            self.before(index, record)
        if should_stop is not None and should_stop():
            return False
        self.robot.angles = move.angles[len(move.angles) // 2].copy()
        record.end = self.robot.angles.copy()
        if self.midway is not None:
            self.midway(index, record)
        if should_stop is not None and should_stop():
            return False
        self.robot.angles = move.angles[-1].copy()
        record.end = self.robot.angles.copy()
        record.completed = True
        if self.after is not None:
            self.after(index, record)
        return True


class FakeTime:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []
        self.on_sleep: Callable[[], None] | None = None  # called after every sleep, where a test delivers a click

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        if self.on_sleep is not None:
            self.on_sleep()

    def clock(self) -> float:
        return self.now


def detected(marker_id: int, x: float, y: float, yaw: float = 0.0, stable: bool = True,
             height: float = 3.0) -> DetectedBlock:
    return DetectedBlock(marker_id=marker_id, x=x, y=y, yaw=yaw, height=height, frames=10,
                         spread=0.1 if stable else 2.0, side_length=2.4, stable=stable)


class FakeHandler:
    """
    the camera over a table with blocks: a block whose id is in the heights is in the tower, at its centre (shifted by
    tower_offset) and the others lie where the table says
    """

    def __init__(self, table: dict[int, tuple[float, float, float]]) -> None:
        self.table = dict(table)
        self.scan_heights: list[dict[int, float]] = []
        self.unstable: set[int] = set()
        self.hidden_in_tower: set[int] = set()
        self.tower_offset = (0.0, 0.0)
        self.board = True
        self.errors: dict[int, Exception] = {}  # scan index -> what that scan raises
        self.on_scan: Callable[[int], None] | None = None
        # what the stacker did to the window, and what it shows now
        self.window_calls: list[tuple[str, Any]] = []
        self.selected: int | None = None
        self.message: str | None = None

    def set_selected(self, marker_id: int | None) -> None:
        self.window_calls.append(('selected', marker_id))
        self.selected = marker_id

    def set_message(self, text: str | None) -> None:
        self.window_calls.append(('message', text))
        self.message = text

    def blocks(self, heights: dict[int, float] | None) -> list[DetectedBlock]:
        heights = heights or {}
        result = []
        for marker_id, (x, y, yaw) in sorted(self.table.items()):
            if marker_id in heights:
                if marker_id in self.hidden_in_tower:
                    continue
                result.append(detected(marker_id, config.tower.x + self.tower_offset[0],
                                       config.tower.y + self.tower_offset[1], height=heights[marker_id]))
            else:
                result.append(detected(marker_id, x, y, yaw, stable=marker_id not in self.unstable))
        return result

    def scan(self, heights: dict[int, float] | None = None, sleep: Any = None) -> list[DetectedBlock]:
        index = len(self.scan_heights)
        self.scan_heights.append(dict(heights or {}))
        if self.on_scan is not None:
            self.on_scan(index)
        if index in self.errors:
            raise self.errors[index]
        return self.blocks(heights)

    def get_blocks(self, heights: dict[int, float] | None = None) -> list[DetectedBlock]:
        return self.blocks(heights)

    def board_visible(self, max_age: float = 1.0) -> bool:
        return self.board


class Setup:
    def __init__(self, table: dict[int, tuple[float, float, float]]) -> None:
        self.robot = FakeRobot()
        self.handler = FakeHandler(table)
        self.time = FakeTime()
        self.executor = FakeExecutor(self.robot)
        self.provider_calls = 0
        self.provider_errors: deque[Exception] = deque()  # raised by the next calls of the provider
        self.main_alive = True
        self.stacker = BlockStacker(self.provide, self.handler, config, sleep=self.time.sleep,  # type: ignore[arg-type]
                                    clock=self.time.clock, execute=self.executor,
                                    main_thread_alive=lambda: self.main_alive)

    def provide(self) -> FakeRobot:
        self.provider_calls += 1
        if self.provider_errors:
            raise self.provider_errors.popleft()
        return self.robot

    def run(self, hover_only: bool = False) -> dict[str, Any]:
        assert self.stacker.start(hover_only)
        self.stacker.wait(60)
        assert not self.stacker.is_running(), 'the run did not end'
        return self.stacker.status()

    def run_click_mode(self) -> dict[str, Any]:
        assert self.stacker.start_click_mode()
        self.stacker.wait(60)
        assert not self.stacker.is_running(), 'the click mode did not end'
        return self.stacker.status()

    @property
    def records(self) -> list[Record]:
        return self.executor.records


def assert_ends_at(test: unittest.TestCase, robot: FakeRobot, record: Record, pose: Any) -> None:
    expected = robot.pose_to_angles(pose)
    test.assertTrue(np.allclose(record.end[1:], expected[1:], atol=1e-5),
                    'the move ends at {} and not at {}'.format(np.round(record.end[1:], 4), np.round(expected[1:], 4)))


class StackerTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.setups: list[Setup] = []

    def tearDown(self) -> None:
        for setup in self.setups:
            setup.stacker.stop()
            setup.stacker.wait(10)

    def setup_for(self, table: dict[int, tuple[float, float, float]]) -> Setup:
        setup = Setup(table)
        self.setups.append(setup)
        return setup

    def assert_left_holding(self, setup: Setup, why: str) -> None:
        """nothing was moved at the end: only the park move was made, the torque is on, the user is told to support"""
        self.assertEqual(1, len(setup.records))
        self.assertEqual(0, setup.robot.disable_calls)
        self.assertTrue(setup.robot.enabled)
        error = setup.stacker.status()['error']
        self.assertIn(why, error)
        self.assertIn('Support the arm' if 'limp' in why else 'support it', error)
        self.assertIn('others still hold', error)
        self.assertNotIn(LOWER_ARM_COMMAND, error)
        self.assertEqual('idle', setup.stacker.status()['state'])
        self.assertIn('not laid down', setup.stacker.status()['activity'])

    def assert_laid_down(self, setup: Setup, hover_only: bool = False, gripper_opens: bool = True) -> None:
        """the last move went to the rest pose and was not stoppable, the gripper opened, then the torque went off"""
        last = setup.records[-1]
        self.assertFalse(last.stoppable)
        self.assertTrue(last.completed)
        self.assertTrue(np.allclose(last.end[1:], setup.robot.rest_angles[1:], atol=1e-6))
        self.assertEqual(1, setup.robot.disable_calls)
        self.assertFalse(setup.robot.enabled)
        events = setup.robot.events
        self.assertEqual('disable', events[-1])
        if not hover_only:
            self.assertEqual('gripper:0.00', events[-2])
            self.assertEqual('move:{}'.format(len(setup.records) - 1), events[-3])
            if gripper_opens:
                self.assertEqual(0.0, setup.robot.state_at_disable)
        self.assertTrue(np.allclose(setup.robot.angles_at_disable[1:], setup.robot.rest_angles[1:], atol=1e-6))


class FullRunTests(StackerTestCase):

    def test_the_arm_of_the_tests_starts_low_in_front_of_the_robot_like_the_rest_pose(self) -> None:
        # the measured angles are only the starting point of the fake arm: by the kinematics their tip is at
        # y=48 with the 15cm gripper (y=47 with the 14cm one), 8cm beyond the rest pose, which does not matter for the
        # moves that start there
        robot = FakeRobot()
        for angles in (robot.angles, robot.rest_angles):
            x, y, z = forward_position_kinematics(angles, robot.robot_config)[-1]
            self.assertLess(abs(x), 1.0)
            self.assertTrue(38 < y < 49)
            self.assertTrue(3 < z < 6)

    def test_three_blocks_are_stacked_nearest_first(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.robot.closing_stops.extend([50, 50, 50])
        robot, handler = setup.robot, setup.handler

        status = setup.run()

        self.assertEqual('idle', status['state'])
        self.assertEqual('done: 3 blocks stacked', status['activity'])
        self.assertIsNone(status['error'])
        self.assertEqual(EXPECTED_ORDER, status['tower'])
        self.assertEqual([], status['plan'])
        self.assertEqual({}, status['skipped'])
        # a scan before every block and one at the end, with the heights of the blocks in the tower
        size = config.blocks.size
        self.assertEqual([{}, {1: size}, {1: size, 3: 2 * size}, {1: size, 3: 2 * size, 2: 3 * size}],
                         handler.scan_heights)

        # the moves: to the park pose, seven per block, and the arm laid down
        records = setup.records
        self.assertEqual(1 + 3 * 7 + 1, len(records))
        assert_ends_at(self, robot, records[0], park_pose(config, True))
        placed: list[int] = []
        for round_index in range(3):
            heights = {marker_id: (level + 1) * size for level, marker_id in enumerate(placed)}
            usable = [block for block in handler.blocks(heights) if block.marker_id not in placed]
            step = plan_stack(usable, config, robot, start_level=len(placed)).steps[0]
            self.assertEqual(EXPECTED_ORDER[round_index], step.marker_id)
            level = step.level
            self.assertEqual(round_index, level)
            poses = [hover_pose(step.pick, config), pick_pose(step.pick, config), lift_pose(step.pick, config, level),
                     above_tower_pose(step.place, config, level), place_pose(step.place, config, level),
                     above_tower_pose(step.place, config, level), park_pose(config, True)]
            for offset, pose in enumerate(poses):
                assert_ends_at(self, robot, records[1 + 7 * round_index + offset], pose)
            # the tower is where the config says
            self.assertAlmostEqual(config.tower.x, poses[4].x)
            self.assertAlmostEqual(config.tower.y, poses[4].y)
            placed.append(step.marker_id)

        # the gripper: open, then close, pull back, open for every block, and open again at the rest pose
        pull_back = 50 + config.gripper.max_lead_positions / POSITIONS_PER_STATE
        expected_commands = [0.0] + [100.0, pull_back, 0.0] * 3 + [0.0]
        self.assertEqual(expected_commands, robot.gripper_commands)
        self.assert_laid_down(setup)
        self.assertEqual(1, robot.enable_calls)

    def test_every_move_starts_where_the_previous_one_ended(self) -> None:
        setup = self.setup_for({1: THREE_BLOCKS[1], 3: THREE_BLOCKS[3]})
        setup.robot.closing_stops.extend([50, 50])
        start = setup.robot.get_current_angles()

        setup.run()

        records = setup.records
        self.assertTrue(len(records) > 10)
        self.assertTrue(np.allclose(records[0].start, start))
        for previous, record in zip(records, records[1:]):
            self.assertTrue(previous.completed)
            self.assertTrue(np.allclose(previous.end, record.start, atol=1e-9),
                            'a move starts away from where the previous one ended')

    def test_every_move_is_stoppable_except_the_rest_move(self) -> None:
        setup = self.setup_for({1: THREE_BLOCKS[1]})
        setup.robot.closing_stops.append(50)
        setup.run()
        self.assertTrue(all(record.stoppable for record in setup.records[:-1]))
        self.assertFalse(setup.records[-1].stoppable)

    def test_the_arm_waits_at_the_park_pose_and_looks_there(self) -> None:
        setup = self.setup_for({1: THREE_BLOCKS[1]})
        setup.robot.closing_stops.append(50)
        positions_at_scan: list[np.ndarray] = []
        original_scan = setup.handler.scan

        def scan(heights: dict[int, float] | None = None, sleep: Any = None) -> list[DetectedBlock]:
            positions_at_scan.append(setup.robot.get_current_angles())
            return original_scan(heights, sleep)

        setup.handler.scan = scan  # type: ignore[method-assign]
        setup.run()
        park = setup.robot.pose_to_angles(park_pose(config, True))
        self.assertEqual(2, len(positions_at_scan))
        for angles in positions_at_scan:
            self.assertTrue(np.allclose(angles[1:], park[1:], atol=1e-5))

    def test_an_unstable_block_is_skipped_with_its_reason(self) -> None:
        setup = self.setup_for({1: THREE_BLOCKS[1], 2: THREE_BLOCKS[2]})
        setup.handler.unstable = {2}
        setup.robot.closing_stops.append(50)
        status = setup.run()
        self.assertEqual([1], status['tower'])
        self.assertEqual({'2': 'its position is not steady in the picture'}, status['skipped'])
        self.assertEqual('done: 1 block stacked', status['activity'])

    def test_a_flip_of_the_plan_moves_to_the_other_park_pose(self) -> None:
        setup = self.setup_for({1: THREE_BLOCKS[1]})
        setup.robot.closing_stops.append(50)
        original = block_stacker.plan_stack

        def other_flip(*args: Any, **kwargs: Any) -> Any:
            plan = original(*args, **kwargs)
            return plan if plan.flip is False else _with_flip(plan, False)

        def _with_flip(plan: Any, flip: bool) -> Any:
            return type(plan)(plan.steps, plan.skipped, flip)

        # the steps stay planned for flip True, which is all this test needs: only the park pose that is moved to counts
        block_stacker.plan_stack = other_flip
        try:
            status = setup.run()
        finally:
            block_stacker.plan_stack = original
        # the arm starts at the park pose of flip True and moves to the one of flip False before the first block
        self.assertIsNone(status['error'], status['error'])
        assert_ends_at(self, setup.robot, setup.records[0], park_pose(config, True))
        assert_ends_at(self, setup.robot, setup.records[1], park_pose(config, False))


class GripperTests(StackerTestCase):

    def test_it_closes_on_a_block_and_pulls_the_target_back(self) -> None:
        setup = self.setup_for({1: THREE_BLOCKS[1]})
        setup.robot.closing_stops.append(50)
        setup.run()
        commands = setup.robot.gripper_commands
        self.assertEqual([0.0, 100.0, 50 + 85 / POSITIONS_PER_STATE, 0.0, 0.0], commands)
        self.assertAlmostEqual(50 + 85 / POSITIONS_PER_STATE, commands[2])

    def test_the_target_is_never_beyond_the_closed_state(self) -> None:
        setup = self.setup_for({1: THREE_BLOCKS[1]})
        setup.robot.closing_stops.append(94)  # held, but close to empty_state: the lead would pass 100
        setup.run()
        self.assertEqual(100.0, setup.robot.gripper_commands[2])

    def test_it_waits_for_the_gripper_to_stop_moving(self) -> None:
        setup = self.setup_for({1: THREE_BLOCKS[1]})
        robot = setup.robot
        robot.closing_stops.append(50)
        reads = [0.0]

        # the gripper closes slowly: 10 states per read until it reaches the block
        original = robot.get_gripper_state

        def slow_state() -> float | None:
            if robot.gripper_commands and robot.gripper_commands[-1] == 100.0 and reads[0] < 50:
                reads[0] += 10
                return reads[0]
            return original()

        robot.get_gripper_state = slow_state  # type: ignore[method-assign]
        setup.run()
        self.assertEqual(4, len(robot.gripper_commands) - 1)
        self.assertAlmostEqual(50 + 85 / POSITIONS_PER_STATE, robot.gripper_commands[2])

    def test_a_gripper_that_closes_on_nothing_is_tried_twice_and_then_given_up(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        # block 1 is closed on nothing twice, the others are fine
        setup.robot.closing_stops.extend([None, None, 50, 50])
        robot = setup.robot

        status = setup.run()

        self.assertIsNone(status['error'])
        self.assertEqual([3, 2], status['tower'])
        self.assertEqual({'1': 'the gripper closed on nothing twice'}, status['skipped'])
        self.assertEqual('done: 2 blocks stacked', status['activity'])
        pull_back = 50 + config.gripper.max_lead_positions / POSITIONS_PER_STATE
        self.assertEqual([0.0, 100.0, 0.0, 100.0, 0.0, 100.0, pull_back, 0.0, 100.0, pull_back, 0.0, 0.0],
                         robot.gripper_commands)
        # after the first failed close: up to the hover pose and back to the park pose, then the next round
        hover = hover_pose(plan_first_step(setup, {}).pick, config)
        records = setup.records
        assert_ends_at(self, robot, records[1], hover)
        assert_ends_at(self, robot, records[2], pick_pose(plan_first_step(setup, {}).pick, config))
        assert_ends_at(self, robot, records[3], hover)
        assert_ends_at(self, robot, records[4], park_pose(config, True))
        self.assert_laid_down(setup)

    def test_the_block_is_tried_again_after_it_failed_once(self) -> None:
        setup = self.setup_for({1: THREE_BLOCKS[1]})
        setup.robot.closing_stops.extend([None, 50])
        status = setup.run()
        self.assertEqual([1], status['tower'])
        self.assertEqual({}, status['skipped'])

    def test_a_block_that_is_lost_during_the_lift_is_let_go_and_tried_again(self) -> None:
        setup = self.setup_for({1: THREE_BLOCKS[1]})
        robot = setup.robot
        robot.closing_stops.extend([50, 50])
        # moves: 0 park, 1 hover, 2 pick, 3 lift: the block falls out when the lift is done
        setup.executor.after = lambda index, record: robot.lose_block() if index == 3 else None

        status = setup.run()

        self.assertIsNone(status['error'])
        self.assertEqual([1], status['tower'])
        pull_back = 50 + config.gripper.max_lead_positions / POSITIONS_PER_STATE
        self.assertEqual([0.0, 100.0, pull_back, 0.0, 100.0, pull_back, 0.0, 0.0], robot.gripper_commands)
        # the arm went to the park pose after the lost block, without carrying anything to the tower
        assert_ends_at(self, robot, setup.records[4], park_pose(config, True))

    def test_a_block_that_is_lost_twice_is_given_up(self) -> None:
        setup = self.setup_for({1: THREE_BLOCKS[1]})
        robot = setup.robot
        robot.closing_stops.extend([50, 50])
        lifts = [3, 7]  # first try: 0 park, 1 hover, 2 pick, 3 lift, 4 park; second try: 5 hover, 6 pick, 7 lift
        setup.executor.after = lambda index, record: robot.lose_block() if index in lifts else None
        status = setup.run()
        self.assertEqual([], status['tower'])
        self.assertEqual({'1': 'the block slipped out of the gripper while it was lifted twice'}, status['skipped'])
        self.assertIsNone(status['error'])

    def test_a_robot_without_a_gripper_servo_counts_as_holding(self) -> None:
        setup = self.setup_for({1: THREE_BLOCKS[1], 3: THREE_BLOCKS[3]})
        robot = setup.robot
        robot.positions_per_state = None
        status = setup.run()
        self.assertEqual([1, 3], status['tower'])
        self.assertEqual(0, robot.gripper_read_count)
        # no pull back without a servo
        self.assertEqual([0.0, 100.0, 0.0, 100.0, 0.0, 0.0], robot.gripper_commands)
        self.assertIn(config.gripper.settle_time, setup.time.sleeps)

    def test_a_failed_read_is_read_again(self) -> None:
        setup = self.setup_for({1: THREE_BLOCKS[1]})
        setup.robot.closing_stops.append(50)
        setup.robot.gripper_read_failures = 3
        status = setup.run()
        self.assertEqual([1], status['tower'])
        self.assertIsNone(status['error'])

    def test_a_gripper_that_can_not_be_read_counts_as_holding_nothing(self) -> None:
        setup = self.setup_for({1: THREE_BLOCKS[1]})
        robot = setup.robot
        robot.closing_stops.extend([50, 50])
        reads_fail = {'on': False}
        original_set = robot.set_gripper

        def set_gripper(state: float) -> bool | None:
            result = original_set(state)
            reads_fail['on'] = state == 100.0
            return result

        original_get = robot.get_gripper_state

        def get_gripper_state() -> float | None:
            return None if reads_fail['on'] else original_get()

        robot.set_gripper = set_gripper  # type: ignore[method-assign]
        robot.get_gripper_state = get_gripper_state  # type: ignore[method-assign]
        status = setup.run()
        self.assertEqual([], status['tower'])
        self.assertEqual({'1': 'the gripper closed on nothing twice'}, status['skipped'])
        # it waited the whole timeout for every try
        self.assertGreaterEqual(setup.time.now, 2 * config.gripper.timeout)

    def test_a_gripper_that_does_not_open_over_the_tower_ends_the_run(self) -> None:
        setup = self.setup_for({1: THREE_BLOCKS[1], 3: THREE_BLOCKS[3]})
        robot = setup.robot
        robot.closing_stops.extend([50, 50])
        # moves: 0 park, 1 to 7 the first block, 8 hover, 9 pick, 10 lift, 11 carry, 12 place: the gripper is stuck from
        # the place of the second block on
        setup.executor.after = lambda index, record: setattr(robot, 'gripper_opens', False) if index == 12 else None
        status = setup.run()
        self.assertIn('did not open', status['error'])
        self.assertEqual([1], status['tower'])
        self.assertEqual(13 + 1, len(setup.records))  # the rest pose from level 1 is one direct move
        self.assert_laid_down(setup, gripper_opens=False)

    def test_a_gripper_that_does_not_open_at_level_0_is_laid_down_from_the_table(self) -> None:
        # the place pose of level 0 is 2.3cm high: the direct move to the rest pose dips below the lowest height
        setup = self.setup_for({1: THREE_BLOCKS[1]})
        robot = setup.robot
        robot.closing_stops.append(50)
        setup.executor.after = lambda index, record: setattr(robot, 'gripper_opens', False) if index == 5 else None
        status = setup.run()
        self.assertIn('did not open', status['error'])
        self.assertNotIn(LOWER_ARM_COMMAND, status['error'])
        self.assertEqual([], status['tower'])
        self.assert_laid_down(setup, gripper_opens=False)
        self.assertTrue(all(not record.stoppable for record in setup.records[6:]))

    def test_a_gripper_that_does_not_close_ends_the_run(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        # it stays within 5 states of the open state: not a block, and not nothing either
        setup.robot.closing_stops.append(4.0)
        status = setup.run()
        self.assertEqual('the gripper did not close', status['error'])
        self.assertEqual([], status['tower'])
        self.assertEqual([0.0, 100.0, 0.0], setup.robot.gripper_commands[:3])
        self.assert_laid_down(setup)

    def test_a_gripper_that_closes_just_beyond_5_states_holds_a_block(self) -> None:
        setup = self.setup_for({1: THREE_BLOCKS[1]})
        setup.robot.closing_stops.append(6.0)
        status = setup.run()
        self.assertIsNone(status['error'])
        self.assertEqual([1], status['tower'])


def plan_first_step(setup: Setup, heights: dict[int, float]) -> Any:
    blocks = [block for block in setup.handler.blocks(heights) if block.marker_id not in heights]
    return plan_stack(blocks, config, setup.robot).steps[0]


class FailureTests(StackerTestCase):

    def test_servos_that_refuse_to_enable_end_the_run_without_a_move(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.robot.enable_result = False
        status = setup.run()
        self.assertIn('could not be switched on', status['error'])
        self.assertEqual([], setup.records)
        self.assertEqual(0, setup.robot.disable_calls)
        self.assertEqual([], setup.robot.gripper_commands)
        self.assertEqual('idle', status['state'])
        self.assertEqual([], setup.handler.scan_heights)

    def test_servos_that_return_none_from_enable_are_fine(self) -> None:
        setup = self.setup_for({1: THREE_BLOCKS[1]})
        setup.robot.enable_result = None
        setup.robot.closing_stops.append(50)
        status = setup.run()
        self.assertIsNone(status['error'])
        self.assertEqual([1], status['tower'])

    def test_a_scan_error_at_the_start_lays_the_arm_down(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.handler.errors[0] = ScanError('the board is not in the picture')
        status = setup.run()
        self.assertEqual('the board is not in the picture', status['error'])
        self.assertEqual([], status['tower'])
        self.assertEqual(2, len(setup.records))
        self.assert_laid_down(setup)

    def test_a_scan_error_in_the_middle_of_the_run_lays_the_arm_down(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.robot.closing_stops.extend([50, 50, 50])
        setup.handler.errors[1] = ScanError('no picture')
        status = setup.run()
        self.assertEqual('no picture', status['error'])
        self.assertEqual([1], status['tower'])
        self.assert_laid_down(setup)

    def test_the_tower_top_missing_ends_the_run(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.robot.closing_stops.extend([50, 50, 50])
        setup.handler.hidden_in_tower = {1}
        status = setup.run()
        self.assertEqual('the tower is not standing where it should: block 1 is not on top', status['error'])
        self.assertEqual([1], status['tower'])
        self.assert_laid_down(setup)

    def test_the_tower_top_off_its_place_ends_the_run(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.robot.closing_stops.extend([50, 50, 50])
        setup.handler.tower_offset = (TOWER_TOLERANCE + 0.5, 0.0)
        status = setup.run()
        self.assertIn('block 1 is not on top', status['error'])
        self.assert_laid_down(setup)

    def test_the_tower_top_within_the_tolerance_is_fine(self) -> None:
        setup = self.setup_for({1: THREE_BLOCKS[1]})
        setup.robot.closing_stops.append(50)
        setup.handler.tower_offset = (TOWER_TOLERANCE - 0.2, 0.0)
        status = setup.run()
        self.assertIsNone(status['error'])
        self.assertEqual([1], status['tower'])

    def test_a_hardware_error_ends_the_run_and_names_the_servo(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.robot.hardware_errors = {2: ['Overload Error']}
        status = setup.run()
        self.assertIn('servos 2 have a hardware error', status['error'])
        self.assertIn('Overload Error', status['error'])
        self.assertEqual([], setup.handler.scan_heights)
        self.assert_left_holding(setup, 'servos 2 have a hardware error and are limp')

    def test_an_unreadable_hardware_status_ends_the_run(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.robot.hardware_errors = None
        status = setup.run()
        self.assertIn('could not be read', status['error'])
        self.assert_left_holding(setup, 'hardware status of the servos could not be read')

    def test_a_hardware_error_that_comes_during_the_run_leaves_the_arm_where_it_is(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.robot.closing_stops.extend([50, 50, 50])
        setup.executor.after = lambda index, record: (
            setattr(setup.robot, 'hardware_errors', {5: ['Overheating Error']}) if index == 3 else None)
        status = setup.run()
        # the health check of the next round finds it, nothing moves after that
        self.assertIn('servos 5 have a hardware error', status['error'])
        self.assertIn('limp', status['error'])
        self.assertEqual(0, setup.robot.disable_calls)
        self.assertTrue(setup.robot.enabled)
        self.assertEqual(8, len(setup.records))

    def test_an_overheating_servo_ends_the_run_and_names_the_servo(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.robot.overheating = {3: 81}
        status = setup.run()
        self.assertIn('servos 3 are too warm', status['error'])
        self.assertIn('81', status['error'])
        self.assert_laid_down(setup)

    def test_overheating_that_can_not_be_read_ends_the_run(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.robot.overheating = None
        status = setup.run()
        self.assertIn('temperatures', status['error'])
        self.assert_laid_down(setup)

    def test_a_hardware_error_in_the_middle_of_the_run_ends_it_after_that_block(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.robot.closing_stops.extend([50, 50, 50])
        setup.executor.after = lambda index, record: (
            setattr(setup.robot, 'overheating', {4: 75}) if index == 7 else None)
        status = setup.run()
        self.assertIn('servos 4 are too warm', status['error'])
        self.assertEqual([1], status['tower'])
        self.assert_laid_down(setup)

    def test_an_unexpected_exception_sets_the_error_and_lays_the_arm_down(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.robot.closing_stops.extend([50, 50, 50])
        setup.handler.errors[1] = RuntimeError('boom')
        with self.assertLogs(LOGGER_NAME, level='ERROR') as logs:
            status = setup.run()
        self.assertIn('boom', status['error'])
        self.assertIn('RuntimeError', status['error'])
        self.assertTrue(any('Traceback' in line for line in logs.output))
        self.assertEqual('idle', status['state'])
        self.assert_laid_down(setup)

    def test_the_rest_move_that_can_not_be_planned_leaves_the_servos_holding(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.handler.errors[0] = ScanError('no picture')
        setup.robot.refuse_rest = True
        status = setup.run()
        self.assertIn('no picture', status['error'])
        self.assertIn(LOWER_ARM_COMMAND, status['error'])
        self.assertEqual(0, setup.robot.disable_calls)
        self.assertTrue(setup.robot.enabled)
        self.assertEqual('idle', status['state'])
        self.assertIn('not laid down', status['activity'])
        # no gripper command after the park move: the gripper is only opened at the rest pose
        self.assertEqual([0.0], setup.robot.gripper_commands)

    def test_angles_that_can_not_be_read_for_the_rest_move_leave_the_servos_holding(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        robot = setup.robot

        def scan_error(index: int) -> None:
            robot.angles = None

        setup.handler.on_scan = scan_error
        setup.handler.errors[0] = ScanError('no picture')
        status = setup.run()
        self.assertIn(LOWER_ARM_COMMAND, status['error'])
        self.assertEqual(0, robot.disable_calls)
        self.assertTrue(robot.enabled)

    def test_a_block_that_lies_on_the_tower_ends_the_run_at_the_first_plan(self) -> None:
        setup = self.setup_for({1: (config.tower.x + 1, config.tower.y, 0.0)})
        status = setup.run()
        self.assertIn('lies where the tower goes', status['error'])
        self.assert_laid_down(setup)

    def test_a_block_that_can_not_be_reached_when_taking_it_is_a_failed_try(self) -> None:
        setup = self.setup_for({1: THREE_BLOCKS[1]})
        setup.robot.closing_stops.extend([50])
        from src.block_stacking import block_stacker as module
        original = module.plan_straight_line
        calls = {'count': 0}

        def failing_line(*args: Any, **kwargs: Any) -> PlannedMove:
            calls['count'] += 1
            if calls['count'] <= 2:  # the line down to the block, twice
                raise MovementException('the block moved out of reach')
            return original(*args, **kwargs)

        module.plan_straight_line = failing_line
        try:
            status = setup.run()
        finally:
            module.plan_straight_line = original
        self.assertEqual([], status['tower'])
        self.assertIn('its position can not be reached: the block moved out of reach twice', status['skipped']['1'])
        self.assertEqual([0.0, 0.0], setup.robot.gripper_commands)
        self.assertIsNone(status['error'])

    def test_a_move_that_fails_with_a_block_in_the_gripper_ends_the_run(self) -> None:
        setup = self.setup_for({1: THREE_BLOCKS[1]})
        setup.robot.closing_stops.extend([50])
        from src.block_stacking import block_stacker as module
        original = module.plan_straight_line
        calls = {'count': 0}

        def failing_line(*args: Any, **kwargs: Any) -> PlannedMove:
            calls['count'] += 1
            if calls['count'] == 3:  # the line down onto the tower, with the block in the gripper
                raise MovementException('out of reach')
            return original(*args, **kwargs)

        module.plan_straight_line = failing_line
        try:
            status = setup.run()
        finally:
            module.plan_straight_line = original
        self.assertIn('block 1 can not be carried to the tower: out of reach', status['error'])
        self.assert_laid_down(setup)

    def test_a_carry_move_that_dips_too_low_ends_the_run_before_it_is_executed(self) -> None:
        setup = self.setup_for({1: THREE_BLOCKS[1]})
        setup.robot.closing_stops.extend([50])
        from src.block_stacking import block_stacker as module
        original = module.min_tip_height
        module.min_tip_height = lambda move, robot_config: 1.0
        try:
            status = setup.run()
        finally:
            module.min_tip_height = original
        self.assertIn('could hit the tower', status['error'])
        # park, hover, pick, lift, and the rest: the carry move was not executed
        self.assertEqual(5, len(setup.records))
        self.assert_laid_down(setup)

    def test_the_carry_move_has_to_stay_close_to_the_travel_height(self) -> None:
        setup = self.setup_for({1: THREE_BLOCKS[1]})
        setup.robot.closing_stops.extend([50])
        status = setup.run()
        self.assertIsNone(status['error'])
        carry = setup.records[4].move
        self.assertGreaterEqual(min_tip_height(carry, setup.robot.robot_config), travel_height(config, 0) - 1.0)


class StopTests(StackerTestCase):

    def test_stop_during_a_move_cuts_it_short_and_lays_the_arm_down(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.robot.closing_stops.extend([50, 50, 50])
        results: list[Any] = []

        def midway(index: int, record: Record) -> None:
            if index == 2:  # the line down to the first block
                results.append(setup.stacker.stop())
                results.append(setup.stacker.status()['state'])
                results.append(setup.stacker.stop())

        setup.executor.midway = midway
        status = setup.run()

        self.assertEqual([True, 'stopping', True], results)
        self.assertIsNone(status['error'])
        self.assertEqual('stopped', status['activity'])
        self.assertEqual('idle', status['state'])
        stopped = setup.records[2]
        self.assertFalse(stopped.completed)
        self.assertTrue(stopped.stoppable)
        # then only the rest move, and that one is not stoppable and was completed
        self.assertEqual(4, len(setup.records))
        self.assert_laid_down(setup)
        # it started from where the cut short move left the arm
        self.assertTrue(np.allclose(setup.records[3].start, stopped.end, atol=1e-9))

    def test_stop_before_the_first_move(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        results: list[bool] = []
        setup.executor.before = lambda index, record: results.append(setup.stacker.stop()) if index == 0 else None
        status = setup.run()
        self.assertEqual([True], results)
        self.assertEqual('stopped', status['activity'])
        self.assertEqual(2, len(setup.records))
        self.assert_laid_down(setup)

    def test_stop_while_holding_a_block_lets_it_go_at_the_rest_pose(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        robot = setup.robot
        robot.closing_stops.extend([50, 50, 50])
        # moves: 0 park, 1 hover, 2 pick, 3 lift, 4 carry, 5 rest
        gripper_before_rest: list[float] = []
        setup.executor.midway = lambda index, record: setup.stacker.stop() if index == 4 else None
        setup.executor.before = lambda index, record: (
            gripper_before_rest.append(robot.gripper_state) if index == 5 else None)
        status = setup.run()
        self.assertEqual('stopped', status['activity'])
        self.assertEqual([], status['tower'])
        self.assertEqual([50.0], gripper_before_rest)  # the block was in the gripper until the rest pose
        self.assertEqual(0.0, robot.state_at_disable)
        events = robot.events
        self.assertEqual(['move:5', 'gripper:0.00', 'disable'], events[-3:])
        self.assertTrue(np.allclose(robot.angles_at_disable[1:], robot.rest_angles[1:], atol=1e-6))

    def test_stop_with_the_gripper_down_at_the_block_goes_up_first_and_lays_the_arm_down(self) -> None:
        # a joint move from the pick pose dips the tip under the lowest height the workspace allows, so the planner
        # refuses it: the arm goes straight up, to the park pose and then down to the rest pose
        setup = self.setup_for(THREE_BLOCKS)
        robot = setup.robot
        robot.closing_stops.extend([50, 50, 50])
        setup.executor.after = lambda index, record: setup.stacker.stop() if index == 2 else None  # at the block
        status = setup.run()
        self.assertIsNone(status['error'])
        self.assertEqual('stopped', status['activity'])
        self.assertEqual(6, len(setup.records))
        up, park, rest = setup.records[3:]
        step = plan_first_step(setup, {})
        self.assertTrue(np.allclose(up.start, setup.records[2].end))
        straight_up = forward_position_kinematics(up.end, robot.robot_config)[-1]
        at_block = forward_position_kinematics(up.start, robot.robot_config)[-1]
        self.assertAlmostEqual(config.motion.hover_height, straight_up[2], places=3)
        self.assertTrue(np.allclose(straight_up[:2], at_block[:2], atol=1e-3))
        assert_ends_at(self, robot, up, hover_pose(step.pick, config))
        assert_ends_at(self, robot, park, park_pose(config, True))
        for record in (up, park, rest):
            self.assertFalse(record.stoppable)
        self.assert_laid_down(setup)

    def test_the_way_round_that_fails_leaves_the_servos_holding_where_it_got_to(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        robot = setup.robot
        robot.closing_stops.extend([50, 50, 50])
        robot.refuse_rest = True
        setup.executor.after = lambda index, record: setup.stacker.stop() if index == 2 else None
        status = setup.run()
        self.assertIn(LOWER_ARM_COMMAND, status['error'])
        self.assertIn('the direct move was refused', status['error'])
        self.assertEqual(0, robot.disable_calls)
        self.assertTrue(robot.enabled)
        # the arm went up and to the park pose, and stopped there
        self.assertEqual(5, len(setup.records))
        self.assertTrue(all(not record.stoppable for record in setup.records[3:]))

    def test_a_move_of_the_way_round_that_does_not_finish_leaves_the_servos_holding(self) -> None:
        setup = self.setup_for({1: THREE_BLOCKS[1]})
        setup.robot.closing_stops.append(50)
        setup.executor.after = lambda index, record: setup.stacker.stop() if index == 2 else None
        original = setup.executor.__call__

        def refusing(robot: FakeRobot, move: PlannedMove, should_stop: Any = None) -> bool:
            if should_stop is None and len(setup.records) == 3:  # the way up
                setup.records.append(Record(move, False))
                return False
            return original(robot, move, should_stop)

        setup.stacker._execute = refusing
        status = setup.run()
        self.assertIn('did not finish', status['error'])
        self.assertEqual(0, setup.robot.disable_calls)

    def test_stop_returns_at_once_and_a_new_run_can_start_after_it(self) -> None:
        setup = self.setup_for({1: THREE_BLOCKS[1]})
        setup.executor.midway = lambda index, record: setup.stacker.stop() if index == 1 else None
        setup.run()
        setup.executor.midway = None
        setup.robot.closing_stops.append(50)
        status = setup.run()
        self.assertIsNone(status['error'])
        self.assertEqual([1], status['tower'])

    def test_stop_when_idle_returns_false(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        self.assertFalse(setup.stacker.stop())
        setup.run()
        self.assertFalse(setup.stacker.stop())

    def test_stop_asked_while_the_arm_is_laid_down_changes_nothing(self) -> None:
        setup = self.setup_for({1: THREE_BLOCKS[1]})
        setup.robot.closing_stops.append(50)
        setup.handler.errors[0] = ScanError('no picture')
        # the last move is the rest move, which does not look at the stop
        setup.executor.before = lambda index, record: setup.stacker.stop() if index == 1 else None
        status = setup.run()
        self.assertEqual('no picture', status['error'])
        self.assert_laid_down(setup)


class RehearsalTests(StackerTestCase):

    def test_hover_only_goes_over_the_blocks_and_the_tower_without_the_gripper(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        robot = setup.robot
        status = setup.run(hover_only=True)

        self.assertIsNone(status['error'])
        self.assertTrue(status['hover_only'])
        self.assertEqual('done: rehearsal of 3 blocks', status['activity'])
        self.assertEqual([], status['tower'])
        self.assertEqual([], robot.gripper_commands)
        self.assertEqual([{}], setup.handler.scan_heights)
        # park, two moves per block, park, rest
        records = setup.records
        self.assertEqual(1 + 3 * 2 + 1 + 1, len(records))
        steps = plan_stack(setup.handler.blocks({}), config, robot).steps
        self.assertEqual(EXPECTED_ORDER, [step.marker_id for step in steps])
        for index, step in enumerate(steps):
            assert_ends_at(self, robot, records[1 + 2 * index], hover_pose(step.pick, config))
            assert_ends_at(self, robot, records[2 + 2 * index], above_tower_pose(step.place, config, step.level))
        assert_ends_at(self, robot, records[7], park_pose(config, True))
        # nothing goes below the hover height, and nothing is done in the rehearsal that is not a joint move
        # (the first move starts at the rest pose on the table, and the last one goes back there)
        for record in records[1:-1]:
            self.assertGreaterEqual(min_tip_height(record.move, robot.robot_config),
                                    config.motion.hover_height - 1.0)
        self.assertEqual(6, setup.time.sleeps.count(block_stacker.REHEARSAL_PAUSE))
        self.assert_laid_down(setup, hover_only=True)

    def test_hover_only_with_nothing_to_do(self) -> None:
        setup = self.setup_for({})
        status = setup.run(hover_only=True)
        self.assertEqual('done: rehearsal of 0 blocks', status['activity'])
        self.assertEqual([], setup.robot.gripper_commands)

    def test_hover_only_can_be_stopped(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.executor.midway = lambda index, record: setup.stacker.stop() if index == 2 else None
        status = setup.run(hover_only=True)
        self.assertEqual('stopped', status['activity'])
        self.assertEqual([], setup.robot.gripper_commands)
        self.assert_laid_down(setup, hover_only=True)

    def test_a_rehearsal_that_dips_too_low_is_refused(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        original = block_stacker.min_tip_height
        block_stacker.min_tip_height = lambda move, robot_config: 1.0
        try:
            status = setup.run(hover_only=True)
        finally:
            block_stacker.min_tip_height = original
        self.assertIn('could hit the tower', status['error'])
        self.assert_laid_down(setup, hover_only=True)


class ScanTests(StackerTestCase):

    def test_scan_fills_the_status_and_moves_nothing(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        self.assertTrue(setup.stacker.scan())
        status = setup.stacker.status()
        self.assertEqual('idle', status['state'])
        self.assertIsNone(status['error'])
        self.assertTrue(status['board_visible'])
        self.assertEqual([1, 2, 3], [block['id'] for block in status['blocks']])
        first = status['blocks'][0]
        self.assertEqual({'id', 'x', 'y', 'yaw', 'frames', 'stable'}, set(first))
        self.assertEqual((10.0, 25.0), (first['x'], first['y']))
        self.assertEqual([{'id': 1, 'level': 0}, {'id': 3, 'level': 1}, {'id': 2, 'level': 2}], status['plan'])
        self.assertEqual({}, status['skipped'])
        self.assertEqual([], status['tower'])
        self.assertEqual([], setup.records)
        self.assertEqual(0, setup.robot.enable_calls)
        self.assertEqual(0, setup.robot.disable_calls)
        self.assertEqual([], setup.robot.gripper_commands)
        self.assertEqual([], setup.robot.events)

    def test_the_yaw_is_in_degrees(self) -> None:
        setup = self.setup_for({1: (10.0, 25.0, math.radians(20))})
        setup.stacker.scan()
        self.assertAlmostEqual(20.0, setup.stacker.status()['blocks'][0]['yaw'])

    def test_an_unstable_block_is_skipped_with_its_reason(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.handler.unstable = {3}
        setup.stacker.scan()
        status = setup.stacker.status()
        self.assertEqual([{'id': 1, 'level': 0}, {'id': 2, 'level': 1}], status['plan'])
        self.assertEqual({'3': 'its position is not steady in the picture'}, status['skipped'])
        self.assertFalse(status['blocks'][2]['stable'])

    def test_a_scan_error_is_the_error(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.handler.errors[0] = ScanError('the board is not in the picture')
        self.assertTrue(setup.stacker.scan())
        self.assertEqual('the board is not in the picture', setup.stacker.status()['error'])
        self.assertEqual([], setup.records)

    def test_a_stacking_error_is_the_error_and_the_next_scan_clears_it(self) -> None:
        setup = self.setup_for({1: (config.tower.x, config.tower.y + 1, 0.0)})
        setup.stacker.scan()
        self.assertIn('lies where the tower goes', setup.stacker.status()['error'])
        self.assertEqual([], setup.stacker.status()['plan'])
        setup.handler.table = dict(THREE_BLOCKS)
        setup.stacker.scan()
        status = setup.stacker.status()
        self.assertIsNone(status['error'])
        self.assertEqual(3, len(status['plan']))

    def test_an_unexpected_exception_in_a_scan_is_the_error(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.handler.errors[0] = RuntimeError('boom')
        with self.assertLogs(LOGGER_NAME, level='ERROR'):
            self.assertTrue(setup.stacker.scan())
        self.assertIn('boom', setup.stacker.status()['error'])

    def test_a_robot_that_can_not_be_opened_is_the_error_of_the_scan(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)

        def broken() -> FakeRobot:
            raise OSError('no port')

        setup.stacker._robot_provider = broken
        with self.assertLogs(LOGGER_NAME, level='ERROR'):
            self.assertTrue(setup.stacker.scan())
        self.assertIn('no port', setup.stacker.status()['error'])

    def test_scan_and_start_during_a_run_return_false(self) -> None:
        setup = self.setup_for({1: THREE_BLOCKS[1]})
        setup.robot.closing_stops.append(50)
        seen: list[Any] = []

        def before(index: int, record: Record) -> None:
            if index == 1:
                seen.extend([setup.stacker.scan(), setup.stacker.start(), setup.stacker.start(True),
                             setup.stacker.is_running(), setup.stacker.read_tip()])

        setup.executor.before = before
        status = setup.run()
        self.assertEqual([False, False, False, True], seen[:4])
        self.assertIn('error', seen[4])
        self.assertIsNone(status['error'])
        self.assertEqual([1], status['tower'])
        # only the scans of the run: the one before the block and the one after it
        self.assertEqual(2, len(setup.handler.scan_heights))

    def test_start_while_scanning_returns_false(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        results: list[bool] = []
        setup.handler.on_scan = lambda index: results.extend([setup.stacker.start(), setup.stacker.scan()])
        self.assertTrue(setup.stacker.scan())
        self.assertEqual([False, False], results)
        self.assertFalse(setup.stacker.is_running())


class ReadTipTests(StackerTestCase):

    def test_the_tip_next_to_the_nearest_block(self) -> None:
        setup = self.setup_for({5: (0.0, 40.0, 0.0), 6: (20.0, 20.0, 0.0)})
        result = setup.stacker.read_tip()
        tip = forward_position_kinematics(setup.robot.angles, setup.robot.robot_config)[-1]
        self.assertEqual([float(tip[0]), float(tip[1]), float(tip[2])], result['tip'])
        self.assertEqual(5, result['nearest_block'])
        self.assertEqual([0.0, 40.0], result['block'])
        self.assertAlmostEqual(0.0 - tip[0], result['difference'][0])
        self.assertAlmostEqual(40.0 - tip[1], result['difference'][1])
        self.assertEqual(0, setup.robot.enable_calls + setup.robot.disable_calls)
        self.assertEqual([], setup.records)
        json.dumps(result)

    def test_the_tip_without_blocks(self) -> None:
        setup = self.setup_for({})
        result = setup.stacker.read_tip()
        self.assertEqual(3, len(result['tip']))
        self.assertIsNone(result['nearest_block'])
        self.assertIsNone(result['block'])
        self.assertIsNone(result['difference'])
        json.dumps(result)

    def test_a_failed_read(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.robot.angles = None
        self.assertEqual({'error': 'the angles could not be read'}, setup.stacker.read_tip())

    def test_the_tip_during_a_run_is_refused(self) -> None:
        setup = self.setup_for({1: THREE_BLOCKS[1]})
        setup.robot.closing_stops.append(50)
        results: list[dict[str, Any]] = []
        setup.executor.before = lambda index, record: results.append(setup.stacker.read_tip()) if index == 0 else None
        setup.run()
        self.assertEqual(1, len(results))
        self.assertEqual(['error'], list(results[0]))


class StatusTests(StackerTestCase):

    def test_the_provider_is_not_called_by_the_constructor(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        self.assertEqual(0, setup.provider_calls)
        self.assertEqual(0, setup.robot.enable_calls)
        setup.stacker.status()
        self.assertEqual(0, setup.provider_calls)

    def test_the_robot_is_asked_for_once(self) -> None:
        setup = self.setup_for({1: THREE_BLOCKS[1]})
        setup.robot.closing_stops.append(50)
        setup.stacker.scan()
        setup.run()
        setup.stacker.read_tip()
        self.assertEqual(1, setup.provider_calls)

    def test_the_status_of_a_new_stacker(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        status = setup.stacker.status()
        self.assertEqual({'state': 'idle', 'activity': 'ready', 'error': None, 'warnings': [], 'hover_only': False,
                          'board_visible': True, 'blocks': [], 'plan': [], 'skipped': {}, 'tower': [],
                          'click_mode': False, 'click_message': None}, status)
        self.assertFalse(setup.stacker.is_running())

    def test_the_status_can_be_serialised_in_every_state(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.robot.closing_stops.extend([50, 50, 50])
        dumped: list[dict[str, Any]] = []
        states: set[str] = set()
        activities: set[str] = set()

        def record_status(index: int, record: Record) -> None:
            status = setup.stacker.status()
            json.dumps(status)
            dumped.append(status)
            states.add(status['state'])
            activities.add(status['activity'])
            if index == 20:
                setup.stacker.stop()

        setup.executor.before = record_status
        json.dumps(setup.stacker.status())
        setup.stacker.scan()
        json.dumps(setup.stacker.status())
        setup.run()
        status = setup.stacker.status()
        json.dumps(status)
        self.assertEqual('idle', status['state'])
        self.assertTrue({'running', 'stopping'} <= states)
        for expected in ('taking block 1', 'carrying block 1 to the tower',
                         'putting block 3 on level 1', 'lifting block 2', 'moving to where the camera looks'):
            self.assertIn(expected, activities)

    def test_the_status_of_an_error_is_serialisable_and_the_next_start_clears_it(self) -> None:
        setup = self.setup_for({1: THREE_BLOCKS[1]})
        setup.handler.errors[0] = ScanError('no picture')
        status = setup.run()
        json.dumps(status)
        self.assertEqual('no picture', status['error'])
        setup.robot.closing_stops.append(50)
        status = setup.run()
        self.assertIsNone(status['error'])
        self.assertEqual([1], status['tower'])

    def test_a_block_where_the_arm_comes_down_is_a_warning(self) -> None:
        # on the line from the base (0, 0) to the rest point (0, 40)
        setup = self.setup_for({1: THREE_BLOCKS[1], 2: (1.0, 25.0, 0.0), 3: (0.0, 25.0, 0.0)})
        setup.robot.closing_stops.extend([50, 50, 50])
        setup.handler.errors[1] = ScanError('no picture')
        status = setup.run()
        self.assertCountEqual(['block 2 lies where the arm comes down', 'block 3 lies where the arm comes down'],
                              status['warnings'])
        self.assertNotIn('block 1 lies where the arm comes down', status['warnings'])

    def test_a_block_in_the_tower_is_no_warning(self) -> None:
        setup = self.setup_for({1: (0.0, 25.0, 0.0)})
        setup.robot.closing_stops.append(50)
        status = setup.run()
        self.assertEqual([1], status['tower'])
        self.assertEqual([], status['warnings'])

    def test_a_block_that_is_held_when_the_arm_lays_down_is_no_warning(self) -> None:
        setup = self.setup_for({1: (0.0, 25.0, 0.0)})
        setup.robot.closing_stops.append(50)
        setup.executor.midway = lambda index, record: setup.stacker.stop() if index == 4 else None
        status = setup.run()
        self.assertEqual([], status['warnings'])

    def test_wait_without_a_run_returns(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.stacker.wait(1)
        setup.stacker.wait()


class LayDownTests(StackerTestCase):

    def left_holding(self) -> Setup:
        """a run that ended with the servos holding, the rest move refused"""
        setup = self.setup_for(THREE_BLOCKS)
        setup.handler.errors[0] = ScanError('no picture')
        setup.robot.refuse_rest = True
        setup.run()
        self.assertTrue(setup.robot.enabled)
        setup.robot.refuse_rest = False
        return setup

    def test_lay_down_brings_the_arm_down_and_switches_the_torque_off(self) -> None:
        setup = self.left_holding()
        setup.robot.gripper_commands.clear()
        moves_before = len(setup.records)
        self.assertTrue(setup.stacker.lay_down())
        setup.stacker.wait(60)
        status = setup.stacker.status()
        self.assertEqual('idle', status['state'])
        self.assertIsNone(status['error'])
        self.assertEqual('the arm is laid down and the servos are switched off', status['activity'])
        self.assertEqual(1, setup.robot.disable_calls)
        self.assertEqual([0.0], setup.robot.gripper_commands)
        self.assertEqual(moves_before + 1, len(setup.records))
        self.assert_laid_down(setup)
        # no enable: the servos hold already
        self.assertEqual(1, setup.robot.enable_calls)

    def test_lay_down_is_running_while_it_goes(self) -> None:
        setup = self.left_holding()
        seen: list[Any] = []

        def before(index: int, record: Record) -> None:
            status = setup.stacker.status()
            seen.append((status['state'], status['activity'], setup.stacker.is_running(), setup.stacker.lay_down(),
                         setup.stacker.disable_servos(), setup.stacker.scan(), setup.stacker.start(),
                         setup.stacker.stop(), record.stoppable))
            json.dumps(status)

        setup.executor.before = before
        self.assertTrue(setup.stacker.lay_down())
        setup.stacker.wait(60)
        self.assertEqual(1, len(seen))
        # a stop request changes nothing: the move is not interruptible
        self.assertEqual(('running', 'laying the arm down', True, False, False, False, False, True, False), seen[0])
        self.assert_laid_down(setup)

    def test_lay_down_uses_the_way_round_from_the_table(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.robot.closing_stops.extend([50, 50, 50])
        setup.executor.after = lambda index, record: setup.stacker.stop() if index == 2 else None
        setup.robot.refuse_rest = True
        setup.run()
        setup.robot.refuse_rest = False
        self.assertTrue(setup.stacker.lay_down())
        setup.stacker.wait(60)
        self.assertIsNone(setup.stacker.status()['error'])
        self.assert_laid_down(setup)

    def test_lay_down_that_fails_sets_the_error(self) -> None:
        setup = self.left_holding()
        setup.robot.refuse_rest = True
        self.assertTrue(setup.stacker.lay_down())
        setup.stacker.wait(60)
        status = setup.stacker.status()
        self.assertIn(LOWER_ARM_COMMAND, status['error'])
        self.assertEqual(0, setup.robot.disable_calls)
        self.assertIn('not laid down', status['activity'])
        self.assertFalse(setup.stacker.is_running())

    def test_lay_down_with_a_hardware_error_moves_nothing(self) -> None:
        setup = self.left_holding()
        moves_before = len(setup.records)
        setup.robot.hardware_errors = {3: ['Overload Error']}
        self.assertTrue(setup.stacker.lay_down())
        setup.stacker.wait(60)
        self.assertIn('servos 3 have a hardware error and are limp', setup.stacker.status()['error'])
        self.assertEqual(moves_before, len(setup.records))
        self.assertEqual(0, setup.robot.disable_calls)

    def test_lay_down_returns_false_during_a_run(self) -> None:
        setup = self.setup_for({1: THREE_BLOCKS[1]})
        setup.robot.closing_stops.append(50)
        seen: list[bool] = []
        setup.executor.before = lambda index, record: seen.append(setup.stacker.lay_down()) if index == 1 else None
        setup.run()
        self.assertEqual([False], seen)
        self.assertEqual(1, setup.robot.disable_calls)

    def test_lay_down_returns_false_during_a_scan(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        results: list[bool] = []
        setup.handler.on_scan = lambda index: results.extend([setup.stacker.lay_down(), setup.stacker.disable_servos()])
        self.assertTrue(setup.stacker.scan())
        self.assertEqual([False, False], results)
        self.assertEqual(0, setup.robot.disable_calls)

    def test_a_new_run_can_start_after_lay_down(self) -> None:
        setup = self.left_holding()
        setup.stacker.lay_down()
        setup.stacker.wait(60)
        setup.robot.closing_stops.append(50)
        setup.handler.errors.clear()
        status = setup.run()
        self.assertIsNone(status['error'])

    def test_disable_servos_switches_the_torque_off_at_once(self) -> None:
        setup = self.left_holding()
        self.assertTrue(setup.stacker.disable_servos())
        self.assertEqual(1, setup.robot.disable_calls)
        self.assertFalse(setup.robot.enabled)
        status = setup.stacker.status()
        self.assertEqual('idle', status['state'])
        # it sets no error of its own: the error of the run is still there
        self.assertIn('no picture', status['error'])
        self.assertIn('switched off', status['activity'])
        self.assertEqual(len(setup.records), 1 + 1)  # the park move and nothing since

    def test_disable_servos_sets_no_error_when_it_goes_well(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        self.assertTrue(setup.stacker.disable_servos())
        self.assertIsNone(setup.stacker.status()['error'])
        self.assertEqual(1, setup.robot.disable_calls)
        self.assertEqual([], setup.records)

    def test_disable_servos_returns_false_during_a_run(self) -> None:
        setup = self.setup_for({1: THREE_BLOCKS[1]})
        setup.robot.closing_stops.append(50)
        seen: list[bool] = []
        setup.executor.before = lambda index, record: seen.append(setup.stacker.disable_servos()) if index == 1 else None
        setup.run()
        self.assertEqual([False], seen)
        self.assertEqual(1, setup.robot.disable_calls)  # only the end of the run

    def test_a_failing_disable_is_the_error(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)

        def failing() -> bool | None:
            raise OSError('the bus is gone')

        setup.robot.disable_servos = failing  # type: ignore[method-assign]
        with self.assertLogs(LOGGER_NAME, level='ERROR'):
            self.assertTrue(setup.stacker.disable_servos())
        self.assertIn('the bus is gone', setup.stacker.status()['error'])
        self.assertFalse(setup.stacker.is_running())


class RobotProviderTests(StackerTestCase):

    def failing_setup(self) -> Setup:
        setup = self.setup_for(THREE_BLOCKS)
        setup.provider_errors.extend([OSError('servos do not answer')] * 5)
        return setup

    def test_a_scan_with_a_provider_that_raises_sets_the_error(self) -> None:
        setup = self.failing_setup()
        with self.assertLogs(LOGGER_NAME, level='ERROR'):
            self.assertTrue(setup.stacker.scan())
        status = setup.stacker.status()
        self.assertEqual('the robot could not be started: servos do not answer', status['error'])
        self.assertEqual('idle', status['state'])
        self.assertEqual([], setup.handler.scan_heights)
        json.dumps(status)

    def test_read_tip_with_a_provider_that_raises_gives_the_error(self) -> None:
        setup = self.failing_setup()
        with self.assertLogs(LOGGER_NAME, level='ERROR'):
            result = setup.stacker.read_tip()
        self.assertEqual({'error': 'the robot could not be started: servos do not answer'}, result)

    def test_lay_down_with_a_provider_that_raises_sets_the_error(self) -> None:
        setup = self.failing_setup()
        with self.assertLogs(LOGGER_NAME, level='ERROR'):
            self.assertTrue(setup.stacker.lay_down())
            setup.stacker.wait(10)
        status = setup.stacker.status()
        self.assertIn('the robot could not be started', status['error'])
        self.assertEqual('idle', status['state'])
        self.assertFalse(setup.stacker.is_running())

    def test_disable_servos_with_a_provider_that_raises_sets_the_error(self) -> None:
        setup = self.failing_setup()
        with self.assertLogs(LOGGER_NAME, level='ERROR'):
            self.assertTrue(setup.stacker.disable_servos())
        self.assertIn('the robot could not be started', setup.stacker.status()['error'])

    def test_a_run_with_a_provider_that_raises_ends_with_the_error(self) -> None:
        setup = self.failing_setup()
        with self.assertLogs(LOGGER_NAME, level='ERROR'):
            status = setup.run()
        self.assertEqual('the robot could not be started: servos do not answer', status['error'])
        self.assertEqual('idle', status['state'])
        self.assertEqual([], setup.records)
        self.assertEqual(0, setup.robot.disable_calls)

    def test_a_later_call_tries_the_provider_again(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.provider_errors.append(OSError('servos do not answer'))
        with self.assertLogs(LOGGER_NAME, level='ERROR'):
            setup.stacker.scan()
        self.assertIn('could not be started', setup.stacker.status()['error'])
        self.assertTrue(setup.stacker.scan())
        status = setup.stacker.status()
        self.assertIsNone(status['error'])
        self.assertEqual(3, len(status['plan']))
        self.assertEqual(2, setup.provider_calls)
        # and from now on the robot is kept
        setup.stacker.read_tip()
        self.assertEqual(2, setup.provider_calls)


WAITING = 'waiting for a click'
IDLE_TIMEOUT = config.click_mode.idle_timeout
FAR_AWAY = (0.0, 95.0, 0.0)  # beyond what the arm reaches


class ClickScript:
    """
    What happens while the arm waits for a click: every poll of the wait takes the next step, a marker id is a click on
    that block (its result goes to results) and a function is called. messages are the click messages the stacker had
    when the step came, so the message a click led to is the one of the next step.
    """

    def __init__(self, setup: Setup, *steps: int | Callable[[], Any]) -> None:
        self.setup = setup
        self.steps = deque(steps)
        self.results: list[bool] = []
        self.messages: list[str | None] = []
        setup.time.on_sleep = self.poll

    def poll(self) -> None:
        if not self.steps:
            return
        status = self.setup.stacker.status()
        if status['activity'] != WAITING:
            return
        self.messages.append(status['click_message'])
        step = self.steps.popleft()
        if isinstance(step, int):
            self.results.append(self.setup.stacker.click(step))
        else:
            step()


def planned_step(setup: Setup, marker_id: int, tower: list[int]) -> BlockStep:
    """the step the stacker plans for this block, with these blocks in the tower"""
    heights = {block: (level + 1) * config.blocks.size for level, block in enumerate(tower)}
    blocks = setup.handler.blocks(heights)
    block = next(candidate for candidate in blocks if candidate.marker_id == marker_id)
    others = [other for other in blocks if other.marker_id != marker_id and other.marker_id not in tower]
    result = plan_one(block, others, config, setup.robot, len(tower))
    assert not isinstance(result, str), result
    return result[0]


class PlanOneSpy:
    """replaces block_stacker.plan_one for a test and notes what it was asked"""

    def __init__(self, flip: bool | None = None) -> None:
        self.calls: list[tuple[int, list[int], int, bool]] = []
        self.flip = flip
        self.original = block_stacker.plan_one

    def __call__(self, block: Any, others: Any, cfg: Any, robot: Any, level: int, preferred_flip: bool = True) -> Any:
        self.calls.append((block.marker_id, [other.marker_id for other in others], level, preferred_flip))
        result = self.original(block, others, cfg, robot, level, preferred_flip)
        if self.flip is None or isinstance(result, str):
            return result
        return result[0], self.flip

    def __enter__(self) -> 'PlanOneSpy':
        block_stacker.plan_one = self  # type: ignore[assignment]
        return self

    def __exit__(self, *args: Any) -> None:
        block_stacker.plan_one = self.original


class ClickModeTests(StackerTestCase):

    def assert_block_moves(self, setup: Setup, first: int, step: BlockStep) -> None:
        """the seven moves of one block start at this record: hover, pick, lift, above, place, above, park"""
        poses = [hover_pose(step.pick, config), pick_pose(step.pick, config), lift_pose(step.pick, config, step.level),
                 above_tower_pose(step.place, config, step.level), place_pose(step.place, config, step.level),
                 above_tower_pose(step.place, config, step.level), park_pose(config, True)]
        for offset, pose in enumerate(poses):
            assert_ends_at(self, setup.robot, setup.records[first + offset], pose)

    def test_one_click_puts_the_block_on_the_tower_and_stop_lays_the_arm_down(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.robot.closing_stops.append(50)
        script = ClickScript(setup, 1, setup.stacker.stop)
        seen: list[dict[str, Any]] = []

        def before(index: int, record: Record) -> None:
            status = setup.stacker.status()
            json.dumps(status)
            seen.append(status)

        setup.executor.before = before
        status = setup.run_click_mode()

        self.assertEqual([True], script.results)
        self.assertEqual([None, 'block 1 is on the tower, level 0'], script.messages)
        self.assertIsNone(status['error'])
        self.assertEqual([1], status['tower'])
        self.assertEqual('stopped', status['activity'])
        self.assertEqual('idle', status['state'])
        self.assertFalse(status['click_mode'])
        self.assertEqual('block 1 is on the tower, level 0', status['click_message'])
        self.assertEqual([{}], setup.handler.scan_heights)
        # park, the seven moves of the block, the rest pose
        self.assertEqual(1 + 7 + 1, len(setup.records))
        assert_ends_at(self, setup.robot, setup.records[0], park_pose(config, True))
        self.assert_block_moves(setup, 1, planned_step(setup, 1, []))
        pull_back = 50 + config.gripper.max_lead_positions / POSITIONS_PER_STATE
        self.assertEqual([0.0, 100.0, pull_back, 0.0, 0.0], setup.robot.gripper_commands)
        self.assert_laid_down(setup)
        self.assertEqual(1, setup.robot.enable_calls)
        # the status in the mode: the mode is on, the plan and the skipped stay empty, the blocks are the last look
        during = [entry for entry in seen if entry['activity'] == 'taking block 1']
        self.assertTrue(during)
        self.assertTrue(during[0]['click_mode'])
        self.assertEqual('running', during[0]['state'])
        self.assertEqual([], during[0]['plan'])
        self.assertEqual({}, during[0]['skipped'])
        self.assertEqual([1, 2, 3], [block['id'] for block in during[0]['blocks']])

    def test_two_clicks_build_a_tower_of_two(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.robot.closing_stops.extend([50, 50])
        script = ClickScript(setup, 1, 3, setup.stacker.stop)
        status = setup.run_click_mode()

        self.assertEqual([True, True], script.results)
        self.assertEqual([1, 3], status['tower'])
        self.assertEqual('block 3 is on the tower, level 1', status['click_message'])
        self.assertIsNone(status['error'])
        self.assertEqual([{}, {1: config.blocks.size}], setup.handler.scan_heights)
        self.assertEqual(1 + 2 * 7 + 1, len(setup.records))
        self.assert_block_moves(setup, 1, planned_step(setup, 1, []))
        self.assert_block_moves(setup, 8, planned_step(setup, 3, [1]))
        self.assert_laid_down(setup)

    def test_a_click_is_only_accepted_while_the_arm_waits_and_has_no_click_yet(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.robot.closing_stops.extend([50, 50])
        stacker = setup.stacker
        before_start = stacker.click(1)
        double: list[bool] = []
        during: list[bool] = []
        after_stop: list[bool] = []

        def stop_and_click() -> None:
            stacker.stop()
            after_stop.append(stacker.click(2))

        # two clicks at once, the second finds the first still waiting to be picked up; one while the block is
        # carried; and one after stop
        script = ClickScript(setup, lambda: double.extend([stacker.click(1), stacker.click(3)]), 3, stop_and_click)
        setup.executor.before = lambda index, record: during.append(stacker.click(3)) if index == 4 else None
        status = setup.run_click_mode()

        self.assertFalse(before_start)
        self.assertEqual([True, False], double)
        self.assertEqual([False], during)
        self.assertEqual([False], after_stop)
        self.assertEqual([True], script.results)
        self.assertFalse(stacker.click(1))
        self.assertEqual([1, 3], status['tower'])

    def test_a_click_just_before_the_idle_timeout_is_handled_and_restarts_the_wait(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        accepted: list[bool] = []

        def on_sleep() -> None:
            if not accepted and setup.time.now > IDLE_TIMEOUT - 0.15:
                accepted.append(setup.stacker.click(9))

        setup.time.on_sleep = on_sleep
        status = setup.run_click_mode()
        self.assertEqual([True], accepted)
        self.assertEqual('block 9 is not in the picture', status['click_message'])
        self.assertEqual(1, len(setup.handler.scan_heights))
        self.assertGreater(setup.time.now, 2 * IDLE_TIMEOUT - 1)

    def test_blocks_that_can_not_be_taken_give_a_message_and_the_mode_goes_on(self) -> None:
        table = dict(THREE_BLOCKS)
        table.update({4: FAR_AWAY, 5: (-10.0, 28.0, 0.0)})
        setup = self.setup_for(table)
        setup.handler.unstable = {5}
        setup.robot.closing_stops.append(50)
        script = ClickScript(setup, 4, 9, 5, 1, 1, setup.stacker.stop)
        status = setup.run_click_mode()

        self.assertEqual([True] * 5, script.results)
        messages = script.messages
        self.assertTrue((messages[1] or '').startswith('block 4 can not be taken: '), messages[1])
        self.assertEqual('block 9 is not in the picture', messages[2])
        self.assertEqual('block 5 can not be taken: its position is not steady in the picture', messages[3])
        self.assertEqual('block 1 is on the tower, level 0', messages[4])
        self.assertEqual('block 1 is in the tower already', messages[5])
        self.assertEqual([1], status['tower'])
        self.assertIsNone(status['error'])
        # only the one good click moved the arm to a block: park, seven moves, rest
        self.assertEqual(1 + 7 + 1, len(setup.records))
        self.assertEqual(1, setup.robot.gripper_commands.count(100.0))
        self.assert_laid_down(setup)

    def test_the_block_that_is_far_away_is_really_out_of_reach(self) -> None:
        setup = self.setup_for({1: THREE_BLOCKS[1], 4: FAR_AWAY})
        blocks = setup.handler.blocks({})
        result = plan_one(blocks[1], [blocks[0]], config, setup.robot, 0)
        self.assertIsInstance(result, str)

    def test_the_other_blocks_are_obstacles_whether_or_not_they_are_steady(self) -> None:
        setup = self.setup_for({1: THREE_BLOCKS[1], 2: (THREE_BLOCKS[1][0] + 3.2, THREE_BLOCKS[1][1], 0.0)})
        setup.handler.unstable = {2}
        with PlanOneSpy() as spy:
            ClickScript(setup, 1, setup.stacker.stop)
            setup.run_click_mode()
        self.assertEqual([(1, [2], 0, True)], spy.calls)

    def test_the_blocks_of_the_tower_are_not_obstacles_and_the_level_is_the_height_of_the_tower(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.robot.closing_stops.extend([50, 50])
        with PlanOneSpy() as spy:
            ClickScript(setup, 1, 3, setup.stacker.stop)
            setup.run_click_mode()
        self.assertEqual([(1, [2, 3], 0, True), (3, [2], 1, True)], spy.calls)

    def test_a_scan_error_is_the_click_message_and_the_mode_goes_on(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.robot.closing_stops.append(50)
        setup.handler.errors[0] = ScanError('the board is not in the picture')
        script = ClickScript(setup, 1, 1, setup.stacker.stop)
        status = setup.run_click_mode()
        self.assertEqual([True, True], script.results)
        self.assertEqual('the board is not in the picture', script.messages[1])
        self.assertEqual([1], status['tower'])
        self.assertIsNone(status['error'])
        self.assertEqual(1 + 7 + 1, len(setup.records))
        self.assert_laid_down(setup)

    def test_a_gripper_that_closes_on_nothing_is_a_message_and_the_block_can_be_clicked_again(self) -> None:
        setup = self.setup_for({1: THREE_BLOCKS[1]})
        setup.robot.closing_stops.extend([None, None, 50])
        script = ClickScript(setup, 1, 1, 1, setup.stacker.stop)
        status = setup.run_click_mode()
        self.assertEqual('block 1 was not taken: the gripper closed on nothing', script.messages[1])
        self.assertEqual('block 1 was not taken: the gripper closed on nothing', script.messages[2])
        # the same block is never given up on, so the third click takes it
        self.assertEqual('block 1 is on the tower, level 0', script.messages[3])
        self.assertEqual([1], status['tower'])
        self.assertEqual({}, status['skipped'])
        self.assertIsNone(status['error'])
        self.assert_laid_down(setup)

    def test_a_block_that_slips_out_is_a_message_too(self) -> None:
        setup = self.setup_for({1: THREE_BLOCKS[1]})
        setup.robot.closing_stops.append(50)
        # moves: 0 park, 1 hover, 2 pick, 3 lift
        setup.executor.after = lambda index, record: setup.robot.lose_block() if index == 3 else None
        ClickScript(setup, 1, setup.stacker.stop)
        status = setup.run_click_mode()
        self.assertEqual('block 1 was not taken: the block slipped out of the gripper while it was lifted',
                         status['click_message'])
        self.assertEqual([], status['tower'])
        self.assertIsNone(status['error'])
        self.assert_laid_down(setup)

    def test_a_block_that_is_out_of_reach_when_taking_it_is_a_message(self) -> None:
        setup = self.setup_for({1: THREE_BLOCKS[1]})
        original = block_stacker.plan_straight_line

        def failing_line(*args: Any, **kwargs: Any) -> PlannedMove:
            raise MovementException('the block moved out of reach')

        block_stacker.plan_straight_line = failing_line
        try:
            ClickScript(setup, 1, setup.stacker.stop)
            status = setup.run_click_mode()
        finally:
            block_stacker.plan_straight_line = original
        self.assertEqual('block 1 was not taken: its position can not be reached: the block moved out of reach',
                         status['click_message'])
        self.assertIsNone(status['error'])
        self.assert_laid_down(setup)

    def test_the_other_flip_moves_to_its_park_pose_first_and_is_preferred_after_that(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.robot.closing_stops.extend([50, 50])
        # the steps stay planned for flip True, which is all this test needs: only the park pose that is moved to counts
        with PlanOneSpy(flip=False) as spy:
            ClickScript(setup, 1, 3, setup.stacker.stop)
            status = setup.run_click_mode()
        self.assertIsNone(status['error'], status['error'])
        self.assertEqual([1, 3], status['tower'])
        self.assertEqual([True, False], [call[3] for call in spy.calls])
        assert_ends_at(self, setup.robot, setup.records[0], park_pose(config, True))
        assert_ends_at(self, setup.robot, setup.records[1], park_pose(config, False))
        # park, park, seven moves, seven moves without another turn of the wrist, rest
        self.assertEqual(2 + 7 + 7 + 1, len(setup.records))

    def test_the_tower_top_missing_at_a_click_ends_the_mode(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.robot.closing_stops.extend([50, 50])
        setup.handler.hidden_in_tower = {1}
        ClickScript(setup, 1, 3, setup.stacker.stop)
        status = setup.run_click_mode()
        self.assertEqual('the tower is not standing where it should: block 1 is not on top', status['error'])
        self.assertEqual([1], status['tower'])
        self.assertFalse(status['click_mode'])
        self.assert_laid_down(setup)

    def test_a_hardware_error_while_waiting_ends_the_mode_and_moves_nothing(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        ClickScript(setup, lambda: setattr(setup.robot, 'hardware_errors', {2: ['Overload Error']}))
        status = setup.run_click_mode()
        self.assertIn('servos 2 have a hardware error', status['error'])
        self.assertFalse(status['click_mode'])
        self.assert_left_holding(setup, 'servos 2 have a hardware error and are limp')
        # it was found by the check of the wait, a few seconds later and not at once
        self.assertGreaterEqual(setup.time.now, CLICK_HEALTH_INTERVAL)
        self.assertEqual([], setup.handler.scan_heights)

    def test_a_hardware_error_at_a_click_ends_the_mode(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)

        def fail_and_click() -> None:
            setup.robot.hardware_errors = {2: ['Overload Error']}
            setup.stacker.click(1)

        ClickScript(setup, fail_and_click)
        status = setup.run_click_mode()
        self.assertIn('servos 2 have a hardware error', status['error'])
        self.assertEqual([], setup.handler.scan_heights)
        self.assert_left_holding(setup, 'servos 2 have a hardware error and are limp')

    def test_an_overheating_servo_while_waiting_ends_the_mode_and_lays_the_arm_down(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        ClickScript(setup, lambda: setattr(setup.robot, 'overheating', {3: 81}))
        status = setup.run_click_mode()
        self.assertIn('servos 3 are too warm', status['error'])
        self.assertGreaterEqual(setup.time.now, CLICK_HEALTH_INTERVAL)
        self.assert_laid_down(setup)

    def test_the_servos_are_checked_every_5_seconds_of_waiting(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        checks: list[float] = []
        original = setup.robot.get_hardware_errors

        def counting() -> dict[int, list[str]] | None:
            checks.append(setup.time.now)
            return original()

        setup.robot.get_hardware_errors = counting  # type: ignore[method-assign]
        setup.run_click_mode()
        # at the start of the wait, then every 5 seconds until the timeout, and once more to lay the arm down
        waiting = checks[:-1]
        self.assertEqual(0.0, waiting[0])
        for earlier, later in zip(waiting, waiting[1:]):
            self.assertAlmostEqual(CLICK_HEALTH_INTERVAL, later - earlier, delta=0.2)
        self.assertGreaterEqual(len(waiting), int(IDLE_TIMEOUT / CLICK_HEALTH_INTERVAL))

    def test_a_gripper_that_does_not_close_ends_the_mode(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.robot.closing_stops.append(4.0)
        ClickScript(setup, 1, 1)
        status = setup.run_click_mode()
        self.assertEqual('the gripper did not close', status['error'])
        self.assertEqual([], status['tower'])
        self.assertFalse(status['click_mode'])
        self.assert_laid_down(setup)

    def test_a_gripper_that_does_not_open_over_the_tower_ends_the_mode(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.robot.closing_stops.append(50)
        # moves: 0 park, 1 hover, 2 pick, 3 lift, 4 carry, 5 place
        setup.executor.after = lambda index, record: (
            setattr(setup.robot, 'gripper_opens', False) if index == 5 else None)
        ClickScript(setup, 1, 3)
        status = setup.run_click_mode()
        self.assertIn('did not open', status['error'])
        self.assertEqual([], status['tower'])
        self.assert_laid_down(setup, gripper_opens=False)

    def test_a_block_that_can_not_be_carried_ends_the_mode(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.robot.closing_stops.append(50)
        original = block_stacker.plan_straight_line
        calls = {'count': 0}

        def failing_line(*args: Any, **kwargs: Any) -> PlannedMove:
            calls['count'] += 1
            if calls['count'] == 3:  # the line down onto the tower, with the block in the gripper
                raise MovementException('out of reach')
            return original(*args, **kwargs)

        block_stacker.plan_straight_line = failing_line
        try:
            ClickScript(setup, 1, 3)
            status = setup.run_click_mode()
        finally:
            block_stacker.plan_straight_line = original
        self.assertIn('block 1 can not be carried to the tower: out of reach', status['error'])
        self.assert_laid_down(setup)

    def test_an_unexpected_exception_ends_the_mode_and_lays_the_arm_down(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.handler.errors[0] = RuntimeError('boom')
        ClickScript(setup, 1, 1)
        with self.assertLogs(LOGGER_NAME, level='ERROR'):
            status = setup.run_click_mode()
        self.assertIn('boom', status['error'])
        self.assertFalse(status['click_mode'])
        self.assert_laid_down(setup)
        self.assertEqual((None, None), (setup.handler.selected, setup.handler.message))

    def test_no_click_for_the_idle_timeout_lays_the_arm_down(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        status = setup.run_click_mode()
        self.assertIsNone(status['error'])
        self.assertEqual('the arm was laid down after {:g} seconds without a click'.format(IDLE_TIMEOUT),
                         status['activity'])
        self.assertFalse(status['click_mode'])
        self.assertEqual('idle', status['state'])
        self.assertGreaterEqual(setup.time.now, IDLE_TIMEOUT)
        self.assertLess(setup.time.now, IDLE_TIMEOUT + 10)
        self.assertEqual(2, len(setup.records))
        self.assert_laid_down(setup)

    def test_a_click_restarts_the_wait_for_the_idle_timeout(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        clicked: list[float] = []

        def on_sleep() -> None:
            if not clicked and setup.time.now >= IDLE_TIMEOUT - 20 and setup.stacker.status()['activity'] == WAITING:
                clicked.append(setup.time.now)
                setup.stacker.click(9)

        setup.time.on_sleep = on_sleep
        status = setup.run_click_mode()
        self.assertEqual('block 9 is not in the picture', status['click_message'])
        self.assertIn('without a click', status['activity'])
        # the wait that began after the click lasted the whole timeout
        self.assertGreaterEqual(setup.time.now, clicked[0] + IDLE_TIMEOUT)
        self.assertLess(setup.time.now, clicked[0] + IDLE_TIMEOUT + 10)
        self.assert_laid_down(setup)

    def test_servos_that_refuse_to_enable_end_the_mode_without_a_move(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.robot.enable_result = False
        status = setup.run_click_mode()
        self.assertIn('could not be switched on', status['error'])
        self.assertEqual([], setup.records)
        self.assertEqual(0, setup.robot.disable_calls)
        self.assertEqual([], setup.robot.gripper_commands)
        self.assertFalse(status['click_mode'])
        self.assertEqual('idle', status['state'])
        self.assertEqual((None, None), (setup.handler.selected, setup.handler.message))

    def test_stop_while_waiting_lays_the_arm_down(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        results: list[bool] = []
        ClickScript(setup, lambda: results.append(setup.stacker.stop()))
        status = setup.run_click_mode()
        self.assertEqual([True], results)
        self.assertEqual('stopped', status['activity'])
        self.assertIsNone(status['error'])
        self.assertFalse(status['click_mode'])
        self.assertEqual(2, len(setup.records))
        self.assert_laid_down(setup)

    def test_a_click_that_is_waiting_when_stop_comes_is_dropped(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)

        def click_and_stop() -> None:
            setup.stacker.click(1)
            setup.stacker.stop()

        ClickScript(setup, click_and_stop)
        status = setup.run_click_mode()
        self.assertEqual([], status['tower'])
        self.assertEqual([], setup.handler.scan_heights)
        self.assertEqual('stopped', status['activity'])
        self.assert_laid_down(setup)

    def test_stop_while_a_block_is_carried_lays_the_arm_down_and_lets_the_block_go(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        robot = setup.robot
        robot.closing_stops.append(50)
        gripper_before_rest: list[float] = []
        # moves: 0 park, 1 hover, 2 pick, 3 lift, 4 carry, 5 rest
        setup.executor.midway = lambda index, record: setup.stacker.stop() if index == 4 else None
        setup.executor.before = lambda index, record: (
            gripper_before_rest.append(robot.gripper_state) if index == 5 else None)
        ClickScript(setup, 1)
        status = setup.run_click_mode()
        self.assertEqual('stopped', status['activity'])
        self.assertIsNone(status['error'])
        self.assertEqual([], status['tower'])
        self.assertEqual([50.0], gripper_before_rest)
        self.assertEqual(6, len(setup.records))
        self.assert_laid_down(setup)

    def test_a_main_thread_that_is_gone_ends_the_mode(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        ClickScript(setup, lambda: setattr(setup, 'main_alive', False))
        status = setup.run_click_mode()
        self.assertEqual('stopped', status['activity'])
        self.assert_laid_down(setup)

    def test_the_window_shows_the_click_and_what_the_arm_does(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.robot.closing_stops.append(50)
        handler = setup.handler
        shown: dict[str, Any] = {}
        waiting_messages: list[str | None] = []

        def before(index: int, record: Record) -> None:
            if index == 1:  # the move to the hover pose
                shown['selected'] = handler.selected
                shown['message'] = handler.message

        setup.executor.before = before
        ClickScript(setup, lambda: waiting_messages.append(handler.message), 1,
                    lambda: waiting_messages.append(handler.message), setup.stacker.stop)
        setup.run_click_mode()

        self.assertEqual(1, shown['selected'])
        self.assertEqual('taking block 1', shown['message'])
        self.assertEqual([CLICK_PROMPT, 'block 1 is on the tower, level 0 - {}'.format(CLICK_PROMPT)],
                         waiting_messages)
        selected = [value for kind, value in handler.window_calls if kind == 'selected']
        self.assertEqual([1, None, None], selected)
        # at the end nothing is shown and nothing is selected
        self.assertIsNone(handler.selected)
        self.assertIsNone(handler.message)
        self.assertEqual(('message', None), handler.window_calls[-1])

    def test_the_window_is_cleared_when_the_mode_ends_with_an_error(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.robot.overheating = {3: 81}
        ClickScript(setup, lambda: None)
        setup.run_click_mode()
        self.assertEqual((None, None), (setup.handler.selected, setup.handler.message))

    def test_the_selection_is_cleared_when_a_click_is_stopped(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.robot.closing_stops.append(50)
        setup.executor.midway = lambda index, record: setup.stacker.stop() if index == 2 else None
        ClickScript(setup, 1)
        setup.run_click_mode()
        self.assertIsNone(setup.handler.selected)
        self.assertEqual(1, [value for kind, value in setup.handler.window_calls if kind == 'selected'][0])

    def test_nothing_else_can_start_while_the_mode_is_on(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.robot.closing_stops.append(50)
        stacker = setup.stacker
        seen: list[tuple[Any, ...]] = []

        def others() -> None:
            status = stacker.status()
            json.dumps(status)
            seen.append((status['state'], status['click_mode'], stacker.is_running(), stacker.start(),
                         stacker.start(True), stacker.scan(), stacker.lay_down(), stacker.disable_servos(),
                         stacker.start_click_mode(), 'error' in stacker.read_tip()))

        expected = ('running', True, True, False, False, False, False, False, False, True)
        ClickScript(setup, others, 1, others, stacker.stop)
        setup.executor.before = lambda index, record: others() if index == 3 else None
        status = setup.run_click_mode()
        self.assertEqual([expected] * 3, seen)
        self.assertIsNone(status['error'])
        self.assertEqual([1], status['tower'])
        self.assertEqual(1, len(setup.handler.scan_heights))

    def test_start_click_mode_is_refused_during_a_run_and_a_scan(self) -> None:
        setup = self.setup_for({1: THREE_BLOCKS[1]})
        setup.robot.closing_stops.append(50)
        results: list[Any] = []

        def before(index: int, record: Record) -> None:
            if index == 1:
                results.extend([setup.stacker.start_click_mode(), setup.stacker.click(1)])

        setup.executor.before = before
        setup.run()
        self.assertEqual([False, False], results)
        results.clear()
        setup.executor.before = None
        setup.handler.on_scan = lambda index: results.append(setup.stacker.start_click_mode())
        self.assertTrue(setup.stacker.scan())
        self.assertEqual([False], results)
        self.assertFalse(setup.stacker.is_running())

    def test_a_new_mode_starts_with_an_empty_tower_and_message(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.robot.closing_stops.extend([50, 50])
        ClickScript(setup, 1, setup.stacker.stop)
        setup.run_click_mode()
        self.assertEqual('block 1 is on the tower, level 0', setup.stacker.status()['click_message'])
        scans_before = len(setup.handler.scan_heights)
        script = ClickScript(setup, 1, setup.stacker.stop)
        status = setup.run_click_mode()
        # the first message of this mode was none, and block 1 went to level 0 again
        self.assertEqual([None, 'block 1 is on the tower, level 0'], script.messages)
        self.assertEqual([1], status['tower'])
        self.assertEqual({}, setup.handler.scan_heights[scans_before])
        self.assertIsNone(status['error'])

    def test_the_normal_run_is_not_in_the_click_mode(self) -> None:
        setup = self.setup_for({1: THREE_BLOCKS[1]})
        setup.robot.closing_stops.append(50)
        seen: list[Any] = []
        setup.executor.before = lambda index, record: seen.append(
            (setup.stacker.click(1), setup.stacker.status()['click_mode'])) if index == 1 else None
        status = setup.run()
        self.assertEqual([(False, False)], seen)
        self.assertFalse(status['click_mode'])
        self.assertIsNone(status['click_message'])
        self.assertEqual([], setup.handler.window_calls)

    def test_the_status_can_be_serialised_in_every_state_of_the_mode(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.robot.closing_stops.extend([50, 50])
        activities: set[str] = set()
        keys: list[set[str]] = []

        def record_status(index: int, record: Record) -> None:
            status = setup.stacker.status()
            json.dumps(status)
            activities.add(status['activity'])
            keys.append(set(status))

        setup.executor.before = record_status
        ClickScript(setup, lambda: json.dumps(setup.stacker.status()), 1, 3, setup.stacker.stop)
        json.dumps(setup.stacker.status())
        status = setup.run_click_mode()
        json.dumps(status)
        for expected in ('taking block 1', 'carrying block 1 to the tower', 'putting block 3 on level 1',
                         'moving to where the camera looks'):
            self.assertIn(expected, activities)
        for names in keys:
            self.assertTrue({'click_mode', 'click_message', 'state', 'tower', 'plan', 'skipped'} <= names)


class MainThreadTests(StackerTestCase):

    def test_a_run_stops_and_lays_the_arm_down_when_the_main_thread_is_gone(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.robot.closing_stops.extend([50, 50, 50])
        # the server is shut down during the first block
        setup.executor.midway = lambda index, record: setattr(setup, 'main_alive', False) if index == 2 else None
        status = setup.run()
        self.assertIsNone(status['error'])
        self.assertEqual('stopped', status['activity'])
        self.assertEqual([], status['tower'])
        self.assertFalse(setup.records[2].completed)
        self.assert_laid_down(setup)

    def test_a_main_thread_that_is_gone_before_the_run_stops_it_after_the_first_move(self) -> None:
        setup = self.setup_for(THREE_BLOCKS)
        setup.executor.before = lambda index, record: setattr(setup, 'main_alive', False) if index == 0 else None
        status = setup.run()
        self.assertEqual('stopped', status['activity'])
        self.assertEqual(2, len(setup.records))
        self.assert_laid_down(setup)

    def test_the_default_check_looks_at_the_main_thread(self) -> None:
        stacker = BlockStacker(lambda: FakeRobot(), FakeHandler({}), config)  # type: ignore[arg-type]
        self.assertFalse(stacker._should_stop())
        self.assertTrue(stacker._main_thread_alive())
        stacker.stop()
        self.assertFalse(stacker._should_stop())


if __name__ == '__main__':
    unittest.main()
