"""
Runs the block stacking: a background thread that looks at the table, plans, moves the arm, works the gripper and
reports what it does. The planning is in stacking_plan, the seeing in camera/block_detection, the moves in
utils/movement; this class puts them in order and keeps the arm safe while it does.

One run goes like this: the servos are enabled, the gripper opens, the arm moves to the park pose where the camera has a
free view of the table, and then block after block: look, check the tower, plan again (the positions are fresh every
round), take the nearest block, put it on the tower and go back to the park pose. Whatever ends the run (done, stop,
an error), the arm is laid down on the table and only then the torque goes off, never with the arm in the air.

The click mode is a run of another kind: the arm waits at the park pose, the user clicks a block in the camera window,
and the arm puts that one block on the tower and goes back to waiting. It ends like a run does, and also when no click
comes for click_mode.idle_timeout seconds.

Only the run thread moves the arm and touches the gripper. The other threads (the web page, the camera) call start,
start_click_mode, click, stop, scan, read_tip and status; the state they share is guarded by one lock.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from src.block_stacking.stacking_config import StackingConfig
from src.block_stacking.stacking_plan import StackingError, StackingPlan, BlockStep, park_pose, hover_pose, pick_pose, \
    lift_pose, above_tower_pose, place_pose, travel_height, in_rest_area, min_tip_height, plan_stack, plan_one
from src.camera.block_detection import BlockImageHandler, DetectedBlock, ScanError
from src.kinematics.kinematics import forward_kinematics, forward_position_kinematics
from src.kinematics.kinematics_utils import Pose
from src.robot_controllers.dynamixel_robot.calibration.record_static_load import REST_POSE
from src.utils.movement import MAX_POSITION_ERROR, PlannedMove, execute_move, plan_move_to_angles, plan_straight_line
from src.utils.movement_exception import MovementException

if TYPE_CHECKING:
    from src.robot_controllers.abstract_robot_controller import AbstractRobotController

logger = logging.getLogger(__name__)

# how far from the centre of the tower (cm) the block on top of it may be seen
TOWER_TOLERANCE = 1.5
# how far (cm) below the travel height a carried block may dip on its way over to the tower: the joints turn along a
# curve, not a straight line
CARRY_MARGIN = 1.0
# how often a block is tried before it is given up on
MAX_TRIES = 2
# the seconds to wait at the hover pose and at the pose above the tower in a rehearsal
REHEARSAL_PAUSE = 1.0
# the seconds the arm gets to settle at the park pose before the camera looks: a picture of a trembling arm or a
# block that still swings in the gripper is of no use
SCAN_SETTLE_TIME = 0.5
# the seconds the arm rests on the table before the gripper opens and the torque goes off
REST_SETTLE_TIME = 0.5
# the gripper is read this often (seconds) while it moves
GRIPPER_POLL_INTERVAL = 0.05
# the gripper has stopped when its state changes less than this (0 to 100) for settle_time
GRIPPER_STEADY_CHANGE = 0.5
# the gripper counts as open when it is this close (states) to the open state
GRIPPER_OPEN_TOLERANCE = 10.0
# a gripper that stopped this close (states) to the open state did not close at all
GRIPPER_NOT_CLOSED_WITHIN = 5.0
# the positions of one gripper state when the robot can not tell, like robot_controllers do for the real servo
DEFAULT_POSITIONS_PER_STATE = 9.5
LOWER_ARM_COMMAND = 'python -m src.robot_controllers.dynamixel_robot.calibration.lower_arm'
NOT_STEADY = 'its position is not steady in the picture'
# the click mode looks at the stop request this often (seconds) while it waits for a click
CLICK_POLL_INTERVAL = 0.1
# and checks the servos this often (seconds) while it waits: the arm may hold for minutes
CLICK_HEALTH_INTERVAL = 5.0
CLICK_PROMPT = 'click a block to put it on the tower'
NOT_LAID_DOWN = 'the arm is not laid down, the servos are still holding it'

IDLE = 'idle'
RUNNING = 'running'
STOPPING = 'stopping'


class _Stopped(Exception):
    """raised in the run thread when the user stopped the run"""


class _RunError(Exception):
    """raised in the run thread when the run can not go on, the message is for the user"""


def _plural(count: int, noun: str) -> str:
    return '{} {}{}'.format(count, noun, '' if count == 1 else 's')


def _angles_of(robot: AbstractRobotController, pose: Pose) -> np.ndarray:
    """
    the angles of the pose from the inverse kinematics
    :raises MovementException: if the arm can not get to the pose
    """
    angles = np.asarray(robot.pose_to_angles(pose), dtype=np.float64)
    if not np.all(np.isfinite(angles)):
        raise MovementException('the inverse kinematics gave no angles')
    reached = forward_position_kinematics(angles, robot.robot_config)[-1]
    if np.linalg.norm(reached - [pose.x, pose.y, pose.z]) > MAX_POSITION_ERROR:
        raise MovementException('the arm can not get to x={:.1f} y={:.1f} z={:.1f}'.format(pose.x, pose.y, pose.z))
    return angles


class BlockStacker:
    """
    Stacks the blocks the camera sees into a tower, in a thread of its own.
    :param robot_provider: gives the robot, called on first need only (the real one opens the serial port) and its
        result is kept
    :param handler: the camera's block finder
    :param sleep: all waiting goes through it, and all time measuring through the clock, so tests need not wait
    :param execute: executes a planned move like execute_move: execute(robot, move, should_stop) returns False when
        the move was stopped
    :param main_thread_alive: whether the main thread of the program still runs. The run thread is not a daemon, so
        that a program that ends still lays the arm down, but when the main thread is gone (the server was shut down
        with Ctrl+C) the run stops like it does on stop(), instead of finishing the whole tower
    """

    def __init__(self, robot_provider: Callable[[], AbstractRobotController], handler: BlockImageHandler,
                 config: StackingConfig, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic,
                 execute: Callable[..., bool] = execute_move,
                 main_thread_alive: Callable[[], bool] = lambda: threading.main_thread().is_alive()) -> None:
        self._robot_provider = robot_provider
        self._handler = handler
        self._config = config
        self._sleep = sleep
        self._clock = clock
        self._execute = execute
        self._main_thread_alive = main_thread_alive
        self._robot: AbstractRobotController | None = None
        self._robot_lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._thread: threading.Thread | None = None

        # the state the web threads read and the run thread writes, guarded by the lock
        self._lock = threading.RLock()
        self._state = IDLE
        self._busy = False
        self._activity = 'ready'
        self._error: str | None = None
        self._warnings: list[str] = []
        self._hover_only = False
        self._scanned: list[DetectedBlock] = []
        self._plan: list[tuple[int, int]] = []  # (marker id, level)
        self._skipped: dict[int, str] = {}
        self._tower: list[int] = []
        # the click mode: whether it is on, what the last click led to, whether the thread waits for a click and the
        # click it has not taken up yet
        self._click_mode = False
        self._click_message: str | None = None
        self._click_waiting = False
        self._pending_click: int | None = None

        # the state of the run, only the run thread uses it
        self._angles: np.ndarray | None = None  # where the last move ended
        self._park_flip: bool | None = None
        self._servos_enabled = False
        self._held: int | None = None  # the block in the gripper
        self._tries: dict[int, list[str]] = {}
        self._given_up: dict[int, str] = {}

    # what the web page calls

    def is_running(self) -> bool:
        """True from a successful start until the thread of the run has ended"""
        with self._lock:
            return self._state != IDLE

    def scan(self) -> bool:
        """
        look at the table and plan, nothing moves and the torque is not touched. What was found is in status.
        :return: False when a run (or another scan) is going on, True when the scan was made, whether or not it ended
            in an error: that is in status
        """
        with self._lock:
            if self._state != IDLE or self._busy:
                return False
            self._busy = True
            self._error = None
            self._warnings = []
            self._activity = 'looking at the table'
        try:
            self._scan_without_run()
        finally:
            with self._lock:
                self._busy = False
        return True

    def start(self, hover_only: bool = False) -> bool:
        """
        start a run in a thread of its own
        :param hover_only: a rehearsal: the arm goes to the hover pose of every block and above the tower, without the
            gripper and without going down
        :return: False when a run is going on already
        """
        with self._lock:
            if self._state != IDLE or self._busy:
                return False
            self._state = RUNNING
            self._stop_requested.clear()
            self._error = None
            self._warnings = []
            self._hover_only = hover_only
            self._plan = []
            self._skipped = {}
            self._tower = []
            self._activity = 'starting'
            self._thread = threading.Thread(target=self._run, args=(hover_only,), name='block stacker')
            try:
                self._thread.start()
            except RuntimeError:
                self._state = IDLE
                self._activity = 'ready'
                raise
        return True

    def start_click_mode(self) -> bool:
        """
        start the click mode in a thread of its own: the arm waits at the park pose and puts the blocks that are clicked
        on the tower, one by one, until stop, an error, or idle_timeout seconds without a click
        :return: False when a run, a scan or anything else is going on
        """
        with self._lock:
            if self._state != IDLE or self._busy:
                return False
            self._state = RUNNING
            self._stop_requested.clear()
            self._error = None
            self._warnings = []
            self._hover_only = False
            self._plan = []
            self._skipped = {}
            self._tower = []
            self._click_mode = True
            self._click_message = None
            self._click_waiting = False
            self._pending_click = None
            self._activity = 'starting'
            self._thread = threading.Thread(target=self._run_click_mode, name='block stacker click mode')
            try:
                self._thread.start()
            except RuntimeError:
                self._state = IDLE
                self._click_mode = False
                self._activity = 'ready'
                raise
        return True

    def click(self, marker_id: int) -> bool:
        """
        the user clicked this block in the camera window. It only stores the click, the thread of the click mode takes
        it up, so it is cheap enough for the camera thread.
        :return: True when the click was accepted: the mode waits for a click and has none yet. False when the mode is
            off, the arm is busy with a block, or the mode is stopping.
        """
        with self._lock:
            if (not self._click_mode or self._state != RUNNING or not self._click_waiting
                    or self._pending_click is not None):
                return False
            self._pending_click = marker_id
            return True

    def lay_down(self) -> bool:
        """
        lay the arm down and switch the torque off, in a thread of its own, for the user to bring the arm down after a
        run left the servos holding (see the error of that run)
        :return: False when a run, a scan or another of these is going on
        """
        with self._lock:
            if self._state != IDLE or self._busy:
                return False
            self._state = RUNNING
            self._stop_requested.clear()
            self._error = None
            self._warnings = []
            self._hover_only = False
            self._activity = 'laying the arm down'
            self._thread = threading.Thread(target=self._run_lay_down, name='block stacker lay down')
            try:
                self._thread.start()
            except RuntimeError:
                self._state = IDLE
                self._activity = 'ready'
                raise
        return True

    def disable_servos(self) -> bool:
        """
        switch the torque off right away, wherever the arm is: for the case that it can not be laid down, with the arm
        supported by hand. It sets no error of its own.
        :return: False when a run, a scan or another of these is going on
        """
        with self._lock:
            if self._state != IDLE or self._busy:
                return False
            self._busy = True
        try:
            robot = self._get_robot()
            robot.disable_servos()
            self._servos_enabled = False
            self._set_activity('the servos are switched off')
        except _RunError as e:
            self._add_error(str(e))
        except Exception as e:
            logger.exception('the servos could not be switched off')
            self._add_error('the servos could not be switched off: {}'.format(e))
        finally:
            with self._lock:
                self._busy = False
        return True

    def stop(self) -> bool:
        """
        ask the run to stop: the move that is going on is cut short, the arm is laid down and the torque goes off. It
        returns at once.
        :return: False when no run is going on
        """
        with self._lock:
            if self._state == IDLE:
                return False
            self._state = STOPPING
            self._stop_requested.set()
            return True

    def wait(self, timeout: float | None = None) -> None:
        """wait until the thread of the run has ended, for the tests"""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def read_tip(self) -> dict[str, Any]:
        """
        where the arm says its tip (the end of the closed fingertips) is, next to the nearest block the camera sees:
        hold the gripper on a block with the torque off and compare. 'difference' is the block minus the arm, in x and
        y. Only when no run is going on, and the torque is not touched.
        """
        with self._lock:
            if self._state != IDLE:
                return {'error': 'the arm is running, read the tip when it is idle'}
        try:
            robot = self._get_robot()
        except _RunError as e:
            return {'error': str(e)}
        angles = robot.get_current_angles()
        if angles is None:
            return {'error': 'the angles could not be read'}
        tip = forward_position_kinematics(angles, robot.robot_config)[-1]
        x, y, z = (float(value) for value in tip)
        result: dict[str, Any] = {'tip': [x, y, z], 'nearest_block': None, 'block': None, 'difference': None}
        blocks = self._handler.get_blocks()
        if blocks:
            nearest = min(blocks, key=lambda block: math.hypot(block.x - x, block.y - y))
            result.update(nearest_block=int(nearest.marker_id), block=[float(nearest.x), float(nearest.y)],
                          difference=[float(nearest.x) - x, float(nearest.y) - y])
        return result

    def status(self) -> dict[str, Any]:
        """what the stacker does and found, all of it json serialisable"""
        with self._lock:
            result: dict[str, Any] = {
                'state': self._state, 'activity': self._activity, 'error': self._error,
                'warnings': list(self._warnings), 'hover_only': self._hover_only,
                'blocks': [{'id': int(block.marker_id), 'x': float(block.x), 'y': float(block.y),
                            'yaw': math.degrees(block.yaw), 'frames': int(block.frames), 'stable': bool(block.stable)}
                           for block in self._scanned],
                'plan': [{'id': marker_id, 'level': level} for marker_id, level in self._plan],
                'skipped': {str(marker_id): reason for marker_id, reason in self._skipped.items()},
                'tower': list(self._tower), 'click_mode': self._click_mode, 'click_message': self._click_message}
        result['board_visible'] = bool(self._handler.board_visible())
        return result

    # the state that is shared

    def _set_activity(self, activity: str) -> None:
        with self._lock:
            self._activity = activity
            show = self._click_mode and not self._click_waiting
        if show:
            # in the click mode the window says what the arm does, only the run thread changes what it shows
            self._handler.set_message(activity)

    def _set_click_message(self, message: str) -> None:
        logger.info('click mode: %s', message)
        with self._lock:
            self._click_message = message

    def _add_error(self, error: str) -> None:
        with self._lock:
            self._error = error if self._error is None else '{}; {}'.format(self._error, error)

    def _add_warning(self, warning: str) -> None:
        with self._lock:
            self._warnings.append(warning)

    def _get_robot(self) -> AbstractRobotController:
        with self._robot_lock:
            if self._robot is None:
                try:
                    self._robot = self._robot_provider()
                except Exception as e:
                    logger.exception('the robot could not be started')
                    raise _RunError('the robot could not be started: {}'.format(e)) from e
            return self._robot

    # looking and planning, for a scan and for every round of a run

    def _scan_without_run(self) -> None:
        try:
            robot = self._get_robot()
            blocks = self._handler.scan({}, sleep=self._sleep)
            plan = self._make_plan(robot, blocks, [], {})
        except (ScanError, StackingError, _RunError) as e:
            with self._lock:
                self._plan = []
                self._skipped = {}
                self._error = str(e)
                self._activity = 'the scan failed'
            return
        except Exception as e:
            logger.exception('the scan failed')
            with self._lock:
                self._error = 'the scan failed: {}'.format(e)
                self._activity = 'the scan failed'
            return
        with self._lock:
            self._activity = 'looked at the table: {} found, {} can be stacked'.format(
                _plural(len(blocks), 'block'), len(plan.steps))

    def _make_plan(self, robot: AbstractRobotController, blocks: list[DetectedBlock], tower: list[int],
                   given_up: Mapping[int, str]) -> StackingPlan:
        """
        plan with the blocks that are steady, not in the tower and not given up on, and show the blocks, the plan and
        why blocks are left out
        :raises StackingError: if the plan is impossible
        """
        skipped: dict[int, str] = {}
        usable: list[DetectedBlock] = []
        for block in blocks:
            if block.marker_id in tower:
                continue
            if block.marker_id in given_up:
                skipped[block.marker_id] = given_up[block.marker_id]
            elif not block.stable:
                skipped[block.marker_id] = NOT_STEADY
            else:
                usable.append(block)
        with self._lock:
            self._scanned = list(blocks)
        plan = plan_stack(usable, self._config, robot, start_level=len(tower))
        skipped.update(plan.skipped)
        with self._lock:
            self._plan = [(step.marker_id, step.level) for step in plan.steps]
            self._skipped = skipped
        return plan

    # the thread of the run

    def _run(self, hover_only: bool) -> None:
        self._run_thread(lambda robot: self._run_rounds(robot, hover_only), hover_only)

    def _run_click_mode(self) -> None:
        """the thread of start_click_mode"""
        self._run_thread(self._click_rounds, False, self._clear_window)

    def _run_thread(self, rounds: Callable[[AbstractRobotController], str], hover_only: bool,
                    on_end: Callable[[], None] | None = None) -> None:
        """
        the thread of every kind of run: whatever ends it, the arm is laid down
        :param rounds: the work of the run, returns the activity of the end when it is done
        :param on_end: called when the arm is laid down, before the run counts as ended
        """
        outcome = 'ended with an error'
        robot: AbstractRobotController | None = None
        self._angles = None
        self._park_flip = None
        self._servos_enabled = False
        self._held = None
        self._tries = {}
        self._given_up = {}
        try:
            robot = self._get_robot()
            outcome = rounds(robot)
        except _Stopped:
            outcome = 'stopped'
        except _RunError as e:
            self._add_error(str(e))
        except Exception as e:
            logger.exception('the block stacking failed')
            self._add_error('unexpected error: {}: {}'.format(type(e).__name__, e))
        if robot is not None and self._servos_enabled and not self._lay_down_safely(robot, hover_only):
            outcome = NOT_LAID_DOWN
        if on_end is not None:
            try:
                on_end()
            except Exception:
                logger.exception('the end of the run could not be shown')
        with self._lock:
            self._activity = outcome
            self._click_mode = False
            self._click_waiting = False
            self._pending_click = None
            self._state = IDLE

    def _clear_window(self) -> None:
        self._handler.set_selected(None)
        self._handler.set_message(None)

    def _run_lay_down(self) -> None:
        """the thread of lay_down"""
        outcome = NOT_LAID_DOWN
        try:
            robot = self._get_robot()
        except _RunError as e:
            self._add_error(str(e))
        else:
            if self._lay_down_safely(robot, False):
                outcome = 'the arm is laid down and the servos are switched off'
        with self._lock:
            self._activity = outcome
            self._state = IDLE

    def _lay_down_safely(self, robot: AbstractRobotController, hover_only: bool) -> bool:
        """_lay_down, which does not raise"""
        try:
            return self._lay_down(robot, hover_only)
        except Exception as e:
            logger.exception('laying the arm down failed')
            self._add_error('laying the arm down failed ({}: {}), the servos are still holding it, {} lowers it'
                            .format(type(e).__name__, e, LOWER_ARM_COMMAND))
            return False

    def _run_rounds(self, robot: AbstractRobotController, hover_only: bool) -> str:
        """the run itself, returns the activity of the end when it is done"""
        self._start_arm(robot, hover_only)

        while True:
            self._check_stop()
            self._check_health(robot)
            blocks = self._look(robot)
            self._check_tower(blocks)
            with self._lock:
                tower = list(self._tower)
            try:
                plan = self._make_plan(robot, blocks, tower, self._given_up)
            except StackingError as e:
                with self._lock:
                    self._plan = []
                raise _RunError(str(e)) from e
            if hover_only:
                self._rehearse(robot, plan)
                return 'done: rehearsal of {}'.format(_plural(len(plan.steps), 'block'))
            if not plan.steps:
                return 'done: {} stacked'.format(_plural(len(tower), 'block'))
            self._check_stop()
            if plan.flip != self._park_flip:
                self._move_to_park(robot, plan.flip)
            self._stack_block(robot, plan.steps[0], plan.flip)

    def _start_arm(self, robot: AbstractRobotController, hover_only: bool) -> None:
        """switch the servos on, open the gripper and move to the park pose of the flip the planner prefers"""
        self._set_activity('switching the servos on')
        if robot.enable_servos() is False:
            raise _RunError('the servos could not be switched on, nothing was moved')
        self._servos_enabled = True
        if not hover_only:
            self._set_activity('opening the gripper')
            if not self._open_gripper(robot):
                logger.warning('the gripper did not confirm that it is open')
        self._check_stop()
        angles = robot.get_current_angles()
        if angles is None:
            raise _RunError('the angles of the arm could not be read')
        self._angles = angles
        # the planner picks flip=True on this arm, and the arm at rest is in the picture of the camera, so this move
        # comes before the first plan; the plan moves on to the park pose of the other flip when it chooses that one
        self._move_to_park(robot, True)

    # the click mode

    def _click_rounds(self, robot: AbstractRobotController) -> str:
        """wait for a click and put that block on the tower, again and again; returns the activity of the end"""
        self._start_arm(robot, False)
        while True:
            marker_id = self._wait_for_click(robot)
            if marker_id is None:
                return 'the arm was laid down after {:g} seconds without a click'.format(
                    self._config.click_mode.idle_timeout)
            self._handler.set_selected(marker_id)
            try:
                self._handle_click(robot, marker_id)
            finally:
                self._handler.set_selected(None)

    def _wait_for_click(self, robot: AbstractRobotController) -> int | None:
        """
        wait at the park pose until a click comes, checking the stop request often and the servos now and then
        :return: the marker id that was clicked, None when idle_timeout seconds passed without one
        """
        started = self._clock()
        next_health = started
        with self._lock:
            self._click_waiting = True
            self._activity = 'waiting for a click'
            message = self._click_message
        self._handler.set_message(CLICK_PROMPT if message is None else '{} - {}'.format(message, CLICK_PROMPT))
        try:
            while True:
                self._check_stop()
                now = self._clock()
                if now >= next_health:
                    self._check_health(robot)
                    next_health = now + CLICK_HEALTH_INTERVAL
                with self._lock:
                    clicked = self._pending_click
                    if clicked is not None or now - started >= self._config.click_mode.idle_timeout:
                        # no click is accepted from here on, so none gets lost
                        self._pending_click = None
                        self._click_waiting = False
                        return clicked
                self._sleep(CLICK_POLL_INTERVAL)
        finally:
            with self._lock:
                self._click_waiting = False

    def _handle_click(self, robot: AbstractRobotController, marker_id: int) -> None:
        """
        put the clicked block on the tower, or say in the click message why not. Only what is wrong with the arm or the
        tower ends the mode, what is wrong with the block does not.
        """
        self._check_health(robot)
        try:
            blocks = self._look_at_table(robot)
        except ScanError as e:
            self._set_click_message(str(e))
            return
        with self._lock:
            self._scanned = list(blocks)
        self._check_tower(blocks)
        with self._lock:
            tower = list(self._tower)
        block = next((candidate for candidate in blocks if candidate.marker_id == marker_id), None)
        if marker_id in tower:
            self._set_click_message('block {} is in the tower already'.format(marker_id))
            return
        if block is None:
            self._set_click_message('block {} is not in the picture'.format(marker_id))
            return
        if not block.stable:
            self._set_click_message('block {} can not be taken: {}'.format(marker_id, NOT_STEADY))
            return
        # every other block on the table is an obstacle to the fingers, steady or not
        others = [other for other in blocks if other.marker_id != marker_id and other.marker_id not in tower]
        planned = plan_one(block, others, self._config, robot, len(tower), preferred_flip=self._park_flip is not False)
        if isinstance(planned, str):
            self._set_click_message('block {} can not be taken: {}'.format(marker_id, planned))
            return
        step, flip = planned
        self._check_stop()
        if flip != self._park_flip:
            self._move_to_park(robot, flip)
        # a block is never given up on here, the user may click it again, so every click starts with no tries
        self._tries = {}
        self._given_up = {}
        self._stack_block(robot, step, flip)
        with self._lock:
            on_tower = marker_id in self._tower
        if on_tower:
            self._set_click_message('block {} is on the tower, level {}'.format(marker_id, step.level))
        else:
            reasons = self._tries.get(marker_id)
            self._set_click_message('block {} was not taken: {}'.format(
                marker_id, reasons[-1] if reasons else 'for an unknown reason'))

    def _should_stop(self) -> bool:
        return self._stop_requested.is_set() or not self._main_thread_alive()

    def _check_stop(self) -> None:
        if self._should_stop():
            raise _Stopped()

    def _check_health(self, robot: AbstractRobotController) -> None:
        errors = robot.get_hardware_errors()
        if errors is None:
            raise _RunError('the hardware status of the servos could not be read')
        if errors:
            raise _RunError('servos {} have a hardware error: {}'.format(
                ', '.join(str(servo) for servo in sorted(errors)),
                '; '.join('{}: {}'.format(servo, ', '.join(names)) for servo, names in sorted(errors.items()))))
        overheating = robot.get_overheating_servos()
        if overheating is None:
            raise _RunError('the temperatures of the servos could not be read')
        if overheating:
            raise _RunError('servos {} are too warm ({}), let them cool down'.format(
                ', '.join(str(servo) for servo in sorted(overheating)),
                ', '.join('{}: {} C'.format(servo, temperature) for servo, temperature in sorted(overheating.items()))))

    def _look(self, robot: AbstractRobotController) -> list[DetectedBlock]:
        """the blocks the camera sees from the park pose, the blocks in the tower at the height they have there"""
        try:
            return self._look_at_table(robot)
        except ScanError as e:
            raise _RunError(str(e)) from e

    def _look_at_table(self, robot: AbstractRobotController) -> list[DetectedBlock]:
        """_look, but a scan that fails raises the ScanError itself"""
        self._set_activity('looking at the table')
        self._sleep(SCAN_SETTLE_TIME)
        with self._lock:
            heights = {marker_id: (level + 1) * self._config.blocks.size for level, marker_id in enumerate(self._tower)}
        return self._handler.scan(heights, sleep=self._sleep)

    def _check_tower(self, blocks: list[DetectedBlock]) -> None:
        """the camera only checks the tower, the blocks are always put down at its configured centre"""
        with self._lock:
            if not self._tower:
                return
            top = self._tower[-1]
        tower = self._config.tower
        for block in blocks:
            if block.marker_id == top and math.hypot(block.x - tower.x, block.y - tower.y) <= TOWER_TOLERANCE:
                return
        raise _RunError('the tower is not standing where it should: block {} is not on top'.format(top))

    def _failed_try(self, marker_id: int, reason: str) -> None:
        tries = self._tries.setdefault(marker_id, [])
        tries.append(reason)
        logger.warning('block %s: %s (try %s of %s)', marker_id, reason, len(tries), MAX_TRIES)
        if len(tries) >= MAX_TRIES:
            if all(previous == reason for previous in tries):
                self._given_up[marker_id] = '{} twice'.format(reason)
            else:
                self._given_up[marker_id] = 'it could not be taken in {} tries: {}'.format(MAX_TRIES, '; '.join(tries))

    # one block

    def _stack_block(self, robot: AbstractRobotController, step: BlockStep, flip: bool) -> None:
        config = self._config
        block, level, pick, place = step.marker_id, step.level, step.pick, step.place
        hover = hover_pose(pick, config)
        at_pick = pick_pose(pick, config)
        lift = lift_pose(pick, config, level)
        above = above_tower_pose(place, config, level)
        at_place = place_pose(place, config, level)

        self._set_activity('taking block {}'.format(block))
        try:
            self._move_joints(robot, hover)
            self._move_line(robot, hover, at_pick)
        except MovementException as e:
            # the position is fresh from the camera, so it can turn out out of reach; the gripper holds nothing
            self._failed_try(block, 'its position can not be reached: {}'.format(e))
            self._move_to_park(robot, flip)
            return

        if not self._close_gripper(robot):
            self._set_activity('the gripper closed on nothing')
            self._open_gripper(robot)
            try:
                self._move_line(robot, at_pick, hover)
            except MovementException as e:
                logger.warning('can not move up from block %s: %s', block, e)
            self._failed_try(block, 'the gripper closed on nothing')
            self._move_to_park(robot, flip)
            return
        self._held = block
        self._check_stop()

        try:
            self._set_activity('lifting block {}'.format(block))
            self._move_line(robot, at_pick, lift)
            if not self._holds_block(robot):
                self._held = None
                self._set_activity('block {} slipped out of the gripper'.format(block))
                self._open_gripper(robot)
                self._failed_try(block, 'the block slipped out of the gripper while it was lifted')
                self._move_to_park(robot, flip)
                return
            self._set_activity('carrying block {} to the tower'.format(block))
            self._carry_to(robot, above, level)
            self._set_activity('putting block {} on level {}'.format(block, level))
            self._move_line(robot, above, at_place)
        except MovementException as e:
            raise _RunError('block {} can not be carried to the tower: {}'.format(block, e)) from e

        if not self._open_gripper(robot):
            raise _RunError('the gripper did not open over the tower, block {} may still be in it'.format(block))
        self._held = None
        with self._lock:
            self._tower.append(block)
        self._set_activity('moving away from the tower')
        try:
            self._move_line(robot, at_place, above)
        except MovementException as e:
            raise _RunError('the arm can not move up from the tower: {}'.format(e)) from e
        self._move_to_park(robot, flip)

    def _rehearse(self, robot: AbstractRobotController, plan: StackingPlan) -> None:
        """the arm goes over every block of the plan and over the tower, without gripper and without going down"""
        config = self._config
        if plan.flip != self._park_flip:
            self._move_to_park(robot, plan.flip)
        for step in plan.steps:
            self._check_stop()
            self._set_activity('rehearsing: over block {}'.format(step.marker_id))
            hover = hover_pose(step.pick, config)
            self._move_joints(robot, hover)
            self._sleep(REHEARSAL_PAUSE)
            self._check_stop()
            self._set_activity('rehearsing: over the tower, level {}'.format(step.level))
            # there is no block in the gripper, but the arm must not come down near the blocks on the way either
            self._carry_to(robot, above_tower_pose(step.place, config, step.level), step.level,
                           lowest=config.motion.hover_height - CARRY_MARGIN)
            self._sleep(REHEARSAL_PAUSE)
        self._move_to_park(robot, plan.flip)

    # moving, every move is planned from where the previous one ended and executed when it is checked

    def _angles_now(self) -> np.ndarray:
        if self._angles is None:
            raise _RunError('the angles of the arm are not known')
        return self._angles

    def _execute_move(self, robot: AbstractRobotController, move: PlannedMove) -> None:
        logger.info('moving: %s', move)
        if not self._execute(robot, move, self._should_stop):
            raise _Stopped()
        self._angles = move.angles[-1].copy()

    def _plan_joints(self, robot: AbstractRobotController, pose: Pose) -> PlannedMove:
        return plan_move_to_angles(robot, self._angles_now(), _angles_of(robot, pose),
                                   angular_speed=self._config.motion.joint_speed)

    def _move_joints(self, robot: AbstractRobotController, pose: Pose) -> None:
        self._execute_move(robot, self._plan_joints(robot, pose))

    def _move_line(self, robot: AbstractRobotController, start: Pose, stop: Pose) -> None:
        # both ends have to be reachable the way the planner checked them, and the line starts where the arm is
        move = plan_straight_line(robot, start, stop, self._config.motion.linear_speed,
                                  current_angles=self._angles_now())
        self._execute_move(robot, move)

    def _carry_to(self, robot: AbstractRobotController, above: Pose, level: int, lowest: float | None = None) -> None:
        """
        move in joint space to the pose above the tower, only when the way there stays high enough: a carried block
        would hit the tower when the joints take a curve that dips
        :param lowest: the lowest z of the tip on the way, by default just under the travel height of the level
        :raises _RunError: if the move dips below it
        """
        move = self._plan_joints(robot, above)
        if lowest is None:
            lowest = travel_height(self._config, level) - CARRY_MARGIN
        dip = min_tip_height(move, robot.robot_config)
        if dip < lowest:
            raise _RunError('the arm would go down to {:.1f}cm on its way over to the tower, higher than {:.1f}cm is '
                            'needed there: the block could hit the tower'.format(dip, lowest))
        self._execute_move(robot, move)

    def _move_to_park(self, robot: AbstractRobotController, flip: bool) -> None:
        """to the park pose of the flip in joint space, where the camera sees the table"""
        self._set_activity('moving to where the camera looks')
        try:
            self._move_joints(robot, park_pose(self._config, flip))
        except MovementException as e:
            raise _RunError('the arm can not move to the park pose: {}'.format(e)) from e
        self._park_flip = flip

    # the end of every run

    def _lay_down(self, robot: AbstractRobotController, hover_only: bool) -> bool:
        """
        lay the arm down on the table, let go of what it holds and switch the torque off: the torque only goes off
        after a move that was completed, never with the arm in the air. None of the moves can be stopped.
        :return: False when the arm could not be laid down and the servos are left holding
        """
        self._set_activity('laying the arm down')
        with self._lock:
            tower = list(self._tower)
            lying = [block for block in self._scanned if block.marker_id not in tower and block.marker_id != self._held]
        for block in lying:
            if in_rest_area(block, self._config, (REST_POSE.x, REST_POSE.y)):
                self._add_warning('block {} lies where the arm comes down'.format(block.marker_id))
        # like lower_to_rest: a servo with a hardware error is limp and the others hold its share of the weight, so
        # moving the arm could make it fall
        errors = robot.get_hardware_errors()
        if errors is None:
            self._add_error('the hardware status of the servos could not be read, so the arm is not moved and the '
                            'servos are left as they are: if one is limp the others still hold the arm, support it '
                            'before the torque is switched off')
            return False
        if errors:
            self._add_error('servos {} have a hardware error and are limp, the others still hold the arm: the arm is '
                            'not moved and the servos are left as they are. Support the arm before the torque is '
                            'switched off'.format(', '.join(str(servo) for servo in sorted(errors))))
            return False
        try:
            angles = robot.get_current_angles()
            if angles is None:
                raise MovementException('the angles of the arm could not be read')
            self._angles = self._move_to_rest(robot, angles)
        except MovementException as e:
            self._add_error('the arm can not be laid down ({}), the servos are still holding it, {} lowers it'
                            .format(e, LOWER_ARM_COMMAND))
            return False
        self._sleep(REST_SETTLE_TIME)
        if not hover_only:
            try:
                self._open_gripper(robot)
            except Exception:
                logger.exception('the gripper could not be opened at the rest pose')
        try:
            robot.disable_servos()
        except Exception as e:
            logger.exception('the servos could not be switched off')
            self._add_error('the servos could not be switched off: {}'.format(e))
        self._servos_enabled = False
        return True

    def _move_to_rest(self, robot: AbstractRobotController, angles: np.ndarray) -> np.ndarray:
        """
        the joint move from the angles to the rest pose; when the planner refuses it (from the height of a pick the
        tip would dip below the lowest height the workspace allows) the way round: straight up, to the park pose, and
        from there to the rest pose
        :return: the angles the arm ends at
        :raises MovementException: if the arm can not be brought to the rest pose, or a move did not finish
        """
        motion = self._config.motion
        target = _angles_of(robot, REST_POSE)
        try:
            direct = plan_move_to_angles(robot, angles, target, angular_speed=motion.joint_speed)
        except MovementException as refused:
            direct_error = str(refused)
            logger.warning('the arm can not be laid down in one move (%s), going up first', refused)
        else:
            return self._execute_unstoppable(robot, direct)
        try:
            pose = forward_kinematics(angles, robot.robot_config)
            height = max(pose.z, motion.hover_height)
            if height > pose.z + 1e-6:
                up = Pose(pose.x, pose.y, height, flip=pose.flip, alpha=pose.alpha, beta=pose.beta, gamma=pose.gamma)
                angles = self._execute_unstoppable(robot, plan_straight_line(
                    robot, pose, up, motion.linear_speed, current_angles=angles))
            park = _angles_of(robot, park_pose(self._config, pose.flip))
            angles = self._execute_unstoppable(
                robot, plan_move_to_angles(robot, angles, park, angular_speed=motion.joint_speed))
            return self._execute_unstoppable(
                robot, plan_move_to_angles(robot, angles, target, angular_speed=motion.joint_speed))
        except MovementException as e:
            raise MovementException('{}; the direct move was refused: {}'.format(e, direct_error)) from e

    def _execute_unstoppable(self, robot: AbstractRobotController, move: PlannedMove) -> np.ndarray:
        """execute a move that stop does not cut short, returns the angles it ends at"""
        logger.info('moving: %s', move)
        if not self._execute(robot, move, None):
            raise MovementException('a move of laying the arm down did not finish')
        return np.asarray(move.angles[-1]).copy()

    # the gripper: set_gripper returns at once and the servo moves by itself, so the state is read to see what it did

    def _positions_per_state(self, robot: AbstractRobotController) -> float | None:
        """the positions of one state, None for a robot without a gripper servo"""
        positions = robot.gripper_positions_per_state()
        if positions is None:
            return None
        return positions if positions > 0 else DEFAULT_POSITIONS_PER_STATE

    def _wait_for_gripper(self, robot: AbstractRobotController,
                          reached: Callable[[float], bool] | None = None) -> float | None:
        """
        read the state of the gripper until it has stopped changing (less than GRIPPER_STEADY_CHANGE for
        gripper.settle_time), reached(state) is true or gripper.timeout has passed. A read that fails is read again.
        :return: the last state that was read, None when it never could be read
        """
        gripper = self._config.gripper
        started = self._clock()
        reference: float | None = None
        steady_since = started
        latest: float | None = None
        while True:
            state = robot.get_gripper_state()
            now = self._clock()
            if state is not None:
                latest = state
                if reached is not None and reached(state):
                    return state
                if reference is None or abs(state - reference) >= GRIPPER_STEADY_CHANGE:
                    reference, steady_since = state, now
                elif now - steady_since >= gripper.settle_time:
                    return state
            if now - started >= gripper.timeout:
                logger.warning('the gripper did not stop moving within %s seconds, its state is %s',
                               gripper.timeout, latest)
                return latest
            self._sleep(GRIPPER_POLL_INTERVAL)

    def _close_gripper(self, robot: AbstractRobotController) -> bool:
        """
        close the gripper on the block and tell whether it holds one. The target is pulled back to just beyond where the
        block stopped the gripper: the grip is as firm as it gets from about 65 positions of lead, and a target that is
        not further away lets go quickly later.
        """
        gripper = self._config.gripper
        positions_per_state = self._positions_per_state(robot)
        self._set_activity('closing the gripper')
        robot.set_gripper(gripper.closed_state)
        if positions_per_state is None:
            self._sleep(gripper.settle_time)
            return True
        state = self._wait_for_gripper(robot)
        if state is None:
            logger.warning('the state of the gripper could not be read, counting it as holding nothing')
            return False
        if state <= gripper.open_state + GRIPPER_NOT_CLOSED_WITHIN:
            # a gripper that closes on nothing goes all the way, so one that stays open does not move at all
            raise _RunError('the gripper did not close')
        if state >= gripper.empty_state:
            return False
        robot.set_gripper(min(state + gripper.max_lead_positions / positions_per_state, gripper.closed_state))
        return True

    def _holds_block(self, robot: AbstractRobotController) -> bool:
        """whether the gripper still holds a block: a fresh reading of its state"""
        if self._positions_per_state(robot) is None:
            return True
        started = self._clock()
        while True:
            state = robot.get_gripper_state()
            if state is not None:
                return state < self._config.gripper.empty_state
            if self._clock() - started >= self._config.gripper.timeout:
                logger.warning('the state of the gripper could not be read, counting it as holding nothing')
                return False
            self._sleep(GRIPPER_POLL_INTERVAL)

    def _open_gripper(self, robot: AbstractRobotController) -> bool:
        """open the gripper and wait until it is open, returns False when it did not get there"""
        gripper = self._config.gripper
        self._set_activity('opening the gripper')
        robot.set_gripper(gripper.open_state)
        if self._positions_per_state(robot) is None:
            self._sleep(gripper.settle_time)
            return True
        state = self._wait_for_gripper(robot, lambda value: abs(value - gripper.open_state) <= GRIPPER_OPEN_TOLERANCE)
        return state is not None and abs(state - gripper.open_state) <= GRIPPER_OPEN_TOLERANCE
