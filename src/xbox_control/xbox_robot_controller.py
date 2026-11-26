from __future__ import division, annotations

import json
import logging
import math
import queue
import re
import threading
from copy import copy
from pathlib import Path
from typing import TYPE_CHECKING

import jsonpickle
import numpy as np
from src.reinforcementlearning.environment.scenario import medium_scenarios

import src.global_constants
from src.kinematics.kinematics import forward_kinematics, forward_position_kinematics
from src.kinematics.kinematics_utils import Pose
from src.utils.movement import plan_move_to_pose, execute_move, check_move, move_between_poses, PlannedMove
from src.utils.movement import MAX_POSITION_ERROR, Movement
from src.utils.decorators import synchronized_with_lock
from src.utils.linalg_utils import get_center
from src.utils.movement import PoseToPoseMovement, SplineMovement
from src.utils.movement_exception import MovementException
from src.utils import workspace
from src.utils.os_utils import get_project_root
from src.xbox_control.xbox360controller.xbox_pose_updater import MAX_STEP_TIME
from time import sleep, perf_counter

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.kinematics.kinematics_utils import RobotConfig
    from src.robot_controllers.abstract_robot_controller import AbstractRobotController
    from src.robot_controllers.combined_robot.combined_robot import CombinedRobot
    from src.robot_controllers.simulated_robot.simulated_robot_controller import SimulatedRobotController
    from src.xbox_control.xbox360controller.xbox_pose_updater import XboxPoseUpdater

logger = logging.getLogger(__name__)

# how often to check if any of the servos shut down because of a hardware error
HARDWARE_CHECK_INTERVAL = 1.0  # seconds
# the range and step size of the maximum speed of the end effector that the d-pad changes, in cm/s
MINIMUM_SPEED = 5
MAXIMUM_SPEED = 50
MAXIMUM_SPEED_STEP = 5
# how much the gripper state (0 to 100) changes every step while a bumper is held
GRIPPER_STEP = 1
# how far the target of the gripper can be beyond its actual state, in servo positions. The lead decides how hard the
# gripper squeezes an object (at most with the goal current of the servo) and how long it takes to let go: when the
# user opens, the target first has to come back through the whole lead. Measured on 25 November 2026 with
# gripper_test.py --measure-lead: at P 600 the current grows by 2.39 mA per position between the target and the
# gripper, and reaches the whole goal current of 150 mA from about 65 positions; more lead adds no grip, only delay.
# 85 is 1.25 times that. Measure again when the goal current or the P gain of the gripper changes: with a larger goal
# current this lead would cap the squeeze (85 positions give about 200 mA). It is in positions and not in states,
# because the states change their meaning whenever the gripper is calibrated again and the positions do not
GRIPPER_MAX_LEAD_POSITIONS = 85
# the positions of one gripper state when the robot can not tell (the simulation): the gripper of the arm has 951
# positions for its 100 states
DEFAULT_GRIPPER_POSITIONS_PER_STATE = 9.5
# a stick step that is too fast is slowed down to this fraction of recommended_max_servo_speed, below the speed check
SLOW_DOWN_MARGIN = 0.95
# the a button flips the wrist with the wrist centre at least this high above the table. Halfway the flip the wrist is
# straight and the gripper swings down along the line of the forearm, around the wrist centre. The gripper is 15 cm
# long, so with the wrist centre 15 cm up it clears the table. Below this height the a button first raises the arm
# straight up to it, and then flips; the arm stays there. Adjust this value by hand: more is safer, less allows the
# flip closer to the table
FLIP_MIN_WRIST_HEIGHT = 15.0  # cm
# the speed of the joint that turns the most while the wrist turns over after the a button, a half turn takes about 2
# seconds. The general limit of the steps of the gamepad (SLOW_DOWN_MARGIN of recommended_max_servo_speed) stays what it
# is: it still holds when this is set above it, and everything else that walks the angles keeps it. Adjust this value
# by hand: less is gentler for the motors, more is quicker
FLIP_JOINT_SPEED = 1.5  # rad/s
# set to False the gamepad sends its steps without velocities, as before 21 November 2026, to compare by hand how the
# arm follows the sticks with and without the velocity lead of the servos
SEND_VELOCITIES = True
# recorded moves are saved to and restored from this directory, next to src, by a name of letters, digits, - and _
RECORDED_MOVES_DIR = get_project_root().parent / 'recorded_moves'
RECORDED_MOVES_NAME = re.compile(r'[A-Za-z0-9_-]{1,100}')


class XboxRobotController:
    """
    Control the robot with an xbox controller.

    Only the control loop thread moves the robot. Requests from other threads that move the robot, like starting the
    find center mode, are queued as commands and handled by the control loop.
    """
    start_pose = Pose(30, 22.0, 4)

    def __init__(self, dynamixel_robot_config: RobotConfig, robot: SimulatedRobotController | CombinedRobot,
                 pose_updater: XboxPoseUpdater) -> None:
        self.pose_updater = pose_updater
        self.dynamixel_robot_config = dynamixel_robot_config
        self.robot = robot
        # guards the state that is changed by both the control loop and the api
        self.lock = threading.RLock()
        # makes sure start and stop don't run at the same time
        self._start_stop_lock = threading.Lock()
        # set when the control loop should stop, an event so stopping doesn't wait for the lock
        self._stop_requested = threading.Event()
        # functions that the control loop calls, so only the control loop moves the robot
        self._commands: queue.Queue[Callable[[], None]] = queue.Queue()
        self._crashed = False
        self._start_move: PlannedMove | None = None
        self._step_skipped = False  # the gamepad asks for a step that fails a check, the robot holds
        # the current pose the a button turned the wrist over for, at FLIP_JOINT_SPEED, until the angles arrive at it.
        # It is the pose object itself: a pose that some other part sets (a reset, a move, a start) is another object,
        # so nothing else is ever slowed by accident
        self._flip_turn_pose: Pose | None = None
        # the pose, the angles and the velocities of the last step of the sticks, what the control loop sends. The
        # velocities are in rad/s like the angles, indexed by servo id, index 0 unused. The step belongs to the
        # current pose only while the pose is this very object, a pose some other part set is not a step
        self._step: tuple[Pose, np.ndarray, np.ndarray] | None = None
        self.recorded_positions: list[Pose] = []
        self.current_pose: Pose = None  # type: ignore[assignment]  # set by start before the control loop uses it
        self.thread: threading.Thread | None = None
        self.find_center_mode = False
        # 3d position in front of the robot to keep the end effector pointed towards
        # (i.e. make sure the camera stays focused on an object)
        self.center: np.ndarray | None = None
        self.move_speed = 10
        self.recorded_moves: list[Movement] = []
        self.gripper_state: float = 0
        self.current_scenario_id: int | None = None
        # dependency inject?
        self.scenarios = medium_scenarios
        # the action of each button, the pressed buttons of a read are all handled, in this order
        self._button_actions = {
            'start': self._play_back,
            'lb': self._decrease_gripper_state,
            'rb': self._increase_gripper_state,
            'b': self._reset_orientation,
            'a': self._toggle_flip,
            'y': self._record_position,
            'x': self._create_move,
            'pad_up': self._increase_maximum_speed,
            'pad_down': self._decrease_maximum_speed,
        }

    def is_running(self) -> bool:
        """
        True from a successful start until stop. This includes a control loop that stopped because of an error,
        the servos keep holding their position until stop disables them.
        """
        return self.thread is not None

    def is_done(self) -> bool:
        return self._stop_requested.is_set()

    def start(self) -> bool:
        """
        Start controlling the robot. The robot first moves in a straight line to the start pose, the robot doesn't
        start if that movement is not possible or there is no gamepad.
        :return: True if the robot started
        """
        with self._start_stop_lock:
            if self.thread is not None:
                return False

            if not self.pose_updater.controller.wait_until_connected():
                logger.error('no gamepad connected, not starting the robot')
                return False

            # plan before enabling the servos, so the robot doesn't do anything when it can't reach the start pose
            start_move = self._plan_move_to(self.start_pose)
            if start_move is None:
                logger.error('can not move to the start pose, not starting the robot')
                return False

            if self.robot.enable_servos() is False:
                logger.error('failed to enable the servos, not starting the robot')
                return False

            # the arm could have been moved by hand while planning
            current_angles = self.robot.get_current_angles()
            try:
                if current_angles is None:
                    raise MovementException('failed to read the current angles')
                check_move(start_move, self.robot, current_angles=current_angles)
            except MovementException as e:
                logger.error('not starting the robot: %s', e)
                self.robot.disable_servos()
                return False

            self._stop_requested.clear()
            self._crashed = False
            self._flip_turn_pose = None
            # the gripper opens after the start move, whatever the target was when the previous run stopped
            self.gripper_state = 0
            self._clear_commands()
            self._start_move = start_move
            # the wrist keeps its flip, so this is the start pose with the flip of the robot
            self.current_pose = forward_kinematics(start_move.angles[-1], self.robot.robot_config)
            self.thread = threading.Thread(target=self.__start_internal, args=())
            self.thread.start()
            return True

    def stop(self) -> bool:
        """
        Stop controlling the robot, it moves back to the start pose and the servos are disabled.
        :return: False if the robot wasn't running
        """
        with self._start_stop_lock:
            if self.thread is None:
                return False
            self._stop_requested.set()
            self.thread.join()
            if self._crashed:
                # the control loop didn't get the chance to move back to the start pose and disable the servos
                self.robot.disable_servos()
            self.thread = None
            self._flip_turn_pose = None
            self._clear_commands()
            return True

    def request_find_center_mode(self) -> bool:
        """
        Reset the orientation of the end effector and record the next 2 positions, the end effector is then kept
        pointed towards the center between them.
        :return: False if the robot isn't running
        """
        if not self.is_running():
            return False
        self._commands.put(self.__start_find_center_mode)
        return True

    # Record 2 poses to define the center the end effector should be pointing towards
    def __start_find_center_mode(self) -> None:
        with self.lock:
            self.recorded_positions = []
            self.center = None
            self.find_center_mode = True
        self._reset_orientation()

    @synchronized_with_lock("lock")
    def clear_center(self) -> None:
        self.center = None

    @synchronized_with_lock("lock")
    def set_profile_velocity_percentage(self, percentage: float) -> None:
        self.robot.set_profile_velocity_percentage(percentage)

    def __start_internal(self) -> None:
        try:
            self.__control_robot()
        except Exception:
            self._crashed = True
            # don't try to move the robot back, whatever failed could fail again while moving
            logger.exception('controlling the robot failed, the servos keep holding their last position. '
                             'Stop to disable them')

    def __control_robot(self) -> None:
        # the start move can be stopped halfway, the control loop then stops right away and moves back
        if execute_move(self.robot, self._start_move, should_stop=self.is_done):  # type: ignore[arg-type]  # none until start
            self.open_gripper()

        pose_update_sleep_time = self.pose_updater.dt
        previous_step_time = perf_counter()
        next_hardware_check = previous_step_time + HARDWARE_CHECK_INTERVAL
        # the start of the next pass, like execute_move the passes follow the clock instead of sleeping a fixed time
        # after the work, so the goals reach the servos evenly spaced whatever a pass costs
        next_pass = previous_step_time

        while True:
            if self.is_done():
                break

            self.__handle_commands()

            # the step covers the time that really passed, so the speeds of the sticks hold at any loop pace
            now = perf_counter()
            step_time = now - previous_step_time
            new_pose = self.pose_updater.get_updated_pose_from_controller(self.current_pose, self.find_center_mode,
                                                                          self.center, dt=step_time)
            previous_step_time = now
            self._take_step(new_pose, step_time)
            self._send_current_pose()

            # after the send: reading the bus takes several ms, a failed read 10 to 20 ms, which must come out of the
            # waiting time and not delay the goal. a hardware error raises, so at most this pass sent its goal
            if perf_counter() > next_hardware_check:
                self.check_hardware_errors()
                next_hardware_check = perf_counter() + HARDWARE_CHECK_INTERVAL

            self.handle_buttons()

            next_pass += pose_update_sleep_time
            time_to_sleep = next_pass - perf_counter()
            if time_to_sleep > 0:
                sleep(time_to_sleep)
            else:
                # a pass that overran doesn't make the next ones hurry, the step of the gamepad covers the time
                # that really passed
                next_pass = perf_counter()

        self.stop_robot()

    def _take_step(self, new_pose: Pose, step_time: float) -> None:
        """
        the pose the gamepad asks for becomes the current pose when the step to it is safe (see _choose_step),
        otherwise the robot holds the current pose: logged as a warning when the holding starts, and at debug level
        while it lasts.

        The velocities of a taken step are the change of the angles of the step divided by step_time, the time that
        really passed, so also when a long pass limited the size of the step: the average speed over the time that
        passed is what the arm does. A held pose gets zero velocities, so the velocity lead of the servos comes down
        at its rate limit instead of dropping at once.
        :param step_time: the seconds that passed since the previous step, 0 or less gives zero velocities
        """
        start_angles = self._step_angles()
        try:
            self.current_pose, angles = self._choose_step(new_pose, step_time)
        except MovementException as e:
            if not self._step_skipped:
                logger.warning('holding the pose, skipping the step the gamepad asks for: %s', e)
            else:
                logger.debug('holding the pose, skipping the step the gamepad asks for: %s', e)
            self._step_skipped = True
            if self._step is not None and self._step[0] is self.current_pose:
                self._step = (self._step[0], self._step[1], np.zeros_like(self._step[2]))
            return
        self._step = (self.current_pose, angles, _step_velocities(start_angles, angles, step_time))
        self._step_skipped = False

    def _step_angles(self) -> np.ndarray:
        """the angles the current pose was sent with"""
        if self._step is not None and self._step[0] is self.current_pose:
            return self._step[1]
        # a pose some other part set (a move, a reset) was sent through the inverse kinematics
        return self.robot.pose_to_angles(self.current_pose)

    def _send_current_pose(self) -> None:
        """
        sends the angles of the last step, which can be between the two wrist solutions while the wrist turns over;
        they are within the joint speed limit for the time of the step, so there is no extra time to wait for. The
        velocities of the step go with them (unless SEND_VELOCITIES is False), a pose some other part set is sent
        with move_to_pose, without velocities.
        """
        if self._step is None or self._step[0] is not self.current_pose:
            self.robot.move_to_pose(self.current_pose)
            return
        if SEND_VELOCITIES:
            self.robot.move_servos(self._step[1], self._step[2])
        else:
            self.robot.move_servos(self._step[1])

    def _choose_step(self, new_pose: Pose, step_time: float) -> tuple[Pose, np.ndarray]:
        """
        The pose to command and the angles to send. The pose keeps the flip of the current pose, only the a button
        changes it. The angles turn towards the angles of the commanded pose at most at recommended_max_servo_speed:
        while they haven't reached them, the gamepad waits and the pose stays, so the arm finishes the move to the
        commanded pose first. That makes it slow down near a singularity instead of stopping, and when the wrist has to
        turn over (a straight wrist passed with the same flip, or the a button) joints 4 and 6 turn by pi at that speed,
        to the other solution of the same pose. When a wrist roll reaches the seam of its one-turn range the same
        walk turns it the long way round, almost a full turn, to the same angle at the other end: the angles of the
        inverse kinematics are taken as they are, from -pi to pi, and a flip across the seam is just a big turn.
        The step is refused when the position can't be reached (the inverse
        kinematics then gives the stretched arm), or the angles are outside the workspace limits or the ranges of the
        servos (the checks of check_move).
        The turn of the wrist after the a button is slower: the angles of that walk turn at most at FLIP_JOINT_SPEED,
        until they have arrived at the flipped pose.
        :param step_time: the seconds the step takes
        :raises MovementException: if the step is not safe
        """
        current_angles = self._step_angles()
        commanded_angles = self._angles_of(self.current_pose)
        if np.abs(commanded_angles[1:] - current_angles[1:]).max() > 1e-9:
            # still on the way to the commanded pose, the gamepad waits
            best_pose, best_angles = self.current_pose, commanded_angles
        else:
            # the angles arrived at the pose, the turn of the wrist after the a button is over
            self._flip_turn_pose = None
            best_pose = _with_flip(new_pose, self.current_pose.flip)
            best_angles = self._angles_of(best_pose)

        # a step that would turn a joint faster than recommended_max_servo_speed goes part of the way, at that speed
        step_time = min(step_time, MAX_STEP_TIME)
        max_speed = SLOW_DOWN_MARGIN * src.global_constants.recommended_max_servo_speed
        if self._flip_turn_pose is not None and self._flip_turn_pose is self.current_pose:
            max_speed = min(max_speed, FLIP_JOINT_SPEED)
        max_change = max_speed * step_time
        change = np.abs(best_angles[1:] - current_angles[1:]).max()
        if change > max_change:
            best_angles = current_angles + (max_change / change) * (best_angles - current_angles)

        # check_move checks the joint speeds with the time of the steps of the planners, so the step is split in
        # steps of that time
        sub_steps = max(math.ceil(step_time * src.global_constants.steps_per_second), 1)
        progress = np.linspace(0.0, 1.0, sub_steps + 1)[:, np.newaxis]
        step_angles = current_angles + progress * (best_angles - current_angles)
        check_move(PlannedMove('step of the gamepad', step_angles, step_time), self.robot)
        return best_pose, best_angles

    def _angles_of(self, pose: Pose) -> np.ndarray:
        """
        the angles of the pose from the inverse kinematics, from -pi to pi
        :raises MovementException: if the pose can't be reached
        """
        ik_angles = self.robot.pose_to_angles(pose)
        if not np.all(np.isfinite(ik_angles)):
            raise MovementException('the inverse kinematics gave no angles')
        reached = forward_position_kinematics(ik_angles, self.robot.robot_config)[-1]
        if np.linalg.norm(reached - [pose.x, pose.y, pose.z]) > MAX_POSITION_ERROR:
            raise MovementException('position {} can not be reached'.format(np.round([pose.x, pose.y, pose.z], 1)))
        return ik_angles

    def __handle_commands(self) -> None:
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return
            command()

    def _clear_commands(self) -> None:
        while True:
            try:
                self._commands.get_nowait()
            except queue.Empty:
                return

    def check_hardware_errors(self) -> None:
        """
        A servo that shut down because of a hardware error doesn't move anymore, so stop controlling the robot right
        away. When a servo gets too warm, the robot stops like it does when it is stopped, while all servos still work.
        """
        errors = self.robot.get_hardware_errors()
        if errors is None:
            logger.warning('failed to read the hardware error status of the servos')
        elif errors:
            raise RuntimeError('servos {} shut down because of a hardware error'.format(sorted(errors)))

        overheating = self.robot.get_overheating_servos()
        if overheating is None:
            logger.warning('failed to read the temperature of the servos')
        elif overheating:
            logger.error('servos %s are too warm, stopping the robot. Stop to start it again after they cooled down',
                         sorted(overheating))
            self._stop_requested.set()

    def _plan_move_to(self, pose: Pose) -> PlannedMove | None:
        """plan a straight line from the current pose of the robot to the pose, returns None if it is not possible"""
        angles = self.robot.get_current_angles()
        if angles is None:
            logger.error('failed to read the current angles of the robot')
            return None
        try:
            return plan_move_to_pose(self.robot, angles, pose)
        except MovementException as e:
            logger.error('can not move to %s: %s', pose, e)
            return None

    def _move_to(self, pose: Pose) -> Pose | None:
        """move in a straight line to the pose, returns the reached pose, or None if the movement is not possible"""
        move = self._plan_move_to(pose)
        if move is None:
            return None
        execute_move(self.robot, move)
        return forward_kinematics(move.angles[-1], self.robot.robot_config)

    def stop_robot(self) -> None:
        if self._move_to(self.start_pose) is None:
            logger.warning('not moving back to the start pose before disabling the servos')
        self.robot.disable_servos()

    def open_gripper(self) -> None:
        self.robot.set_gripper(self.gripper_state)

    def __move_gripper(self, delta: float) -> None:
        """
        Move the target state of the gripper, it stays within GRIPPER_MAX_LEAD_POSITIONS (converted to states with the
        calibrated range of the gripper) of the actual state. When an object blocks the gripper it squeezes harder the
        further the target is beyond the object, and when the gripper is opened again it lets go within that lead,
        instead of first undoing everything the target would have run ahead. When the actual state can not be read
        the target is not clipped to it for this pass.
        :param delta: the change of the target state
        """
        new_state = self.gripper_state + delta
        actual_state = self.robot.get_gripper_state()
        if actual_state is not None:
            max_lead = GRIPPER_MAX_LEAD_POSITIONS / self.__positions_per_state()
            new_state = np.clip(new_state, actual_state - max_lead, actual_state + max_lead)
        new_state = float(np.clip(new_state, 0, 100))
        if new_state != self.gripper_state:
            self.robot.set_gripper(new_state)
            self.gripper_state = new_state

    def __positions_per_state(self) -> float:
        """the positions of one state of the gripper, the default when the robot can not tell or tells nonsense"""
        positions_per_state = self.robot.gripper_positions_per_state()
        if positions_per_state is None or not positions_per_state > 0:
            return DEFAULT_GRIPPER_POSITIONS_PER_STATE
        return positions_per_state

    def handle_buttons(self) -> None:
        """
        perform the action of every pressed button, in the order of _button_actions. The momentary buttons are reset
        when they are read, so a press is lost when it isn't handled in this read, like when a bumper is held.
        """
        buttons = self.pose_updater.controller.get_buttons()
        for button, action in self._button_actions.items():
            if self.is_done():
                return
            if getattr(buttons, button):
                action()

    def _play_back(self) -> None:
        if len(self.recorded_moves) < 1:
            return
        self.current_pose = self.playback_recorded_moves(self.recorded_moves)

    def _decrease_gripper_state(self) -> None:
        self.__move_gripper(-GRIPPER_STEP)

    def _increase_gripper_state(self) -> None:
        self.__move_gripper(GRIPPER_STEP)

    def _reset_orientation(self) -> None:
        """
        turn the gripper to the neutral orientation around the wrist centre, which stays where it is: the tip moves
        to the height of the wrist centre. The pose stays when the turn is not possible.
        """
        try:
            # the wrist centre of the angles the arm was sent to, there is no bus read in a button press
            self.current_pose = reset_orientation(self.current_pose, self.dynamixel_robot_config, self.robot,
                                                  self._step_angles())
        except MovementException as e:
            logger.warning('can not reset the orientation: %s', e)

    def _toggle_flip(self) -> None:
        angles = self._step_angles()
        # halfway the flip the gripper swings down along the forearm, so the wrist centre has to be high enough above
        # the table; the angles are the ones the arm was sent to, there is no bus read in a button press
        wrist_height = float(forward_position_kinematics(angles, self.robot.robot_config)[3][2])
        # after a raise the wrist centre is at the minimum give or take rounding, and the next press must not raise
        # again by a rounding error: a wrist centre that close below the minimum is high enough, like the workspace
        # limits accept a pose that close beyond them
        if wrist_height < FLIP_MIN_WRIST_HEIGHT - workspace.TOLERANCE:
            raised_angles = self._raise_for_flip(angles, wrist_height)
            if raised_angles is None:
                return
            angles = raised_angles
        # the next steps turn the wrist over at the joint speed limit, starting from the angles it was sent with; a step
        # set by hand has no speed, the next step measures its own velocities
        self._step =(self.current_pose, angles, np.zeros_like(angles))
        self.current_pose.flip = not self.current_pose.flip
        # the turn of the wrist is slow, also when the a button is pressed again during the turn, to turn back
        self._flip_turn_pose = self.current_pose

    def _raise_for_flip(self, angles: np.ndarray, wrist_height: float) -> np.ndarray | None:
        """
        Raise the arm in a straight line straight up until the wrist centre is FLIP_MIN_WRIST_HEIGHT above the table:
        the tip rises by the missing height, and x, y, the orientation and the flip stay. The move is planned and
        checked like the start move and can be stopped like it. The current pose becomes the pose the arm ended at, the
        flip of the current pose is not changed here.
        :param angles: the angles the arm was sent to, the raise starts there; there is no bus read in a button press
        :param wrist_height: the height of the wrist centre at these angles
        :return: the angles the raise ended at, or None when the wrist must not be flipped: the raise is not possible
            (nothing moved) or it was stopped half way (the current pose is where the arm stopped)
        """
        missing = FLIP_MIN_WRIST_HEIGHT - wrist_height
        robot_config = self.robot.robot_config
        target = forward_kinematics(angles, robot_config)
        target.z += missing
        try:
            move = plan_move_to_pose(self.robot, angles, target)
        except MovementException as e:
            logger.warning('not flipping the wrist: the wrist centre is %.1f cm above the table, the minimum is '
                           '%.1f cm, and the arm can not go straight up by %.1f cm: %s. Move the arm to where it can '
                           'go up first', wrist_height, FLIP_MIN_WRIST_HEIGHT, missing, e)
            return None
        logger.info('raising the arm by %.1f cm for the flip of the wrist: the wrist centre is %.1f cm above the '
                    'table, the minimum is %.1f cm', missing, wrist_height, FLIP_MIN_WRIST_HEIGHT)

        # execute_move checks should_stop before every step and sends the step when it is False, so the count is the
        # number of steps that were sent
        steps_sent = 0

        def should_stop() -> bool:
            nonlocal steps_sent
            if self.is_done():
                return True
            steps_sent += 1
            return False

        completed = execute_move(self.robot, move, should_stop=should_stop)
        if steps_sent > 0:
            # the wrist keeps the flip the current pose is meant to have, also when it was between the solutions
            reached = forward_kinematics(move.angles[steps_sent - 1], robot_config)
            reached.flip = self.current_pose.flip
            self.current_pose = reached
        if not completed:
            logger.info('the raise of the arm for the flip was stopped, not flipping the wrist')
            return None
        return move.angles[-1]

    def _recording_flip(self) -> bool | None:
        """the flip of the first pose of the recording, None when nothing is recorded"""
        if self.recorded_moves:
            return self.recorded_moves[0].poses[0].flip
        if self.recorded_positions:
            return self.recorded_positions[0].flip
        return None

    def _record_position(self) -> None:
        # a move between differently flipped poses turns joints 4 and 6 by pi, so every pose of a recording keeps the
        # flip of its first pose; the poses of find center mode only define the center, they never become a move
        flip = None if self.find_center_mode else self._recording_flip()
        if flip is not None and self.current_pose.flip != flip:
            logger.warning('not recording the pose: its flip is %s, the recording started with flip %s; press a to '
                           'flip back', self.current_pose.flip, flip)
            return
        self.recorded_positions.append(self.current_pose)
        if self.should_set_center():
            self.set_center()
        print("added position!")
        print(self.current_pose)

    def _create_move(self) -> None:
        if not self.has_enough_recorded_positions():
            print("not enough positions for a movement")
            return
        move = create_move(self.robot, self.recorded_positions, self.move_speed, self.center)
        self.store_move_or_go_back(move)

    def _increase_maximum_speed(self) -> None:
        self.__change_maximum_speed(MAXIMUM_SPEED_STEP)

    def _decrease_maximum_speed(self) -> None:
        self.__change_maximum_speed(-MAXIMUM_SPEED_STEP)

    def __change_maximum_speed(self, delta: float) -> None:
        self.pose_updater.maximum_speed = np.clip(self.pose_updater.maximum_speed + delta,
                                                  MINIMUM_SPEED, MAXIMUM_SPEED)

    def has_enough_recorded_positions(self) -> bool:
        return len(self.recorded_positions) > 1

    def store_move_or_go_back(self, move: Movement | None) -> None:
        if move is None:
            print('move outside workspace limits, not adding this move!')
            self.recorded_positions = [self.recorded_positions[0]]
            reached_pose = self._move_to(self.recorded_positions[0])
            if reached_pose is not None:
                self.current_pose = reached_pose
        else:
            print('created move')
            self.recorded_moves.append(move)
            self.recorded_positions = [copy(self.recorded_positions[-1])]

    @synchronized_with_lock("lock")
    def should_set_center(self) -> bool:
        return self.find_center_mode and len(self.recorded_positions) == 2

    @synchronized_with_lock("lock")
    def set_center(self) -> None:
        center = get_center(self.recorded_positions[0], self.recorded_positions[1])
        self.recorded_positions = []
        if center is None:
            # get_center logged why, stay in find center mode so two new poses can be recorded
            logger.warning('the two poses do not define a center, record two new ones')
            return
        self.center = center
        print('setting center x={}, y={}, z={}'.format(center[0], center[1], center[2]))
        self.find_center_mode = False

    @synchronized_with_lock("lock")
    def save_recorded_moves_to_file(self, name: str) -> bool:
        """
        save the recorded moves as RECORDED_MOVES_DIR/name.json
        :return: False if there is nothing to save
        :raises ValueError: if the name is not allowed
        """
        path = recorded_moves_path(name)
        if len(self.recorded_moves) < 1:
            return False
        json_string = jsonpickle.encode(self.recorded_moves, make_refs=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as outfile:
            json.dump(json.loads(json_string), outfile, indent=4)
        return True

    @synchronized_with_lock("lock")
    def restore_recorded_moves_from_file(self, name: str) -> None:
        """
        restore the recorded moves from RECORDED_MOVES_DIR/name.json
        :raises ValueError: if the name is not allowed or the file doesn't hold a list of recorded moves
        :raises FileNotFoundError: if there is no such file
        """
        with open(recorded_moves_path(name), 'r') as infile:
            string = infile.read()
        try:
            moves = jsonpickle.decode(string, safe=True)
        except Exception as e:
            raise ValueError('the file does not hold recorded moves: {}'.format(e)) from e
        if not isinstance(moves, list) or not all(isinstance(move, Movement) for move in moves):
            raise ValueError('the file does not hold a list of recorded moves')
        for move in moves:
            poses = getattr(move, 'poses', None)
            if not isinstance(poses, (list, tuple)) or len(poses) < 2 or                     not all(isinstance(pose, Pose) for pose in poses):
                raise ValueError('the file does not hold recorded moves of poses')
        self.recorded_moves = moves

    @synchronized_with_lock("lock")
    def clear_recorded_moves_and_positions(self) -> None:
        self.recorded_moves = []
        self.recorded_positions = []
        print('cleared recorded moves and positions!')

    @synchronized_with_lock("lock")
    def set_maximum_speed(self, new_maximum_speed: float) -> None:
        self.pose_updater.maximum_speed = new_maximum_speed

    @synchronized_with_lock("lock")
    def set_scenario(self, scenario_id: int) -> bool:
        if scenario_id < 0 or scenario_id >= len(self.scenarios):
            return False

        physics_client = self.robot.physics_client

        if self.current_scenario_id is not None:
            self.scenarios[self.current_scenario_id].destroy_scenario(physics_client)
            self.current_scenario_id = None

        # only a built scenario is the current one
        self.scenarios[scenario_id].build_scenario(physics_client)
        self.current_scenario_id = scenario_id

        return True

    def playback_recorded_moves(self, recorded_moves: list[Movement]) -> Pose:
        """play back the recorded moves, the playback stops between moves when the robot is stopped"""
        recorded_moves[0].go_to_start_of_move(self.robot)
        try:
            for move in recorded_moves:
                if self.is_done():
                    return self.current_pose
                move.move(self.robot)
        except MovementException as e:
            logger.warning(e)
            reached_pose = self._move_to(self.start_pose)
            if reached_pose is None:
                raise RuntimeError('playback failed and can not move back to the start pose') from e
            return reached_pose

        return recorded_moves[-1].poses[-1]


def _step_velocities(start_angles: np.ndarray, angles: np.ndarray, step_time: float) -> np.ndarray:
    """
    the velocities of a step in rad/s, indexed by servo id like the angles: the change of the angles over the time
    the step covers. Zero when there is no time, or when an angle is not a number (the step was checked, but the angles
    it started from came from the inverse kinematics of a pose some other part set).
    """
    velocities = np.zeros_like(angles, dtype=np.float64)
    if step_time > 0:
        velocities = (angles - start_angles) / step_time
        velocities[0] = 0.0  # servo id 0 doesn't exist
        if not np.all(np.isfinite(velocities)):
            velocities = np.zeros_like(velocities)
    return velocities


def _with_flip(pose: Pose, flip: bool) -> Pose:
    """a copy of the pose with the given wrist solution"""
    result = copy(pose)
    result.flip = flip
    return result


def recorded_moves_path(name: str) -> Path:
    """
    the file of the recorded moves with this name, always in RECORDED_MOVES_DIR
    :raises ValueError: if the name has anything but letters, digits, - and _
    """
    if not isinstance(name, str) or RECORDED_MOVES_NAME.fullmatch(name) is None:
        raise ValueError('the name should be 1 to 100 letters, digits, - or _')
    return RECORDED_MOVES_DIR / (name + '.json')


def create_move(servo_controller: AbstractRobotController, poses: list[Pose], speed: float,
                center: np.ndarray | None) -> Movement | None:
    """a movement through the poses at the speed, or None when it leaves the workspace limits of the robot"""
    time = determine_time(poses, speed)

    if len(poses) == 2:
        move: Movement = PoseToPoseMovement(poses, time, center)  # orientation adjustment
    else:
        move = SplineMovement(poses, time, center)
    try:
        is_ok = move.is_possible(servo_controller)
    except MovementException:
        print("invalid move")
        return None
    return move if is_ok else None


def determine_time(poses: list[Pose], speed: float) -> float:
    if speed == 0:
        raise ValueError("Cannot have zero speed")

    dr = 0
    prev_x, prev_y, prev_z = poses[0].x, poses[0].y, poses[0].z

    for pose in poses:
        x, y, z = pose.x, pose.y, pose.z
        dx = abs(x - prev_x)
        dy = abs(y - prev_y)
        dz = abs(z - prev_z)
        dr += np.sqrt(np.power(dx, 2) + np.power(dy, 2) + np.power(dz, 2))
        prev_x, prev_y, prev_z = x, y, z

    if dr < 0.1:
        return 0.5

    return dr / speed


def reset_orientation(current_pose: Pose, dynamixel_robot_config: RobotConfig,
                      dynamixel_servo_controller: AbstractRobotController,
                      angles: np.ndarray | None = None) -> Pose:
    """
    Turn the gripper to the neutral orientation (level, pointing along +y of the robot) in a planned move of a second.
    The wrist centre stays where it is and only the wrist joints turn: the new tip is the wrist centre plus the length
    of the gripper along the neutral direction, so the tip ends at the height of the wrist centre. Keeping the tip in
    place instead would swing the wrist centre down around it, into the table when the gripper points down.
    :param angles: the angles the robot was sent to, they give the wrist centre; when None the angles of the current
        pose
    :return: the new pose, which has the flip of the current pose
    :raises MovementException: if the move is not possible, the robot then has not moved
    """
    dynamixel_robot_config.restore_initial_values()
    robot_config = dynamixel_servo_controller.robot_config
    if angles is None:
        angles = dynamixel_servo_controller.pose_to_angles(current_pose)
    wrist_centre = forward_position_kinematics(angles, robot_config)[3]
    # the wrist keeps its flip, changing it in place would make the wrist turn around
    new_orientation = copy(current_pose)
    new_orientation.reset_orientation()
    # the inverse kinematics finds the wrist centre as the tip minus d6 along the third column of the orientation
    direction = new_orientation.get_euler_matrix()[:3, 2]
    tip = wrist_centre + robot_config.d6 * direction
    new_orientation.x, new_orientation.y, new_orientation.z = float(tip[0]), float(tip[1]), float(tip[2])
    return move_between_poses(dynamixel_servo_controller, current_pose, new_orientation, duration=1)

