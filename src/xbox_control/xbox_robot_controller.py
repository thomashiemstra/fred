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
# how far the target of the gripper can be beyond its actual state, this determines how hard the gripper squeezes an
# object, at most with the goal current of the servo. 25 states is 256 positions, which is about 500mA, so the goal
# current of the gripper (400mA) limits the squeeze
GRIPPER_MAX_LEAD = 25
# a stick step that is too fast is slowed down to this fraction of recommended_max_servo_speed, below the speed check
SLOW_DOWN_MARGIN = 0.95
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
        # the pose and the angles of the last step of the sticks, the angles the control loop sends
        self._step: tuple[Pose, np.ndarray] | None = None
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
        next_hardware_check = perf_counter() + HARDWARE_CHECK_INTERVAL
        previous_step_time = perf_counter()

        while True:
            if self.is_done():
                break

            if perf_counter() > next_hardware_check:
                self.check_hardware_errors()
                next_hardware_check = perf_counter() + HARDWARE_CHECK_INTERVAL

            self.__handle_commands()

            # the step covers the time that really passed, so the speeds of the sticks hold at any loop pace
            now = perf_counter()
            step_time = now - previous_step_time
            new_pose = self.pose_updater.get_updated_pose_from_controller(self.current_pose, self.find_center_mode,
                                                                          self.center, dt=step_time)
            previous_step_time = now
            self._take_step(new_pose, step_time)
            recommended_time, time_taken = self._send_current_pose()
            self.handle_buttons()

            time_to_sleep = np.maximum(np.maximum(recommended_time, pose_update_sleep_time) - time_taken, 0)
            sleep(time_to_sleep)

        self.stop_robot()

    def _take_step(self, new_pose: Pose, step_time: float) -> None:
        """
        the pose the gamepad asks for becomes the current pose when the step to it is safe (see _choose_step),
        otherwise the robot holds the current pose: logged as a warning when the holding starts, and at debug level
        while it lasts
        """
        try:
            self.current_pose, angles = self._choose_step(new_pose, step_time)
        except MovementException as e:
            if not self._step_skipped:
                logger.warning('holding the pose, skipping the step the gamepad asks for: %s', e)
            else:
                logger.debug('holding the pose, skipping the step the gamepad asks for: %s', e)
            self._step_skipped = True
            return
        self._step = (self.current_pose, angles)
        self._step_skipped = False

    def _step_angles(self) -> np.ndarray:
        """the angles the current pose was sent with"""
        if self._step is not None and self._step[0] is self.current_pose:
            return self._step[1]
        # a pose some other part set (a move, a reset) was sent through the inverse kinematics
        return self.robot.pose_to_angles(self.current_pose)

    def _send_current_pose(self) -> tuple[float, float]:
        """
        sends the angles of the last step, which can be between the two wrist solutions while the wrist turns over;
        they are within the joint speed limit for the time of the step, so there is no extra time to wait for
        :return: the recommended wait time and the time sending took, in seconds
        """
        if self._step is None or self._step[0] is not self.current_pose:
            return self.robot.move_to_pose(self.current_pose)
        time_taken = self.robot.move_servos(self._step[1])
        return 0.0, time_taken or 0.0

    def _choose_step(self, new_pose: Pose, step_time: float) -> tuple[Pose, np.ndarray]:
        """
        The pose to command and the angles to send. The pose keeps the flip of the current pose, only the a button
        changes it. The angles turn towards the angles of the commanded pose at most at recommended_max_servo_speed:
        while they haven't reached them, the gamepad waits and the pose stays, so the arm finishes the move to the
        commanded pose first. That makes it slow down near a singularity instead of stopping, and when the wrist has to
        turn over (a straight wrist passed with the same flip, or the a button) joints 4 and 6 turn by pi at that speed,
        to the other solution of the same pose. The step is refused when the position can't be reached (the inverse
        kinematics then gives the stretched arm), or the angles are outside the workspace limits or the ranges of the
        servos (the checks of check_move).
        :param step_time: the seconds the step takes
        :raises MovementException: if the step is not safe
        """
        current_angles = self._step_angles()
        commanded_angles = self._angles_towards(self.current_pose, current_angles)
        if np.abs(commanded_angles[1:] - current_angles[1:]).max() > 1e-9:
            # still on the way to the commanded pose, the gamepad waits
            best_pose, best_angles = self.current_pose, commanded_angles
        else:
            best_pose = _with_flip(new_pose, self.current_pose.flip)
            best_angles = self._angles_towards(best_pose, current_angles)

        # a step that would turn a joint faster than recommended_max_servo_speed goes part of the way, at that speed
        step_time = min(step_time, MAX_STEP_TIME)
        max_change = SLOW_DOWN_MARGIN * src.global_constants.recommended_max_servo_speed * step_time
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

    def _angles_towards(self, pose: Pose, current_angles: np.ndarray) -> np.ndarray:
        """
        the angles of the pose, with each angle on the side of the current one
        :raises MovementException: if the pose can't be reached
        """
        ik_angles = self.robot.pose_to_angles(pose)
        if not np.all(np.isfinite(ik_angles)):
            raise MovementException('the inverse kinematics gave no angles')
        unwrapped_angles = unwrap_towards(ik_angles, current_angles)
        reached = forward_position_kinematics(unwrapped_angles, self.robot.robot_config)[-1]
        if np.linalg.norm(reached - [pose.x, pose.y, pose.z]) > MAX_POSITION_ERROR:
            raise MovementException('position {} can not be reached'.format(np.round([pose.x, pose.y, pose.z], 1)))
        return unwrapped_angles

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
        Move the target state of the gripper, it stays within GRIPPER_MAX_LEAD of the actual state. When an object
        blocks the gripper it squeezes harder the further the target is beyond the object, and when the gripper is
        opened again it lets go within GRIPPER_MAX_LEAD steps, instead of first undoing everything the target would
        have run ahead.
        """
        new_state = self.gripper_state + delta
        actual_state = self.robot.get_gripper_state()
        if actual_state is not None:
            new_state = np.clip(new_state, actual_state - GRIPPER_MAX_LEAD, actual_state + GRIPPER_MAX_LEAD)
        new_state = float(np.clip(new_state, 0, 100))
        if new_state != self.gripper_state:
            self.robot.set_gripper(new_state)
            self.gripper_state = new_state

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
        """turn the end effector to the neutral orientation, in place; the pose stays when the turn is not possible"""
        try:
            self.current_pose = reset_orientation(self.current_pose, self.dynamixel_robot_config, self.robot)
        except MovementException as e:
            logger.warning('can not reset the orientation: %s', e)

    def _toggle_flip(self) -> None:
        # the next steps turn the wrist over at the joint speed limit, starting from the angles it was sent with
        self._step = (self.current_pose, self._step_angles())
        self.current_pose.flip = not self.current_pose.flip

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


def unwrap_towards(ik_angles: np.ndarray, current_angles: np.ndarray) -> np.ndarray:
    """
    The inverse kinematics gives angles from -pi to pi, so a turn of almost 2 pi is the same angle just past the end
    of the range: this takes that one, the range check then holds the joint at its end instead of turning it all the
    way round. A turn of about pi is the wrist turning over to the other solution, that one stays.
    """
    turn = ik_angles - current_angles
    unwrapped: np.ndarray = np.where(np.abs(turn) > 1.5 * np.pi, ik_angles - np.sign(turn) * 2 * np.pi, ik_angles)
    return unwrapped


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
                      dynamixel_servo_controller: AbstractRobotController) -> Pose:
    # the wrist keeps its flip, changing it in place would make the wrist turn around
    dynamixel_robot_config.restore_initial_values()
    new_orientation = copy(current_pose)
    new_orientation.reset_orientation()
    return move_between_poses(dynamixel_servo_controller, current_pose, new_orientation, duration=1)

