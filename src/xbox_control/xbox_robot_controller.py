from __future__ import division

import json
import logging
import queue
import threading
from copy import copy

import jsonpickle
import numpy as np
from src.reinforcementlearning.environment.scenario import medium_scenarios

from src.kinematics.kinematics import forward_kinematics
from src.kinematics.kinematics_utils import Pose
from src.utils.movement import plan_move_to_pose, execute_move, check_move, move_between_poses
from src.utils.decorators import synchronized_with_lock
from src.utils.linalg_utils import get_center
from src.utils.movement import PoseToPoseMovement, SplineMovement
from src.utils.movement_exception import MovementException
from time import sleep, perf_counter

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


class XboxRobotController:
    """
    Control the robot with an xbox controller.

    Only the control loop thread moves the robot. Requests from other threads that move the robot, like starting the
    find center mode, are queued as commands and handled by the control loop.
    """
    start_pose = Pose(30, 22.0, 4)

    def __init__(self, dynamixel_robot_config, robot, pose_updater):
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
        self._commands = queue.Queue()
        self._crashed = False
        self._start_move = None
        self._outside_workspace = False  # the gamepad asks for a pose outside the workspace, the robot holds
        self.recorded_positions = []
        self.current_pose = None
        self.thread = None
        self.find_center_mode = False
        # 3d position in front of the robot to keep the end effector pointed towards
        # (i.e. make sure the camera stays focused on an object)
        self.center = None
        self.move_speed = 10
        self.recorded_moves = []
        self.gripper_state = 0
        self.current_scenario_id = None
        # dependency inject?
        self.scenarios = medium_scenarios
        # the action of each button, when multiple buttons are pressed only the first one in this order is handled
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

    def is_running(self):
        """
        True from a successful start until stop. This includes a control loop that stopped because of an error,
        the servos keep holding their position until stop disables them.
        """
        return self.thread is not None

    def is_done(self):
        return self._stop_requested.is_set()

    def start(self):
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
            self._clear_commands()
            self._start_move = start_move
            # the wrist keeps its flip, so this is the start pose with the flip of the robot
            self.current_pose = forward_kinematics(start_move.angles[-1], self.robot.robot_config)
            self.thread = threading.Thread(target=self.__start_internal, args=())
            self.thread.start()
            return True

    def stop(self):
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

    def request_find_center_mode(self):
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
    def __start_find_center_mode(self):
        with self.lock:
            self.recorded_positions = []
            self.center = None
            self.find_center_mode = True
        self._reset_orientation()

    @synchronized_with_lock("lock")
    def clear_center(self):
        self.center = None

    @synchronized_with_lock("lock")
    def set_profile_velocity_percentage(self, percentage):
        self.robot.set_profile_velocity_percentage(percentage)

    def __start_internal(self):
        try:
            self.__control_robot()
        except Exception:
            self._crashed = True
            # don't try to move the robot back, whatever failed could fail again while moving
            logger.exception('controlling the robot failed, the servos keep holding their last position. '
                             'Stop to disable them')

    def __control_robot(self):
        # the start move can be stopped halfway, the control loop then stops right away and moves back
        if execute_move(self.robot, self._start_move, should_stop=self.is_done):
            self.open_gripper()

        pose_update_sleep_time = self.pose_updater.dt
        next_hardware_check = perf_counter() + HARDWARE_CHECK_INTERVAL

        while True:
            if self.is_done():
                break

            if perf_counter() > next_hardware_check:
                self.check_hardware_errors()
                next_hardware_check = perf_counter() + HARDWARE_CHECK_INTERVAL

            self.__handle_commands()

            new_pose = self.pose_updater.get_updated_pose_from_controller(self.current_pose, self.find_center_mode,
                                                                          self.center)
            self._update_pose_within_the_workspace(new_pose)
            recommended_time, time_taken = self.robot.move_to_pose(self.current_pose)
            self.handle_buttons()

            time_to_sleep = np.maximum(np.maximum(recommended_time, pose_update_sleep_time) - time_taken, 0)
            sleep(time_to_sleep)

        self.stop_robot()

    def _update_pose_within_the_workspace(self, new_pose):
        """the pose the gamepad asks for becomes the current pose when it is inside the workspace limits of the
        robot, otherwise the robot holds the current pose (logged once)"""
        violation = self.robot.workspace_violation(self.robot.pose_to_angles(new_pose))
        if violation is None:
            self.current_pose = new_pose
            self._outside_workspace = False
        elif not self._outside_workspace:
            logger.warning('holding the pose, the gamepad asks for one outside the workspace: %s', violation)
            self._outside_workspace = True

    def __handle_commands(self):
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return
            command()

    def _clear_commands(self):
        while True:
            try:
                self._commands.get_nowait()
            except queue.Empty:
                return

    def check_hardware_errors(self):
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

    def _plan_move_to(self, pose):
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

    def _move_to(self, pose):
        """move in a straight line to the pose, returns the reached pose, or None if the movement is not possible"""
        move = self._plan_move_to(pose)
        if move is None:
            return None
        execute_move(self.robot, move)
        return forward_kinematics(move.angles[-1], self.robot.robot_config)

    def stop_robot(self):
        if self._move_to(self.start_pose) is None:
            logger.warning('not moving back to the start pose before disabling the servos')
        self.robot.disable_servos()

    def open_gripper(self):
        self.robot.set_gripper(self.gripper_state)

    def __move_gripper(self, delta):
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

    def handle_buttons(self):
        """perform the action of the first pressed button, in the order of _button_actions"""
        buttons = self.pose_updater.controller.get_buttons()
        for button, action in self._button_actions.items():
            if getattr(buttons, button):
                action()
                return

    def _play_back(self):
        if len(self.recorded_moves) < 1:
            return
        self.current_pose = self.playback_recorded_moves(self.recorded_moves)

    def _decrease_gripper_state(self):
        self.__move_gripper(-GRIPPER_STEP)

    def _increase_gripper_state(self):
        self.__move_gripper(GRIPPER_STEP)

    def _reset_orientation(self):
        """turn the end effector to the neutral orientation, in place; the pose stays when the turn is not possible"""
        try:
            self.current_pose = reset_orientation(self.current_pose, self.dynamixel_robot_config, self.robot)
        except MovementException as e:
            logger.warning('can not reset the orientation: %s', e)

    def _toggle_flip(self):
        self.current_pose.flip = not self.current_pose.flip

    def _record_position(self):
        self.recorded_positions.append(self.current_pose)
        if self.should_set_center():
            self.set_center()
        print("added position!")
        print(self.current_pose)

    def _create_move(self):
        if not self.has_enough_recorded_positions():
            print("not enough positions for a movement")
            return
        move = create_move(self.robot, self.recorded_positions, self.move_speed, self.center)
        self.store_move_or_go_back(move)

    def _increase_maximum_speed(self):
        self.__change_maximum_speed(MAXIMUM_SPEED_STEP)

    def _decrease_maximum_speed(self):
        self.__change_maximum_speed(-MAXIMUM_SPEED_STEP)

    def __change_maximum_speed(self, delta):
        self.pose_updater.maximum_speed = np.clip(self.pose_updater.maximum_speed + delta,
                                                  MINIMUM_SPEED, MAXIMUM_SPEED)

    def has_enough_recorded_positions(self):
        return len(self.recorded_positions) > 1

    def store_move_or_go_back(self, move):
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
    def should_set_center(self):
        return self.find_center_mode and len(self.recorded_positions) == 2

    @synchronized_with_lock("lock")
    def set_center(self):
        self.center = get_center(self.recorded_positions[0], self.recorded_positions[1])
        print('setting center x={}, y={}, z={}'.format(self.center[0], self.center[1], self.center[2]))
        self.find_center_mode = False
        self.recorded_positions = []

    @synchronized_with_lock("lock")
    def save_recorded_moves_to_file(self, filename):
        if len(self.recorded_moves) < 1:
            return False
        json_string = jsonpickle.encode(self.recorded_moves, make_refs=False)
        with open(filename, 'w') as outfile:
            json.dump(json.loads(json_string), outfile, indent=4)
        return True

    @synchronized_with_lock("lock")
    def restore_recorded_moves_from_file(self, filename):
        with open(filename, 'r') as infile:
            string = infile.read()
        moves = jsonpickle.decode(string)
        self.recorded_moves = moves

    @synchronized_with_lock("lock")
    def clear_recorded_moves_and_positions(self):
        self.recorded_moves = []
        self.recorded_positions = []
        print('cleared recorded moves and positions!')

    @synchronized_with_lock("lock")
    def set_maximum_speed(self, new_maximum_speed):
        self.pose_updater.maximum_speed = new_maximum_speed

    @synchronized_with_lock("lock")
    def set_scenario(self, scneario_id):
        if scneario_id < 0 or scneario_id > len(self.scenarios):
            return False

        physics_client = self.robot.physics_client

        if self.current_scenario_id is not None:
            self.scenarios[self.current_scenario_id].destroy_scenario(physics_client)

        self.current_scenario_id = scneario_id
        self.scenarios[self.current_scenario_id].build_scenario(physics_client)

        return True

    def playback_recorded_moves(self, recorded_moves):
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


def create_move(servo_controller, poses, speed, center):
    """a movement through the poses at the speed, or None when it leaves the workspace limits of the robot"""
    time = determine_time(poses, speed)

    if len(poses) == 2:
        move = PoseToPoseMovement(poses, time, center)  # orientation adjustment
    else:
        move = SplineMovement(poses, time, center)
    try:
        is_ok = move.is_possible(servo_controller)
    except MovementException:
        print("invalid move")
        return None
    return move if is_ok else None


def determine_time(poses, speed):
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


def reset_orientation(current_pose, dynamixel_robot_config, dynamixel_servo_controller):
    # the wrist keeps its flip, changing it in place would make the wrist turn around
    dynamixel_robot_config.restore_initial_values()
    new_orientation = copy(current_pose)
    new_orientation.reset_orientation()
    return move_between_poses(dynamixel_servo_controller, current_pose, new_orientation, duration=1)

