from __future__ import division

import json
import threading
from copy import copy

import jsonpickle
import numpy as np

from src.global_constants import WorkSpaceLimits
from src.kinematics.kinematics_utils import Pose
from src.utils.decorators import synchronized_with_lock
from src.utils.linalg_utils import get_center
from src.utils.movement import PoseToPoseMovement, SplineMovement
from src.utils.movement_exception import MovementException
from src.utils.movement_utils import pose_to_pose, from_current_angles_to_pose
from time import sleep
import logging as log
from timeit import default_timer as timer


class XboxRobotController:
    start_pose = Pose(-26, 21.0, 6)

    def __init__(self, dynamixel_robot_config, servo_controller, pose_updater):
        self.pose_updater = pose_updater
        self.dynamixel_robot_config = dynamixel_robot_config
        self.servo_controller = servo_controller
        self.lock = threading.RLock()
        self.done = False
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

    @synchronized_with_lock("lock")
    def is_done(self):
        return self.done

    def stop(self):
        self.set_done()
        self.thread.join()
        with self.lock:
            self.thread = None
        self.reset()

    @synchronized_with_lock("lock")
    def set_done(self):
        self.done = True

    @synchronized_with_lock("lock")
    def reset(self):
        self.done = False

    # Record 2 poses to define the center the end effector should be pointing towards
    @synchronized_with_lock("lock")
    def start_find_center_mode(self):
        self.recorded_positions = []
        self.center = None
        self.find_center_mode = True
        reset_orientation(self.current_pose, self.dynamixel_robot_config,
                          self.servo_controller)

    @synchronized_with_lock("lock")
    def clear_center(self):
        self.center = None

    @synchronized_with_lock("lock")
    def clear_center(self):
        self.center = None

    @synchronized_with_lock("lock")
    def start(self):
        if self.done:
            return False

        self.servo_controller.enable_servos()
        self.current_pose = copy(self.start_pose)
        if self.thread is None:
            self.thread = threading.Thread(target=self.__start_internal, args=())
            self.thread.start()
            return True
        else:
            return False

    def __start_internal(self):
        # The robot could be anywhere, first move it from it's current position to the target pose
        # It would be easier to get a get_current_pose(), but I'm too lazy to write that
        from_current_angles_to_pose(self.current_pose, self.servo_controller, 1)
        self.servo_controller.set_gripper(self.gripper_state)
        # self.current_pose = pose_to_pose(self.current_pose, Pose(0, 25, 10), self.servo_controller, 2)
        self.pose_updater.reset_buttons()

        pose_update_sleep_time = self.pose_updater.dt
        steps_to_take_without_pose_update = 4
        steps_taken_since_pose_update = 0
        default_sleep_time = pose_update_sleep_time / steps_to_take_without_pose_update

        while True:
            if self.is_done():
                break

            recommended_time, time_taken = 0, 0
            if steps_taken_since_pose_update == steps_to_take_without_pose_update - 1:
                self.current_pose = self.pose_updater.get_updated_pose_from_controller(self.current_pose,
                                                                                       self.find_center_mode, self.center)
                recommended_time, time_taken = self.servo_controller.move_to_pose(self.current_pose)
                steps_taken_since_pose_update = 0
            else:
                steps_taken_since_pose_update += 1

            self.handle_buttons()

            time_to_sleep = np.maximum(np.maximum(recommended_time, default_sleep_time) - time_taken, 0)
            sleep(time_to_sleep)

        self.stop_robot()

    def stop_robot(self):
        from_current_angles_to_pose(self.start_pose, self.servo_controller, 4)
        self.servo_controller.disable_servos()

    def handle_buttons(self):
        buttons = self.pose_updater.controller_state_manager.get_buttons()
        if buttons.start:
            if len(self.recorded_moves) < 1:
                return
            self.current_pose = self.playback_recorded_moves(self.recorded_moves)
        elif buttons.lb:
            self.gripper_state = np.clip(self.gripper_state - 10, 0, 100)
            self.servo_controller.set_gripper(self.gripper_state)
            return
        elif buttons.rb:
            self.gripper_state = np.clip(self.gripper_state + 10, 0, 100)
            self.servo_controller.set_gripper(self.gripper_state)
        elif buttons.b:
            self.current_pose = reset_orientation(self.current_pose, self.dynamixel_robot_config,
                                                  self.servo_controller)
        elif buttons.a:
            self.current_pose.flip = not self.current_pose.flip
        elif buttons.y:
            self.recorded_positions.append(self.current_pose)
            if self.should_set_center():
                self.set_center()
            print("added position!")
            print(self.current_pose)
        elif buttons.x:
            if self.has_enough_recorded_positions():
                move = create_move(self.servo_controller, self.recorded_positions,
                                   self.move_speed, self.center, WorkSpaceLimits)
                self.store_move_or_go_back(move)
            else:
                print("not enough positions for a movement")
        elif buttons.pad_ud != 0:
            old_speed = self.pose_updater.maximum_speed
            new_speed = np.clip(old_speed + buttons.pad_ud * 5, 5, 50)
            self.pose_updater.maximum_speed = new_speed

    def has_enough_recorded_positions(self):
        return len(self.recorded_positions) > 1

    def store_move_or_go_back(self, move):
        if move is None:
            print('move outside workspace limits, not adding this move!')
            self.recorded_positions = [self.recorded_positions[0]]
            from_current_angles_to_pose(self.recorded_positions[0], self.servo_controller, 4)
            self.current_pose = self.recorded_positions[0]
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

    def playback_recorded_moves(self, recorded_moves):
        recorded_moves[0].go_to_start_of_move(self.servo_controller)
        try:
            for move in recorded_moves:
                move.move(self.servo_controller)
        except MovementException as e:
            log.warning(e)
            from_current_angles_to_pose(self.start_pose, self.servo_controller, 4)
            return self.start_pose

        return recorded_moves[-1].poses[-1]


def create_move(servo_controller, poses, speed, center, workspace_limits):
    if len(poses) == 2 and np.allclose([poses[0].x, poses[0].y, poses[0].z], [poses[1].x, poses[1].y, poses[1].z]):
        return PoseToPoseMovement(poses, 0.5, center, workspace_limits)  # orientation adjustment

    time = determine_time(poses, speed)
    move = SplineMovement(poses, time, center, workspace_limits)
    is_ok = move.check_workspace_limits(servo_controller, workspace_limits)
    return move if is_ok else None


def determine_time(poses, speed):
    dr = 0
    prev_x, prev_y, prev_z = poses[0].x, poses[0].y, poses[0].z

    for pose in poses:
        x, y, z = pose.x, pose.y, pose.z
        dx = abs(x - prev_x)
        dy = abs(y - prev_y)
        dz = abs(z - prev_z)
        dr += np.sqrt(np.power(dx, 2) + np.power(dy, 2) + np.power(dz, 2))
        prev_x, prev_y, prev_z = x, y, z

    return dr / speed


def reset_orientation(current_pose, dynamixel_robot_config, dynamixel_servo_controller):
    current_pose.flip = False
    dynamixel_robot_config.restore_initial_values()
    new_orientation = copy(current_pose)
    new_orientation.reset_orientation()
    pose_to_pose(current_pose, new_orientation, dynamixel_servo_controller, time=1)
    return new_orientation
