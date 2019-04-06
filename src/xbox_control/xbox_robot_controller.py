from __future__ import division

import threading
from copy import copy

from yaml import dump

from src.kinematics.kinematics_utils import Pose
from src.utils.decorators import synchronized_with_lock
from src.utils.linalg_utils import get_center
from src.utils.movement_utils import pose_to_pose
from src.xbox_control.xbox360controller.xbox_pose_updater import XboxPoseUpdater
from time import sleep


class XboxRobotController:
    start_pose = Pose(-26, 14.0, 6)

    def __init__(self, dynamixel_robot_config, dynamixel_servo_controller):
        self.pose_poller = XboxPoseUpdater()
        self.dynamixel_robot_config = dynamixel_robot_config
        self.dynamixel_servo_controller = dynamixel_servo_controller
        self.lock = threading.RLock()
        self.done = False
        self.recorded_positions = []
        self.current_pose = None
        self.thread = None
        self.find_center_mode = False
        self.center = None

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
                          self.dynamixel_servo_controller)

    @synchronized_with_lock("lock")
    def clear_center(self):
        self.center = None

    @synchronized_with_lock("lock")
    def start(self):
        if self.done:
            return False

        self.dynamixel_servo_controller.enable_servos()
        self.current_pose = copy(self.start_pose)
        if self.thread is None:
            self.thread = threading.Thread(target=self.__start_internal, args=())
            self.thread.start()
            return True
        else:
            return False

    def __start_internal(self):
        self.dynamixel_servo_controller.from_current_angles_to_pose(self.current_pose, 1)

        while True:
            if self.is_done():
                break
            self.current_pose = self.pose_poller.get_updated_pose_from_controller(self.current_pose,
                                                                                  self.find_center_mode, self.center)
            self.dynamixel_servo_controller.move_to_pose(self.current_pose)

            buttons = self.pose_poller.get_buttons()
            self.handle_buttons(buttons)
            sleep(self.pose_poller.dt)

        self.current_pose = reset_orientation(self.current_pose, self.dynamixel_robot_config,
                                              self.dynamixel_servo_controller)

        pose_to_pose(self.current_pose, self.start_pose,
                     self.dynamixel_servo_controller, time=3)

        # self.pose_poller.stop()
        self.dynamixel_servo_controller.disable_servos()

    def handle_buttons(self, buttons):
        if buttons.start:
            # with open('recorded_positions.yml', 'w') as outfile:
            #     dump(self.recorded_positions, outfile)
            # self.recorded_positions = []
            pass
        elif buttons.b:
            self.current_pose = reset_orientation(self.current_pose, self.dynamixel_robot_config,
                                                  self.dynamixel_servo_controller)
        elif buttons.a:
            self.current_pose.flip = not self.current_pose.flip
        elif buttons.y:
            self.recorded_positions.append(self.current_pose)
            with self.lock:
                if self.find_center_mode and len(self.recorded_positions) == 2:
                    self.center = get_center(self.recorded_positions[0], self.recorded_positions[1])
                    print('setting center x={}, y={}, z={}'.format(self.center[0], self.center[1], self.center[2]))
                    self.find_center_mode = False

            print("added position!")
        elif buttons.x:
            pass
        elif buttons.lb:
            pass
        elif buttons.rb:
            pass


def reset_orientation(current_pose, dynamixel_robot_config, dynamixel_servo_controller):
    current_pose.flip = False
    dynamixel_robot_config.restore_initial_values()
    new_orientation = copy(current_pose)
    new_orientation.reset_orientation()
    pose_to_pose(current_pose, new_orientation, dynamixel_servo_controller, time=1)
    return new_orientation
