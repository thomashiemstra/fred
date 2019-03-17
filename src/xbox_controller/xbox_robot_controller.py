from __future__ import division

import threading
from copy import copy

from yaml import dump

from src.kinematics.kinematics_utils import Pose
from src.utils.decorators import synchronized_with_lock
from src.utils.movement_utils import pose_to_pose
from src.xbox_controller.pose_poller import PosePoller


class XboxRobotController:
    start_pose = Pose(-26, 11.0, 6)

    def __init__(self, dynamixel_robot_config, dynamixel_servo_controller):
        self.pose_poller = PosePoller()
        self.dynamixel_robot_config = dynamixel_robot_config
        self.dynamixel_servo_controller = dynamixel_servo_controller
        self.lock = threading.RLock()
        self.done = False
        self.recorded_positions = []
        self.current_pose = None

    @synchronized_with_lock("lock")
    def is_done(self):
        return self.done

    @synchronized_with_lock("lock")
    def stop(self):
        self.done = True

    @synchronized_with_lock("lock")
    def reset(self):
        self.done = False

    def start(self, latch):
        with self.lock:
            self.done = False
        self.dynamixel_servo_controller.enable_servos()
        self.current_pose = copy(self.start_pose)
        self.dynamixel_servo_controller.from_current_angles_to_pose(self.current_pose, 2)
        latch.count_down()  # indicate that we are ready to go

        while True:
            if self.is_done:
                break
            self.current_pose = self.pose_poller.get_updated_pose_from_controller(self.current_pose)
            self.dynamixel_servo_controller.move_to_pose(self.current_pose)

            buttons = self.pose_poller.get_buttons()
            self.handle_buttons(buttons)

        self.current_pose = reset_orientation(self.current_pose, self.dynamixel_robot_config,
                                              self.dynamixel_servo_controller)

        pose_to_pose(self.current_pose, self.start_pose,
                     self.dynamixel_robot_config, self.dynamixel_servo_controller, time=3)

        self.pose_poller.stop()
        self.dynamixel_servo_controller.disable_servos()

    def handle_buttons(self, buttons):
        if buttons.start:
            with open('recorded_positions.yml', 'w') as outfile:
                dump(self.recorded_positions, outfile)
            self.recorded_positions = []
        elif buttons.b:
            self.current_pose = reset_orientation(self.current_pose, self.dynamixel_robot_config,
                                                  self.dynamixel_servo_controller)
        elif buttons.a:
            self.current_pose.flip = not self.current_pose.flip
        elif buttons.y:
            self.recorded_positions.append(self.current_pose)
            print("added position!")
        elif buttons.x:
            pass


def reset_orientation(current_pose, dynamixel_robot_config, dynamixel_servo_controller):
    current_pose.flip = False
    dynamixel_robot_config.restore_initial_values()
    new_orientation = copy(current_pose)
    new_orientation.reset_orientation()
    pose_to_pose(current_pose, new_orientation, dynamixel_robot_config, dynamixel_servo_controller, time=1)
    return new_orientation
