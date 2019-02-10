from __future__ import division
from flask import jsonify

import time

from flask import Blueprint

from kinematics.kinematics_utils import RobotConfig
from movement_utils import point_to_point
from servo_handling.servo_controller import ServoController
from xbox_controller.xbox_utils import PosePoller
from xbox_controller.xbox_poller import Buttons
import threading
from utils.threading_utils import CountDownLatch
from copy import copy
import numpy as np

xbox_api = Blueprint('xbox_api', __name__)

api_lock = threading.Lock()
started = False
done = False
running_thread = None


@xbox_api.route('/start', methods=['POST'])
def start():
    global started, done, running_thread
    with api_lock:
        if started:
            return "already started"
        else:
            started = True
            done = False

    latch = CountDownLatch(1)
    running_thread = threading.Thread(target=run_xbox_poller, args=(latch,))
    running_thread.start()

    resp = jsonify(success=True)
    return resp


# TODO convert to class
def run_xbox_poller(countdown_latch):
    pose_poller = PosePoller()
    dynamixel_robot_config = RobotConfig(d1=9.1, a2=15.8, d4=22.0, d6=2.0)
    dynamixel_servo_controller = ServoController("COM5", dynamixel_robot_config)
    dynamixel_servo_controller.enable_servos()

    start_pose = dynamixel_servo_controller.get_current_pose()

    lift_pose = copy(start_pose)
    lift_pose.z += 5
    lift_pose.reset_orientation()

    current_pose = copy(lift_pose)
    point_to_point(start_pose, current_pose, 1, dynamixel_robot_config, dynamixel_servo_controller)
    countdown_latch.count_down()

    global done
    while True:
        with api_lock:
            if done:
                break
        current_pose = pose_poller.get_updated_pose_from_controller(current_pose)
        dynamixel_servo_controller.move_to_pose(current_pose)

        # Button handling
        buttons = pose_poller.get_buttons()
        if buttons.start:
            current_pose = back_to_start(current_pose, lift_pose, dynamixel_robot_config, dynamixel_servo_controller)
        elif buttons.b:
            current_pose = reset_orientation(current_pose, dynamixel_robot_config, dynamixel_servo_controller)

        if buttons.rb:
            back_it_up(current_pose, 1, dynamixel_robot_config, dynamixel_servo_controller)
        if buttons.lb:
            back_it_up(current_pose, -1, dynamixel_robot_config, dynamixel_servo_controller)

        time.sleep(pose_poller.dt)

    pose_poller.stop()
    reset_orientation(current_pose, dynamixel_robot_config, dynamixel_servo_controller)

    current_pose = point_to_point(current_pose, lift_pose, 3, dynamixel_robot_config, dynamixel_servo_controller)
    point_to_point(current_pose, start_pose, 2, dynamixel_robot_config, dynamixel_servo_controller)
    time.sleep(1)
    dynamixel_servo_controller.disable_servos()


# this changes current_pose and d6 of robot_config as a side effect, refactor?
# move to pose_poller?
def back_it_up(current_pose, direction, dynamixel_robot_config, dynamixel_servo_controller):
    world_gripper_vector = current_pose.orientation.dot(np.array([0, 0, 1]))
    old_pose = copy(current_pose)
    current_pose.x += direction * world_gripper_vector[0]
    current_pose.y += direction * world_gripper_vector[1]
    current_pose.z += direction * world_gripper_vector[2]
    point_to_point(old_pose, current_pose, 0.1, dynamixel_robot_config, dynamixel_servo_controller)
    dynamixel_robot_config.d6 -= direction


def back_to_start(current_pose, lift_pose, dynamixel_robot_config, dynamixel_servo_controller):
    point_to_point(current_pose, lift_pose, 2, dynamixel_robot_config, dynamixel_servo_controller)
    return copy(lift_pose)


def reset_orientation(current_pose, dynamixel_robot_config, dynamixel_servo_controller):
    dynamixel_robot_config.restore_initial_values()
    new_orientation = copy(current_pose)
    new_orientation.reset_orientation()
    point_to_point(current_pose, new_orientation, 1, dynamixel_robot_config, dynamixel_servo_controller)
    return new_orientation



@xbox_api.route('/stop', methods=['POST'])
def stop():
    global started, done, running_thread
    with api_lock:
        if started:
            started = False
            done = True
        else:
            return "already stopped"

    running_thread.join()

    resp = jsonify(success=True)
    return resp


if __name__ == '__main__':
    latch = CountDownLatch(1)
    run_xbox_poller(latch)

