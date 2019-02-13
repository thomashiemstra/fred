from __future__ import division

import threading
import time
from copy import copy

import numpy as np
from flask import Blueprint
from flask import jsonify

from kinematics.kinematics_utils import Pose
from kinematics.kinematics_utils import RobotConfig
from utils.movement_utils import pose_to_pose, line
from servo_handling.servo_controller import ServoController
from utils.threading_utils import CountDownLatch
from xbox_controller.xbox_utils import PosePoller
from time import sleep

from ruamel.yaml import load, dump, round_trip_dump
import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

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

    start_pose = Pose(-26, 11.0, 6)
    current_pose = copy(start_pose)

    dynamixel_servo_controller.from_current_angles_to_pose(current_pose, 2)

    countdown_latch.count_down()

    recorded_positions = []

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
            with open('recorded_positions.yml', 'w') as outfile:
                round_trip_dump(recorded_positions, outfile)
        elif buttons.b:
            current_pose = reset_orientation(current_pose, dynamixel_robot_config, dynamixel_servo_controller)
        elif buttons.a:
            current_pose.flip = not current_pose.flip
        elif buttons.y:
            recorded_positions.append(current_pose)
            print("added position!")
        elif buttons.x:
            playback_recorded_positions(current_pose, recorded_positions,
                                        dynamixel_robot_config, dynamixel_servo_controller)

        if buttons.rb:
            back_it_up(current_pose, 1, dynamixel_robot_config, dynamixel_servo_controller)
        if buttons.lb:
            back_it_up(current_pose, -1, dynamixel_robot_config, dynamixel_servo_controller)

        time.sleep(pose_poller.dt)

    pose_poller.stop()
    reset_orientation(current_pose, dynamixel_robot_config, dynamixel_servo_controller)

    current_pose = pose_to_pose(current_pose, start_pose, dynamixel_robot_config, dynamixel_servo_controller, time=3)

    time.sleep(1)
    dynamixel_servo_controller.disable_servos()


# this changes current_pose and d6 of robot_config as a side effect, refactor?
# move to pose_poller?
def back_it_up(current_pose, direction, dynamixel_robot_config, dynamixel_servo_controller):
    if dynamixel_robot_config.d6 + direction < dynamixel_robot_config.initial_d6:
        return

    world_gripper_vector = current_pose.orientation.dot(np.array([0, 0, 1]))
    old_pose = copy(current_pose)
    current_pose.x -= direction * world_gripper_vector[0]
    current_pose.y -= direction * world_gripper_vector[1]
    current_pose.z -= direction * world_gripper_vector[2]
    pose_to_pose(old_pose, current_pose, dynamixel_robot_config, dynamixel_servo_controller, time=0.5)
    dynamixel_robot_config.d6 += direction


def reset_orientation(current_pose, dynamixel_robot_config, dynamixel_servo_controller):
    current_pose.flip = False
    dynamixel_robot_config.restore_initial_values()
    new_orientation = copy(current_pose)
    new_orientation.reset_orientation()
    pose_to_pose(current_pose, new_orientation, dynamixel_robot_config, dynamixel_servo_controller, time=1)
    return new_orientation


def playback_recorded_positions(current_pose, recorded_poses, dynamixel_robot_config, dynamixel_servo_controller):
    if len(recorded_poses) < 1:
        return
    current__playback_pose = pose_to_pose(current_pose, recorded_poses[0], dynamixel_robot_config,
                                          dynamixel_servo_controller, time=2)

    for pose in recorded_poses[1::]:
        current__playback_pose = line(current__playback_pose, pose, dynamixel_robot_config, dynamixel_servo_controller)
        sleep(1)

    pose_to_pose(current__playback_pose, current_pose, dynamixel_robot_config, dynamixel_servo_controller, time=2)


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
