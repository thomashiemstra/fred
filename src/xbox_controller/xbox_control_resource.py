from __future__ import division

import threading
import time
from copy import copy

import numpy as np
from flask import Blueprint
from flask import jsonify

from src.kinematics.kinematics_utils import Pose
from src.utils.movement_utils import pose_to_pose, line
from src.xbox_controller.pose_poller import PosePoller
from time import sleep
import src.globals as globals

from yaml import dump


try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

xbox_api = Blueprint('xbox_api', __name__)

api_lock = threading.Lock()
started = False
done = False
running_thread = None


@xbox_api.route('/test', methods=['GET'])
def test():
    resp = jsonify(Pose(5.5, 2.0, 3.0).__dict__)
    return resp


@xbox_api.route('/testpost', methods=['POST'])
def test_post(json):
    print(json)


@xbox_api.route('/start', methods=['POST'])
def start():
    global started
    with api_lock:
        if started:
            return "already started"
        else:
            started = True

    xbox_robot_controller = globals.get_xbox_robot_controller(globals.dynamixel_robot_arm_port)
    success = xbox_robot_controller.start()
    if success:
        xbox_robot_controller.dynamixel_servo_controller.change_status(True)
    resp = jsonify(success=success)
    return resp


@xbox_api.route('/stop', methods=['POST'])
def stop():
    global started
    with api_lock:
        if started:
            started = False
        else:
            return "already stopped"
    xbox_robot_controller = globals.get_xbox_robot_controller(globals.dynamixel_robot_arm_port)
    xbox_robot_controller.stop()
    xbox_robot_controller.dynamixel_servo_controller.change_status(False)

    resp = jsonify(success=True)
    return resp


@xbox_api.route('/runfile/<file>', methods=['POST'])
def run_file(file):
    file = file + '.yml'
    return "doing " + file


# TODO convert to class
def run_xbox_poller(countdown_latch):
    # xbox_robot_controller = XboxRobotController(dynamixel_robot_config, dynamixel_servo_controller)
    pose_poller = PosePoller()
    dynamixel_servo_controller = globals.get_robot(globals.dynamixel_robot_arm_port)
    dynamixel_servo_controller.change_status(True)
    dynamixel_robot_config = globals.dynamixel_robot_config
    dynamixel_servo_controller.enable_servos()
    print(dynamixel_servo_controller)
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
                dump(recorded_positions, outfile)
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
    current_pose = reset_orientation(current_pose, dynamixel_robot_config, dynamixel_servo_controller)

    pose_to_pose(current_pose, start_pose, dynamixel_robot_config, dynamixel_servo_controller, time=3)

    time.sleep(1)
    dynamixel_servo_controller.disable_servos()
    dynamixel_servo_controller.change_status(False)


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


