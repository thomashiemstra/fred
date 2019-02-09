from __future__ import division
from flask import jsonify

import time

from flask import Blueprint

from kinematics.kinematics_utils import RobotConfig
from movement_utils import point_to_point
from servo_handling.servo_controller import ServoController
from xbox_controller.xbox_utill import PosePoller
import threading
from utils.threading_utils import CountDownLatch
from copy import copy

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


def run_xbox_poller(countdown_latch):
    pose_poller = PosePoller()
    dynamixel_robot_config = RobotConfig(d1=9.1, a2=15.8, d4=22.0, d6=2.0)
    dynamixel_servo_controller = ServoController("COM5", dynamixel_robot_config)
    dynamixel_servo_controller.enable_servos()

    start_pose = dynamixel_servo_controller.get_current_pose()

    xbox_pose = copy(start_pose)
    xbox_pose.z += 5
    xbox_pose.reset_orientation()

    point_to_point(start_pose, xbox_pose, 1, dynamixel_robot_config, dynamixel_servo_controller)
    countdown_latch.count_down()

    global done
    while True:
        with api_lock:
            if done:
                break
        pose_poller.update_positions_of_pose(xbox_pose)

        dynamixel_servo_controller.move_to_pose(xbox_pose)

        time.sleep(pose_poller.dt)

    pose_poller.stop()

    current_pose = dynamixel_servo_controller.get_current_pose()
    lift_pose = copy(start_pose)
    lift_pose.z += 5

    current_pose = point_to_point(current_pose, lift_pose, 5, dynamixel_robot_config, dynamixel_servo_controller)
    point_to_point(current_pose, start_pose, 2, dynamixel_robot_config, dynamixel_servo_controller)
    time.sleep(1)
    dynamixel_servo_controller.disable_servos()


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
