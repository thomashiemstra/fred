from __future__ import division

import threading

from flask import Blueprint
from flask import jsonify

import src.global_constants
from src.kinematics.kinematics_utils import Pose
import src.global_objects as global_objects


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

    xbox_robot_controller = global_objects.get_xbox_robot_controller(src.global_constants.dynamixel_robot_arm_port)
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
    xbox_robot_controller = global_objects.get_xbox_robot_controller(src.global_constants.dynamixel_robot_arm_port)
    xbox_robot_controller.stop()
    xbox_robot_controller.dynamixel_servo_controller.change_status(False)

    resp = jsonify(success=True)
    return resp


@xbox_api.route('/findcenter', methods=['POST'])
def define_center():
    xbox_robot_controller = global_objects.get_xbox_robot_controller(src.global_constants.dynamixel_robot_arm_port)
    xbox_robot_controller.start_find_center_mode()
    resp = jsonify(success=True)
    return resp


@xbox_api.route('/clearcenter', methods=['POST'])
def clear_center():
    xbox_robot_controller = global_objects.get_xbox_robot_controller(src.global_constants.dynamixel_robot_arm_port)
    xbox_robot_controller.clear_center()
    resp = jsonify(success=True)
    return resp


@xbox_api.route('/runfile/<file>', methods=['POST'])
def run_file(file):
    file = file + '.yml'
    return "doing " + file
