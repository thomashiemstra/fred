from __future__ import division

import threading

from flask import Blueprint, request
from flask import jsonify

import src.global_constants
from src.kinematics.kinematics_utils import Pose
import src.global_objects as global_objects
import numpy as np
import cv2
import pybullet as p
from flask import Response

from src.simulation.simulation_utils import start_simulated_robot
from src.utils.decorators import timer
import time

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
        xbox_robot_controller.servo_controller.change_status(True)
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
    xbox_robot_controller.servo_controller.change_status(False)

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


@xbox_api.route('/saveToFile', methods=['POST'])
def save_to_file():
    filename = get_filename()
    if filename is None:
        return jsonify(success=False)

    xbox_robot_controller = global_objects.get_xbox_robot_controller(src.global_constants.dynamixel_robot_arm_port)
    success = xbox_robot_controller.save_recorded_moves_to_file(filename)
    return jsonify(success=success)


@xbox_api.route('/restoreFromFile', methods=['POST'])
def play_from_file():
    filename = get_filename()
    if filename is None:
        return jsonify(success=False)

    xbox_robot_controller = global_objects.get_xbox_robot_controller(src.global_constants.dynamixel_robot_arm_port)
    xbox_robot_controller.restore_recorded_moves_from_file(filename)
    resp = jsonify(success=True)
    return resp


@xbox_api.route('/clearAll',  methods=['POST'])
def clear_all():
    xbox_robot_controller = global_objects.get_xbox_robot_controller(src.global_constants.dynamixel_robot_arm_port)
    xbox_robot_controller.clear_recorded_moves_and_positions()
    resp = jsonify(success=True)
    return resp


@xbox_api.route('/setSpeed', methods=['POST'])
def set_speed():
    raw_speed = get_parameter('speed')
    if raw_speed is None:
        return jsonify(success=False)
    speed = int(raw_speed)
    if speed < 0 or speed > 50:
        return jsonify(success=False)
    xbox_robot_controller = global_objects.get_xbox_robot_controller(src.global_constants.dynamixel_robot_arm_port)
    xbox_robot_controller.set_maximum_speed(speed)
    return jsonify(success=True)


@xbox_api.route('/video_feed')
def video_feed():
    return Response(get_image(), mimetype='multipart/x-mixed-replace; boundary=frame')


def get_filename():
    try:
        return request.get_json()['filename'] + '.json'
    except KeyError:
        return None


def get_parameter(param_name):
    try:
        return request.get_json()[param_name]
    except KeyError:
        return None

@timer
def get_image():
    while True:
        width = 320
        height = 240

        fov = 60
        aspect = width / height
        near = 0.02
        far = 10

        camTargetPos = [-0.2, 0.3, 0.4]
        # camTargetPos=[0, 0, 0]
        camDistance = 0.5
        yaw = -60.0
        pitch = -40.0
        roll = 0
        upAxisIndex = 2

        viewMatrix = p.computeViewMatrixFromYawPitchRoll(camTargetPos,
                                                         camDistance,
                                                         yaw,
                                                         pitch,
                                                         roll,
                                                         upAxisIndex)

        projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

        # Get depth values using the OpenGL renderer
        width, height, rgbImg, depthImg, segImg = p.getCameraImage(width,
                                                                   height,
                                                                   viewMatrix,
                                                                   projection_matrix,
                                                                   renderer=p.ER_BULLET_HARDWARE_OPENGL)
        # ER_TINY_RENDERER

        img = np.array(rgbImg)[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        frame = cv2.imencode('.jpg', img)[1].tobytes()

        return b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        # yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#
#
robot = start_simulated_robot(use_gui=False)


for _ in range(100):
    get_image()