from __future__ import division

import threading

from flask import Blueprint
from flask import jsonify

import src.global_objects as global_objects
from src.camera.image_handlers import get_default_aurco_image_handler, \
    get_default_board_to_board_image_handler
from src.camera_control.board_to_board_robot_controller import get_board_to_board_controller

camera_api = Blueprint('camera_api', __name__)

api_lock = threading.Lock()
started = False
aruco_started = False
board_to_board_started = False
aruco_image_handler = None
board_to_board_handler = None


@camera_api.route('/start', methods=['POST'])
def start():
    global started
    with api_lock:
        if started:
            return "already started"
    camera = global_objects.get_camera()
    camera.start_camera()
    with api_lock:
        started = True
    resp = jsonify(success=True)
    print("camera started")
    return resp


@camera_api.route('/stop', methods=['POST'])
def stop():
    global started, aruco_started
    with api_lock:
        if started:
            started = False
            aruco_started = False
        else:
            return "already stopped"

    camera = global_objects.get_camera()
    camera.stop_camera()
    resp = jsonify(success=True)
    print("camera stopped")
    return resp


@camera_api.route('/start_aruco', methods=['POST'])
def start_aruco():
    global aruco_started, board_to_board_started, started

    with api_lock:
        if not started:
            print("camera not started!")
            return jsonify(success=False)
        if aruco_started:
            print('already started')
            return jsonify(success=True)
        if board_to_board_started:
            print("let's not start 2 detectors at the same time, board to board already started!")
            return jsonify(success=True)
        else:
            aruco_started = True
    camera = global_objects.get_camera()

    global aruco_image_handler
    aruco_image_handler = get_default_aurco_image_handler()
    camera.add_image_handler(aruco_image_handler)
    resp = jsonify(success=True)
    print("aruco started")
    return resp


@camera_api.route('/start_board_to_board', methods=['POST'])
def start_board_to_board():
    global aruco_started, board_to_board_started, started

    with api_lock:
        if not started:
            print("camera not started!")
            return jsonify(success=False)
        if aruco_started:
            print("let's not start 2 detectors at the same time, aruco already started!")
            return jsonify(success=True)
        if board_to_board_started:
            print('already started')
            return jsonify(success=True)
        else:
            board_to_board_started = True
    camera = global_objects.get_camera()

    global board_to_board_handler
    board_to_board_handler = get_default_board_to_board_image_handler()
    camera.add_image_handler(board_to_board_handler)
    resp = jsonify(success=True)
    print("board to board started")
    return resp


@camera_api.route('/start_board_to_board_controller', methods=['POST'])
def start_board_to_board_controller():
    global board_to_board_started, started
    with api_lock:
        if not started:
            print("camera not started!")
            return jsonify(success=False)
        if not board_to_board_started:
            print('no board to board image handler started')
            return jsonify(success=True)

    global board_to_board_handler
    controller = get_board_to_board_controller(board_to_board_handler)
    controller.start()
    print("board to board controller stopped")
    return jsonify(success=True)


@camera_api.route('/stop_board_to_board_controller', methods=['POST'])
def stop_board_to_board_controller():
    controller = get_board_to_board_controller(board_to_board_handler)
    controller.stop()
    return jsonify(success=True)
