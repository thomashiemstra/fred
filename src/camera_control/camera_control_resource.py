from __future__ import division

import threading

from flask import Blueprint, request
from flask import jsonify

import src.global_constants
import src.global_objects as global_objects
from src.camera.image_handlers import ArucoImageHandler, get_default_aurco_image_handler
from src.camera.util import get_calibrations

camera_api = Blueprint('camera_api', __name__)

api_lock = threading.Lock()
started = False
aruco_started = False
aruco_image_handler = None

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
    return resp


@camera_api.route('/start_aruco', methods=['POST'])
def start_aruco():
    global aruco_started, started

    with api_lock:
        if not started:
            print("camera not started!")
            return jsonify(success=False)
        if aruco_started:
            return jsonify(success=True)
        else:
            aruco_started = True
    camera = global_objects.get_camera()

    global aruco_image_handler
    aruco_image_handler = get_default_aurco_image_handler()
    camera.add_image_handler(aruco_image_handler)
    resp = jsonify(success=True)
    return resp