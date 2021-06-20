from __future__ import division

import threading

from flask import Blueprint, request
from flask import jsonify

import src.global_constants
import src.global_objects as global_objects


camera_api = Blueprint('camera_api', __name__)

api_lock = threading.Lock()
started = False
done = False
running_thread = None


@camera_api.route('/start', methods=['POST'])
def start():
    global started
    with api_lock:
        if started:
            return "already started"
        else:
            started = True
    camera = global_objects.get_camera()
    camera.start_camera()
    resp = jsonify(success=True)
    return resp


@camera_api.route('/stop', methods=['POST'])
def stop():
    global started
    with api_lock:
        if started:
            started = False
        else:
            return "already stopped"

    camera = global_objects.get_camera()
    camera.stop_camera()
    resp = jsonify(success=True)
    return resp
