"""
The api of the block stacking page. Every answer is json with 'success'; an error also has 'error' and a status that
fits: 400 for a bad request, 409 when it is not possible now (the arm is busy), 503 when the feature can not be set up.
The page only asks and shows, what the arm does is decided and kept safe by the BlockStacker.
"""
import threading
from typing import Any

from flask import Blueprint, Response, jsonify, request

import src.global_objects as global_objects
from src.block_stacking.block_stacker import BlockStacker
from src.block_stacking.stacking_config import StackingConfigError
from src.xbox_control.xbox_control_resource import InvalidRequest

block_api = Blueprint('block_api', __name__)

BUSY = 'the arm is busy, wait until it is done or stop it'

# the camera keeps its image handlers over stop and start, so the block handler is added once per process
_camera_lock = threading.Lock()
_handler_added = False


@block_api.errorhandler(InvalidRequest)
def handle_invalid_request(error: InvalidRequest) -> tuple[Response, int]:
    return jsonify(success=False, error=error.message), error.status


def get_stacker() -> BlockStacker:
    """
    the block stacker, which does not open the robot or the camera when it is built
    :raises InvalidRequest: with status 503 when the calibration of the camera or the stacking config is not usable
    """
    try:
        return global_objects.get_block_stacker()
    except (FileNotFoundError, StackingConfigError) as e:
        raise InvalidRequest('block stacking can not be set up: {}'.format(e), 503) from e


def refuse_while_gamepad_runs() -> None:
    """
    two programs must not move the arm at once
    :raises InvalidRequest: with status 409 when the gamepad controls the robot
    """
    if global_objects.xbox_controller_running():
        raise InvalidRequest('the gamepad is controlling the robot, stop it first', 409)


def status_response(stacker: BlockStacker) -> Response:
    return jsonify(success=True, **stacker.status())


@block_api.route('/camera/start', methods=['POST'])
def start_camera() -> Response:
    global _handler_added
    camera = global_objects.get_camera()
    with _camera_lock:
        if not _handler_added:
            # the handler first: a missing calibration or a bad config must not leave a camera running that can not
            # find blocks, and the first pictures are already searched
            try:
                handler = global_objects.get_block_image_handler()
            except (FileNotFoundError, StackingConfigError) as e:
                raise InvalidRequest('block detection can not be set up: {}'.format(e), 503) from e
            camera.add_image_handler(handler)
            _handler_added = True
        camera.start_camera()
    return jsonify(success=True)


@block_api.route('/camera/stop', methods=['POST'])
def stop_camera() -> Response:
    global_objects.get_camera().stop_camera()
    return jsonify(success=True)


@block_api.route('/status', methods=['GET'])
def status() -> Response:
    return status_response(get_stacker())


@block_api.route('/scan', methods=['POST'])
def scan() -> Response:
    stacker = get_stacker()
    # the scan looks for seconds, and answers when it is done
    if not stacker.scan():
        raise InvalidRequest(BUSY, 409)
    return status_response(stacker)


@block_api.route('/start', methods=['POST'])
def start() -> Response:
    hover_only = get_hover_only()
    refuse_while_gamepad_runs()
    if not get_stacker().start(hover_only=hover_only):
        raise InvalidRequest(BUSY, 409)
    return jsonify(success=True)


@block_api.route('/click_mode', methods=['POST'])
def click_mode() -> Response:
    on = get_click_mode_switch()
    if on:
        refuse_while_gamepad_runs()
        if not get_stacker().start_click_mode():
            raise InvalidRequest('the arm is busy, stop it first', 409)
    else:
        stacker = get_stacker()
        # the switch must not stop a normal stacking run, that is what the Stop button is for
        if not stacker.status()['click_mode']:
            raise InvalidRequest('the click mode is not on', 409)
        if not stacker.stop():
            raise InvalidRequest('the click mode has just ended, there is nothing to stop', 409)
    return jsonify(success=True)


@block_api.route('/stop', methods=['POST'])
def stop() -> Response:
    if not get_stacker().stop():
        raise InvalidRequest('nothing is running, there is nothing to stop', 409)
    return jsonify(success=True)


@block_api.route('/lay_down', methods=['POST'])
def lay_down() -> Response:
    # laying down moves the arm, which the gamepad must not do at the same time
    refuse_while_gamepad_runs()
    if not get_stacker().lay_down():
        raise InvalidRequest(BUSY, 409)
    return jsonify(success=True)


@block_api.route('/disable_servos', methods=['POST'])
def disable_servos() -> Response:
    if not get_stacker().disable_servos():
        raise InvalidRequest(BUSY, 409)
    return jsonify(success=True)


@block_api.route('/tip', methods=['POST'])
def tip() -> Response:
    result = get_stacker().read_tip()
    if 'error' in result:
        raise InvalidRequest(str(result['error']), 409)
    return jsonify(success=True, **result)


def get_click_mode_switch() -> bool:
    """
    the required 'on' of the json body of the click mode switch
    :raises InvalidRequest: if the body is not a json object with a bool 'on'
    """
    body: Any = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise InvalidRequest('the body should be a json object')
    on = body.get('on')
    if not isinstance(on, bool):
        raise InvalidRequest('on should be true or false')
    return on


def get_hover_only() -> bool:
    """
    the optional hover_only of the json body, a rehearsal when true
    :raises InvalidRequest: if the body is not a json object or hover_only is not a bool
    """
    if not request.get_data():
        return False
    body: Any = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise InvalidRequest('the body should be a json object')
    hover_only = body.get('hover_only', False)
    if not isinstance(hover_only, bool):
        raise InvalidRequest('hover_only should be true or false')
    return hover_only
