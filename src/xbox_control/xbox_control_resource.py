from __future__ import division


from typing import Any

from flask import Blueprint, Response, request
from flask import jsonify

import src.global_objects as global_objects
from src.kinematics.kinematics_utils import Pose

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper  # type: ignore[assignment]  # only the fallback differs from the C versions

xbox_api = Blueprint('xbox_api', __name__)

# the range of the maximum speed of the end effector in cm/s, the one the d-pad changes it in
MINIMUM_SPEED = 5
MAXIMUM_SPEED = 50


class InvalidRequest(Exception):
    """the request is wrong, the answer is the message with status 400 (or the given status)"""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


@xbox_api.errorhandler(InvalidRequest)
def handle_invalid_request(error: InvalidRequest) -> tuple[Response, int]:
    return jsonify(success=False, error=error.message), error.status



@xbox_api.route('/test', methods=['GET'])
def test() -> Response:
    resp = jsonify(Pose(5.5, 2.0, 3.0).__dict__)
    return resp

@xbox_api.route('/start', methods=['POST'])
def start() -> str | Response:
    # the xbox robot controller keeps track of whether it is running, and makes sure it only starts once
    xbox_robot_controller = global_objects.get_xbox_robot_controller()
    if xbox_robot_controller.is_running():
        return "already started"
    return jsonify(success=xbox_robot_controller.start())


@xbox_api.route('/stop', methods=['POST'])
def stop() -> str | Response:
    xbox_robot_controller = global_objects.get_xbox_robot_controller()
    if not xbox_robot_controller.stop():
        return "already stopped"
    return jsonify(success=True)


@xbox_api.route('/findcenter', methods=['POST'])
def define_center() -> Response:
    # the control loop moves the robot, so it can't be done in this request
    xbox_robot_controller = global_objects.get_xbox_robot_controller()
    return jsonify(success=xbox_robot_controller.request_find_center_mode())


@xbox_api.route('/clearcenter', methods=['POST'])
def clear_center() -> Response:
    xbox_robot_controller = global_objects.get_xbox_robot_controller()
    xbox_robot_controller.clear_center()
    resp = jsonify(success=True)
    return resp


@xbox_api.route('/saveToFile', methods=['POST'])
def save_to_file() -> Response:
    name = get_filename()
    xbox_robot_controller = global_objects.get_xbox_robot_controller()
    try:
        success = xbox_robot_controller.save_recorded_moves_to_file(name)
    except ValueError as e:
        raise InvalidRequest(str(e)) from e
    return jsonify(success=success)


@xbox_api.route('/restoreFromFile', methods=['POST'])
def play_from_file() -> Response:
    name = get_filename()
    xbox_robot_controller = global_objects.get_xbox_robot_controller()
    try:
        xbox_robot_controller.restore_recorded_moves_from_file(name)
    except FileNotFoundError as e:
        raise InvalidRequest('no recorded moves with this name', 404) from e
    except ValueError as e:
        raise InvalidRequest(str(e)) from e
    resp = jsonify(success=True)
    return resp


@xbox_api.route('/clearAll',  methods=['POST'])
def clear_all() -> Response:
    xbox_robot_controller = global_objects.get_xbox_robot_controller()
    xbox_robot_controller.clear_recorded_moves_and_positions()
    resp = jsonify(success=True)
    return resp


@xbox_api.route('/setSpeed', methods=['POST'])
def set_speed() -> Response:
    speed = get_int_parameter('speed')
    if speed < MINIMUM_SPEED or speed > MAXIMUM_SPEED:
        raise InvalidRequest('speed should be between {} and {}'.format(MINIMUM_SPEED, MAXIMUM_SPEED))
    xbox_robot_controller = global_objects.get_xbox_robot_controller()
    xbox_robot_controller.set_maximum_speed(speed)
    return jsonify(success=True)


@xbox_api.route('/setScenarioId', methods=['POST'])
def set_scenario() -> Response:
    scenario_id = get_int_parameter('scenario_id')
    xbox_robot_controller = global_objects.get_xbox_robot_controller()
    success = xbox_robot_controller.set_scenario(scenario_id)
    return jsonify(success=success)


@xbox_api.route('/speed', methods=['POST'])
def speed() -> Response:
    percentage = get_int_parameter('percentage')
    if not (0 < percentage <= 100):
        raise InvalidRequest('percentage should be between 1 and 100')

    xbox_robot_controller = global_objects.get_xbox_robot_controller()
    xbox_robot_controller.set_profile_velocity_percentage(percentage)
    resp = jsonify(success=True)
    return resp

# @xbox_api.route('/video_feed')
# def video_feed():
#     return Response(get_image(), mimetype='multipart/x-mixed-replace; boundary=frame')


def get_filename() -> str:
    """the name of the recorded moves, the controller decides whether it is allowed"""
    name = get_parameter('filename')
    if not isinstance(name, str):
        raise InvalidRequest('filename should be a string')
    return name


def get_parameter(param_name: str) -> Any:
    """:raises InvalidRequest: if the body is not a json object or doesn't have the parameter"""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise InvalidRequest('the body should be a json object')
    if param_name not in body:
        raise InvalidRequest('{} is missing'.format(param_name))
    return body[param_name]


def get_int_parameter(param_name: str) -> int:
    """a whole number, or a string of one (the forms of the ui send strings)"""
    value = get_parameter(param_name)
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise InvalidRequest('{} should be a whole number'.format(param_name))
    try:
        return int(value)
    except ValueError as e:
        raise InvalidRequest('{} should be a whole number'.format(param_name)) from e
