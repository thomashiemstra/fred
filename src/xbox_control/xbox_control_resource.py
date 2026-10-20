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
    filename = get_filename()
    if filename is None:
        return jsonify(success=False)

    xbox_robot_controller = global_objects.get_xbox_robot_controller()
    success = xbox_robot_controller.save_recorded_moves_to_file(filename)
    return jsonify(success=success)


@xbox_api.route('/restoreFromFile', methods=['POST'])
def play_from_file() -> Response:
    filename = get_filename()
    if filename is None:
        return jsonify(success=False)

    xbox_robot_controller = global_objects.get_xbox_robot_controller()
    xbox_robot_controller.restore_recorded_moves_from_file(filename)
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
    raw_speed = get_parameter('speed')
    if raw_speed is None:
        return jsonify(success=False)
    speed = int(raw_speed)
    if speed < 0 or speed > 50:
        return jsonify(success=False)
    xbox_robot_controller = global_objects.get_xbox_robot_controller()
    xbox_robot_controller.set_maximum_speed(speed)
    return jsonify(success=True)


@xbox_api.route('/setScenarioId', methods=['POST'])
def set_scenario() -> Response:
    scenario_id = get_parameter('scenario_id')
    if scenario_id is None:
        return jsonify(success=False)
    xbox_robot_controller = global_objects.get_xbox_robot_controller()
    success = xbox_robot_controller.set_scenario(int(scenario_id))
    return jsonify(success=success)


@xbox_api.route('/speed', methods=['POST'])
def speed() -> Response:
    raw_percentage = get_parameter("percentage")
    if raw_percentage is None:
        return jsonify(success=False)
    percentage = int(raw_percentage)
    if not (0 < percentage <= 100):
        return jsonify(success=False)

    xbox_robot_controller = global_objects.get_xbox_robot_controller()
    xbox_robot_controller.set_profile_velocity_percentage(percentage)
    resp = jsonify(success=True)
    return resp

# @xbox_api.route('/video_feed')
# def video_feed():
#     return Response(get_image(), mimetype='multipart/x-mixed-replace; boundary=frame')


def get_filename() -> str | None:
    try:
        return request.get_json()['filename'] + '.json'
    except KeyError:
        return None


def get_parameter(param_name: str) -> Any:
    try:
        return request.get_json()[param_name]
    except KeyError:
        return None
