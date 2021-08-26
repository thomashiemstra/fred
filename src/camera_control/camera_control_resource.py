from __future__ import division

import threading

from flask import Blueprint
from flask import jsonify

import src.global_objects as global_objects
from src import global_constants
from src.camera.image_handlers import get_default_aurco_image_handler, \
    get_default_board_to_board_image_handler
from src.camera_control.board_to_board_robot_controller import get_board_to_board_controller
from src.utils.decorators import synchronized_with_lock

camera_api = Blueprint('camera_api', __name__)

api_lock = threading.Lock()
started = False


class Handlers:

    def __init__(self):
        self.lock = threading.RLock()
        self._obstacle_avoidance = None
        self._aruco_image_handler = None
        self._board_to_board_handler = None

    @synchronized_with_lock("lock")
    def set_board_to_board_handler(self, val):
        self._board_to_board_handler = val

    @synchronized_with_lock("lock")
    def get_board_to_board_handler(self):
        return self._board_to_board_handler

    @synchronized_with_lock("lock")
    def set_aruco_image_handler(self, val):
        self._aruco_image_handler = val

    @synchronized_with_lock("lock")
    def get_aruco_image_handler(self):
        return self._aruco_image_handler

    @synchronized_with_lock("lock")
    def set_obstacle_avoidance(self, val):
        self._obstacle_avoidance = val

    @synchronized_with_lock("lock")
    def get_obstacle_avoidance(self):
        return self._obstacle_avoidance


object_handler = Handlers()


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
    global started
    with api_lock:
        if started:
            started = False
        else:
            return "already stopped"

    camera = global_objects.get_camera()
    camera.stop_camera()
    resp = jsonify(success=True)
    print("camera stopped")
    return resp


@camera_api.route('/start_aruco', methods=['POST'])
def start_aruco():
    global object_handler, started

    with api_lock:
        if not started:
            print("camera not started!")
            return jsonify(success=False)
        if object_handler.get_aruco_image_handler() is not None:
            print('already started')
            return jsonify(success=True)
        if object_handler.get_board_to_board_handler() is not None:
            print("let's not start 2 detectors at the same time, board to board already started!")
            return jsonify(success=True)
    camera = global_objects.get_camera()

    aruco_image_handler = get_default_aurco_image_handler()
    object_handler.set_aruco_image_handler(aruco_image_handler)
    camera.add_image_handler(aruco_image_handler)
    resp = jsonify(success=True)
    print("aruco started")
    return resp


@camera_api.route('/disable_aruco_draw', methods=['POST'])
def disable_aruco_draw():
    global object_handler, started

    with api_lock:
        if not started:
            print("camera not started!")
            return jsonify(success=False)
        if object_handler.get_aruco_image_handler() is None:
            print("aruco image handler not started")
            return jsonify(success=False)

    object_handler.get_aruco_image_handler().disable_draw()
    return jsonify(success=True)


@camera_api.route('/enable_aruco_draw', methods=['POST'])
def enable_aruco_draw():
    global object_handler, started

    with api_lock:
        if not started:
            print("camera not started!")
            return jsonify(success=False)
        if object_handler.get_aruco_image_handler() is None:
            print("aruco image handler not started")
            return jsonify(success=False)

    object_handler.get_aruco_image_handler().enable_draw()
    return jsonify(success=True)


@camera_api.route('/start_board_to_board', methods=['POST'])
def start_board_to_board():
    global started, object_handler

    with api_lock:
        if not started:
            print("camera not started!")
            return jsonify(success=False)
        if object_handler.get_aruco_image_handler() is not None:
            print("let's not start 2 detectors at the same time, aruco already started!")
            return jsonify(success=True)
        if object_handler.get_obstacle_avoidance() is not None:
            print('already started obstacle avoidance')
            return jsonify(success=True)

    camera = global_objects.get_camera()

    board_to_board_handler = get_default_board_to_board_image_handler()
    object_handler.set_board_to_board_handler(board_to_board_handler)
    camera.add_image_handler(board_to_board_handler)
    resp = jsonify(success=True)
    print("board to board started")
    return resp


@camera_api.route('/start_board_to_board_controller', methods=['POST'])
def start_board_to_board_controller():
    global object_handler, started
    with api_lock:
        if not started:
            print("camera not started!")
            return jsonify(success=False)
        if object_handler.get_board_to_board_handler() is None:
            print('no board to board image handler started')
            return jsonify(success=True)

    board_to_board_handler = object_handler.get_board_to_board_handler()
    controller = get_board_to_board_controller(board_to_board_handler)
    controller.start()
    return jsonify(success=True)


@camera_api.route('/stop_board_to_board_controller', methods=['POST'])
def stop_board_to_board_controller():
    global object_handler
    board_to_board_handler = object_handler.get_board_to_board_handler()
    if board_to_board_handler is None:
        print("no board to board controller running")
        return jsonify(success=True)

    controller = get_board_to_board_controller(board_to_board_handler)
    controller.stop()
    return jsonify(success=True)


@camera_api.route('/enable_filter_board_to_board_controller', methods=['POST'])
def enable_filter_board_to_board_controller():
    global object_handler
    board_to_board_handler = object_handler.get_board_to_board_handler()
    if board_to_board_handler is None:
        print("no board to board controller running")
        return jsonify(success=True)

    controller = get_board_to_board_controller(board_to_board_handler)
    controller.set_should_filter(True)
    return jsonify(success=True)


@camera_api.route('/disable_filter_board_to_board_controller', methods=['POST'])
def disable_filter_board_to_board_controller():
    global object_handler
    board_to_board_handler = object_handler.get_board_to_board_handler()
    if board_to_board_handler is None:
        print("no board to board controller running")
        return jsonify(success=True)

    controller = get_board_to_board_controller(board_to_board_handler)
    controller.set_should_filter(False)
    return jsonify(success=True)


@camera_api.route('/enable_servos', methods=['POST'])
def enable_servos():
    robot = global_objects.get_robot(global_constants.dynamixel_robot_arm_port)
    robot.enable_servos()
    return jsonify(success=True)


@camera_api.route('/disable_servos', methods=['POST'])
def disable_servos():
    robot = global_objects.get_robot(global_constants.dynamixel_robot_arm_port)
    robot.disable_servos()
    return jsonify(success=True)


@camera_api.route('/start_obstacle_avoidance', methods=['POST'])
def start_obstacle_avoidance():
    global object_handler
    with api_lock:
        if object_handler.get_aruco_image_handler() is None:
            print("aruco image handler not started")
            return jsonify(success=False)
        if object_handler.get_obstacle_avoidance() is not None:
            print("already started obstacle_avoidance")
            return jsonify(success=False)

    robot = global_objects.get_robot(global_constants.dynamixel_robot_arm_port)
    aruco_image_handler = object_handler.get_aruco_image_handler()

    from src.camera_control.obstacle_avoidance import ObstacleAvoidance
    obstacle_avoidance = ObstacleAvoidance(aruco_image_handler, robot)
    object_handler.set_obstacle_avoidance(obstacle_avoidance)

    print("obstacle avoidance started")
    return jsonify(success=True)


@camera_api.route('/create_and_set_scenario', methods=['POST'])
def create_and_set_scenario():
    global object_handler
    with api_lock:
        if object_handler.get_obstacle_avoidance() is None:
            print("no obstacle avoidance running!")
            return jsonify(success=False)

    object_handler.get_obstacle_avoidance().create_and_set_scenario()
    return jsonify(success=True)


@camera_api.route('/obstacle_avoidance_gradient_descent', methods=['POST'])
def obstacle_avoidance_gradient_descent():
    global object_handler
    with api_lock:
        if object_handler.get_obstacle_avoidance() is None:
            print("no obstacle avoidance running!")
            return jsonify(success=False)

    object_handler.get_obstacle_avoidance().obstacle_avoidance_gradient_descent()
    return jsonify(success=True)


@camera_api.route('/obstacle_avoidance_sac', methods=['POST'])
def obstacle_avoidance_sac():
    global object_handler
    with api_lock:
        if object_handler.get_obstacle_avoidance() is None:
            print("no obstacle avoidance running!")
            return jsonify(success=False)

    object_handler.get_obstacle_avoidance().obstacle_avoidance_sac()
    return jsonify(success=True)


@camera_api.route('/stop_obstacle_avoidance', methods=['POST'])
def stop_obstacle_avoidance():
    global object_handler
    with api_lock:
        if object_handler.get_obstacle_avoidance() is None:
            print('not obstacle avoidance running, nothing to stop')
            return jsonify(success=True)

    object_handler.get_obstacle_avoidance().stop()
    return jsonify(success=True)


