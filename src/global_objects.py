import logging
import sys

from src import global_constants

from functools import lru_cache

from src.xbox_control.xbox360controller.XboxController import XboxController


@lru_cache(maxsize=1)
def get_robot(port):
    if global_constants.use_simulation:
        from src.simulation.simulation_utils import start_simulated_robot
        return start_simulated_robot(use_gui=True)
    else:
        from src.robot_controllers.dynamixel_robot.dynamixel_robot_controller import DynamixelRobotController
        return DynamixelRobotController(port, global_constants.dynamixel_robot_config)


def get_servo_config(servo_config_path):
    try:
        with open(servo_config_path, 'r') as servo_config_file:
            return servo_config_file.read()
    except FileNotFoundError:
        logging.error("error getting servo config, exiting")
        sys.exit()


@lru_cache(maxsize=1)
def get_xbox_robot_controller(port):
    from src.xbox_control.xbox360controller.xbox_pose_updater import XboxPoseUpdater
    from src.xbox_control.xbox_robot_controller import XboxRobotController

    dynamixel_servo_controller = get_robot(port)
    controller = XboxController(dead_zone=30, scale=100)
    pose_updater = XboxPoseUpdater(controller)

    return XboxRobotController(global_constants.dynamixel_robot_config, dynamixel_servo_controller, pose_updater)


@lru_cache(maxsize=1)
def get_camera(camera=0):
    from src.camera.capture import CameraCapture
    return CameraCapture(camera, [])