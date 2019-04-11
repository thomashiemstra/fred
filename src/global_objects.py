import src.global_constants
from src.camera.capture import CameraCapture
from src.dynamixel_robot.servo_controller import DynamixelRobotArm
from functools import lru_cache

from src.xbox_control.xbox360controller.controller_state_manager import ControllerStateManager
from src.xbox_control.xbox360controller.xbox_pose_updater import XboxPoseUpdater
from src.xbox_control.xbox_robot_controller import XboxRobotController


@lru_cache(maxsize=1)
def get_robot(port):
    return DynamixelRobotArm(port, src.global_constants.dynamixel_robot_config)


@lru_cache(maxsize=1)
def get_xbox_robot_controller(port):
    dynamixel_servo_controller = get_robot(port)
    controller_state_manager = ControllerStateManager()
    pose_poller = XboxPoseUpdater(controller_state_manager)

    return XboxRobotController(src.global_constants.dynamixel_robot_config, dynamixel_servo_controller, pose_poller)


@lru_cache(maxsize=1)
def get_camera(camera):
    return CameraCapture(camera)
