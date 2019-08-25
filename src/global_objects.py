from src import global_constants
from src.camera.capture import CameraCapture

from functools import lru_cache

from src.simulation.simulation_utils import start_simulated_robot
from src.xbox_control.xbox360controller.controller_state_manager import ControllerStateManager
from src.xbox_control.xbox360controller.xbox_pose_updater import XboxPoseUpdater
from src.xbox_control.xbox_robot_controller import XboxRobotController


@lru_cache(maxsize=1)
def get_robot(port):
    if global_constants.use_simulation:
        return start_simulated_robot(use_gui=True)
    else:
        from src.robot_controllers.dynamixel_robot.dynamixel_robot_controller import DynamixelRobotController
        return DynamixelRobotController(port, global_constants.dynamixel_robot_config)


@lru_cache(maxsize=1)
def get_xbox_robot_controller(port):
    dynamixel_servo_controller = get_robot(port)
    controller_state_manager = ControllerStateManager()
    pose_poller = XboxPoseUpdater(controller_state_manager)

    return XboxRobotController(global_constants.dynamixel_robot_config, dynamixel_servo_controller, pose_poller)


@lru_cache(maxsize=1)
def get_camera(camera):
    return CameraCapture(camera)
