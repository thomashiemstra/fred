from src.camera.capture import CameraCapture
from src.dynamixel_robot.servo_controller import DynamixelRobotArm
from src.kinematics.kinematics_utils import RobotConfig
from functools import lru_cache

from src.xbox_control.xbox360controller.xbox_poller import XboxPoller
from src.xbox_control.xbox360controller.xbox_pose_updater import XboxPoseUpdater
from src.xbox_control.xbox_robot_controller import XboxRobotController

dynamixel_robot_arm_port = 'COM5'
dynamixel_robot_config = RobotConfig(d1=9.0, a2=15.8, d4=22.0, d6=5.5)
steps_per_second = 15


@lru_cache(maxsize=1)
def get_robot(port):
    global dynamixel_robot_config
    return DynamixelRobotArm(port, dynamixel_robot_config)


@lru_cache(maxsize=1)
def get_xbox_robot_controller(port):
    global dynamixel_robot_config
    dynamixel_servo_controller = get_robot(port)
    poller = XboxPoller()
    pose_poller = XboxPoseUpdater(poller)

    return XboxRobotController(dynamixel_robot_config, dynamixel_servo_controller, pose_poller)


@lru_cache(maxsize=1)
def get_camera(camera):
    return CameraCapture(camera)
