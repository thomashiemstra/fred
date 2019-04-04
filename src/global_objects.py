from src.camera.capture import CameraCapture
from src.dynamixel_robot.servo_controller import DynamixelRobotArm
from src.kinematics.kinematics_utils import RobotConfig
from functools import lru_cache

from src.xbox_controller.xbox_robot_controller import XboxRobotController

dynamixel_robot_arm_port = 'COM5'
dynamixel_robot_config = RobotConfig(d1=9.1, a2=15.8, d4=22.0, d6=5.0)
steps_per_second = 10


@lru_cache(maxsize=1)
def get_robot(port):
    global dynamixel_robot_config
    return DynamixelRobotArm(port, dynamixel_robot_config)


@lru_cache(maxsize=1)
def get_xbox_robot_controller(port):
    global dynamixel_robot_config
    dynamixel_servo_controller = get_robot(port)
    return XboxRobotController(dynamixel_robot_config, dynamixel_servo_controller)


@lru_cache(maxsize=1)
def get_camera(camera):
    return CameraCapture(camera)
