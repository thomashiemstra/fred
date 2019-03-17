from src.servo_handling.servo_controller import ServoController
from src.kinematics.kinematics_utils import RobotConfig
from functools import lru_cache

from src.xbox_controller.xbox_robot_controller import XboxRobotController

dynamixel_robot_arm_port = 'COM5'
dynamixel_robot_config = RobotConfig(d1=9.1, a2=15.8, d4=22.0, d6=2.0)


@lru_cache(maxsize=1)
def get_robot(port):
    global dynamixel_robot_config
    return ServoController(port, dynamixel_robot_config)


@lru_cache(maxsize=1)
def get_xbox_robot_controller(port):
    global dynamixel_robot_config
    dynamixel_servo_controller = get_robot(port)
    return XboxRobotController(dynamixel_robot_config, dynamixel_servo_controller)
