from servo_handling.servo_controller import ServoController
from kinematics.kinematics_utils import RobotConfig
from functools import lru_cache


dynamixel_robot_arm_port = 'COM5'
dynamixel_robot_config = RobotConfig(d1=9.1, a2=15.8, d4=22.0, d6=2.0)


@lru_cache(maxsize=10)
def get_robot(port):
    global dynamixel_robot_config
    return ServoController(port, dynamixel_robot_config)

