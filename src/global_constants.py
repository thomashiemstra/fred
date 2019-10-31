from src.kinematics.kinematics_utils import RobotConfig
from src.utils.os_utils import is_linux

dynamixel_robot_arm_port = '/dev/ttyUSB0' if is_linux() else 'COM5'
dynamixel_robot_config = RobotConfig(d1=9.1, a2=15.8, d4=21.9, d6=11)
simulated_robot_config = RobotConfig(d1=9.05, a2=15.8, d4=21.9, d6=10.0)
steps_per_second = 15
recommended_max_servo_speed = 4  # rads/sec
use_simulation = False
root_dir = None


class WorkSpaceLimits:
    x_min = -40
    x_max = 40
    y_min = 10
    y_max = 40
    z_min = 0
    z_max = 40
