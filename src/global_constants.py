from src.kinematics.kinematics_utils import RobotConfig
from src.utils.os_utils import is_linux

dynamixel_robot_arm_port = '/dev/ttyUSB0' if is_linux() else 'COM3'
# dynamixel_robot_config = RobotConfig(d1=9.1, a2=15.8, d4=21.9, d6=11)
dynamixel_robot_config = RobotConfig(d1=13.92, a2=20, d4=22, d6=12)
simulated_robot_config = RobotConfig(d1=13.92, a2=20, d4=22, d6=12)

servo_config_file = 'src/robot_controllers/dynamixel_robot/resources/servo_config.json'
steps_per_second = 25
recommended_max_servo_speed = 4  # rads/sec
use_simulation = False
root_dir = None


class WorkSpaceLimits:
    radius_min = 15
    radius_max = 57
    y_min = 10
    z_min = 0
