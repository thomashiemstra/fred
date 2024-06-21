import os

from src.kinematics.kinematics_utils import RobotConfig
from src.utils.os_utils import is_linux, get_project_root

dynamixel_robot_arm_port = '/dev/ttyUSB0' if is_linux() else 'COM3'

d6 = 12  # Length of the gripper
dynamixel_robot_config = RobotConfig(d1=13.92, a2=20, d4=22, d6=d6)
simulated_robot_config = RobotConfig(d1=13.92, a2=20, d4=22, d6=d6)

SERVO_1_LOW_CURRENT = 40
SERVO_2_LOW_CURRENT = 150
SERVO_1_HIGH_CURRENT = 1193
SERVO_2_HIGH_CURRENT = 2047

POSITION_CONTROL_MODE = 3
EXTENDED_POSITION_CONTROL = 4
CURRENT_BASED_POSITION_CONTROL_MODE = 5


steps_per_second = 100
recommended_max_servo_speed = 4  # rads/sec
use_simulation = False
root_dir = None

sac_network_weights = os.path.expanduser(os.path.dirname(get_project_root()) +
                                         '/src/reinforcementlearning/softActorCritic/'
                                         'checkpoints/rs_01_grid_new_network/train')


class WorkSpaceLimits:
    radius_min = 10
    radius_max = 57
    y_min = d6 + 0.1  # y=0 is bad for the IK calculation, so let's make sure that never happens
    z_min = -5
