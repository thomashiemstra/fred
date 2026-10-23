import os

from src.robot_config import load_robot_configs
from src.utils.os_utils import get_project_root


# the link lengths of the robots come from src/robot_config.yaml
_robot_configs = load_robot_configs()
dynamixel_robot_config = _robot_configs['dynamixel_robot']
simulated_robot_config = _robot_configs['simulated_robot']
d6 = dynamixel_robot_config.d6  # Length of the gripper

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
root_dir: str | None = None

sac_network_weights = os.path.expanduser(os.path.dirname(get_project_root()) +
                                         '/src/reinforcementlearning/softActorCritic/'
                                         'checkpoints/rs_01_grid_new_network/train')

class WorkSpaceLimits:
    """
    Where the arm may go, in cm from the origin on the ground under the base. The gripper tip has to stay between
    radius_min and radius_max from the origin, at y >= y_min and at z >= z_min. The wrist centre (the axis of joint 5)
    has to stay at z >= wrist_z_min: when the gripper points up the wrist servo body hangs about 1.8cm below the
    wrist centre, so this keeps it off the table while the tip may come down to the table. It can't be higher than
    the wrist centre of the arm resting on the table (the wrist servo lies on it), or no move can start from rest.

    The check is one place, AbstractRobotController.check_workspace(angles) (src/utils/workspace.py): whoever moves
    the robot runs it before executing a move. The planner (src/utils/movement.py) checks every step of a planned move,
    the xbox control loop and the playback check every pose, the recording script checks its poses. move_servos
    itself checks nothing, so it stays fast.
    """
    radius_min = 10
    radius_max = 60
    y_min = d6 + 0.1  # y=0 is bad for the IK calculation, so let's make sure that never happens
    z_min = -1  # the gripper tip, the fingers may rest on the table
    wrist_z_min = 1.5  # the wrist centre, just under the resting height of the wrist servo lying on the table
