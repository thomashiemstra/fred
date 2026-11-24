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
    Where the arm may go, in cm from the origin on the ground under the base. The gripper tip has to stay at least
    radius_min from the origin, at y >= y_min and at z >= z_min. The wrist centre (the axis of joint 5) has to stay
    within wrist_reach_max of the shoulder (the axis of joint 2, d1 above the origin) and at z >= wrist_z_min: when the
    gripper points up the wrist servo body hangs about 1.8cm below the wrist centre, so this keeps it off the table.
    The tip of the model (d6 from the wrist centre) is the point where the gripper grips, the closed fingers reach
    about 1cm beyond it (150mm from the wrist centre, measured on 24 November 2026), so z_min keeps the fingertips 1cm
    above the table when the gripper points straight down. The geometry the heights come from (d1, a2, d4 and the zeros
    of joints 1 and 2) was confirmed in the jig on that day. The arm rests on the table with the gripper pointing
    forward, its tip at about 3.7cm and its wrist centre at about 2.3cm by the kinematics. Neither limit may be higher
    than where the arm rests, or no move can start from rest.

    The reach is measured from the shoulder because that is where the arm reaches from: the wrist centre can get at
    most a2 + d4 = 42cm from it, with the elbow straight. wrist_reach_max keeps the elbow bent at least about 11
    degrees, away from the straight elbow where the two solutions of the inverse kinematics meet and a straight line
    needs unbounded joint speed (check_move's speed check refuses those moves too). Near the straight elbow the reach
    hardly changes with the bend, 40cm would already keep it bent 35 degrees and refuse the poses of the gain tuning.
    There is no maximum height, nothing hangs above the robot. Until October 2026 the tip had to stay within a sphere
    around the origin on the ground instead, see "Workspace limits.md".

    The check is one place, AbstractRobotController.check_workspace(angles) (src/utils/workspace.py): whoever moves
    the robot runs it before executing a move. The planner (src/utils/movement.py) checks every step of a planned move,
    the xbox control loop and the playback check every pose, the recording script checks its poses. move_servos
    itself checks nothing, so it stays fast.
    """
    radius_min = 10
    wrist_reach_max = 41.8  # the wrist centre from the shoulder, the elbow bent at least about 11 degrees
    y_min = d6 + 0.1  # y=0 is bad for the IK calculation, so let's make sure that never happens
    # the gripper tip of the model, the point where the gripper grips: the closed fingers reach 1cm beyond it (150mm
    # from the wrist centre, 24 November 2026), so with the gripper pointing down the fingertips stay 1cm above the
    # table. The arm rests with its tip at about 3.7cm, so moves can start from there
    z_min = 2
    wrist_z_min = 1  # the wrist centre, just under the resting height of the wrist servo lying on the table
