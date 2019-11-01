from src.global_objects import get_robot
from src.kinematics.kinematics_utils import Pose
from src.utils.movement_utils import from_current_angles_to_pose
import numpy as np

robot = get_robot('COM5')
robot.enable_servos()

pose = Pose(0, 21.9, 13.9, gamma=-np.pi/2)
from_current_angles_to_pose(pose, robot, 2)

angles = robot.pose_to_angles(pose)

print(angles*(180 / np.pi))
