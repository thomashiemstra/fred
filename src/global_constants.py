from src.kinematics.kinematics_utils import RobotConfig

dynamixel_robot_arm_port = 'COM5'
dynamixel_robot_config = RobotConfig(d1=9.05, a2=15.8, d4=21.9, d6=5.5)
steps_per_second = 15


class WorkSpaceLimits:
    x_min = -40
    x_max = 40
    y_min = 16
    y_max = 40
    z_min = 5
    z_max = 40
