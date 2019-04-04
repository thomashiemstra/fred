from __future__ import division

from src.kinematics.kinematics import forward_orientation_kinematics
from src.kinematics.kinematics import forward_position_kinematics
from src.kinematics.kinematics_utils import Pose, RobotConfig
from src.dynamixel_robot.servo_controller import DynamixelRobotArm
from yaml import load, dump

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


if __name__ == '__main__':
    dynamixel_robot_config = RobotConfig(d1=9.1, a2=15.8, d4=22.0, d6=2.0)
    dynamixel_servo_controller = DynamixelRobotArm("COM5", dynamixel_robot_config)
    # dynamixel_servo_controller.enable_servos()
    dynamixel_servo_controller.set_velocity_profile()
    dynamixel_servo_controller.set_pid()

    angles = dynamixel_servo_controller.get_angles()
    p1, p2, p3, p4, p6 = forward_position_kinematics(angles, dynamixel_robot_config)
    rot_matrix = forward_orientation_kinematics(angles)

    start_pose = Pose(p6[0], p6[1], p6[2])
    start_pose.orientation = rot_matrix.copy()

    lift_pose = Pose(start_pose.x, start_pose.y, start_pose.z + 10)

    pose_1 = Pose(-20, 20, 15, time=2)
    pose_2 = Pose(20, 20, 15, time=2)
    pose_3 = Pose(0, 20, 15, time=2)
    pose_4 = Pose(0, 20, 30, time=2)
    pose_5 = Pose(0, 15, 5, time=2)
    pose_6 = Pose(15, 30, 5, time=2)

    positions = [pose_1, pose_2, pose_3, pose_4, pose_5, pose_6]

    for pose in positions:
        print(pose)

    print("--------------")
    with open('test.yml', 'w') as outfile:
        dump(positions, outfile)

    with open('test.yml', 'r') as infile:
        read_pos = load(infile)

    print(read_pos[0])
    # for pose in read_pos:
    #     print(pose)
    # current_pose = point_to_point(start_pose, lift_pose, 1, dynamixel_robot_config, dynamixel_servo_controller)
    # current_pose = point_to_point(current_pose, pose_5, 2, dynamixel_robot_config, dynamixel_servo_controller)
    #
    # for pose in positions:
    #     current_pose = line(current_pose, pose, dynamixel_robot_config, dynamixel_servo_controller)
    #     # input("Press Enter to continue...")
    #
    # current_pose = point_to_point(current_pose, lift_pose, 1, dynamixel_robot_config, dynamixel_servo_controller)
    # current_pose = point_to_point(current_pose, start_pose, 2, dynamixel_robot_config, dynamixel_servo_controller)
    #
    # dynamixel_servo_controller.disable_servos()
