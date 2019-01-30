from __future__ import division
from kinematics.kinematics_utils import Pose, RobotConfig
from kinematics.kinematics import inverse_kinematics
from kinematics.kinematics import forward_position_kinematics
from kinematics.kinematics import forward_orientation_kinematics
from servo_handling.servo_controller import ServoController
from time import sleep
import numpy as np
from numpy import pi
from math import ceil


def line(start_pose, stop_pose, time, robot_config, servo_controller):
    """go from start to stop pose in time amount of seconds"""
    flip = start_pose.flip if start_pose.flip == stop_pose.flip else stop_pose.flip
    dx = stop_pose.x - start_pose.x
    dy = stop_pose.y - start_pose.y
    dz = stop_pose.z - start_pose.z
    d_alpha = stop_pose.alpha - start_pose.alpha
    d_beta = stop_pose.beta - start_pose.beta
    d_gamma = stop_pose.gamma - start_pose.gamma

    steps = 20
    total_steps = ceil(time * steps)  # 50 steps per second
    dt = 1.0 / steps

    current_angles = inverse_kinematics(start_pose, robot_config)
    servo_controller.move_servos(current_angles)

    for i in range(total_steps + 1):
        t = i / total_steps

        curve_value = (6 * np.power(t, 5) - 15 * np.power(t, 4) + 10 * np.power(t, 3))
        x = start_pose.x + dx * curve_value
        y = start_pose.y + dy * curve_value
        z = start_pose.z + dz * curve_value

        alpha = start_pose.alpha + d_alpha * curve_value
        beta = start_pose.beta + d_beta * curve_value
        gamma = start_pose.gamma + d_gamma * curve_value

        temp_pose = Pose(x, y, z, flip, alpha, beta, gamma)

        current_angles = inverse_kinematics(temp_pose, robot_config)
        # angles = dynamixel_servo_controller.get_angles()

        servo_controller.move_servos(current_angles)
        sleep(dt)

        # diff_2 = abs(current_angles[2] - angles[2])
        # diff_3 = abs(current_angles[3] - angles[3])
        # print(angles[2], current_angles[2])
        # print(diff_2*57)

    angles = servo_controller.get_angles()
    # for i in range(1, 7):
    #     print("angle{} difference = {}pi".format(i, round((current_angles[i] -angles[i])/pi, 4)))

    return stop_pose


def point_to_point(start_pose, stop_pose, time, robot_config, servo_controller):
    start_angles = inverse_kinematics(start_pose, robot_config)
    stop_angles = inverse_kinematics(stop_pose, robot_config)

    """go from start to stop angles in time amount of seconds"""
    delta_angle = np.zeros(7, dtype=np.float64)

    for i in range(1, 7):
        delta_angle[i] = stop_angles[i] - start_angles[i]

    current_angles = start_angles.copy()

    steps = 10
    total_steps = ceil(time*steps)  # 50 steps per second
    dt = 1.0/steps

    for i in range(total_steps + 1):
        t = i / total_steps

        for j in range(1, 7):
            current_angles[j] = start_angles[j] + delta_angle[j]*(6*np.power(t, 5) - 15*np.power(t, 4) + 10*np.power(t, 3))

        sleep(dt)

        servo_controller.move_servos(current_angles)

    return stop_pose


def read_stuff():
    dynamixel_robot_config = RobotConfig(d1=9.1, a2=15.8, d4=21.9, d6=2)
    servo_controller = ServoController("COM5")

    angles = servo_controller.get_angles()

    for i in range(1, 7):
        print("angle{} = {}pi".format(i, round(angles[i] / pi, 2)))

    p1, p2, p3, p4, p6 = forward_position_kinematics(angles, dynamixel_robot_config)
    rot_matrix = forward_orientation_kinematics(angles)

    print(p6)

    # print(Pose(0, 0, 0).orientation)

    print(rot_matrix)


def ik_test():
    dynamixel_robot_config = RobotConfig()
    pose = Pose(0, 25, 9.1, flip=False)
    print(pose.orientation)
    angles = inverse_kinematics(pose, dynamixel_robot_config)
    p1, p2, p3, p4, p6 = forward_position_kinematics(angles, dynamixel_robot_config)
    for i in range(1, 7):
        print("angle{} = {}pi".format(i, round(angles[i] / pi, 2)))


if __name__ == '__main__':
    dynamixel_robot_config = RobotConfig(d1=9.1, a2=15.8, d4=22.0, d6=2.0)
    dynamixel_servo_controller = ServoController("COM5")
    dynamixel_servo_controller.enable_servos()
    dynamixel_servo_controller.set_velocity_profile()
    dynamixel_servo_controller.set_pid()

    angles = dynamixel_servo_controller.get_angles()
    p1, p2, p3, p4, p6 = forward_position_kinematics(angles, dynamixel_robot_config)
    rot_matrix = forward_orientation_kinematics(angles)

    start_pose = Pose(p6[0], p6[1], p6[2])
    start_pose.orientation = rot_matrix.copy()

    lift_pose = Pose(start_pose.x, start_pose.y, start_pose.z + 5)

    pose_1 = Pose(-20, 20, 15)
    pose_2 = Pose(20, 20, 15)
    pose_3 = Pose(0, 20, 15)
    pose_4 = Pose(0, 20, 30)
    pose_5 = Pose(0, 15, 5)
    pose_6 = Pose(0, 35, 5)

    # positions = [(pose_1, 2), (pose_2, 3), (pose_3, 2), (pose_4, 2), (pose_5, 3), (pose_6, 2), (pose_3, 3), (pose_1, 3)]

    positions = [(pose_5, 2), (pose_6, 3), (pose_5, 3)]

    current_pose = point_to_point(start_pose, lift_pose, 2, dynamixel_robot_config, dynamixel_servo_controller)

    for position in positions:
        pose, time = position
        current_pose = line(current_pose, pose, time, dynamixel_robot_config, dynamixel_servo_controller)
        # input("Press Enter to continue...")

    current_pose = point_to_point(current_pose, lift_pose, 2, dynamixel_robot_config, dynamixel_servo_controller)
    current_pose = point_to_point(current_pose, start_pose, 2, dynamixel_robot_config, dynamixel_servo_controller)

    dynamixel_servo_controller.disable_servos()
