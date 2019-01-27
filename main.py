from __future__ import division
from kinematics.kinematics_utils import Pose, RobotConfig
from kinematics.kinematics import inverse_kinematics
from kinematics.kinematics import forward_position_kinematics
from kinematics.kinematics import forward_orientation_kinematics
from servo_handling.servo_controller import ServoController
from time import sleep
from time import time as timing
import numpy as np
from numpy import pi
from math import ceil


def line(start_pose, stop_pose, flip, time, robot_config, servo_controller):
    """go from start to stop pose in time amount of seconds"""
    dx = stop_pose.x - start_pose.x
    dy = stop_pose.y - start_pose.y
    dz = stop_pose.z - start_pose.z

    steps = 50
    total_steps = ceil(time * steps)  # 50 steps per second
    dt = 1.0 / steps

    for i in range(total_steps + 1):
        t = i / total_steps

        x = start_pose.x + 3 * dx * (6 * np.power(t, 5) - 15 * np.power(t, 4) + 10 * np.power(t, 3))
        y = start_pose.y + 3 * dy * (6 * np.power(t, 5) - 15 * np.power(t, 4) + 10 * np.power(t, 3))
        z = start_pose.z + 3 * dz * (6 * np.power(t, 5) - 15 * np.power(t, 4) + 10 * np.power(t, 3))

        current_angles = inverse_kinematics(Pose(x, y, z, flip), robot_config)

        sleep(dt)
        servo_controller.move_servos(current_angles)


def point_to_point(start_pose, stop_pose, time, servo_controller, robot_config):
    start_angles = inverse_kinematics(start_pose, robot_config)
    stop_angles = inverse_kinematics(stop_pose, robot_config)

    """go from start to stop angles in time amount of seconds"""
    delta_angle = np.zeros(7, dtype=np.float64)

    for i in range(1, 7):
        delta_angle[i] = stop_angles[i] - start_angles[i]

    current_angles = start_angles.copy()

    steps = 50
    total_steps = ceil(time*steps)  # 50 steps per second
    dt = 1.0/steps

    start = timing()

    for i in range(total_steps + 1):
        t = i / total_steps

        for j in range(1, 7):
            current_angles[j] = start_angles[j] + delta_angle[j]*(6*np.power(t, 5) - 15*np.power(t, 4) + 10*np.power(t, 3))

        sleep(dt)

        servo_controller.move_servos(current_angles)

    stop = timing()
    print(stop-start)


def read_stuff():
    dynamixel_robot_config = RobotConfig(d1=9.1, a2=15.8, d4=21.9, d6=0)
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
    dynamixel_robot_config = RobotConfig(d1=9.1, a2=15.8, d4=21.9, d6=0)
    pose = Pose(0, 37.7, 9.1, flip=False, gamma=pi/2, beta=-pi/2)
    print(pose.orientation)
    angles = inverse_kinematics(pose, dynamixel_robot_config)
    for i in range(1, 7):
        print("angle{} = {}pi".format(i, round(angles[i] / pi, 2)))


if __name__ == '__main__':
    dynamixel_robot_config = RobotConfig(d1=9.1, a2=15.8, d4=21.9, d6=2)
    dynamixel_servo_controller = ServoController("COM5")
    dynamixel_servo_controller.enable_servos()
    dynamixel_servo_controller.set_velocity_profile()

    angles = dynamixel_servo_controller.get_angles()
    p1, p2, p3, p4, p6 = forward_position_kinematics(angles, dynamixel_robot_config)
    rot_matrix = forward_orientation_kinematics(angles)

    start_pose = Pose(p6[0], p6[1], p6[2])
    start_pose.orientation = rot_matrix.copy()
    goal_pose = Pose(-20, 10, 15)
    goal_pose_1 = Pose(0, 30, 15)

    time = 2

    point_to_point(start_pose, goal_pose, time, dynamixel_servo_controller, dynamixel_robot_config)
    point_to_point(goal_pose, goal_pose_1, time*2, dynamixel_servo_controller, dynamixel_robot_config)
    point_to_point(goal_pose_1, goal_pose, time*2, dynamixel_servo_controller, dynamixel_robot_config)
    point_to_point(goal_pose, start_pose, time, dynamixel_servo_controller, dynamixel_robot_config)

    dynamixel_servo_controller.disable_servos()
