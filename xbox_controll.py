from __future__ import division

import time

from src.kinematics.kinematics import forward_orientation_kinematics
from src.kinematics.kinematics import forward_position_kinematics
from src.kinematics.kinematics import inverse_kinematics
from src.kinematics.kinematics_utils import Pose, RobotConfig
from src.utils.movement_utils import pose_to_pose
from src.dynamixel_robot.servo_controller import DynamixelRobotArm
from src.xbox_controller.xbox_poller import XboxPoller


def input_to_delta_velocity(controller_input, velocity, maximum_velocity):
    new_velocity = 0
    if controller_input > 0:
        new_velocity = velocity + dv if velocity < maximum_velocity else maximum_velocity
    elif controller_input < 0:
        new_velocity = velocity - dv if velocity > maximum_velocity else maximum_velocity
    else:
        if velocity > 0:
            new_velocity = velocity - dv if velocity - dv > 0 else 0
        elif velocity < 0:
            new_velocity = velocity + dv if velocity + dv < 0 else 0
    return new_velocity


def limit_position(position, min, max):
    if position > max:
        position = max
    if position < min:
        position = min
    return position


def get_xyz():
    x_in, y_in = poller.get_left_thumb()
    z_in = -poller.get_lr_trigger()
    return x_in, y_in, z_in


if __name__ == '__main__':
    v_x = 0
    v_y = 0
    v_z = 0

    steps_per_second = 15
    dt = 1.0 / steps_per_second
    maximum_speed = 20.0  # cm/sec
    ramp_up_time = 0.1  # 1 second to reach max speed
    dv = maximum_speed / (ramp_up_time * steps_per_second)  # v/step

    poller = XboxPoller()

    # ________________ Dynamixel stuff ________________
    dynamixel_robot_config = RobotConfig(d1=9.1, a2=15.8, d4=22.0, d6=2.0)
    dynamixel_servo_controller = DynamixelRobotArm("COM5")
    dynamixel_servo_controller.enable_servos()
    dynamixel_servo_controller.set_velocity_profile()
    dynamixel_servo_controller.set_pid()

    angles = dynamixel_servo_controller.get_angles()
    p1, p2, p3, p4, p6 = forward_position_kinematics(angles, dynamixel_robot_config)
    rot_matrix = forward_orientation_kinematics(angles)
    pos_x, pos_y, pos_z = p6[0], p6[1], p6[2]

    start_pose = Pose(pos_x, pos_y, pos_z)
    start_pose.orientation = rot_matrix.copy()

    pos_z += 5
    xbox_pose = Pose(pos_x, pos_y, pos_z)

    current_pose = pose_to_pose(start_pose, xbox_pose, 1, dynamixel_robot_config, dynamixel_servo_controller)

    try:
        while True:
            x, y, z = get_xyz()

            v_x_max = maximum_speed * (x / 100)
            v_y_max = maximum_speed * (y / 100)
            v_z_max = maximum_speed * (z / 100)

            v_x = input_to_delta_velocity(x, v_x, v_x_max)
            v_y = input_to_delta_velocity(y, v_y, v_y_max)
            v_z = input_to_delta_velocity(z, v_z, v_z_max)

            pos_x += dt * v_x
            pos_y += dt * v_y
            pos_z += dt * v_z

            pos_x = limit_position(pos_x, -30, 30)
            pos_y = limit_position(pos_y, 10, 40)
            pos_z = limit_position(pos_z, 5, 35)

            xbox_pose.x = pos_x
            xbox_pose.y = pos_y
            xbox_pose.z = pos_z

            current_angles = inverse_kinematics(xbox_pose, dynamixel_robot_config)
            dynamixel_servo_controller.move_servos(current_angles)

            time.sleep(dt)

    except KeyboardInterrupt:
        print("stopped")

    finally:
        poller.stop()
