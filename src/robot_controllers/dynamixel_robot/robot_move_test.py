"""
Test script that moves the physical robot, run it from the project root:
    python -m src.robot_controllers.dynamixel_robot.robot_move_test

1. reads the current pose of the robot with forward kinematics
2. plans a straight line to the start pose, the whole movement is checked before anything moves. The wrist keeps
   its current flip, both flips reach the same pose
3. asks for confirmation, enables the torque and moves the robot
4. reads the pose again and compares it to the start pose

Ctrl+C stops the movement, the servos keep holding their last position until the torque is disabled.
"""
import logging
import sys
from time import sleep

import numpy as np

from src import global_constants
from src.kinematics.kinematics import forward_kinematics
from src.kinematics.kinematics_utils import Pose
from src.robot_controllers.dynamixel_robot.dynamixel_robot_controller import DynamixelRobotController
from src.utils.movement import plan_move_to_pose, execute_move, check_move
from src.utils.movement_exception import MovementException

# all the way to the right, the same start pose as the xbox controller
START_POSE = Pose(30, 25.0, 5)
# the angle of a servo outside of its calibrated range is clamped, the first step would make it jump this far
MAX_OUTSIDE_CALIBRATED_RANGE = 50  # positions, 4096 positions is a full turn


def print_pose(name, pose):
    print('{}: x={:.1f} y={:.1f} z={:.1f} alpha={:.2f} beta={:.2f} gamma={:.2f} flip={}'.format(
        name, pose.x, pose.y, pose.z, pose.alpha, pose.beta, pose.gamma, pose.flip))


def print_angles(name, angles):
    print('{}: {}'.format(name, ' '.join('{}={:.1f}deg'.format(joint, np.degrees(angles[joint]))
                                         for joint in range(1, 7))))


def check_calibrated_range(robot):
    outside = robot.get_servos_outside_calibrated_range()
    if outside is None:
        print('failed to read the servo positions')
        return False

    ok = True
    for servo_id, (position, (lower, upper)) in outside.items():
        distance = lower - position if position < lower else position - upper
        print('servo {} is at position {}, {} positions outside of its calibrated range [{}, {}]'.format(
            servo_id, position, distance, lower, upper))
        if distance > MAX_OUTSIDE_CALIBRATED_RANGE:
            ok = False
    if not ok:
        print('the servos would jump at the start of the movement, move the arm closer to its range by hand')
    return ok


def compare_to_target(robot, target_pose):
    pose = robot.get_current_pose()
    if pose is None:
        print('failed to read the pose after the movement')
        return
    print_pose('reached pose', pose)
    position_error = np.linalg.norm([pose.x - target_pose.x, pose.y - target_pose.y, pose.z - target_pose.z])
    print('position error: {:.2f}cm'.format(position_error))


def disable_torque_on_request(robot):
    try:
        input('\npress enter to disable the torque, SUPPORT THE ARM, it will drop. '
              'Ctrl+C to exit and keep the torque enabled\n')
    except KeyboardInterrupt:
        print('\nexiting, the servos keep holding their position')
        return
    robot.disable_servos()


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

    robot = DynamixelRobotController(None, global_constants.dynamixel_robot_config)

    if not check_calibrated_range(robot):
        return 1

    current_angles = robot.get_current_angles()
    if current_angles is None:
        print('failed to read the current angles')
        return 1
    print_angles('current angles', current_angles)
    print_pose('current pose', forward_kinematics(current_angles, robot.robot_config))
    print_pose('target pose', START_POSE)

    try:
        move = plan_move_to_pose(robot, current_angles, START_POSE, linear_speed=5.0)
    except MovementException as e:
        print('can not move to the start pose: {}'.format(e))
        return 1

    print('\nplanned movement: {}'.format(move))
    if input('\ntype yes to move the robot: ').strip().lower() != 'yes':
        print('not moving')
        return 0

    if not robot.enable_servos():
        print('failed to enable the servos')
        return 1

    try:
        # the arm could have been moved by hand while waiting for the confirmation
        angles_after_enabling = robot.get_current_angles()
        if angles_after_enabling is None:
            raise MovementException('failed to read the angles after enabling the servos')
        check_move(move, robot, current_angles=angles_after_enabling)
        execute_move(robot, move)
        sleep(0.5)  # let the servos settle
        compare_to_target(robot, START_POSE)
    except MovementException as e:
        print('stopped: {}'.format(e))
    except KeyboardInterrupt:
        print('\nstopped, the servos are holding their last position')

    disable_torque_on_request(robot)
    return 0


if __name__ == '__main__':
    sys.exit(main())
