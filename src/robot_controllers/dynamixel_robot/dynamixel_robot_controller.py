import sys
import threading
from timeit import default_timer as timer

import numpy as np

from src.kinematics.kinematics import inverse_kinematics, forward_position_kinematics
from src.robot_controllers.abstract_robot_controller import AbstractRobotController
from src.robot_controllers.dynamixel_robot import dynamixel_x_config as cfg
from src.robot_controllers.dynamixel_robot.dynamixel_utils import setup_dynamixel_handlers
from src.robot_controllers.dynamixel_robot.servo_configurations import servo_configs
from src.robot_controllers.dynamixel_robot.servo_handler import ServoHandler
from src.utils.decorators import synchronized_with_lock
from src.utils.movement_utils import from_current_angles_to_pose
from src.utils.robot_controller_utils import get_recommended_wait_time, servo_2_check, servo_1_check
from time import sleep


# Facade for the robot as a whole, abstracting away the servo handling
class DynamixelRobotController(AbstractRobotController):

    def __init__(self, port, robot_config, servos=servo_configs, perform_safety_checks=True):
        """
        :param port: a string representing the usb port the robot is connected to
        :param robot_config: a RobotConfig object
        :param servos: (dict of str:json) servo name and servo config
        """
        self.robot_config = robot_config
        self.perform_safety_checks = perform_safety_checks

        self.servo1 = servos[0]
        self.servo2 = servos[1]
        self.servo3 = servos[2]
        self.servo4 = servos[3]
        self.servo5 = servos[4]
        self.servo6 = servos[5]
        self.servo7 = servos[6]
        self._servos = servos

        port_handler, packet_handler, group_bulk_write, group_bulk_read = setup_dynamixel_handlers(port, cfg)

        base_servos = {1: self.servo1, 2: self.servo2, 3: self.servo3}
        self.base_servo_handler = ServoHandler(base_servos, cfg, port_handler,
                                               packet_handler, group_bulk_write, group_bulk_read)

        wrist_servos = {4: self.servo4, 5: self.servo5, 6: self.servo6}
        self.wrist_servo_handler = ServoHandler(wrist_servos, cfg, port_handler,
                                                packet_handler, group_bulk_write, group_bulk_read)

        gripper_servos = {7: self.servo7}
        self.gripper_servo_handler = ServoHandler(gripper_servos, cfg, port_handler,
                                                  packet_handler, group_bulk_write, group_bulk_read)

        self.initialize_servos_or_exit()
        self._current_angles = self.get_current_angles_or_exit()
        self.status = False
        self.lock = threading.RLock()

        self.gripper_state = 0  # 0 is completely open 100 is completely closed
        self.counter = 0
        self._recorder = None

    def set_recorder(self, recorder):
        self._recorder = recorder

    def initialize_servos_or_exit(self):
        success = self.set_control_mode()
        success = success & self.set_velocity_profile()
        success = success & self.set_pid()
        # TODO set min and max pos in servo directly
        if self.perform_safety_checks and not success:
            print('failed to setup the dynamixel robot, exiting')
            sys.exit()

    def get_current_angles_or_exit(self):
        angles = self.get_current_angles()
        if angles is None and self.perform_safety_checks:
            print('failed to setup the dynamixel robot, exiting')
            sys.exit()
        return angles

    def enable_servos(self):
        if self.perform_safety_checks and not self.safety_check():
            print('failed the safety check, disabling servos!')
            self.disable_servos()
            return

        self.base_servo_handler.set_torque(enable=True)
        self.wrist_servo_handler.set_torque(enable=True)
        self.gripper_servo_handler.set_torque(enable=True)

    def safety_check(self):
        positions = self.get_current_positions()
        return servo_1_check(positions) and servo_2_check(positions)

    def disable_servos(self):
        self.base_servo_handler.set_torque(enable=False)
        self.wrist_servo_handler.set_torque(enable=False)
        self.gripper_servo_handler.set_torque(enable=False)

    def move_to_pose(self, pose):
        angles = inverse_kinematics(pose, self.robot_config)
        recommended_time = get_recommended_wait_time(self._current_angles, angles)
        time_taken = self.move_servos(angles)
        return recommended_time, time_taken

    def move_to_pose_and_give_new_angles(self, pose):
        angles = inverse_kinematics(pose, self.robot_config)
        self.move_servos(angles)
        return angles

    def get_current_gripper_position(self):
        angles = self.get_current_angles()
        _, _, _, _, p6 = forward_position_kinematics(angles, self.robot_config)
        return p6

    # returns the time in seconds it took to move the servos
    def move_servos(self, angles):
        start = timer()
        # if self.counter % 10 == 0:
        #     current_positions = self.get_current_positions()
        #
        #     previous_target_positions = np.zeros(7, dtype=np.long)
        #     previous_target_positions[1] = self.servo1.unmodified_target_position
        #     previous_target_positions[2] = self.servo2.unmodified_target_position
        #     previous_target_positions[3] = self.servo3.unmodified_target_position
        #     previous_target_positions[4] = self.servo4.unmodified_target_position
        #     previous_target_positions[5] = self.servo5.unmodified_target_position
        #     previous_target_positions[6] = self.servo6.unmodified_target_position
        #
        #     diff = np.zeros(7, dtype=np.long)
        #     for i in range(1, 7):
        #         diff[i] = current_positions[i] - previous_target_positions[i]
        #
        #     print("1: {} 2:{} 3:{}, 4:{}, 5:{}, 6:{}".format(diff[1], diff[2], diff[3], diff[4], diff[5], diff[6]))
        #     self.counter = 0
        # self.counter += 1

        if self._recorder is not None:
            self.base_servo_handler.read_current_pos()
            self.wrist_servo_handler.read_current_pos()
            self._recorder.record(self._servos)

        self._current_angles = angles
        # First set the target_position variable of all servos
        self.base_servo_handler.set_angle(1, angles[1])
        self.base_servo_handler.set_angle(2, angles[2], angles)
        self.base_servo_handler.set_angle(3, angles[3], angles)

        self.wrist_servo_handler.set_angle(4, angles[4])
        self.wrist_servo_handler.set_angle(5, angles[5])
        self.wrist_servo_handler.set_angle(6, angles[6])

        # Next physically move the servos to their target_position
        self.base_servo_handler.move_to_angles()
        self.wrist_servo_handler.move_to_angles()
        end = timer()
        return end - start

    def set_gripper(self, new_gripper_state):
        """
        directly control the gripper on the robot
        :param new_gripper_state: value between 0 and 100 0 being fully closed 100 fully open
        :return:
        """
        self.gripper_servo_handler.read_current_pos()

        current_state = self.gripper_servo_handler.get_angle(7, self.servo7.current_position)

        delta = int(new_gripper_state - current_state)
        if abs(delta) > 10:
            for i in range(abs(delta)):
                current_state = current_state + np.sign(delta)
                self.gripper_servo_handler.set_angle(7, int(current_state))
                self.gripper_servo_handler.move_to_angles()
                sleep(0.1)
        else:
            self.gripper_servo_handler.set_angle(7, new_gripper_state)
            self.gripper_servo_handler.move_to_angles()

    def set_velocity_profile(self):
        success = self.base_servo_handler.set_profile_velocity_and_acceleration()
        success = success & self.wrist_servo_handler.set_profile_velocity_and_acceleration()
        success = success & self.gripper_servo_handler.set_profile_velocity_and_acceleration()
        return success

    def set_goal_current(self):
        success = self.base_servo_handler.set_configured_goal_current()
        success = success & self.wrist_servo_handler.set_configured_goal_current()
        success = success & self.gripper_servo_handler.set_configured_goal_current()
        return success

    def set_control_mode(self):
        self.base_servo_handler.set_configured_operating_mode()
        self.wrist_servo_handler.set_configured_operating_mode()
        self.gripper_servo_handler.set_configured_operating_mode()
        return True

    def set_pid(self):
        success = self.base_servo_handler.set_pid()
        success = success & self.wrist_servo_handler.set_pid()
        success = success & self.gripper_servo_handler.set_pid()
        return success

    def set_profile_velocity_percentage(self, percentage):
        success = self.base_servo_handler.set_profile_velocity_percentage(percentage)
        success = success & self.wrist_servo_handler.set_profile_velocity_percentage(percentage)
        success = success & self.gripper_servo_handler.set_profile_velocity_percentage(percentage)
        return success

    # debug function to control single servo
    def move_servo(self, servo_id, angle, all_angles=None):
        if servo_id <= 3:
            self.base_servo_handler.set_angle(servo_id, angle, all_angles)
            self.base_servo_handler.move_servo_to_angle(servo_id)
        else:
            self.wrist_servo_handler.set_angle(servo_id, angle, all_angles)
            self.wrist_servo_handler.move_servo_to_angle(servo_id)

    def get_current_angles(self):
        success = self.base_servo_handler.read_current_pos()
        success = success & self.wrist_servo_handler.read_current_pos()
        if not success:
            return None

        angles = np.zeros(7, dtype=np.float64)

        angles[1] = self.base_servo_handler.get_angle(1, self.servo1.current_position)
        angles[2] = self.base_servo_handler.get_angle(2, self.servo2.current_position)
        angles[3] = self.base_servo_handler.get_angle(3, self.servo3.current_position)

        angles[4] = self.wrist_servo_handler.get_angle(4, self.servo4.current_position)
        angles[5] = self.wrist_servo_handler.get_angle(5, self.servo5.current_position)
        angles[6] = self.wrist_servo_handler.get_angle(6, self.servo6.current_position)

        return angles

    def get_current_positions(self):
        self.base_servo_handler.read_current_pos()
        self.wrist_servo_handler.read_current_pos()
        positions = np.zeros(7, dtype=np.int_)

        positions[1] = self.servo1.current_position
        positions[2] = self.servo2.current_position
        positions[3] = self.servo3.current_position
        positions[4] = self.servo4.current_position
        positions[5] = self.servo5.current_position
        positions[6] = self.servo6.current_position

        return positions

    def pose_to_angles(self, pose):
        return inverse_kinematics(pose, self.robot_config)

    @synchronized_with_lock("lock")
    def get_status(self):
        return self.status

    @synchronized_with_lock("lock")
    def change_status(self, new_state):
        self.status = new_state

    def set_pid_single_servo(self, servo_id, p, i, d):
        if servo_id <= 3:
            self.base_servo_handler.set_pid_single_servo(servo_id, p, i, d)
        else:
            self.wrist_servo_handler.set_pid_single_servo(servo_id, p, i, d)

    def set_servo_position(self, servo_id, pos):
        if servo_id <= 3:
            self.base_servo_handler.move_servo_to_pos(servo_id, pos)
        elif 3 < servo_id <= 6:
            self.wrist_servo_handler.move_servo_to_pos(servo_id, pos)
        elif servo_id == 7:
            self.gripper_servo_handler.move_servo_to_pos(servo_id, pos)

    def get_servo_position(self, servo_id):
        if servo_id <= 3:
            return self.base_servo_handler.read_current_pos_single_servo(servo_id)
        else:
            return self.wrist_servo_handler.read_current_pos_single_servo(servo_id)

    def reset_to_pose(self, pose):
        from_current_angles_to_pose(pose, self, 4)
