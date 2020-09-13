import threading
from timeit import default_timer as timer

import jsonpickle
import numpy as np
from Arduino import Arduino

from src.kinematics.kinematics import inverse_kinematics
from src.robot_controllers.abstract_robot_controller import AbstractRobotController
from src.robot_controllers.dynamixel_robot import dynamixel_x_config as cfg
from src.robot_controllers.dynamixel_robot.dynamixel_utils import setup_dynamixel_handlers
from src.robot_controllers.dynamixel_robot.servo_handler import ServoHandler
from src.utils.decorators import synchronized_with_lock
# Facade for the robot as a whole, abstracting away the servo handling
from src.utils.robot_controller_utils import get_recommended_wait_time

gripper_min_pwm = 60
gripper_max_pwm = 160
gripper_servo_pin = 9


def convert_gripper_state_to_pwm(state):
    return gripper_min_pwm + ((gripper_max_pwm - gripper_min_pwm) * (state / 100))


class DynamixelRobotController(AbstractRobotController):

    def __init__(self, port, robot_config, servo_config):
        """
        :param port: a string representing the usb port the robot is connected to
        :param robot_config: a RobotConfig object
        :param servo_config: (dict of str:json) servo name and servo config
        """
        self.servo1 = servo_config["servo1"]
        self.servo2 = servo_config["servo2"]
        self.servo3 = servo_config["servo3"]
        self.servo4 = servo_config["servo4"]
        self.servo5 = servo_config["servo5"]
        self.servo6 = servo_config["servo6"]

        port_handler, packet_handler, group_bulk_write, group_bulk_read = setup_dynamixel_handlers(port, cfg)

        base_servos = {1: self.servo1, 2: self.servo2, 3: self.servo3}
        self.base_servo_handler = ServoHandler(base_servos, cfg, port_handler,
                                               packet_handler, group_bulk_write, group_bulk_read)

        wrist_servos = {4: self.servo4, 5: self.servo5, 6: self.servo6}
        self. wrist_servo_handler = ServoHandler(wrist_servos, cfg, port_handler,
                                                 packet_handler, group_bulk_write, group_bulk_read)

        self.robot_config = robot_config
        self.set_velocity_profile()
        self.set_pid()
        self.status = False
        self.lock = threading.RLock()
        self._current_angles = self.get_current_angles()
        self.board = Arduino()
        self.gripper_state = 0  # 0 is completely open 100 is completely closed

    def get_servo_config(self):
        try:
            with open(self.servo_config_path, 'r') as servo_config_file:
                string = servo_config_file.read()
                return jsonpickle.decode(string)
        except FileNotFoundError:
            return None

    def enable_servos(self):
        self.base_servo_handler.set_torque(enable=True)
        self.wrist_servo_handler.set_torque(enable=True)
        self.board.Servos.attach(gripper_servo_pin, min=720, max=1240)

    def disable_servos(self):
        self.base_servo_handler.set_torque(enable=False)
        self.wrist_servo_handler.set_torque(enable=False)
        self.board.Servos.detach(9)

    def move_to_pose(self, pose):
        angles = inverse_kinematics(pose, self.robot_config)
        recommended_time = get_recommended_wait_time(self._current_angles, angles)
        time_taken = self.move_servos(angles)
        return recommended_time, time_taken

    # returns the time in seconds it took to move the servos
    def move_servos(self, angles):
        start = timer()
        self._current_angles = angles
        # First set the target_position variable of all servos
        self.base_servo_handler.set_angle(1, angles[1])
        self.base_servo_handler.set_angle(2, angles[2])
        self.base_servo_handler.set_angle(3, angles[3])

        self.wrist_servo_handler.set_angle(4, angles[4])
        self.wrist_servo_handler.set_angle(5, angles[5])
        self.wrist_servo_handler.set_angle(6, angles[6])

        # Next physically move the servos to their target_position
        self.base_servo_handler.move_to_angles()
        self.wrist_servo_handler.move_to_angles()
        end = timer()
        return end - start

    def set_gripper(self, new_gripper_state):
        pwm = convert_gripper_state_to_pwm(new_gripper_state)
        self.board.Servos.write(gripper_servo_pin, pwm)
        self.gripper_state = new_gripper_state

    def set_velocity_profile(self):
        self.base_servo_handler.set_profile_velocity_and_acceleration()
        self.wrist_servo_handler.set_profile_velocity_and_acceleration()

    def set_pid(self):
        self.base_servo_handler.set_pid()
        self.wrist_servo_handler.set_pid()

    def set_servo_torque(self, servo_id, enable):
        self.wrist_servo_handler.set_servo_torque(servo_id, enable)

    # debug function to control single servo
    def move_servo(self, servo_id, angle):
        if servo_id <= 3:
            self.base_servo_handler.set_angle(servo_id, angle)
            self.base_servo_handler.move_servo_to_angle(servo_id)
        else:
            self.wrist_servo_handler.set_angle(servo_id, angle)
            self.wrist_servo_handler.move_servo_to_angle(servo_id)

    def get_current_angles(self):
        self.base_servo_handler.read_current_pos()
        self.wrist_servo_handler.read_current_pos()
        angles = np.zeros(7, dtype=np.float64)

        angles[1] = self.base_servo_handler.get_angle(1, self.servo1.current_position)
        angles[2] = self.base_servo_handler.get_angle(2, self.servo2.current_position)
        angles[3] = self.base_servo_handler.get_angle(3, self.servo3.current_position)

        angles[4] = self.wrist_servo_handler.get_angle(4, self.servo4.current_position)
        angles[5] = self.wrist_servo_handler.get_angle(5, self.servo5.current_position)
        angles[6] = self.wrist_servo_handler.get_angle(6, self.servo6.current_position)

        return angles

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
        else:
            self.wrist_servo_handler.move_servo_to_pos(servo_id, pos)

    def get_servo_position(self, servo_id):
        if servo_id <= 3:
            return self.base_servo_handler.read_current_pos_single_servo(servo_id)
        else:
            return self.wrist_servo_handler.read_current_pos_single_servo(servo_id)

