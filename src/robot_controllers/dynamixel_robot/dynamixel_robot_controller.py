from src.robot_controllers.abstract_robot_controller import AbstractRobotController
from src.robot_controllers.dynamixel_robot import dynamixel_x_config as cfg
from src.robot_controllers.dynamixel_robot.dynamixel_utils import setup_dynamixel_handlers
from src.robot_controllers.dynamixel_robot.servo import Servo
from src.robot_controllers.dynamixel_robot.servo_handler import ServoHandler
from src.kinematics.kinematics import inverse_kinematics
from src.utils.decorators import synchronized_with_lock
import threading

import numpy as np
from numpy import pi

# todo make global and check it, it's too late now....
recommended_max_servo_speed = 0.001  # rads/sec


def get_recommended_wait_time(current_angles, new_angles):
    recommended_time = 0
    for current_angle, new_angle in zip(current_angles, new_angles):
        delta_angle = current_angle - new_angle
        delta_angle = np.abs(delta_angle)
        time = delta_angle / recommended_max_servo_speed
        recommended_time = np.max(time, recommended_time)

    return recommended_time


# Facade for the robot as a whole, abstracting away the servo handling
class DynamixelRobotController(AbstractRobotController):

    def __init__(self, port, robot_config):
        self.robot_config = robot_config
        port_handler, packet_handler, group_bulk_write, group_bulk_read = setup_dynamixel_handlers(port, cfg)

        # todo these PID and speed values should be in a file...
        self.servo1 = Servo(1024, 3072, 0, pi, 80, 30, p=800, i=0, d=2500)
        self.servo2 = Servo(1024, 3072, 0, pi, 80, 30, p=1500, i=0, d=500)
        self.servo3 = Servo(1024, 3072, -pi/2, pi/2, 80, 30, p=1500, i=100, d=500)
        base_servos = {1: self.servo1, 2: self.servo2, 3: self.servo3}

        self.base_servo_handler = ServoHandler(base_servos, cfg, port_handler,
                                               packet_handler, group_bulk_write, group_bulk_read)

        self.servo4 = Servo(0, 4095, -pi, pi, 150, 50, p=2500, i=0, d=3500)
        self.servo5 = Servo(0, 4095, -pi, pi, 150, 50, p=2500, i=0, d=3500)
        self.servo6 = Servo(0, 4095, -pi, pi, 150, 50, p=2500, i=0, d=3500)
        wrist_servos = {4: self.servo4, 5: self.servo5, 6: self.servo6}

        self. wrist_servo_handler = ServoHandler(wrist_servos, cfg, port_handler,
                                                 packet_handler, group_bulk_write, group_bulk_read)

        self.set_velocity_profile()
        self.set_pid()
        self.status = False
        self.lock = threading.RLock()
        self._current_angles = self.get_current_angles()

    def enable_servos(self):
        self.base_servo_handler.set_torque(enable=True)
        self.wrist_servo_handler.set_torque(enable=True)

    def disable_servos(self):
        self.base_servo_handler.set_torque(enable=False)
        self.wrist_servo_handler.set_torque(enable=False)

    def move_to_pose(self, pose):
        new_angles = inverse_kinematics(pose, self.robot_config)
        self.move_servos(new_angles)
        recommended_time = get_recommended_wait_time(self._current_angles, new_angles)
        self._current_angles = new_angles
        return recommended_time

    def move_servos(self, angles):
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

