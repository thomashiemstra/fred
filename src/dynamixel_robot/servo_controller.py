from src.dynamixel_robot import dynamixel_x_config as cfg
from src.dynamixel_robot.dynamixel_utils import setup_dynamixel_handlers
from src.dynamixel_robot.servo import Servo
from src.dynamixel_robot.servo_handler import ServoHandler
from src.kinematics.kinematics import inverse_kinematics
from src.utils.movement_utils import angles_to_angles
from src.utils.decorators import synchronized_with_lock
import threading

import numpy as np
from numpy import pi


# Facade for the robot as a whole, abstracting away the servo handling
class DynamixelRobotArm:

    def __init__(self, port, robot_config):
        self.robot_config = robot_config
        port_handler, packet_handler, group_bulk_write, group_bulk_read = setup_dynamixel_handlers(port, cfg)

        self.servo1 = Servo(1024, 3072, 0, pi, 40, 10, p=500, i=0, d=800)
        self.servo2 = Servo(1024, 3072, 0, pi, 40, 10, p=1500, i=0, d=800)
        self.servo3 = Servo(1024, 3072, -pi/2, pi/2, 40, 10, p=1500, i=0, d=800)
        base_servos = {1: self.servo1, 2: self.servo2, 3: self.servo3}

        self.base_servo_handler = ServoHandler(base_servos, cfg, port_handler,
                                               packet_handler, group_bulk_write, group_bulk_read)

        self.servo4 = Servo(0, 4095, -pi, pi, 150, 50, p=2500, i=500, d=3500, offset=-10)
        self.servo5 = Servo(0, 4095, -pi, pi, 150, 50, p=2500, i=500, d=3500, offset=60)
        self.servo6 = Servo(0, 4095, -pi, pi, 150, 50, p=2500, i=500, d=3500)
        wrist_servos = {4: self.servo4, 5: self.servo5, 6: self.servo6}

        self. wrist_servo_handler = ServoHandler(wrist_servos, cfg, port_handler,
                                                 packet_handler, group_bulk_write, group_bulk_read)

        self.set_velocity_profile()
        self.set_pid()
        self.status = False
        self.lock = threading.RLock()

    def enable_servos(self):
        self.base_servo_handler.set_torque(enable=True)
        self.wrist_servo_handler.set_torque(enable=True)

    def disable_servos(self):
        self.base_servo_handler.set_torque(enable=False)
        self.wrist_servo_handler.set_torque(enable=False)

    def move_to_pose(self, pose):
        angles = inverse_kinematics(pose, self.robot_config)
        self.move_servos(angles)

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

    def get_angles(self):
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

    def from_current_angles_to_pose(self, pose, time):
        current_angles = self.get_angles()
        target_angles = inverse_kinematics(pose, self.robot_config)

        angles_to_angles(current_angles, target_angles, time, self)

    @synchronized_with_lock("lock")
    def get_status(self):
        return self.status

    @synchronized_with_lock("lock")
    def change_status(self, new_state):
        self.status = new_state
