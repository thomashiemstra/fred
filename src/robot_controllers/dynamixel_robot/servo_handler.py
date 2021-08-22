# Handler for dynamixel X servos
import src.robot_controllers.dynamixel_robot.dynamixel_sdk as dynamixel
import src.robot_controllers.dynamixel_robot.dynamixel_utils as utils

MAX_INT = 4294967296

class ServoHandler(object):
    """class for handling servos

    This class is responsible for controlling multiple servos who all share the same usb connection
    """

    def __init__(self, servos, config, port_handler, packet_handler, group_bulk_write, group_bulk_read):
        """"
        Args:
            servos: a mapping of servo id to servo object
            config: a python file containing dynamixel config values, see dynamixel_x_config.py
            port_handler: a dynamixel port handler instance
            packet_handler: a dynamixel packet handler instance
            group_bulk_write: a dynamixel group bulk write instance
            group_bulk_read: a dynamixel group bulk read instance
        """
        self.servo_map = servos  # {id1, servo1, id2, servo2, etc...}
        self.port_handler = port_handler
        self.packet_handler = packet_handler
        self.group_bulk_write = group_bulk_write
        self.group_bulk_read = group_bulk_read
        self.config = config

    def set_torque(self, enable):
        """Enable or disable the torque for all the servos"""
        for servo_id in self.servo_map:
            utils.set_torque(self.packet_handler, self.port_handler, self.config, servo_id, enable)

    def move_to_angles(self):
        """set the physical servos to their target position"""
        self.group_bulk_write.clearParam()

        # set the params of group_bulk_write to the target position for the servos
        for servo_id in self.servo_map:
            servo = self.servo_map[servo_id]
            self.__add_to_write(self.config.ADDR_GOAL_POSITION,
                                self.config.LEN_GOAL_POSITION, servo_id, servo.target_position)

        return self.__write_and_clear()

    def set_profile_velocity_and_acceleration(self):
        """set the velocity profile of the servos, this get's wiped after a reboot"""
        self.group_bulk_write.clearParam()

        for servo_id in self.servo_map:
            servo = self.servo_map[servo_id]
            self.__add_to_write(self.config.ADDR_PROFILE_VELOCITY,
                                self.config.LEN_PROFILE_VELOCITY, servo_id, servo.profile_velocity)

        success = self.__write_and_clear()
        if not success:
            return success

        for servo_id in self.servo_map:
            servo = self.servo_map[servo_id]

            self.__add_to_write(self.config.ADDR_PROFILE_ACCELERATION,
                                self.config.LEN_PROFILE_ACCELERATION, servo_id, servo.profile_acceleration)

        return self.__write_and_clear()

    def set_profile_velocity_percentage(self, percentage):
        """
        set the profile velocity as percentage of the maximum configured
        :param percentage: a value between 0 and 100
        :return:
        """
        if not (0 < percentage <= 100):
            raise ValueError("A percentage outside the range (0,100]!? I don't think so!")

        self.group_bulk_write.clearParam()

        for servo_id in self.servo_map:
            servo = self.servo_map[servo_id]
            val = int((servo.profile_velocity / 100) * percentage)

            self.__add_to_write(self.config.ADDR_PROFILE_VELOCITY,
                                self.config.LEN_PROFILE_VELOCITY, servo_id, val)

        return self.__write_and_clear()

    def set_configured_goal_current(self):
        self.group_bulk_write.clearParam()

        for servo_id in self.servo_map:
            servo = self.servo_map[servo_id]
            goal_current = servo.goal_current
            if goal_current is None:
                continue
            self.__add_to_write(self.config.ADDR_GOAL_CURRENT,
                                self.config.LEN_GOAL_POSITION, servo_id, servo.goal_current)

        return self.__write_and_clear()

    def set_goal_current(self, servo_id, val):
        self.group_bulk_write.clearParam()
        self.__add_to_write(self.config.ADDR_GOAL_CURRENT,
                            self.config.LEN_GOAL_POSITION, servo_id, val)

        return self.__write_and_clear()

    def set_configured_operating_mode(self):
        for servo_id in self.servo_map:
            servo = self.servo_map[servo_id]
            operating_mode = servo.operating_mode
            self.set_operating_mode(servo_id, operating_mode)

    def set_operating_mode(self, servo_id, mode):
        utils.write_1_byte(self.packet_handler, self.port_handler, self.config.ADDR_OPERATING_MODE, servo_id, mode)

    def set_pid(self):
        """set the velocity profile of the servos, this get's wiped after a reboot"""
        self.group_bulk_write.clearParam()

        for servo_id in self.servo_map:
            servo = self.servo_map[servo_id]
            self.__add_to_write(self.config.ADDR_P,
                                self.config.LEN_PID, servo_id, servo.p)

        success = self.__write_and_clear()
        if not success:
            return success

        for servo_id in self.servo_map:
            servo = self.servo_map[servo_id]

            self.__add_to_write(self.config.ADDR_I,
                                self.config.LEN_PID, servo_id, servo.i)

        success = self.__write_and_clear()
        if not success:
            return success

        for servo_id in self.servo_map:
            servo = self.servo_map[servo_id]

            self.__add_to_write(self.config.ADDR_D,
                                self.config.LEN_PID, servo_id, servo.d)

        return self.__write_and_clear()

    def read_current_pos(self):
        """update the current_position parameter of all the servo objects"""
        self.group_bulk_read.clearParam()

        for servo_id in self.servo_map:
            res = self.group_bulk_read.addParam(servo_id, self.config.ADDR_PRESENT_POSITION,
                                                self.config.LEN_PRESENT_POSITION)
            if not res:
                raise RuntimeError("[ID:%03d] groupBulkRead addparam failed" % id)

        self.__send_read_packet()

        for servo_id in self.servo_map:
            position_result = self.__get_read_res(servo_id, self.config.ADDR_PRESENT_POSITION,
                                                  self.config.LEN_PRESENT_POSITION)
            if position_result is None:
                return False

            if position_result is not None and position_result > MAX_INT / 2:
                position_result = position_result - MAX_INT
            self.servo_map[servo_id].current_position = position_result

        self.group_bulk_read.clearParam()
        return True

    def get_angle(self, servo_id, position):
        """converts the position to a servo angle"""
        return self.servo_map[servo_id].get_angle_from_position(position)

    def __write_and_clear(self):
        success = self.__send_write_packet()
        self.group_bulk_write.clearParam()
        return success 

    def __add_to_write(self, address, address_length, servo_id, value):
        utils.add_group_write(self.group_bulk_write, servo_id, address,
                              address_length, value)

    def __send_write_packet(self):
        """send all the write commands to all the servos"""
        comm_result = self.group_bulk_write.txPacket()
        if comm_result != dynamixel.COMM_SUCCESS:
            print("%s" % self.packet_handler.getTxRxResult(comm_result))
            return False
        return True

    def __send_read_packet(self):
        """send all the read commands to all the servos"""
        comm_result = self.group_bulk_read.txRxPacket()
        if comm_result != dynamixel.COMM_SUCCESS:
            print("%s" % self.packet_handler.getTxRxResult(comm_result))

    def __get_read_res(self, servo_id, address, address_len):
        """get the response for a specific servo from group_bulk_read"""
        data_result = self.group_bulk_read.isAvailable(servo_id, address, address_len)
        if not data_result:
            print("[ID:%03d] groupBulkRead getdata failed" % servo_id)
            return None

        return self.group_bulk_read.getData(servo_id, address, address_len)

    # Used in tests
    def get_servo(self, servo_id):
        """return the servo object corresponding to the id"""
        return self.servo_map[servo_id]

    def set_angle(self, servo_id, angle, all_angles=None):
        """Set the target position based on the angle for the specific servo"""
        self.servo_map[servo_id].set_target_position_from_angle(angle, all_angles)

    # debug function to move a single servo
    def move_servo_to_angle(self, servo_id):
        self.group_bulk_write.clearParam()

        servo = self.servo_map[servo_id]
        self.__add_to_write(self.config.ADDR_GOAL_POSITION,
                            self.config.LEN_GOAL_POSITION, servo_id, servo.target_position)

        return self.__write_and_clear()

    # debug function to set the torque of a single servo
    def set_servo_torque(self, servo_id, enable):
        utils.set_torque(self.packet_handler, self.port_handler, self.config, servo_id, enable)

    def move_servo_to_pos(self, servo_id, position):
        self.group_bulk_write.clearParam()

        self.__add_to_write(self.config.ADDR_GOAL_POSITION,
                            self.config.LEN_GOAL_POSITION, servo_id, position)

        return self.__write_and_clear()

    def read_current_pos_single_servo(self, servo_id):
        """update the current_position parameter of all the servo objects"""
        self.group_bulk_read.clearParam()

        res = self.group_bulk_read.addParam(servo_id, self.config.ADDR_PRESENT_POSITION,
                                            self.config.LEN_PRESENT_POSITION)
        if not res:
            raise RuntimeError("[ID:%03d] groupBulkRead addparam failed" % id)

        self.__send_read_packet()

        position_result = self.__get_read_res(servo_id, self.config.ADDR_PRESENT_POSITION,
                                              self.config.LEN_PRESENT_POSITION)
        self.servo_map[servo_id].current_position = position_result

        self.group_bulk_read.clearParam()

        return position_result

    def set_pid_single_servo(self, servo_id, p, i, d):
        """set the velocity profile of the servos, this get's wiped after a reboot"""
        self.group_bulk_write.clearParam()

        self.__add_to_write(self.config.ADDR_P,
                            self.config.LEN_PID, servo_id, p)

        success = self.__write_and_clear()
        if not success:
            return success

        self.__add_to_write(self.config.ADDR_I,
                            self.config.LEN_PID, servo_id, i)

        success = self.__write_and_clear()
        if not success:
            return success

        self.__add_to_write(self.config.ADDR_D,
                            self.config.LEN_PID, servo_id, d)

        return self.__write_and_clear()