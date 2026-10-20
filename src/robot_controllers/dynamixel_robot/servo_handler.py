# Handler for dynamixel X servos
import logging
import threading
import time

import dynamixel_sdk as dynamixel
import src.robot_controllers.dynamixel_robot.dynamixel_utils as utils
from src.utils.decorators import synchronized_with_lock

logger = logging.getLogger(__name__)


class ServoHandler(object):
    """class for handling servos

    This class is responsible for controlling all servos that share the same usb connection.

    Methods that act on multiple servos take an optional list of servo ids, by default they act on all servos.
    Packets sent by the usb adapter reach the servos on all its connectors (ttl and rs485), so all servos are
    written in a single sync packet. The responses of the servos on one connector are not seen by the servos on the
    other connector though. In a sync read every servo waits for the response of the previous servo, so the servos
    are read with one sync read per bus.
    """

    def __init__(self, servos, config, port_handler, packet_handler, buses=None):
        """"
        Args:
            servos: a mapping of servo id to servo object
            config: a python file containing dynamixel config values, see dynamixel_x_config.py
            port_handler: a dynamixel port handler instance
            packet_handler: a dynamixel packet handler instance
            buses: a list of groups of servo ids that are on the same bus (connector of the usb adapter), by default
                all servos are on the same bus
        """
        self.servo_map = servos  # {id1, servo1, id2, servo2, etc...}
        self.port_handler = port_handler
        self.packet_handler = packet_handler
        self.config = config
        self.buses = [tuple(bus) for bus in buses] if buses is not None else [tuple(servos)]
        bus_ids = sorted(servo_id for bus in self.buses for servo_id in bus)
        if bus_ids != sorted(servos):
            raise ValueError("every servo should be on exactly one bus, got buses %s for servos %s"
                             % (self.buses, sorted(servos)))
        # adjacent registers are written in a single packet, see set_profile_velocity_and_acceleration and set_pid
        if (config.ADDR_PROFILE_ACCELERATION + config.LEN_PROFILE_ACCELERATION != config.ADDR_PROFILE_VELOCITY or
                config.ADDR_D + config.LEN_PID != config.ADDR_I or config.ADDR_I + config.LEN_PID != config.ADDR_P):
            raise ValueError("the profile or pid registers in the config are not adjacent")
        # guards the usb connection, all communication with the servos goes through this lock
        self.bus_lock = threading.RLock()

    def get_servo(self, servo_id):
        """return the servo object corresponding to the id"""
        return self.servo_map[servo_id]

    def set_angle(self, servo_id, angle, all_angles=None):
        """Set the target position based on the angle for the specific servo"""
        self.servo_map[servo_id].set_target_position_from_angle(angle, all_angles)

    def get_angle(self, servo_id, position):
        """converts the position to a servo angle"""
        return self.servo_map[servo_id].get_angle_from_position(position)

    @synchronized_with_lock("bus_lock")
    def reboot(self, servo_ids=None, timeout=3.0):
        """reboot the servos and wait until they respond again, returns True if all of them came back

        A reboot disables the torque and wipes all settings that are not stored in the eeprom (pid, profile, etc.)
        """
        servo_ids = self.__ids(servo_ids)
        for servo_id in servo_ids:
            comm_result, error = self.packet_handler.reboot(self.port_handler, servo_id)
            utils.check_result(self.packet_handler, servo_id, comm_result, error)

        deadline = time.monotonic() + timeout
        waiting = set(servo_ids)
        while waiting and time.monotonic() < deadline:
            time.sleep(0.1)
            for servo_id in list(waiting):
                _, comm_result, error = self.packet_handler.ping(self.port_handler, servo_id)
                if comm_result == dynamixel.COMM_SUCCESS and error == 0:
                    waiting.remove(servo_id)

        for servo_id in waiting:
            logger.error("[ID:%03d] did not respond after reboot", servo_id)
        return not waiting

    @synchronized_with_lock("bus_lock")
    def set_torque(self, enable, servo_ids=None):
        """Enable or disable the torque of the servos, returns True if it succeeded for all of them

        The servos are written one at a time so every servo confirms the write.
        """
        success = True
        # don't stop at the first failure, when disabling we want to reach as many servos as possible
        for servo_id in self.__ids(servo_ids):
            success &= utils.set_torque(self.packet_handler, self.port_handler, self.config, servo_id, enable)
        return success

    @synchronized_with_lock("bus_lock")
    def set_configured_operating_mode(self, servo_ids=None):
        """set the configured operating mode of the servos, returns True if it succeeded for all of them"""
        success = True
        for servo_id in self.__ids(servo_ids):
            success &= self.set_operating_mode(servo_id, self.servo_map[servo_id].operating_mode)
        return success

    @synchronized_with_lock("bus_lock")
    def set_operating_mode(self, servo_id, mode):
        """set the operating mode of a servo, returns True on success

        The operating mode can only be changed while the torque is disabled, so the write is skipped when the servo
        is already in the requested mode. This allows reinitializing while the servos are holding their position.
        """
        current_mode = utils.read_1_byte(self.packet_handler, self.port_handler,
                                         self.config.ADDR_OPERATING_MODE, servo_id)
        if current_mode is None:
            return False
        if current_mode == mode:
            return True
        return utils.write_1_byte(self.packet_handler, self.port_handler, self.config.ADDR_OPERATING_MODE,
                                  servo_id, mode)

    def set_profile_velocity_and_acceleration(self, servo_ids=None):
        """set the velocity profile of the servos, this get's wiped after a reboot"""
        # the acceleration and velocity registers are adjacent, so they are written in a single packet
        data = {}
        for servo_id in self.__ids(servo_ids):
            servo = self.servo_map[servo_id]
            data[servo_id] = (utils.to_bytes(servo.profile_acceleration, self.config.LEN_PROFILE_ACCELERATION) +
                              utils.to_bytes(servo.profile_velocity, self.config.LEN_PROFILE_VELOCITY))
        return self.__sync_write(self.config.ADDR_PROFILE_ACCELERATION,
                                 self.config.LEN_PROFILE_ACCELERATION + self.config.LEN_PROFILE_VELOCITY, data)

    def set_profile_velocity_percentage(self, percentage, servo_ids=None):
        """
        set the profile velocity as percentage of the maximum configured
        :param percentage: a value between 0 and 100
        :return:
        """
        if not (0 < percentage <= 100):
            raise ValueError("A percentage outside the range (0,100]!? I don't think so!")

        values = {}
        for servo_id in self.__ids(servo_ids):
            servo = self.servo_map[servo_id]
            # a profile velocity of 0 means unlimited velocity, so never go below 1
            values[servo_id] = max(1, int((servo.profile_velocity / 100) * percentage))

        return self.__sync_write_values(self.config.ADDR_PROFILE_VELOCITY, self.config.LEN_PROFILE_VELOCITY, values)

    def set_configured_goal_current(self, servo_ids=None):
        """set the goal current for the servos that have one configured"""
        values = {servo_id: self.servo_map[servo_id].goal_current for servo_id in self.__ids(servo_ids)
                  if self.servo_map[servo_id].goal_current is not None}
        return self.__sync_write_values(self.config.ADDR_GOAL_CURRENT, self.config.LEN_GOAL_CURRENT, values)

    def set_goal_current(self, servo_id, val):
        return self.__sync_write_values(self.config.ADDR_GOAL_CURRENT, self.config.LEN_GOAL_CURRENT, {servo_id: val})

    def set_pid(self, servo_ids=None):
        """set the configured pid values of the servos, this get's wiped after a reboot"""
        pid = {}
        for servo_id in self.__ids(servo_ids):
            servo = self.servo_map[servo_id]
            pid[servo_id] = (servo.p, servo.i, servo.d)
        return self.__write_pid(pid)

    def set_pid_single_servo(self, servo_id, p, i, d):
        """set the pid values of a single servo, this get's wiped after a reboot"""
        return self.__write_pid({servo_id: (p, i, d)})

    def move_to_angles(self, servo_ids=None):
        """set the physical servos to their target position"""
        values = {}
        for servo_id in self.__ids(servo_ids):
            self.__clamp_target_position(servo_id)
            values[servo_id] = self.servo_map[servo_id].target_position
        return self.__sync_write_values(self.config.ADDR_GOAL_POSITION, self.config.LEN_GOAL_POSITION, values)

    # debug function to move a single servo to a raw position, this is not clamped to the calibrated range
    def move_servo_to_pos(self, servo_id, position):
        return self.__sync_write_values(self.config.ADDR_GOAL_POSITION, self.config.LEN_GOAL_POSITION,
                                        {servo_id: position})

    def read_current_pos(self, servo_ids=None):
        """update the current_position parameter of the servo objects

        The positions are only updated when all servos could be read, so they are never partially stale.
        Returns True on success.
        """
        positions = self.__sync_read(self.config.ADDR_PRESENT_POSITION, self.config.LEN_PRESENT_POSITION,
                                     self.__ids(servo_ids))
        if positions is None:
            return False

        for servo_id, position in positions.items():
            self.servo_map[servo_id].current_position = utils.to_signed(position, self.config.LEN_PRESENT_POSITION)
        return True

    @synchronized_with_lock("bus_lock")
    def hold_current_position(self, servo_ids=None):
        """set the target and goal position of the servos to their present position, returns True on success

        This keeps the servos in place when the torque is enabled, instead of moving to an old goal position.
        """
        servo_ids = self.__ids(servo_ids)
        if not self.read_current_pos(servo_ids):
            return False
        values = {}
        for servo_id in servo_ids:
            servo = self.servo_map[servo_id]
            servo.target_position = servo.current_position
            servo.compensation_positions = 0  # the present position is held as it is, nothing was added to it
            values[servo_id] = servo.current_position
        return self.__sync_write_values(self.config.ADDR_GOAL_POSITION, self.config.LEN_GOAL_POSITION, values)

    def read_current_pos_single_servo(self, servo_id):
        """update the current_position parameter of a single servo, returns the position or None on failure"""
        if not self.read_current_pos([servo_id]):
            return None
        return self.servo_map[servo_id].current_position

    def read_profile_velocity(self, servo_ids=None):
        """returns a mapping of servo id to profile velocity, or None if the servos could not be read"""
        return self.__sync_read(self.config.ADDR_PROFILE_VELOCITY, self.config.LEN_PROFILE_VELOCITY,
                                self.__ids(servo_ids))

    def read_hardware_errors(self, servo_ids=None):
        """returns a mapping of servo id to a list of the names of its hardware errors, for the servos that have one

        A servo with a hardware error disables its torque and ignores goal positions until it is rebooted.
        Returns None if the servos could not be read.
        """
        statuses = self.__sync_read(self.config.ADDR_HARDWARE_ERROR_STATUS, self.config.LEN_HARDWARE_ERROR_STATUS,
                                    self.__ids(servo_ids))
        if statuses is None:
            return None
        errors = {}
        for servo_id, status in statuses.items():
            if status:
                errors[servo_id] = [name for bit, name in self.config.HARDWARE_ERRORS.items() if status & (1 << bit)]
        return errors

    def read_present_current(self, servo_ids=None):
        """returns a mapping of servo id to the signed value of its present current register, or None on failure

        The unit depends on the model: 2.69mA for the XM430 and XM540, 1mA for the XL330. The XC430 has no current
        sensor, its register is the present load in 0.1% of the maximum (see CURRENT_UNITS in
        calibration/record_static_load.py).
        """
        values = self.__sync_read(self.config.ADDR_PRESENT_CURRENT, self.config.LEN_PRESENT_CURRENT,
                                  self.__ids(servo_ids))
        if values is None:
            return None
        return {servo_id: utils.to_signed(value, self.config.LEN_PRESENT_CURRENT)
                for servo_id, value in values.items()}

    def read_temperatures(self, servo_ids=None):
        """returns a mapping of servo id to temperature in degrees celsius, or None on failure"""
        return self.__sync_read(self.config.ADDR_PRESENT_TEMPERATURE, self.config.LEN_PRESENT_TEMPERATURE,
                                self.__ids(servo_ids))

    def read_temperature_and_voltage(self, servo_ids=None):
        """returns a mapping of servo id to (temperature in degrees celsius, input voltage), or None on failure"""
        servo_ids = self.__ids(servo_ids)
        temperatures = self.read_temperatures(servo_ids)
        voltages = self.__sync_read(self.config.ADDR_PRESENT_INPUT_VOLTAGE, self.config.LEN_PRESENT_INPUT_VOLTAGE,
                                    servo_ids)
        if temperatures is None or voltages is None:
            return None
        # the input voltage is in units of 0.1V
        return {servo_id: (temperatures[servo_id], voltages[servo_id] / 10.0) for servo_id in servo_ids}

    def __ids(self, servo_ids):
        if servo_ids is None:
            return list(self.servo_map)
        return list(servo_ids)

    def __clamp_target_position(self, servo_id):
        """clamp the target position to both the calibrated and the physical range of the servo

        The calibrated range applies to the target before its static load compensation, the compensation is kept at
        the ends of the range; the physical range applies to the goal position that is sent.
        """
        servo = self.servo_map[servo_id]
        clamped = servo.clamp_target_position()
        clamped = min(max(clamped, self.config.MIN_POSITION), self.config.MAX_POSITION)
        if clamped != servo.target_position:
            logger.warning("[ID:%03d] target position %d is out of range, clamped to %d",
                           servo_id, servo.target_position, clamped)
            servo.target_position = clamped

    def __write_pid(self, pid):
        """write a mapping of servo id to (p, i, d)"""
        # the d, i and p registers are adjacent (in that order), so they are written in a single packet
        data = {}
        for servo_id, (p, i, d) in pid.items():
            data[servo_id] = (utils.to_bytes(d, self.config.LEN_PID) +
                              utils.to_bytes(i, self.config.LEN_PID) +
                              utils.to_bytes(p, self.config.LEN_PID))
        return self.__sync_write(self.config.ADDR_D, 3 * self.config.LEN_PID, data)

    def __sync_write_values(self, address, length, values):
        """write a mapping of servo id to value to the given address"""
        data = {servo_id: utils.to_bytes(value, length) for servo_id, value in values.items()}
        return self.__sync_write(address, length, data)

    @synchronized_with_lock("bus_lock")
    def __sync_write(self, address, length, data):
        """write a mapping of servo id to byte list to the given address in a single packet, returns True on success

        The servos don't respond to a sync write, so only failures to send the packet are detected.
        """
        if not data:
            return True

        group_sync_write = dynamixel.GroupSyncWrite(self.port_handler, self.packet_handler, address, length)
        for servo_id, byte_list in data.items():
            if not group_sync_write.addParam(servo_id, byte_list):
                raise utils.DynamixelError("[ID:%03d] groupSyncWrite addparam failed" % servo_id)

        comm_result = group_sync_write.txPacket()
        if comm_result != dynamixel.COMM_SUCCESS:
            logger.warning("sync write to address %d failed: %s", address,
                           self.packet_handler.getTxRxResult(comm_result))
            return False
        return True

    @synchronized_with_lock("bus_lock")
    def __sync_read(self, address, length, servo_ids):
        """read the given address of all servos with a single packet per bus

        returns a mapping of servo id to the unsigned value, or None if any of the servos could not be read
        """
        values = {}
        for bus in self.buses:
            bus_servo_ids = [servo_id for servo_id in servo_ids if servo_id in bus]
            if not bus_servo_ids:
                continue
            bus_values = self.__sync_read_bus(address, length, bus_servo_ids)
            if bus_values is None:
                return None
            values.update(bus_values)
        return values

    def __sync_read_bus(self, address, length, servo_ids):
        """read the given address of servos that are on the same bus in a single packet"""
        group_sync_read = dynamixel.GroupSyncRead(self.port_handler, self.packet_handler, address, length)
        for servo_id in servo_ids:
            if not group_sync_read.addParam(servo_id):
                raise utils.DynamixelError("[ID:%03d] groupSyncRead addparam failed" % servo_id)

        comm_result = group_sync_read.txRxPacket()
        if comm_result != dynamixel.COMM_SUCCESS:
            logger.warning("sync read of address %d failed: %s", address,
                           self.packet_handler.getTxRxResult(comm_result))
            return None

        values = {}
        for servo_id in servo_ids:
            if not group_sync_read.isAvailable(servo_id, address, length):
                logger.warning("[ID:%03d] groupSyncRead getdata failed", servo_id)
                return None
            values[servo_id] = group_sync_read.getData(servo_id, address, length)
        return values
