# Handler for dynamixel X servos
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterable, Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Any, NamedTuple

import dynamixel_sdk as dynamixel
import src.robot_controllers.dynamixel_robot.dynamixel_utils as utils
from src.utils.decorators import synchronized_with_lock

if TYPE_CHECKING:
    import numpy as np

    from src.robot_controllers.dynamixel_robot.servo import Servo

logger = logging.getLogger(__name__)

# a field of a block read: address, length and whether the value is signed
Field = tuple[int, int, bool]


class ServoState(NamedTuple):
    """the registers of the control loop of a servo, read in one packet by ServoHandler.read_state"""
    tick: int  # the realtime tick of the servo in ms, wraps at 32767
    moving: int
    moving_status: int  # see MOVING_STATUS_* in dynamixel_x_config
    pwm: int  # signed, 885 is 100 % duty
    current: int  # signed, the present current (XM, XL servos) or the present load (XC430)
    velocity: int  # signed, 0.229 rpm per unit
    position: int  # signed
    velocity_trajectory: int  # signed, the velocity the profile of the servo asks for
    position_trajectory: int  # signed, the position the profile asks for, the setpoint of the position loop
    voltage: float  # input voltage in volts
    temperature: int  # degrees celsius


class ServoHandler(object):
    """class for handling servos

    This class is responsible for controlling all servos that share the same usb connection.

    Methods that act on multiple servos take an optional list of servo ids, by default they act on all servos.
    Packets sent by the usb adapter reach the servos on all its connectors (ttl and rs485), so all servos are
    written in a single sync packet. The responses of the servos on one connector are not seen by the servos on the
    other connector though. In a sync read every servo waits for the response of the previous servo, so the servos
    are read with one sync read per bus.
    """

    def __init__(self, servos: dict[int, Servo], config: ModuleType, port_handler: Any, packet_handler: Any,
                 buses: Iterable[Iterable[int]] | None = None) -> None:
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
        # the state block of read_state runs from the realtime tick to the temperature
        if (config.ADDR_PRESENT_TEMPERATURE + config.LEN_PRESENT_TEMPERATURE - config.ADDR_REALTIME_TICK
                != config.LEN_STATE_BLOCK):
            raise ValueError("the state block in the config doesn't cover the tick to the temperature")
        # guards the usb connection, all communication with the servos goes through this lock
        self.bus_lock = threading.RLock()

    def get_servo(self, servo_id: int) -> Servo:
        """return the servo object corresponding to the id"""
        return self.servo_map[servo_id]

    def set_angle(self, servo_id: int, angle: float, all_angles: np.ndarray | None = None) -> None:
        """Set the target position based on the angle for the specific servo"""
        self.servo_map[servo_id].set_target_position_from_angle(angle, all_angles)

    def get_angle(self, servo_id: int, position: int) -> float:
        """converts the position to a servo angle"""
        return self.servo_map[servo_id].get_angle_from_position(position)

    @synchronized_with_lock("bus_lock")
    def reboot(self, servo_ids: Iterable[int] | None = None, timeout: float = 3.0) -> bool:
        """reboot the servos and wait until they respond again, returns True if all of them came back

        A servo that does not confirm the reboot command counts as failed, even if it still answers a ping.
        A reboot disables the torque and wipes all settings that are not stored in the eeprom (pid, profile, etc.)
        """
        servo_ids = self.__ids(servo_ids)
        failed = set()
        for servo_id in servo_ids:
            comm_result, error = self.packet_handler.reboot(self.port_handler, servo_id)
            if not utils.check_result(self.packet_handler, servo_id, comm_result, error):
                failed.add(servo_id)

        deadline = time.monotonic() + timeout
        waiting = set(servo_ids) - failed
        while waiting and time.monotonic() < deadline:
            time.sleep(0.1)
            for servo_id in list(waiting):
                _, comm_result, error = self.packet_handler.ping(self.port_handler, servo_id)
                if comm_result == dynamixel.COMM_SUCCESS and error == 0:
                    waiting.remove(servo_id)

        for servo_id in sorted(failed):
            logger.error("[ID:%03d] reboot command failed", servo_id)
        for servo_id in sorted(waiting):
            logger.error("[ID:%03d] did not respond after reboot", servo_id)
        return not failed and not waiting

    @synchronized_with_lock("bus_lock")
    def set_torque(self, enable: bool, servo_ids: Iterable[int] | None = None) -> bool:
        """Enable or disable the torque of the servos, returns True if it succeeded for all of them

        The servos are written one at a time so every servo confirms the write.
        """
        success = True
        # don't stop at the first failure, when disabling we want to reach as many servos as possible
        for servo_id in self.__ids(servo_ids):
            success &= utils.set_torque(self.packet_handler, self.port_handler, self.config, servo_id, enable)
        return success

    @synchronized_with_lock("bus_lock")
    def set_configured_operating_mode(self, servo_ids: Iterable[int] | None = None) -> bool:
        """set the configured operating mode of the servos, returns True if it succeeded for all of them"""
        success = True
        for servo_id in self.__ids(servo_ids):
            success &= self.set_operating_mode(servo_id, self.servo_map[servo_id].operating_mode)
        return success

    @synchronized_with_lock("bus_lock")
    def set_operating_mode(self, servo_id: int, mode: int) -> bool:
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

    def set_profile_velocity_and_acceleration(self, servo_ids: Iterable[int] | None = None) -> bool:
        """set the velocity profile of the servos, this get's wiped after a reboot"""
        # the acceleration and velocity registers are adjacent, so they are written in a single packet
        data = {}
        for servo_id in self.__ids(servo_ids):
            servo = self.servo_map[servo_id]
            data[servo_id] = (utils.to_bytes(servo.profile_acceleration, self.config.LEN_PROFILE_ACCELERATION) +
                              utils.to_bytes(servo.profile_velocity, self.config.LEN_PROFILE_VELOCITY))
        return self.__sync_write(self.config.ADDR_PROFILE_ACCELERATION,
                                 self.config.LEN_PROFILE_ACCELERATION + self.config.LEN_PROFILE_VELOCITY, data)

    def set_profile(self, servo_id: int, velocity: int, acceleration: int) -> bool:
        """
        set the profile of one servo to the given values instead of the configured ones, the Servo object keeps
        its configured profile so set_profile_velocity_and_acceleration restores it; 0 means unlimited
        """
        if velocity < 0 or acceleration < 0:
            raise ValueError("the profile velocity and acceleration can't be negative")
        data = {servo_id: (utils.to_bytes(acceleration, self.config.LEN_PROFILE_ACCELERATION) +
                           utils.to_bytes(velocity, self.config.LEN_PROFILE_VELOCITY))}
        return self.__sync_write(self.config.ADDR_PROFILE_ACCELERATION,
                                 self.config.LEN_PROFILE_ACCELERATION + self.config.LEN_PROFILE_VELOCITY, data)

    def set_profile_velocity_percentage(self, percentage: float, servo_ids: Iterable[int] | None = None) -> bool:
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

    def set_configured_goal_current(self, servo_ids: Iterable[int] | None = None) -> bool:
        """set the goal current for the servos that have one configured"""
        values = {servo_id: self.servo_map[servo_id].goal_current for servo_id in self.__ids(servo_ids)
                  if self.servo_map[servo_id].goal_current is not None}
        return self.__sync_write_values(self.config.ADDR_GOAL_CURRENT, self.config.LEN_GOAL_CURRENT, values)  # type: ignore[arg-type]  # none filtered out above

    def set_goal_current(self, servo_id: int, val: int) -> bool:
        return self.__sync_write_values(self.config.ADDR_GOAL_CURRENT, self.config.LEN_GOAL_CURRENT, {servo_id: val})

    def set_pid(self, servo_ids: Iterable[int] | None = None) -> bool:
        """set the configured pid values of the servos, this get's wiped after a reboot"""
        pid = {}
        for servo_id in self.__ids(servo_ids):
            servo = self.servo_map[servo_id]
            pid[servo_id] = (servo.p, servo.i, servo.d)
        return self.__write_pid(pid)

    def set_pid_single_servo(self, servo_id: int, p: int, i: int, d: int) -> bool:
        """set the pid values of a single servo, this get's wiped after a reboot; the Servo object keeps the
        configured values, so set_pid restores them"""
        return self.__write_pid({servo_id: (p, i, d)})

    def set_goal_pwm(self, servo_id: int, value: int) -> bool:
        """
        set the goal pwm of one servo: in position mode a limit on the pwm output, so on the torque, between
        -MAX_PWM and MAX_PWM (885, 100 %); it is reset to the pwm limit by a reboot or a change of operating mode
        """
        if abs(value) > self.config.MAX_PWM:
            raise ValueError("the goal pwm %d is outside -%d..%d" % (value, self.config.MAX_PWM, self.config.MAX_PWM))
        return self.__sync_write_values(self.config.ADDR_GOAL_PWM, self.config.LEN_GOAL_PWM, {servo_id: value})

    def move_to_angles(self, servo_ids: Iterable[int] | None = None) -> bool:
        """set the physical servos to their target position"""
        values = {}
        for servo_id in self.__ids(servo_ids):
            self.__clamp_target_position(servo_id)
            values[servo_id] = self.servo_map[servo_id].target_position
        return self.__sync_write_values(self.config.ADDR_GOAL_POSITION, self.config.LEN_GOAL_POSITION, values)

    # debug function to move a single servo to a raw position, this is not clamped to the calibrated range
    def move_servo_to_pos(self, servo_id: int, position: int) -> bool:
        return self.__sync_write_values(self.config.ADDR_GOAL_POSITION, self.config.LEN_GOAL_POSITION,
                                        {servo_id: position})

    def read_current_pos(self, servo_ids: Iterable[int] | None = None) -> bool:
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
    def hold_current_position(self, servo_ids: Iterable[int] | None = None) -> bool:
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

    def read_current_pos_single_servo(self, servo_id: int) -> int | None:
        """update the current_position parameter of a single servo, returns the position or None on failure"""
        if not self.read_current_pos([servo_id]):
            return None
        return self.servo_map[servo_id].current_position

    def read_torque_enabled(self, servo_ids: Iterable[int] | None = None) -> dict[int, bool] | None:
        """returns a mapping of servo id to whether its torque is enabled, or None on failure"""
        values = self.__sync_read(self.config.ADDR_TORQUE_ENABLE, 1, self.__ids(servo_ids))
        if values is None:
            return None
        return {servo_id: bool(value) for servo_id, value in values.items()}

    def read_profile_velocity(self, servo_ids: Iterable[int] | None = None) -> dict[int, int] | None:
        """returns a mapping of servo id to profile velocity, or None if the servos could not be read"""
        return self.__sync_read(self.config.ADDR_PROFILE_VELOCITY, self.config.LEN_PROFILE_VELOCITY,
                                self.__ids(servo_ids))

    def read_profile(self, servo_ids: Iterable[int] | None = None) -> dict[int, tuple[int, int]] | None:
        """returns a mapping of servo id to (profile velocity, profile acceleration), or None on failure"""
        c = self.config
        fields: Sequence[Field] = ((c.ADDR_PROFILE_VELOCITY, c.LEN_PROFILE_VELOCITY, False),
                                   (c.ADDR_PROFILE_ACCELERATION, c.LEN_PROFILE_ACCELERATION, False))
        values = self.__sync_read_fields(c.ADDR_PROFILE_ACCELERATION,
                                         c.LEN_PROFILE_ACCELERATION + c.LEN_PROFILE_VELOCITY, fields,
                                         self.__ids(servo_ids))
        if values is None:
            return None
        return {servo_id: (value[0], value[1]) for servo_id, value in values.items()}

    def read_pid(self, servo_ids: Iterable[int] | None = None) -> dict[int, tuple[int, int, int]] | None:
        """returns a mapping of servo id to the (p, i, d) gains the servo has, or None on failure; a sync write
        isn't acknowledged, so this is the only way to know a gain write took"""
        c = self.config
        fields: Sequence[Field] = ((c.ADDR_P, c.LEN_PID, False), (c.ADDR_I, c.LEN_PID, False),
                                   (c.ADDR_D, c.LEN_PID, False))
        values = self.__sync_read_fields(c.ADDR_D, 3 * c.LEN_PID, fields, self.__ids(servo_ids))
        if values is None:
            return None
        return {servo_id: (value[0], value[1], value[2]) for servo_id, value in values.items()}

    def read_goal_pwm(self, servo_ids: Iterable[int] | None = None) -> dict[int, int] | None:
        """returns a mapping of servo id to its signed goal pwm, or None on failure"""
        values = self.__sync_read(self.config.ADDR_GOAL_PWM, self.config.LEN_GOAL_PWM, self.__ids(servo_ids))
        if values is None:
            return None
        return {servo_id: utils.to_signed(value, self.config.LEN_GOAL_PWM) for servo_id, value in values.items()}

    def read_state(self, servo_ids: Iterable[int] | None = None) -> dict[int, ServoState] | None:
        """
        returns a mapping of servo id to the ServoState of the servo, read in one packet per bus (addresses
        120 to 146: the realtime tick, moving, moving status, present pwm, current or load, velocity and position,
        the velocity and position trajectory of the profile, the input voltage and the temperature), or None on
        failure. About 2 ms for one servo, 1 ms more per servo on the same bus.
        """
        c = self.config
        fields: Sequence[Field] = (
            (c.ADDR_REALTIME_TICK, c.LEN_REALTIME_TICK, False), (c.ADDR_MOVING, c.LEN_MOVING, False),
            (c.ADDR_MOVING_STATUS, c.LEN_MOVING_STATUS, False), (c.ADDR_PRESENT_PWM, c.LEN_PRESENT_PWM, True),
            (c.ADDR_PRESENT_CURRENT, c.LEN_PRESENT_CURRENT, True),
            (c.ADDR_PRESENT_VELOCITY, c.LEN_PRESENT_VELOCITY, True),
            (c.ADDR_PRESENT_POSITION, c.LEN_PRESENT_POSITION, True),
            (c.ADDR_VELOCITY_TRAJECTORY, c.LEN_VELOCITY_TRAJECTORY, True),
            (c.ADDR_POSITION_TRAJECTORY, c.LEN_POSITION_TRAJECTORY, True),
            (c.ADDR_PRESENT_INPUT_VOLTAGE, c.LEN_PRESENT_INPUT_VOLTAGE, False),
            (c.ADDR_PRESENT_TEMPERATURE, c.LEN_PRESENT_TEMPERATURE, False))
        values = self.__sync_read_fields(c.ADDR_REALTIME_TICK, c.LEN_STATE_BLOCK, fields, self.__ids(servo_ids))
        if values is None:
            return None
        return {servo_id: ServoState(tick=value[0], moving=value[1], moving_status=value[2], pwm=value[3],
                                     current=value[4], velocity=value[5], position=value[6],
                                     velocity_trajectory=value[7], position_trajectory=value[8],
                                     voltage=value[9] / 10.0, temperature=value[10])
                for servo_id, value in values.items()}

    def read_hardware_errors(self, servo_ids: Iterable[int] | None = None) -> dict[int, list[str]] | None:
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

    def read_present_current(self, servo_ids: Iterable[int] | None = None) -> dict[int, int] | None:
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

    def read_temperatures(self, servo_ids: Iterable[int] | None = None) -> dict[int, int] | None:
        """returns a mapping of servo id to temperature in degrees celsius, or None on failure"""
        return self.__sync_read(self.config.ADDR_PRESENT_TEMPERATURE, self.config.LEN_PRESENT_TEMPERATURE,
                                self.__ids(servo_ids))

    def read_temperature_and_voltage(
            self, servo_ids: Iterable[int] | None = None
    ) -> dict[int, tuple[int, float]] | None:
        """returns a mapping of servo id to (temperature in degrees celsius, input voltage), or None on failure"""
        c = self.config
        # the voltage and temperature registers are adjacent, so they are read in one packet per bus
        fields: Sequence[Field] = ((c.ADDR_PRESENT_TEMPERATURE, c.LEN_PRESENT_TEMPERATURE, False),
                                   (c.ADDR_PRESENT_INPUT_VOLTAGE, c.LEN_PRESENT_INPUT_VOLTAGE, False))
        values = self.__sync_read_fields(c.ADDR_PRESENT_INPUT_VOLTAGE,
                                         c.LEN_PRESENT_INPUT_VOLTAGE + c.LEN_PRESENT_TEMPERATURE, fields,
                                         self.__ids(servo_ids))
        if values is None:
            return None
        # the input voltage is in units of 0.1V
        return {servo_id: (value[0], value[1] / 10.0) for servo_id, value in values.items()}

    def __ids(self, servo_ids: Iterable[int] | None) -> list[int]:
        if servo_ids is None:
            return list(self.servo_map)
        return list(servo_ids)

    def __clamp_target_position(self, servo_id: int) -> None:
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

    def __write_pid(self, pid: dict[int, tuple[int, int, int]]) -> bool:
        """write a mapping of servo id to (p, i, d)"""
        # the d, i and p registers are adjacent (in that order), so they are written in a single packet
        data = {}
        for servo_id, (p, i, d) in pid.items():
            for name, gain in (('p', p), ('i', i), ('d', d)):
                # the servos drop a gain outside their range silently, the sync write isn't acknowledged
                if not 0 <= gain <= self.config.MAX_GAIN:
                    raise ValueError("[ID:%03d] the %s gain %d is outside 0..%d" % (servo_id, name, gain,
                                                                                    self.config.MAX_GAIN))
            data[servo_id] = (utils.to_bytes(d, self.config.LEN_PID) +
                              utils.to_bytes(i, self.config.LEN_PID) +
                              utils.to_bytes(p, self.config.LEN_PID))
        return self.__sync_write(self.config.ADDR_D, 3 * self.config.LEN_PID, data)

    def __sync_write_values(self, address: int, length: int, values: dict[int, int]) -> bool:
        """write a mapping of servo id to value to the given address"""
        data = {servo_id: utils.to_bytes(value, length) for servo_id, value in values.items()}
        return self.__sync_write(address, length, data)

    @synchronized_with_lock("bus_lock")
    def __sync_write(self, address: int, length: int, data: dict[int, list[int]]) -> bool:
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

    def __sync_read(self, address: int, length: int, servo_ids: list[int]) -> dict[int, int] | None:
        """read the given address of all servos with a single packet per bus

        returns a mapping of servo id to the unsigned value, or None if any of the servos could not be read
        """
        values = self.__sync_read_fields(address, length, ((address, length, False),), servo_ids)
        if values is None:
            return None
        return {servo_id: fields[0] for servo_id, fields in values.items()}

    @synchronized_with_lock("bus_lock")
    def __sync_read_fields(self, address: int, length: int, fields: Sequence[Field],
                           servo_ids: list[int]) -> dict[int, tuple[int, ...]] | None:
        """read a block of registers of all servos with a single packet per bus and pick fields out of it

        :param address: the start of the block
        :param length: the length of the block in bytes
        :param fields: the (address, length, signed) of every value wanted from the block, 1, 2 or 4 bytes each
        :return: a mapping of servo id to the values of the fields, or None if any of the servos could not be read
        """
        values = {}
        for bus in self.buses:
            bus_servo_ids = [servo_id for servo_id in servo_ids if servo_id in bus]
            if not bus_servo_ids:
                continue
            bus_values = self.__sync_read_bus(address, length, fields, bus_servo_ids)
            if bus_values is None:
                return None
            values.update(bus_values)
        return values

    def __sync_read_bus(self, address: int, length: int, fields: Sequence[Field],
                        servo_ids: list[int]) -> dict[int, tuple[int, ...]] | None:
        """read a block of registers of servos that are on the same bus in a single packet, see __sync_read_fields"""
        group_sync_read = dynamixel.GroupSyncRead(self.port_handler, self.packet_handler, address, length)
        for servo_id in servo_ids:
            if not group_sync_read.addParam(servo_id):
                raise utils.DynamixelError("[ID:%03d] groupSyncRead addparam failed" % servo_id)

        comm_result = group_sync_read.txRxPacket()
        if comm_result != dynamixel.COMM_SUCCESS:
            logger.warning("sync read of address %d from servos %s failed: %s", address, servo_ids,
                           self.packet_handler.getTxRxResult(comm_result))
            return None

        values = {}
        for servo_id in servo_ids:
            if not group_sync_read.isAvailable(servo_id, address, length):
                logger.warning("[ID:%03d] groupSyncRead getdata failed", servo_id)
                return None
            # the sdk picks any 1, 2 or 4 byte value out of the block it received
            values[servo_id] = tuple(
                utils.to_signed(group_sync_read.getData(servo_id, field_address, field_length), field_length)
                if signed else group_sync_read.getData(servo_id, field_address, field_length)
                for field_address, field_length, signed in fields)
        return values
