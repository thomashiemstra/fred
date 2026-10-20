import copy
import os
import unittest

import numpy as np
from unittest import mock

import dynamixel_sdk as dxl
from src.robot_controllers.dynamixel_robot import dynamixel_robot_controller as drc
from src.robot_controllers.dynamixel_robot import dynamixel_x_config as cfg
from src.robot_controllers.dynamixel_robot.dynamixel_utils import DynamixelError
from src.robot_controllers.dynamixel_robot.servo_config import load_servo_config
from src import global_constants
from src.kinematics.kinematics import forward_kinematics

ACCESS_ERROR = 0x07
# the tests use their own servo configuration, so recalibrating the robot doesn't affect them
TEST_SERVO_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_servo_config.yaml')


def calibrated_range(servo):
    """the lowest and highest position of the calibrated range of the servo, including its offset"""
    return servo.clamp_position(-2 ** 31), servo.clamp_position(2 ** 31)


def position_of(servo, angle):
    """the goal position the servo gets for the angle, including its offset"""
    servo = copy.deepcopy(servo)
    servo.set_target_position_from_angle(angle)
    return servo.target_position


class FakeBus:
    """Fake packet handler that stores the control table of every servo in memory

    The real sdk group sync read/write classes are used on top of it, so the packets are built like on the robot.
    Like on the robot, servos on different buses can't hear each other's responses, so a sync read with servos of
    both buses fails.
    """

    def __init__(self, servos, buses):
        self.memory = {servo_id: bytearray(256) for servo_id in range(1, len(servos) + 1)}
        for servo_id, servo in enumerate(servos, start=1):
            self.memory[servo_id][cfg.ADDR_OPERATING_MODE] = servo.operating_mode
            self.set_position(servo_id, 2048)
        self.unresponsive = set()
        self.packets = []  # (instruction, address, ids)
        self.rebooted = set()
        self.buses = buses
        self._sync_read_ids = []

    # helpers for the tests
    def get(self, servo_id, address, length, signed=False):
        return int.from_bytes(self.memory[servo_id][address:address + length], 'little', signed=signed)

    def set_position(self, servo_id, position):
        self.memory[servo_id][cfg.ADDR_PRESENT_POSITION:cfg.ADDR_PRESENT_POSITION + 4] = \
            (position & 0xFFFFFFFF).to_bytes(4, 'little')

    def torque(self, servo_id):
        return self.memory[servo_id][cfg.ADDR_TORQUE_ENABLE]

    def packets_to(self, address):
        return [p for p in self.packets if p[1] == address]

    # packet handler interface
    def getProtocolVersion(self):
        return 2.0

    def getTxRxResult(self, result):
        return 'comm result %d' % result

    def getRxPacketError(self, error):
        return 'packet error %d' % error

    def syncWriteTxOnly(self, port, start_address, data_length, param, param_length):
        ids = []
        for i in range(0, param_length, data_length + 1):
            servo_id = param[i]
            ids.append(servo_id)
            if servo_id not in self.unresponsive:
                self.memory[servo_id][start_address:start_address + data_length] = \
                    bytes(param[i + 1:i + 1 + data_length])
        self.packets.append(('sync_write', start_address, ids))
        return dxl.COMM_SUCCESS

    def syncReadTx(self, port, start_address, data_length, param, param_length, fast_option=False):
        self._sync_read = (start_address, data_length)
        self._sync_read_ids = list(param[:param_length])
        self.packets.append(('sync_read', start_address, list(param)))
        return dxl.COMM_SUCCESS

    def readRx(self, port, servo_id, length):
        if servo_id in self.unresponsive or not self._heard_previous_responses(servo_id):
            return [], dxl.COMM_RX_TIMEOUT, 0
        start_address, _ = self._sync_read
        return list(self.memory[servo_id][start_address:start_address + length]), dxl.COMM_SUCCESS, 0

    def _heard_previous_responses(self, servo_id):
        """in a sync read a servo only responds after it has seen the responses of the servos before it"""
        bus = next(bus for bus in self.buses if servo_id in bus)
        previous_ids = self._sync_read_ids[:self._sync_read_ids.index(servo_id)]
        return all(previous_id in bus for previous_id in previous_ids)

    def read1ByteTxRx(self, port, servo_id, address):
        if servo_id in self.unresponsive:
            return 0, dxl.COMM_RX_TIMEOUT, 0
        return self.memory[servo_id][address], dxl.COMM_SUCCESS, 0

    def write1ByteTxRx(self, port, servo_id, address, value):
        if servo_id in self.unresponsive:
            return dxl.COMM_RX_TIMEOUT, 0
        if address == cfg.ADDR_OPERATING_MODE and self.torque(servo_id):
            return dxl.COMM_SUCCESS, ACCESS_ERROR
        self.memory[servo_id][address] = value
        return dxl.COMM_SUCCESS, 0

    def reboot(self, port, servo_id):
        self.rebooted.add(servo_id)
        self.memory[servo_id][cfg.ADDR_TORQUE_ENABLE] = 0
        return dxl.COMM_SUCCESS, 0

    def ping(self, port, servo_id):
        if servo_id in self.unresponsive:
            return 0, dxl.COMM_RX_TIMEOUT, 0
        return 1020, dxl.COMM_SUCCESS, 0


class TestDynamixelRobotController(unittest.TestCase):

    def setUp(self):
        self.servo_config = load_servo_config(TEST_SERVO_CONFIG_PATH)
        self.servos = list(self.servo_config.servos.values())
        self.bus = FakeBus(self.servos, self.servo_config.buses)
        self.robot = self.make_robot()

    def make_robot(self, **kwargs):
        return drc.DynamixelRobotController(global_constants.dynamixel_robot_config, mock.Mock(), self.bus,
                                            servo_config=self.servo_config, **kwargs)

    def test_robot_starts_with_the_real_servo_configuration(self):
        # only checks that the configuration of the robot can be used, not its values
        real_config = load_servo_config()
        self.bus = FakeBus(list(real_config.servos.values()), real_config.buses)
        self.servo_config = real_config
        self.assertIsNotNone(self.make_robot().get_current_angles())

    def test_setup_writes_the_configured_values(self):
        for servo_id, servo in enumerate(self.servos, start=1):
            self.assertEqual(servo.profile_acceleration, self.bus.get(servo_id, cfg.ADDR_PROFILE_ACCELERATION, 4))
            self.assertEqual(servo.profile_velocity, self.bus.get(servo_id, cfg.ADDR_PROFILE_VELOCITY, 4))
            self.assertEqual(servo.p, self.bus.get(servo_id, cfg.ADDR_P, 2))
            self.assertEqual(servo.i, self.bus.get(servo_id, cfg.ADDR_I, 2))
            self.assertEqual(servo.d, self.bus.get(servo_id, cfg.ADDR_D, 2))
        # only the gripper has a goal current configured
        self.assertEqual(self.servos[6].goal_current, self.bus.get(7, cfg.ADDR_GOAL_CURRENT, 2))
        self.assertEqual([('sync_write', cfg.ADDR_GOAL_CURRENT, [7])], self.bus.packets_to(cfg.ADDR_GOAL_CURRENT))
        # adjacent registers are written in a single packet
        self.assertEqual(1, len(self.bus.packets_to(cfg.ADDR_PROFILE_ACCELERATION)))
        self.assertEqual(1, len(self.bus.packets_to(cfg.ADDR_D)))
        self.assertEqual([], self.bus.packets_to(cfg.ADDR_PROFILE_VELOCITY))
        self.assertEqual([], self.bus.packets_to(cfg.ADDR_P))

    def test_setup_raises_when_a_servo_does_not_respond(self):
        self.bus.unresponsive = {5}
        with self.assertRaises(DynamixelError):
            self.make_robot()

    def test_setup_without_safety_checks_does_not_raise(self):
        self.bus.unresponsive = {5}
        robot = self.make_robot(perform_safety_checks=False)
        self.assertIsNone(robot.get_current_angles())

    def test_move_to_pose_refuses_while_the_current_angles_are_unknown(self):
        pose = forward_kinematics(self.robot.get_current_angles(), self.robot.robot_config)
        self.bus.unresponsive = {5}
        robot = self.make_robot(perform_safety_checks=False)

        with self.assertRaises(DynamixelError):
            robot.move_to_pose(pose)

        # once the servos answer the current angles are read before moving
        self.bus.unresponsive = set()
        recommended_time, _ = robot.move_to_pose(pose)
        self.assertGreaterEqual(recommended_time, 0)

    def test_gripper_position_is_none_when_the_servos_can_not_be_read(self):
        self.assertEqual(3, len(self.robot.get_current_gripper_position()))
        self.bus.unresponsive = {5}
        self.assertIsNone(self.robot.get_current_gripper_position())

    def test_operating_mode_is_only_written_when_it_changes(self):
        for servo_id in range(1, 8):
            self.bus.memory[servo_id][cfg.ADDR_TORQUE_ENABLE] = 1
        self.assertTrue(self.robot.set_control_mode())

        self.bus.memory[1][cfg.ADDR_OPERATING_MODE] = 1
        self.assertFalse(self.robot.set_control_mode())

        self.bus.memory[1][cfg.ADDR_TORQUE_ENABLE] = 0
        self.assertTrue(self.robot.set_control_mode())
        self.assertEqual(self.servos[0].operating_mode, self.bus.memory[1][cfg.ADDR_OPERATING_MODE])

    def test_move_servos_is_a_single_packet(self):
        self.bus.packets.clear()
        angles = [0, 1.5, 1.5, 0.0, 0.5, -0.5, 1.0]
        self.robot.move_servos(angles)

        self.assertEqual([('sync_write', cfg.ADDR_GOAL_POSITION, [1, 2, 3, 4, 5, 6])], self.bus.packets)
        for servo_id in range(1, 7):
            self.assertEqual(self.servos[servo_id - 1].target_position,
                             self.bus.get(servo_id, cfg.ADDR_GOAL_POSITION, 4))

    def test_move_servos_clamps_out_of_range_positions(self):
        self.robot.move_servos([0, 1.5, 1.5, 0.0, -10.0, 0.0, 0.0])
        # servo 4 goes to the low end of its calibrated range, but never below the physical range
        lower, _ = calibrated_range(self.servos[3])
        self.assertEqual(max(lower, cfg.MIN_POSITION), self.bus.get(4, cfg.ADDR_GOAL_POSITION, 4))

    def test_move_servos_clamps_to_the_physical_range(self):
        # with an offset that moves the calibrated range below 0, a position below 0 would be rejected by the servo
        self.servos[3].constant_offset = -100
        self.robot.move_servos([0, 1.5, 1.5, 0.0, -10.0, 0.0, 0.0])
        self.assertEqual(cfg.MIN_POSITION, self.bus.get(4, cfg.ADDR_GOAL_POSITION, 4))

    def test_current_angles_are_read_with_a_packet_per_bus(self):
        self.bus.packets.clear()
        angles = self.robot.get_current_angles()

        self.assertEqual([('sync_read', cfg.ADDR_PRESENT_POSITION, [1, 2, 3]),
                          ('sync_read', cfg.ADDR_PRESENT_POSITION, [4, 5, 6])], self.bus.packets)
        self.assertAlmostEqual(self.servos[0].get_angle_from_position(2048), angles[1])

    def test_all_servos_are_read_with_a_packet_per_bus(self):
        self.bus.packets.clear()
        self.assertTrue(self.robot.servo_handler.read_current_pos())
        self.assertEqual([('sync_read', cfg.ADDR_PRESENT_POSITION, [1, 2, 3]),
                          ('sync_read', cfg.ADDR_PRESENT_POSITION, [4, 5, 6, 7])], self.bus.packets)

    def test_reading_servos_of_both_buses_in_one_packet_fails(self):
        # makes sure the fake bus behaves like the robot, which is why the handler reads per bus
        self.robot.servo_handler.buses = [(1, 2, 3, 4, 5, 6, 7)]
        self.assertFalse(self.robot.servo_handler.read_current_pos())

    def test_failed_read_does_not_update_positions(self):
        self.bus.set_position(1, 1500)
        self.assertEqual(1500, self.robot.get_current_positions()[1])

        self.bus.set_position(1, 2048)
        self.bus.unresponsive = {3}
        self.assertIsNone(self.robot.get_current_positions())
        self.assertIsNone(self.robot.get_current_angles())
        self.assertEqual(1500, self.servos[0].current_position)

    def test_failed_read_fails_the_safety_check(self):
        self.bus.unresponsive = {3}
        self.assertFalse(self.robot.safety_check())
        self.assertFalse(self.robot.enable_servos())

    def test_single_servo_position(self):
        self.bus.set_position(4, -20)
        self.assertEqual(-20, self.robot.get_servo_position(4))
        self.assertEqual(2048, self.robot.get_servo_position(7))

        self.bus.unresponsive = {4}
        self.assertIsNone(self.robot.get_servo_position(4))
        self.assertEqual(-20, self.servos[3].current_position)

        with self.assertRaises(ValueError):
            self.robot.get_servo_position(8)

    def test_enable_servos(self):
        self.assertTrue(self.robot.enable_servos())
        self.assertTrue(all(self.bus.torque(servo_id) for servo_id in range(1, 8)))

    def test_hardware_errors(self):
        self.assertEqual({}, self.robot.get_hardware_errors())

        self.bus.memory[5][cfg.ADDR_HARDWARE_ERROR_STATUS] = 0b00100100
        self.bus.memory[5][cfg.ADDR_PRESENT_TEMPERATURE] = 71
        self.bus.memory[5][cfg.ADDR_PRESENT_INPUT_VOLTAGE] = 112
        with self.assertLogs(drc.logger, 'ERROR') as logs:
            self.assertEqual({5: ['overheating', 'overload']}, self.robot.get_hardware_errors())
        self.assertIn('servo 5 shut down because of: overheating, overload (temperature 71C, input voltage 11.2V)',
                      logs.output[0])

    def test_hardware_errors_of_both_buses_are_read(self):
        self.bus.memory[2][cfg.ADDR_HARDWARE_ERROR_STATUS] = 0b00000001
        self.bus.memory[7][cfg.ADDR_HARDWARE_ERROR_STATUS] = 0b00010000
        self.assertEqual({2: ['input voltage error'], 7: ['electrical shock']}, self.robot.get_hardware_errors())

    def set_temperature(self, servo_id, temperature):
        self.bus.memory[servo_id][cfg.ADDR_PRESENT_TEMPERATURE] = temperature

    def test_overheating_servos(self):
        for servo_id in range(1, 8):
            self.set_temperature(servo_id, 30)
        self.assertEqual({}, self.robot.get_overheating_servos())

        self.set_temperature(2, self.servo_config.temperature_limit + 1)
        self.set_temperature(5, self.servo_config.temperature_limit)
        self.assertEqual({2: self.servo_config.temperature_limit + 1}, self.robot.get_overheating_servos())

    def test_warm_servo_is_logged_once(self):
        self.set_temperature(3, self.servo_config.temperature_warning + 1)
        with self.assertLogs(drc.logger, 'WARNING') as logs:
            self.robot.get_overheating_servos()
            self.robot.get_overheating_servos()
        self.assertEqual(1, len([line for line in logs.output if 'servo 3 is getting warm' in line]))

        # and again after it cooled down and warmed up again
        self.set_temperature(3, self.servo_config.temperature_warning)
        self.robot.get_overheating_servos()
        self.set_temperature(3, self.servo_config.temperature_warning + 2)
        with self.assertLogs(drc.logger, 'WARNING') as logs:
            self.robot.get_overheating_servos()
        self.assertIn('servo 3 is getting warm', logs.output[0])

    def test_servos_are_not_enabled_when_too_warm(self):
        self.set_temperature(2, self.servo_config.temperature_limit + 5)
        self.assertFalse(self.robot.enable_servos())
        self.assertFalse(any(self.bus.torque(servo_id) for servo_id in range(1, 8)))

    def test_servos_are_not_enabled_with_a_hardware_error(self):
        self.bus.memory[5][cfg.ADDR_HARDWARE_ERROR_STATUS] = 0b00100000
        self.assertFalse(self.robot.enable_servos())
        self.assertFalse(any(self.bus.torque(servo_id) for servo_id in range(1, 8)))

    def test_enable_servos_holds_the_current_position(self):
        self.bus.memory[2][cfg.ADDR_GOAL_POSITION:cfg.ADDR_GOAL_POSITION + 4] = (3000).to_bytes(4, 'little')
        self.bus.set_position(2, 1500)
        self.assertTrue(self.robot.enable_servos())
        self.assertEqual(1500, self.bus.get(2, cfg.ADDR_GOAL_POSITION, 4))
        self.assertEqual(1500, self.servos[1].target_position)

    def test_current_pose(self):
        angles = self.robot.get_current_angles()
        pose = self.robot.get_current_pose()
        expected = forward_kinematics(angles, global_constants.dynamixel_robot_config)
        self.assertEqual((expected.x, expected.y, expected.z), (pose.x, pose.y, pose.z))

    def test_servos_outside_calibrated_range(self):
        self.assertEqual({}, self.robot.get_servos_outside_calibrated_range())
        # the calibrated range includes the offset of the servo
        lower, upper = calibrated_range(self.servos[1])
        self.bus.set_position(2, lower)
        self.assertEqual({}, self.robot.get_servos_outside_calibrated_range())
        self.bus.set_position(2, lower - 15)
        self.assertEqual({2: (lower - 15, (lower, upper))}, self.robot.get_servos_outside_calibrated_range())

    def test_recalibrated_offset_is_used(self):
        servo = self.servos[1]
        self.bus.set_position(2, position_of(servo, 1.0))
        self.assertAlmostEqual(1.0, self.robot.get_current_angles()[2], places=2)

        # after recalibrating, the same physical position is a different angle, and the same angle a different position
        servo.constant_offset += 100
        self.assertAlmostEqual(servo.get_angle_from_position(position_of(servo, 1.0) - 100),
                               self.robot.get_current_angles()[2])
        self.robot.move_servo(2, 1.0)
        self.assertEqual(position_of(servo, 1.0), self.bus.get(2, cfg.ADDR_GOAL_POSITION, 4))

    def test_enable_servos_disables_all_servos_when_one_fails(self):
        self.bus.unresponsive = {5}
        with mock.patch.object(self.robot, 'safety_check', return_value=True):
            self.assertFalse(self.robot.enable_servos())
        self.assertFalse(any(self.bus.torque(servo_id) for servo_id in range(1, 8)))

    def test_profile_velocity_percentage_never_writes_zero(self):
        self.robot.set_profile_velocity_percentage(0.1)
        for servo_id in range(1, 8):
            self.assertEqual(1, self.bus.get(servo_id, cfg.ADDR_PROFILE_VELOCITY, 4))

    def test_single_servo_pid(self):
        self.robot.set_pid_single_servo(1, 1500, 10, 500)
        self.assertEqual((1500, 10, 500), (self.bus.get(1, cfg.ADDR_P, 2), self.bus.get(1, cfg.ADDR_I, 2),
                                            self.bus.get(1, cfg.ADDR_D, 2)))
        self.assertEqual(self.servos[1].p, self.bus.get(2, cfg.ADDR_P, 2))

    def test_raw_servo_position_is_not_clamped(self):
        self.robot.set_servo_position(1, 1024)
        self.assertEqual(1024, self.bus.get(1, cfg.ADDR_GOAL_POSITION, 4))

    def test_gripper_is_moved_with_a_single_write(self):
        self.bus.packets.clear()
        gripper = self.servos[6]
        self.assertTrue(self.robot.set_gripper(0))
        self.assertEqual([('sync_write', cfg.ADDR_GOAL_POSITION, [7])], self.bus.packets)
        self.assertEqual(position_of(gripper, 0), self.bus.get(7, cfg.ADDR_GOAL_POSITION, 4))

        # a large change isn't split up, the servo moves smoothly at its profile velocity
        self.bus.packets.clear()
        self.assertTrue(self.robot.set_gripper(100))
        self.assertEqual([('sync_write', cfg.ADDR_GOAL_POSITION, [7])], self.bus.packets)
        self.assertEqual(position_of(gripper, 100), self.bus.get(7, cfg.ADDR_GOAL_POSITION, 4))

    def test_gripper_state(self):
        self.bus.set_position(7, position_of(self.servos[6], 50))
        self.assertAlmostEqual(50, self.robot.get_gripper_state(), delta=0.1)
        self.bus.unresponsive = {7}
        self.assertIsNone(self.robot.get_gripper_state())

    def test_reboot_reinitializes_the_servos(self):
        self.bus.memory[1][cfg.ADDR_P:cfg.ADDR_P + 2] = bytes(2)
        with mock.patch('src.robot_controllers.dynamixel_robot.servo_handler.time.sleep'):
            self.robot.reboot_all_servos()
        self.assertEqual(set(range(1, 8)), self.bus.rebooted)
        self.assertEqual(self.servos[0].p, self.bus.get(1, cfg.ADDR_P, 2))

    def test_servo_configuration_without_all_servos_raises_before_touching_the_bus(self):
        del self.servo_config.servos[4]
        bus = mock.Mock()
        with self.assertRaisesRegex(DynamixelError, r'no servos with ids \[4\]'):
            drc.DynamixelRobotController(global_constants.dynamixel_robot_config, mock.Mock(), bus,
                                         servo_config=self.servo_config)
        self.assertEqual([], bus.mock_calls)

    def test_servos_are_not_enabled_when_the_hardware_errors_can_not_be_read(self):
        with mock.patch.object(self.robot, 'safety_check', return_value=True), \
                mock.patch.object(self.robot.servo_handler, 'read_hardware_errors', return_value=None):
            self.assertFalse(self.robot.enable_servos())
        self.assertFalse(any(self.bus.torque(servo_id) for servo_id in range(1, 8)))

    def test_servos_are_not_enabled_when_the_temperatures_can_not_be_read(self):
        with mock.patch.object(self.robot, 'safety_check', return_value=True), \
                mock.patch.object(self.robot.servo_handler, 'read_temperatures', return_value=None):
            self.assertIsNone(self.robot.get_overheating_servos())
            self.assertFalse(self.robot.enable_servos())
        self.assertFalse(any(self.bus.torque(servo_id) for servo_id in range(1, 8)))

    def test_failed_safety_check_disables_the_servos(self):
        self.assertTrue(self.robot.enable_servos())
        with mock.patch.object(self.robot, 'safety_check', return_value=False):
            self.assertFalse(self.robot.enable_servos())
        self.assertFalse(any(self.bus.torque(servo_id) for servo_id in range(1, 8)))

    def test_servos_are_not_enabled_when_the_goal_position_can_not_be_set(self):
        with mock.patch.object(self.robot.servo_handler, 'hold_current_position', return_value=False):
            self.assertFalse(self.robot.enable_servos())
        self.assertFalse(any(self.bus.torque(servo_id) for servo_id in range(1, 8)))

    def test_servo_at_the_warning_temperature_is_not_logged(self):
        for servo_id in range(1, 8):
            self.set_temperature(servo_id, self.servo_config.temperature_warning)
        with mock.patch.object(drc.logger, 'warning') as warning:
            self.assertEqual({}, self.robot.get_overheating_servos())
        warning.assert_not_called()

    def test_temperature_limits_come_from_the_servo_configuration(self):
        self.servo_config.temperature_warning, self.servo_config.temperature_limit = 30, 35
        self.set_temperature(1, 36)
        self.set_temperature(2, 35)
        self.assertEqual({1: 36}, self.robot.get_overheating_servos())

    def test_move_servos_does_not_move_the_gripper(self):
        gripper_goal = self.bus.get(7, cfg.ADDR_GOAL_POSITION, 4)
        self.robot.move_servos([0, 1.5, 1.5, 0.0, 0.5, -0.5, 1.0])
        self.assertEqual(gripper_goal, self.bus.get(7, cfg.ADDR_GOAL_POSITION, 4))

    def test_debug_functions_reject_unknown_servo_ids(self):
        self.bus.packets.clear()
        for function, args in ((self.robot.move_servo, (8, 0.0)), (self.robot.set_servo_position, (0, 2048)),
                               (self.robot.set_pid_single_servo, (9, 1, 2, 3))):
            with self.assertRaises(ValueError):
                function(*args)
        self.assertEqual([], self.bus.packets)

    def test_angle_limits(self):
        limits = self.robot.get_angle_limits()
        self.assertEqual(list(drc.ARM_IDS), list(limits))
        for servo_id, (low, high) in limits.items():
            servo = self.servos[servo_id - 1]
            self.assertEqual((min(servo.min_angle, servo.max_angle), max(servo.min_angle, servo.max_angle)),
                             (low, high))

    def test_reboot_raises_when_a_servo_does_not_come_back(self):
        self.bus.unresponsive = {2}
        with mock.patch('src.robot_controllers.dynamixel_robot.servo_handler.time.sleep'):
            with self.assertRaises(DynamixelError):
                self.robot.reboot_all_servos()

    # static load compensation

    def test_present_currents(self):
        self.bus.memory[2][cfg.ADDR_PRESENT_CURRENT:cfg.ADDR_PRESENT_CURRENT + 2] = (-50 & 0xFFFF).to_bytes(2, 'little')
        currents = self.robot.get_present_currents()
        self.assertEqual(-50, currents[2])
        self.assertEqual([1, 2, 3, 4, 5, 6], sorted(currents))
        self.assertEqual({7: 0}, self.robot.get_present_currents([7]))
        self.bus.unresponsive = {4}
        self.assertIsNone(self.robot.get_present_currents())

    def test_temperatures_and_voltages(self):
        self.set_temperature(3, 41)
        self.bus.memory[3][cfg.ADDR_PRESENT_INPUT_VOLTAGE:cfg.ADDR_PRESENT_INPUT_VOLTAGE + 2] = (116).to_bytes(2, 'little')
        self.assertEqual((41, 11.6), self.robot.get_temperatures_and_voltages()[3])
        self.bus.unresponsive = {3}
        self.assertIsNone(self.robot.get_temperatures_and_voltages())

    def test_compensation_is_sent_with_the_goal_and_left_out_of_the_position_error(self):
        from src.robot_controllers.dynamixel_robot.static_compensation import StaticCompensation
        servo = self.servos[1]  # servo 2
        servo.compensation = StaticCompensation(2, {'cos_q2': 20.0}, 60)
        angles = np.array([0, 0.5, 0.0, 1.0, 0.2, 0.3, 0.1])  # cos(q2 = 0) = 1: 20 positions
        self.robot.move_servos(angles)
        self.assertEqual(20, servo.compensation_positions)
        self.assertEqual(position_of(self.servo_config.servos[2], 0.0) + 20, self.bus.get(2, cfg.ADDR_GOAL_POSITION, 4))
        # the servo reaches the uncompensated target: the printed error is 0
        self.bus.set_position(2, servo.target_position - 20)
        with mock.patch('builtins.print') as printed:
            self.robot._print_position_error()
        self.assertIn('2:0', printed.call_args.args[0])
        self.assertAlmostEqual(0.0, self.robot.get_current_angles()[2], places=3)


if __name__ == '__main__':
    unittest.main()
