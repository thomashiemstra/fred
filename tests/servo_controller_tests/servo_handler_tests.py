import math
import types
import unittest
from unittest import mock

import dynamixel_sdk as dxl
from numpy import pi

from src.global_constants import POSITION_CONTROL_MODE, EXTENDED_POSITION_CONTROL, \
    CURRENT_BASED_POSITION_CONTROL_MODE
from src.robot_controllers.dynamixel_robot import dynamixel_x_config
from src.robot_controllers.dynamixel_robot import dynamixel_x_config as cfg
from src.robot_controllers.dynamixel_robot import servo_handler as servo_handler_module
from src.robot_controllers.dynamixel_robot.servo import Servo
from src.robot_controllers.dynamixel_robot.servo_handler import ServoHandler, ServoState
from tests.servo_controller_tests.dynamixel_robot_controller_tests import FakeBus


class TestServoHandlerAngleConversion(unittest.TestCase):

    def setUp(self):
        servo1 = Servo(0, 4095, 0, 2 * pi, POSITION_CONTROL_MODE)
        servo2 = Servo(1024, 3072, 0, pi, POSITION_CONTROL_MODE)
        servo3 = Servo(0, 4095, -pi, pi, POSITION_CONTROL_MODE)
        # if a servo is mounted backwards it's min will correspond to the max angle and vice versa
        servo4 = Servo(3072, 1024, -pi/2, pi/2, POSITION_CONTROL_MODE)
        servos = {1: servo1, 2: servo2, 3: servo3, 4: servo4}
        self.servo_handler = ServoHandler(servos, dynamixel_x_config, None, None)

    def test_angle_to_position_full_turn_servo(self):
        expected_position = math.floor(4095 / 2)
        self.servo_handler.set_angle(1, pi)
        self.assertEqual(expected_position, self.servo_handler.get_servo(1).target_position)

    def test_angle_to_position_half_turn_servo(self):
        expected_position = 1024 + (3072 - 1024) / 4
        self.servo_handler.set_angle(2, pi/4)
        self.assertEqual(expected_position, self.servo_handler.get_servo(2).target_position)

    def test_angle_to_position_is_rounded_to_a_whole_position(self):
        expected_position = 1280
        self.servo_handler.set_angle(2, pi/8)
        self.assertEqual(expected_position, self.servo_handler.get_servo(2).target_position)

    def test_angle_to_position_symmetric_range(self):
        expected_position = math.floor(3*4095 / 4)
        self.servo_handler.set_angle(3, pi/2)
        self.assertEqual(expected_position, self.servo_handler.get_servo(3).target_position)

    def test_angle_to_position_reversed_servo_middle(self):
        expected_position = math.ceil(4095 / 2)
        self.servo_handler.set_angle(4, 0)
        self.assertEqual(expected_position, self.servo_handler.get_servo(4).target_position)

    def test_angle_to_position_reversed_servo_positive_angle(self):
        expected_position = math.floor(1024 + (3072 - 1024)/4)
        self.servo_handler.set_angle(4, pi/4)
        self.assertEqual(expected_position, self.servo_handler.get_servo(4).target_position)

    def test_angle_to_position_reversed_servo_negative_angle(self):
        expected_position = math.floor(1024 + 3*(3072 - 1024) / 4)
        self.servo_handler.set_angle(4, - pi/4)
        self.assertEqual(expected_position, self.servo_handler.get_servo(4).target_position)

    def test_position_to_angle_full_turn_servo(self):
        self.assertAlmostEqual(pi, self.servo_handler.get_angle(1, math.floor(4095/2)), delta=0.01)
        self.assertAlmostEqual(pi/2, self.servo_handler.get_angle(1, math.floor(4095/4)), delta=0.01)

    def test_position_to_angle_half_turn_servo(self):
        angle = self.servo_handler.get_angle(2, math.floor(1024 + (3072-1024)/4))
        self.assertAlmostEqual(pi/4, angle, delta=0.01)

    def test_position_to_angle_symmetric_range(self):
        self.assertAlmostEqual(0, self.servo_handler.get_angle(3, math.floor(4095/2)), delta=0.01)
        self.assertAlmostEqual(pi/2, self.servo_handler.get_angle(3, math.floor(3*4095/4)), delta=0.01)

    def test_position_to_angle_reversed_servo(self):
        self.assertAlmostEqual(-pi/4, self.servo_handler.get_angle(4, math.ceil(1024 + 3*(3072-1024)/4)), delta=0.01)
        self.assertAlmostEqual(0, self.servo_handler.get_angle(4, math.ceil(1024 + (3072-1024)/2)), delta=0.01)
        self.assertAlmostEqual(-pi/2, self.servo_handler.get_angle(4, 3072), delta=0.01)
        self.assertAlmostEqual(pi/2, self.servo_handler.get_angle(4, 1024), delta=0.01)


class TestServoHandlerCommunication(unittest.TestCase):
    """tests of the packets the handler sends, on a fake bus with two buses of two servos"""

    BUSES = ((1, 2), (3, 4))

    def setUp(self):
        self.servos = [
            Servo(1024, 3072, 0, pi, POSITION_CONTROL_MODE, profile_velocity=250, profile_acceleration=100,
                  p=1500, i=20, d=500, offset=100),
            # mounted the other way around
            Servo(3072, 1024, -pi / 2, pi / 2, POSITION_CONTROL_MODE, p=0x1234, i=0x0102, d=0x0A0B),
            Servo(0, 4096, -pi, pi, EXTENDED_POSITION_CONTROL, profile_velocity=0x01020304,
                  profile_acceleration=0x0A0B0C0D),
            Servo(1024, 2048, 0, 100, CURRENT_BASED_POSITION_CONTROL_MODE, goal_current=400),
        ]
        self.bus = FakeBus(self.servos, self.BUSES)
        self.handler = ServoHandler({servo_id: servo for servo_id, servo in enumerate(self.servos, start=1)},
                                    cfg, mock.Mock(), self.bus, self.BUSES)

    def raw(self, servo_id, address, length):
        return list(self.bus.memory[servo_id][address:address + length])

    def goal_position(self, servo_id):
        return self.bus.get(servo_id, cfg.ADDR_GOAL_POSITION, 4, signed=True)

    # construction

    def test_every_servo_must_be_on_exactly_one_bus(self):
        servos = {1: self.servos[0], 2: self.servos[1]}
        with self.assertRaises(ValueError):
            ServoHandler(servos, cfg, None, None, [(1,)])
        with self.assertRaises(ValueError):
            ServoHandler(servos, cfg, None, None, [(1, 2), (2,)])
        # by default all servos are on the same bus
        self.assertEqual([(1, 2)], ServoHandler(servos, cfg, None, None).buses)

    # reading

    def test_positions_are_read_with_a_packet_per_bus(self):
        self.assertTrue(self.handler.read_current_pos())
        self.assertEqual([('sync_read', cfg.ADDR_PRESENT_POSITION, [1, 2]),
                          ('sync_read', cfg.ADDR_PRESENT_POSITION, [3, 4])], self.bus.packets)

    def test_only_the_buses_of_the_requested_servos_are_read(self):
        self.assertTrue(self.handler.read_current_pos([3]))
        self.assertEqual([('sync_read', cfg.ADDR_PRESENT_POSITION, [3])], self.bus.packets)

    def test_present_position_is_converted_to_a_signed_value(self):
        for servo_id, position in ((1, -1), (2, -4096), (3, -2 ** 31), (4, 2 ** 31 - 1)):
            self.bus.set_position(servo_id, position)
        self.assertTrue(self.handler.read_current_pos())
        self.assertEqual([-1, -4096, -2 ** 31, 2 ** 31 - 1], [servo.current_position for servo in self.servos])

    def test_failed_read_of_the_second_bus_does_not_update_the_first_bus(self):
        self.bus.set_position(1, 1500)
        self.assertTrue(self.handler.read_current_pos())
        self.assertEqual(1500, self.servos[0].current_position)

        # the first bus is read successfully, but the positions are only updated when all servos could be read
        self.bus.set_position(1, 1600)
        self.bus.unresponsive = {4}
        self.assertFalse(self.handler.read_current_pos())
        self.assertEqual(1500, self.servos[0].current_position)
        self.assertIsNone(self.handler.read_current_pos_single_servo(4))

    def test_failed_sync_read_packet_returns_none(self):
        with mock.patch.object(dxl.GroupSyncRead, 'txRxPacket', return_value=dxl.COMM_TX_FAIL):
            self.assertFalse(self.handler.read_current_pos())
            self.assertIsNone(self.handler.read_temperatures())
            self.assertIsNone(self.handler.read_hardware_errors())
            self.assertIsNone(self.handler.read_profile_velocity())

    def test_a_corrupt_status_packet_is_read_again(self):
        original = dxl.GroupSyncRead.txRxPacket
        results = [dxl.COMM_RX_CORRUPT, dxl.COMM_RX_CORRUPT]  # two bad packets, then the read works

        def flaky(group_sync_read):
            return results.pop(0) if results else original(group_sync_read)
        with mock.patch.object(dxl.GroupSyncRead, 'txRxPacket', flaky), self.assertLogs(level='WARNING'):
            self.assertIsNotNone(self.handler.read_temperatures())
        self.assertEqual([], results)
        self.assertEqual(2, self.handler.failed_reads)
        # every attempt failing still fails the read
        with mock.patch.object(dxl.GroupSyncRead, 'txRxPacket', return_value=dxl.COMM_RX_CORRUPT) as packet, \
                self.assertLogs(level='WARNING'):
            self.assertIsNone(self.handler.read_temperatures())
        self.assertEqual(servo_handler_module.SYNC_READ_ATTEMPTS, packet.call_count,
                         'the first bus, attempted to the end')

    def test_a_failed_read_says_whether_it_ran_into_the_timeout(self):
        # the sdk zeroes packet_timeout when it runs out, the port handler keeps the value it was set to
        self.handler.port_handler.packet_timeout = 0.0
        self.handler.port_handler.last_packet_timeout = 10.0
        clock = mock.Mock()
        with mock.patch.object(servo_handler_module, 'time', clock), \
                mock.patch.object(dxl.GroupSyncRead, 'txRxPacket', return_value=dxl.COMM_RX_CORRUPT):
            # every attempt: 10.5 ms, the reply was cut off by the timeout
            clock.perf_counter.side_effect = [0.0, 0.0105] * servo_handler_module.SYNC_READ_ATTEMPTS
            with self.assertLogs(servo_handler_module.logger, 'WARNING') as logs:
                self.assertIsNone(self.handler.read_temperatures())
            self.assertIn('failed after 10.5 ms, at the timeout of 10.0 ms', logs.output[0])
            self.assertEqual((3, 3), (self.handler.failed_reads, self.handler.late_reads))
            # 1.2 ms: the reply was there but its bytes were wrong
            clock.perf_counter.side_effect = [0.0, 0.0012] * servo_handler_module.SYNC_READ_ATTEMPTS
            with self.assertLogs(servo_handler_module.logger, 'WARNING') as logs:
                self.assertIsNone(self.handler.read_temperatures())
            self.assertIn('failed after 1.2 ms, before the timeout of 10.0 ms', logs.output[0])
            self.assertEqual((6, 3), (self.handler.failed_reads, self.handler.late_reads))
        # read_temperatures reads the first bus first and gives up there
        self.assertEqual({(1, 2): 6}, self.handler.failed_reads_by_bus)

    def test_hardware_error_bits(self):
        self.bus.memory[1][cfg.ADDR_HARDWARE_ERROR_STATUS] = 0
        self.bus.memory[2][cfg.ADDR_HARDWARE_ERROR_STATUS] = 1 << 3
        self.bus.memory[3][cfg.ADDR_HARDWARE_ERROR_STATUS] = (1 << 0) | (1 << 2) | (1 << 5)
        errors = self.handler.read_hardware_errors()
        self.assertEqual({2: [cfg.HARDWARE_ERRORS[3]],
                          3: [cfg.HARDWARE_ERRORS[0], cfg.HARDWARE_ERRORS[2], cfg.HARDWARE_ERRORS[5]]}, errors)

    def test_unknown_hardware_error_bits_are_still_reported(self):
        # a servo with an error status is reported even when none of its bits are known, so it is never enabled
        unknown_bits = [bit for bit in range(8) if bit not in cfg.HARDWARE_ERRORS]
        self.bus.memory[4][cfg.ADDR_HARDWARE_ERROR_STATUS] = 1 << unknown_bits[0]
        self.assertEqual({4: []}, self.handler.read_hardware_errors())

    def test_temperature_and_voltage(self):
        self.bus.memory[2][cfg.ADDR_PRESENT_TEMPERATURE] = 41
        self.bus.memory[2][cfg.ADDR_PRESENT_INPUT_VOLTAGE:cfg.ADDR_PRESENT_INPUT_VOLTAGE + 2] = \
            (121).to_bytes(2, 'little')
        self.assertEqual({2: (41, 12.1)}, self.handler.read_temperature_and_voltage([2]))

        self.bus.unresponsive = {2}
        self.assertIsNone(self.handler.read_temperature_and_voltage([2]))

    def test_profile_velocity_is_read(self):
        self.handler.set_profile_velocity_and_acceleration()
        self.assertEqual({servo_id: servo.profile_velocity for servo_id, servo in enumerate(self.servos, start=1)},
                         self.handler.read_profile_velocity())

    # writing

    def test_pid_is_written_in_a_single_packet_as_d_i_p(self):
        self.assertTrue(self.handler.set_pid([2]))
        self.assertEqual([('sync_write', cfg.ADDR_D, [2])], self.bus.packets)
        # little endian d, i, p in adjacent registers
        self.assertEqual([0x0B, 0x0A, 0x02, 0x01, 0x34, 0x12], self.raw(2, cfg.ADDR_D, 6))

    def test_single_servo_pid_only_writes_that_servo(self):
        self.assertTrue(self.handler.set_pid_single_servo(3, 800, 0, 60))
        self.assertEqual([('sync_write', cfg.ADDR_D, [3])], self.bus.packets)
        self.assertEqual((800, 0, 60), (self.bus.get(3, cfg.ADDR_P, 2), self.bus.get(3, cfg.ADDR_I, 2),
                                        self.bus.get(3, cfg.ADDR_D, 2)))

    def test_profile_is_written_in_a_single_packet_as_acceleration_and_velocity(self):
        self.assertTrue(self.handler.set_profile_velocity_and_acceleration())
        self.assertEqual([('sync_write', cfg.ADDR_PROFILE_ACCELERATION, [1, 2, 3, 4])], self.bus.packets)
        self.assertEqual([0x0D, 0x0C, 0x0B, 0x0A, 0x04, 0x03, 0x02, 0x01],
                         self.raw(3, cfg.ADDR_PROFILE_ACCELERATION, 8))
        self.assertEqual((100, 250), (self.bus.get(1, cfg.ADDR_PROFILE_ACCELERATION, 4),
                                      self.bus.get(1, cfg.ADDR_PROFILE_VELOCITY, 4)))

    def test_non_adjacent_registers_are_rejected(self):
        # a combined write to registers that are not adjacent would write garbage into the registers in between
        config_values = {name: getattr(cfg, name) for name in dir(cfg) if name.isupper()}
        for changes in ({'ADDR_PROFILE_VELOCITY': cfg.ADDR_PROFILE_VELOCITY + 1}, {'ADDR_I': cfg.ADDR_I + 2},
                        {'ADDR_P': cfg.ADDR_P - 1}):
            config = types.SimpleNamespace(**dict(config_values, **changes))
            with self.assertRaises(ValueError):
                ServoHandler({1: self.servos[0]}, config, None, None)

    def test_profile_velocity_percentage(self):
        self.assertTrue(self.handler.set_profile_velocity_percentage(50, [1]))
        self.assertEqual(125, self.bus.get(1, cfg.ADDR_PROFILE_VELOCITY, 4))
        # the other registers are not touched
        self.assertEqual(0, self.bus.get(1, cfg.ADDR_PROFILE_ACCELERATION, 4))
        self.assertEqual(0, self.bus.get(2, cfg.ADDR_PROFILE_VELOCITY, 4))

    def test_profile_velocity_percentage_out_of_range(self):
        for percentage in (0, -10, 100.1, 200):
            with self.assertRaises(ValueError):
                self.handler.set_profile_velocity_percentage(percentage)
        self.assertEqual([], self.bus.packets)

    def test_configured_goal_current_only_writes_servos_that_have_one(self):
        self.assertTrue(self.handler.set_configured_goal_current())
        self.assertEqual([('sync_write', cfg.ADDR_GOAL_CURRENT, [4])], self.bus.packets)
        self.assertEqual(400, self.bus.get(4, cfg.ADDR_GOAL_CURRENT, 2))

        # nothing to write is not a failure, and doesn't send an empty packet
        self.bus.packets.clear()
        self.assertTrue(self.handler.set_configured_goal_current([1, 2]))
        self.assertEqual([], self.bus.packets)

    def test_negative_goal_current_is_written_as_twos_complement(self):
        self.assertTrue(self.handler.set_goal_current(4, -300))
        self.assertEqual(-300, self.bus.get(4, cfg.ADDR_GOAL_CURRENT, 2, signed=True))

    def test_failed_sync_write_returns_false(self):
        with mock.patch.object(dxl.GroupSyncWrite, 'txPacket', return_value=dxl.COMM_TX_FAIL):
            self.assertFalse(self.handler.move_to_angles())
            self.assertFalse(self.handler.set_pid())
            self.assertFalse(self.handler.set_profile_velocity_and_acceleration())

    def test_move_to_angles_writes_the_target_positions_in_one_packet(self):
        self.handler.set_angle(1, pi / 2)
        self.handler.set_angle(2, pi / 4)
        self.assertTrue(self.handler.move_to_angles([1, 2]))
        self.assertEqual([('sync_write', cfg.ADDR_GOAL_POSITION, [1, 2])], self.bus.packets)
        self.assertEqual(2048 + 100, self.goal_position(1))
        self.assertEqual(1536, self.goal_position(2))

    def test_target_position_is_clamped_to_the_calibrated_range_including_the_offset(self):
        self.servos[0].target_position = 5000
        self.servos[1].target_position = 0  # reversed servo, its range is 1024 to 3072
        with self.assertLogs(servo_handler_module.logger, 'WARNING'):
            self.handler.move_to_angles([1, 2])
        self.assertEqual(3072 + 100, self.goal_position(1))
        self.assertEqual(1024, self.goal_position(2))
        # the clamped target is stored, so position errors are computed against what the servo actually got
        self.assertEqual((3072 + 100, 1024), (self.servos[0].target_position, self.servos[1].target_position))

    def test_target_position_is_clamped_to_the_physical_range(self):
        # the calibrated range of servo 3 (0 to 4096) is beyond the physical range of one turn
        self.servos[2].target_position = -50
        self.handler.move_to_angles([3])
        self.assertEqual(cfg.MIN_POSITION, self.goal_position(3))
        self.servos[2].target_position = 4096
        self.handler.move_to_angles([3])
        self.assertEqual(cfg.MAX_POSITION, self.goal_position(3))

    def test_target_position_in_range_is_not_changed_or_logged(self):
        self.servos[0].target_position = 2000
        with mock.patch.object(servo_handler_module.logger, 'warning') as warning:
            self.handler.move_to_angles([1])
        warning.assert_not_called()
        self.assertEqual(2000, self.goal_position(1))

    def test_raw_position_is_not_clamped(self):
        self.assertTrue(self.handler.move_servo_to_pos(1, 10))
        self.assertEqual(10, self.goal_position(1))
        self.assertEqual(0, self.servos[0].target_position)

    # holding the position

    def test_hold_current_position(self):
        for servo_id, position in ((1, 1500), (2, 2500), (3, -20), (4, 1100)):
            self.bus.set_position(servo_id, position)
        self.assertTrue(self.handler.hold_current_position())
        self.assertEqual([1500, 2500, -20, 1100], [self.goal_position(servo_id) for servo_id in range(1, 5)])
        self.assertEqual([1500, 2500, -20, 1100], [servo.target_position for servo in self.servos])

    def test_hold_current_position_does_not_write_when_the_read_fails(self):
        self.bus.set_position(1, 1500)
        self.bus.unresponsive = {3}
        self.assertFalse(self.handler.hold_current_position())
        self.assertEqual([], self.bus.packets_to(cfg.ADDR_GOAL_POSITION))
        self.assertEqual(0, self.servos[0].target_position)

    # torque and operating mode

    def test_set_torque_tries_every_servo_when_one_fails(self):
        for servo_id in range(1, 5):
            self.bus.memory[servo_id][cfg.ADDR_TORQUE_ENABLE] = 1
        self.bus.unresponsive = {2}
        self.assertFalse(self.handler.set_torque(enable=False))
        self.assertEqual([0, 0, 0], [self.bus.torque(servo_id) for servo_id in (1, 3, 4)])

    def test_set_torque_of_some_servos(self):
        self.assertTrue(self.handler.set_torque(True, [3]))
        self.assertEqual([0, 0, 1, 0], [self.bus.torque(servo_id) for servo_id in range(1, 5)])

    def test_operating_mode_is_not_written_when_it_is_already_set(self):
        with mock.patch.object(self.bus, 'write1ByteTxRx', wraps=self.bus.write1ByteTxRx) as write:
            self.assertTrue(self.handler.set_configured_operating_mode())
        write.assert_not_called()

    def test_operating_mode_is_written_when_it_differs(self):
        self.bus.memory[3][cfg.ADDR_OPERATING_MODE] = POSITION_CONTROL_MODE
        with mock.patch.object(self.bus, 'write1ByteTxRx', wraps=self.bus.write1ByteTxRx) as write:
            self.assertTrue(self.handler.set_configured_operating_mode())
        write.assert_called_once_with(mock.ANY, 3, cfg.ADDR_OPERATING_MODE, EXTENDED_POSITION_CONTROL)
        self.assertEqual(EXTENDED_POSITION_CONTROL, self.bus.memory[3][cfg.ADDR_OPERATING_MODE])

    def test_operating_mode_is_not_written_when_it_can_not_be_read(self):
        self.bus.unresponsive = {2}
        with mock.patch.object(self.bus, 'write1ByteTxRx', wraps=self.bus.write1ByteTxRx) as write:
            self.assertFalse(self.handler.set_operating_mode(2, EXTENDED_POSITION_CONTROL))
        write.assert_not_called()

    # reboot

    def test_reboot_returns_once_all_servos_respond(self):
        with mock.patch.object(servo_handler_module.time, 'sleep') as sleep:
            self.assertTrue(self.handler.reboot())
        self.assertEqual({1, 2, 3, 4}, self.bus.rebooted)
        self.assertEqual(1, sleep.call_count)

    def test_reboot_waits_for_slow_servos(self):
        pings = {'count': 0}
        ping = self.bus.ping

        def slow_ping(port, servo_id):
            pings['count'] += 1
            if servo_id == 3 and pings['count'] < 10:
                return 0, dxl.COMM_RX_TIMEOUT, 0
            return ping(port, servo_id)

        with mock.patch.object(self.bus, 'ping', slow_ping), \
                mock.patch.object(servo_handler_module.time, 'sleep'):
            self.assertTrue(self.handler.reboot([1, 3]))
        self.assertEqual({1, 3}, self.bus.rebooted)

    def test_reboot_times_out_when_a_servo_does_not_respond(self):
        self.bus.unresponsive = {4}
        clock = iter(range(0, 1000))
        with mock.patch.object(servo_handler_module.time, 'monotonic', lambda: next(clock) * 0.5), \
                mock.patch.object(servo_handler_module.time, 'sleep'), \
                self.assertLogs(servo_handler_module.logger, 'ERROR') as logs:
            self.assertFalse(self.handler.reboot(timeout=3.0))
        self.assertIn('[ID:004] did not respond after reboot', logs.output[0])
        self.assertEqual(1, len(logs.output))

    def test_servo_responding_with_an_error_has_not_come_back(self):
        ping = self.bus.ping
        with mock.patch.object(self.bus, 'ping',
                               lambda port, servo_id: (1020, dxl.COMM_SUCCESS, 0x80) if servo_id == 2
                               else ping(port, servo_id)), \
                mock.patch.object(servo_handler_module.time, 'sleep'), \
                mock.patch.object(servo_handler_module.time, 'monotonic', side_effect=[0.0, 1.0, 5.0]), \
                self.assertLogs(servo_handler_module.logger, 'ERROR'):
            self.assertFalse(self.handler.reboot(timeout=3.0))

    def fail_reboot_of(self, failing, comm_result=dxl.COMM_RX_TIMEOUT, error=0):
        """patch the fake bus so the reboot command of the given servos fails, the servos still answer pings"""
        reboot = self.bus.reboot
        return mock.patch.object(
            self.bus, 'reboot',
            lambda port, servo_id: (comm_result, error) if servo_id in failing else reboot(port, servo_id))

    def test_reboot_fails_when_the_reboot_command_fails_even_if_the_servo_answers_pings(self):
        with self.fail_reboot_of({3}), mock.patch.object(servo_handler_module.time, 'sleep'), \
                self.assertLogs(servo_handler_module.logger, 'ERROR') as logs:
            self.assertFalse(self.handler.reboot())
        self.assertEqual({1, 2, 4}, self.bus.rebooted)
        self.assertTrue(any('[ID:003] reboot command failed' in line for line in logs.output))
        self.assertFalse(any('did not respond' in line for line in logs.output))

    def test_reboot_fails_when_the_servo_answers_the_reboot_command_with_an_error(self):
        with self.fail_reboot_of({2}, dxl.COMM_SUCCESS, 0x01), \
                mock.patch.object(servo_handler_module.time, 'sleep'), \
                self.assertLogs(servo_handler_module.logger, 'ERROR'):
            self.assertFalse(self.handler.reboot([1, 2]))

    def test_reboot_still_waits_for_the_other_servos_when_one_command_fails(self):
        self.bus.unresponsive = {4}
        clock = iter(range(0, 1000))
        with self.fail_reboot_of({1}), mock.patch.object(servo_handler_module.time, 'sleep'), \
                mock.patch.object(servo_handler_module.time, 'monotonic', lambda: next(clock) * 0.5), \
                self.assertLogs(servo_handler_module.logger, 'ERROR') as logs:
            self.assertFalse(self.handler.reboot(timeout=3.0))
        self.assertTrue(any('[ID:001] reboot command failed' in line for line in logs.output))
        self.assertTrue(any('[ID:004] did not respond after reboot' in line for line in logs.output))

    # static load compensation

    def test_present_current_is_read_signed_with_a_packet_per_bus(self):
        self.bus.memory[1][cfg.ADDR_PRESENT_CURRENT:cfg.ADDR_PRESENT_CURRENT + 2] = (-30 & 0xFFFF).to_bytes(2, 'little')
        self.bus.memory[3][cfg.ADDR_PRESENT_CURRENT:cfg.ADDR_PRESENT_CURRENT + 2] = (100).to_bytes(2, 'little')
        self.assertEqual({1: -30, 2: 0, 3: 100, 4: 0}, self.handler.read_present_current())
        self.assertEqual([('sync_read', cfg.ADDR_PRESENT_CURRENT, [1, 2]),
                          ('sync_read', cfg.ADDR_PRESENT_CURRENT, [3, 4])], self.bus.packets)
        self.assertEqual({3: 100}, self.handler.read_present_current([3]))
        self.bus.unresponsive = {2}
        self.assertIsNone(self.handler.read_present_current())

    def test_the_compensation_is_kept_at_the_end_of_the_calibrated_range(self):
        # servo 1: positions 1024-3072 with offset 100, a target beyond the range with 40 positions of compensation
        servo = self.servos[0]
        servo.target_position = 3300
        servo.compensation_positions = 40
        with self.assertLogs(servo_handler_module.logger, 'WARNING'):
            self.assertTrue(self.handler.move_to_angles([1]))
        self.assertEqual(3072 + 100 + 40, self.goal_position(1))
        self.assertEqual(3072 + 100 + 40, servo.target_position)

    def test_hold_current_position_drops_the_compensation(self):
        servo = self.servos[0]
        servo.compensation_positions = 40
        self.bus.set_position(1, 2000)
        self.assertTrue(self.handler.hold_current_position([1]))
        self.assertEqual(2000, servo.target_position)
        self.assertEqual(0, servo.compensation_positions)
        self.assertEqual(2000, servo.clamp_target_position())

    def test_the_compensation_never_leaves_the_physical_range(self):
        servo = self.servos[2]  # 0-4096
        servo.target_position = 4110
        servo.compensation_positions = 20
        with self.assertLogs(servo_handler_module.logger, 'WARNING'):
            self.assertTrue(self.handler.move_to_angles([3]))
        self.assertEqual(cfg.MAX_POSITION, self.goal_position(3))



class TestServoHandlerBlockReads(unittest.TestCase):
    """the block reads and the single servo setters of the gain tuning, on the fake bus of two buses"""

    BUSES = ((1, 2), (3, 4))

    def setUp(self):
        self.servos = [
            Servo(1024, 3072, 0, pi, POSITION_CONTROL_MODE, profile_velocity=250, profile_acceleration=100,
                  p=1500, i=20, d=500, offset=100),
            Servo(3072, 1024, -pi / 2, pi / 2, POSITION_CONTROL_MODE, p=0x1234, i=0x0102, d=0x0A0B),
            Servo(0, 4096, -pi, pi, EXTENDED_POSITION_CONTROL, profile_velocity=300, profile_acceleration=50),
            Servo(1024, 2048, 0, 100, CURRENT_BASED_POSITION_CONTROL_MODE, goal_current=400),
        ]
        self.bus = FakeBus(self.servos, self.BUSES)
        self.handler = ServoHandler({servo_id: servo for servo_id, servo in enumerate(self.servos, start=1)},
                                    cfg, mock.Mock(), self.bus, self.BUSES)

    def set(self, servo_id, address, length, value):
        self.bus.memory[servo_id][address:address + length] = (value & ((1 << (8 * length)) - 1)).to_bytes(
            length, 'little')

    def test_state_is_read_in_one_block_per_bus(self):
        self.set(1, cfg.ADDR_REALTIME_TICK, 2, 1234)
        self.set(1, cfg.ADDR_MOVING, 1, 1)
        self.set(1, cfg.ADDR_MOVING_STATUS, 1, cfg.MOVING_STATUS_IN_POSITION | cfg.MOVING_STATUS_FOLLOWING_ERROR)
        self.set(1, cfg.ADDR_PRESENT_PWM, 2, -300)
        self.set(1, cfg.ADDR_PRESENT_CURRENT, 2, -20)
        self.set(1, cfg.ADDR_PRESENT_VELOCITY, 4, -5)
        self.bus.set_position(1, 2000)
        self.set(1, cfg.ADDR_VELOCITY_TRAJECTORY, 4, 7)
        self.set(1, cfg.ADDR_POSITION_TRAJECTORY, 4, -2010)
        self.set(1, cfg.ADDR_PRESENT_INPUT_VOLTAGE, 2, 118)
        self.set(1, cfg.ADDR_PRESENT_TEMPERATURE, 1, 33)
        states = self.handler.read_state()
        self.assertEqual(ServoState(tick=1234, moving=1, moving_status=9, pwm=-300, current=-20, velocity=-5,
                                    position=2000, velocity_trajectory=7, position_trajectory=-2010, voltage=11.8,
                                    temperature=33), states[1])
        self.assertEqual(2048, states[3].position)
        self.assertEqual([('sync_read', cfg.ADDR_REALTIME_TICK, [1, 2]), ('sync_read', cfg.ADDR_REALTIME_TICK, [3, 4])],
                         self.bus.packets)
        self.bus.packets.clear()
        self.assertEqual({3: states[3]}, self.handler.read_state([3]))
        self.assertEqual([('sync_read', cfg.ADDR_REALTIME_TICK, [3])], self.bus.packets)
        self.bus.unresponsive = {2}
        self.assertIsNone(self.handler.read_state())

    def test_pid_is_read_back(self):
        self.assertTrue(self.handler.set_pid_single_servo(3, 800, 0, 60))
        self.bus.packets.clear()
        self.assertEqual({3: (800, 0, 60)}, self.handler.read_pid([3]))
        self.assertEqual([('sync_read', cfg.ADDR_D, [3])], self.bus.packets)
        self.assertTrue(self.handler.set_pid())
        self.assertEqual({1: (1500, 20, 500), 2: (0x1234, 0x0102, 0x0A0B), 3: (1000, 600, 500), 4: (1000, 600, 500)},
                         self.handler.read_pid())
        self.bus.unresponsive = {4}
        self.assertIsNone(self.handler.read_pid())

    def test_gains_outside_the_firmware_range_are_refused(self):
        for p, i, d in ((cfg.MAX_GAIN + 1, 0, 0), (800, -1, 0), (800, 0, 70000)):
            with self.assertRaises(ValueError):
                self.handler.set_pid_single_servo(3, p, i, d)
        self.assertEqual([], self.bus.packets)
        self.assertTrue(self.handler.set_pid_single_servo(3, cfg.MAX_GAIN, 0, 0))

    def test_goal_pwm_is_written_signed_and_read_back(self):
        self.assertTrue(self.handler.set_goal_pwm(2, -400))
        self.assertEqual([('sync_write', cfg.ADDR_GOAL_PWM, [2])], self.bus.packets)
        self.assertEqual(-400, self.bus.get(2, cfg.ADDR_GOAL_PWM, 2, signed=True))
        self.assertEqual({2: -400}, self.handler.read_goal_pwm([2]))
        with self.assertRaises(ValueError):
            self.handler.set_goal_pwm(2, cfg.MAX_PWM + 1)
        self.bus.unresponsive = {2}
        self.assertIsNone(self.handler.read_goal_pwm([2]))

    def test_profile_of_one_servo_is_set_without_changing_the_configured_one(self):
        self.assertTrue(self.handler.set_profile(1, 0, 0))
        self.assertEqual([('sync_write', cfg.ADDR_PROFILE_ACCELERATION, [1])], self.bus.packets)
        self.assertEqual({1: (0, 0)}, self.handler.read_profile([1]))
        self.assertEqual((250, 100), (self.servos[0].profile_velocity, self.servos[0].profile_acceleration))
        self.assertTrue(self.handler.set_profile_velocity_and_acceleration([1]))
        self.assertEqual({1: (250, 100)}, self.handler.read_profile([1]))
        with self.assertRaises(ValueError):
            self.handler.set_profile(1, -1, 0)
        self.bus.unresponsive = {1}
        self.assertIsNone(self.handler.read_profile([1]))

    def test_torque_enabled_is_read(self):
        self.bus.memory[2][cfg.ADDR_TORQUE_ENABLE] = 1
        self.assertEqual({1: False, 2: True, 3: False, 4: False}, self.handler.read_torque_enabled())
        self.assertEqual([('sync_read', cfg.ADDR_TORQUE_ENABLE, [1, 2]), ('sync_read', cfg.ADDR_TORQUE_ENABLE, [3, 4])],
                         self.bus.packets)
        self.bus.unresponsive = {3}
        self.assertIsNone(self.handler.read_torque_enabled())

    def test_temperature_and_voltage_are_read_in_one_packet_per_bus(self):
        self.set(4, cfg.ADDR_PRESENT_TEMPERATURE, 1, 39)
        self.set(4, cfg.ADDR_PRESENT_INPUT_VOLTAGE, 2, 52)
        self.assertEqual({1: (0, 0.0), 2: (0, 0.0), 3: (0, 0.0), 4: (39, 5.2)},
                         self.handler.read_temperature_and_voltage())
        self.assertEqual([('sync_read', cfg.ADDR_PRESENT_INPUT_VOLTAGE, [1, 2]),
                          ('sync_read', cfg.ADDR_PRESENT_INPUT_VOLTAGE, [3, 4])], self.bus.packets)


class TestServoHandlerTimeBasedProfile(unittest.TestCase):
    """the drive mode, the profile registers, the feedforward and the goal writes of servos with a move time; servos 1
    and 3 are time-based, 2 and 4 velocity-based"""

    BUSES = ((1, 2), (3, 4))

    def setUp(self):
        self.servos = [
            Servo(0, 4095, 0, 2 * pi, POSITION_CONTROL_MODE, profile_velocity=250, profile_acceleration=100,
                  move_time=15, ff1=40),
            Servo(0, 4095, 0, 2 * pi, POSITION_CONTROL_MODE, profile_velocity=250, profile_acceleration=100,
                  ff1=cfg.MAX_GAIN),
            Servo(0, 4095, 0, 2 * pi, POSITION_CONTROL_MODE, profile_velocity=500, profile_acceleration=50,
                  move_time=20),
            Servo(0, 4095, 0, 2 * pi, POSITION_CONTROL_MODE, profile_velocity=300, profile_acceleration=70),
        ]
        self.bus = FakeBus(self.servos, self.BUSES)
        self.handler = ServoHandler({servo_id: servo for servo_id, servo in enumerate(self.servos, start=1)},
                                    cfg, mock.Mock(), self.bus, self.BUSES)

    def raw(self, servo_id, address, length):
        return list(self.bus.memory[servo_id][address:address + length])

    def goal_packet(self, servo_id):
        """the (acceleration, velocity, goal) registers of the servo"""
        return (self.bus.get(servo_id, cfg.ADDR_PROFILE_ACCELERATION, 4),
                self.bus.get(servo_id, cfg.ADDR_PROFILE_VELOCITY, 4),
                self.bus.get(servo_id, cfg.ADDR_GOAL_POSITION, 4, signed=True))

    @staticmethod
    def packet_bytes(acceleration, time_ms, goal):
        return [b for value in (acceleration, time_ms, goal) for b in (value & 0xFFFFFFFF).to_bytes(4, 'little')]

    # construction

    def test_the_goal_and_feedforward_registers_have_to_be_adjacent_too(self):
        config_values = {name: getattr(cfg, name) for name in dir(cfg) if name.isupper()}
        for changes in ({'ADDR_GOAL_POSITION': cfg.ADDR_GOAL_POSITION + 4},
                        {'ADDR_FEEDFORWARD_1ST_GAIN': cfg.ADDR_FEEDFORWARD_1ST_GAIN + 2},
                        {'ADDR_FEEDFORWARD_2ND_GAIN': cfg.ADDR_FEEDFORWARD_2ND_GAIN - 2}):
            config = types.SimpleNamespace(**dict(config_values, **changes))
            with self.assertRaises(ValueError):
                ServoHandler({1: self.servos[0]}, config, None, None)

    def test_a_new_handler_has_no_last_goals_and_the_full_speed(self):
        self.assertEqual(({}, 1.0), (self.handler.last_goal, self.handler.speed_factor))

    # drive mode

    def test_drive_mode_is_read(self):
        self.bus.memory[3][cfg.ADDR_DRIVE_MODE] = 5
        self.assertEqual({1: 0, 2: 0, 3: 5, 4: 0}, self.handler.read_drive_mode())
        self.assertEqual({3: 5}, self.handler.read_drive_mode([3]))
        self.bus.unresponsive = {4}
        self.assertIsNone(self.handler.read_drive_mode())

    def test_drive_mode_is_not_written_when_it_already_matches(self):
        for servo_id, mode in ((1, 0b0101), (2, 0b0001), (3, cfg.DRIVE_MODE_TIME_BASED_PROFILE), (4, 0)):
            self.bus.memory[servo_id][cfg.ADDR_DRIVE_MODE] = mode
            self.bus.memory[servo_id][cfg.ADDR_TORQUE_ENABLE] = 1  # the servos are holding their position
        with mock.patch.object(self.bus, 'write1ByteTxRx', wraps=self.bus.write1ByteTxRx) as write:
            self.assertTrue(self.handler.set_configured_drive_mode())
        write.assert_not_called()

    def test_the_time_based_bit_is_set_and_cleared_and_the_other_bits_are_kept(self):
        self.bus.memory[1][cfg.ADDR_DRIVE_MODE] = 0b0001
        self.bus.memory[2][cfg.ADDR_DRIVE_MODE] = 0b0101
        self.bus.memory[3][cfg.ADDR_DRIVE_MODE] = 0
        with self.assertLogs(servo_handler_module.logger, 'INFO') as logs:
            self.assertTrue(self.handler.set_configured_drive_mode())
        self.assertEqual([0b0101, 0b0001, cfg.DRIVE_MODE_TIME_BASED_PROFILE, 0],
                         [self.bus.memory[servo_id][cfg.ADDR_DRIVE_MODE] for servo_id in range(1, 5)])
        self.assertEqual(['INFO:' + servo_handler_module.logger.name + ':the drive mode of servo %d changed from %d '
                          'to %d' % change for change in ((1, 1, 5), (2, 5, 1), (3, 0, 4))], logs.output)

    def test_drive_mode_of_some_servos(self):
        self.assertTrue(self.handler.set_configured_drive_mode([3]))
        self.assertEqual([0, 0, 4, 0], [self.bus.memory[servo_id][cfg.ADDR_DRIVE_MODE] for servo_id in range(1, 5)])

    def test_drive_mode_is_not_changed_with_the_torque_on(self):
        self.bus.memory[1][cfg.ADDR_TORQUE_ENABLE] = 1
        with mock.patch.object(self.bus, 'write1ByteTxRx', wraps=self.bus.write1ByteTxRx) as write, \
                self.assertLogs(servo_handler_module.logger, 'ERROR') as logs:
            self.assertFalse(self.handler.set_configured_drive_mode([1]))
        write.assert_not_called()
        self.assertEqual(0, self.bus.memory[1][cfg.ADDR_DRIVE_MODE])
        self.assertIn('servo 1 has drive mode 0 and should have 4', logs.output[0])
        self.assertIn('torque of this servo has to be off', logs.output[0])

    def test_drive_mode_is_not_changed_when_the_torque_can_not_be_read(self):
        read = self.bus.read1ByteTxRx

        def torque_unreadable(port, servo_id, address):
            return (0, dxl.COMM_RX_TIMEOUT, 0) if address == cfg.ADDR_TORQUE_ENABLE else read(port, servo_id, address)

        with mock.patch.object(self.bus, 'read1ByteTxRx', torque_unreadable), \
                mock.patch.object(self.bus, 'write1ByteTxRx') as write, \
                self.assertLogs(servo_handler_module.logger, 'ERROR') as logs:
            self.assertFalse(self.handler.set_configured_drive_mode([1]))
        write.assert_not_called()
        self.assertIn("can't be read", logs.output[-1])
        self.assertEqual(0, self.bus.memory[1][cfg.ADDR_DRIVE_MODE])

    def test_drive_mode_fails_when_the_servo_does_not_answer(self):
        self.bus.unresponsive = {1}
        self.assertFalse(self.handler.set_configured_drive_mode([1]))

    def test_drive_mode_fails_when_the_read_back_differs(self):
        write = self.bus.write1ByteTxRx

        def dropped(port, servo_id, address, value):
            return (dxl.COMM_SUCCESS, 0) if address == cfg.ADDR_DRIVE_MODE else write(port, servo_id, address, value)

        with mock.patch.object(self.bus, 'write1ByteTxRx', dropped), \
                self.assertLogs(servo_handler_module.logger, 'ERROR') as logs:
            self.assertFalse(self.handler.set_configured_drive_mode([1]))
        self.assertIn('servo 1 did not take the drive mode 4', logs.output[0])

    def test_a_failing_servo_does_not_stop_the_others(self):
        self.bus.memory[1][cfg.ADDR_TORQUE_ENABLE] = 1  # servo 1 can't be changed
        self.bus.memory[2][cfg.ADDR_DRIVE_MODE] = 0b0100  # servo 2 and 3 can
        with self.assertLogs(servo_handler_module.logger, 'INFO'):
            self.assertFalse(self.handler.set_configured_drive_mode())
        self.assertEqual([0, 0, 4, 0], [self.bus.memory[servo_id][cfg.ADDR_DRIVE_MODE] for servo_id in range(1, 5)])

    # profile registers

    def test_a_time_based_servo_gets_its_move_time_and_no_acceleration_time(self):
        self.assertTrue(self.handler.set_profile_velocity_and_acceleration())
        self.assertEqual([('sync_write', cfg.ADDR_PROFILE_ACCELERATION, [1, 2, 3, 4])], self.bus.packets)
        self.assertEqual([(0, 15), (100, 250), (0, 20), (70, 300)],
                         [(self.bus.get(servo_id, cfg.ADDR_PROFILE_ACCELERATION, 4),
                           self.bus.get(servo_id, cfg.ADDR_PROFILE_VELOCITY, 4)) for servo_id in range(1, 5)])
        # the configured profile of the servo is kept for the goals that are far away
        self.assertEqual((250, 100), (self.servos[0].profile_velocity, self.servos[0].profile_acceleration))
        self.assertEqual({1: (15, 0), 3: (20, 0)}, self.handler.read_profile([1, 3]))

    def test_the_raw_profile_is_written_as_it_is_to_a_time_based_servo(self):
        self.assertTrue(self.handler.set_profile(1, 33, 12))
        self.assertEqual({1: (33, 12)}, self.handler.read_profile([1]))

    # speed percentage

    def test_the_speed_percentage_writes_the_velocity_of_the_velocity_based_servos_only(self):
        self.handler.set_profile_velocity_and_acceleration()
        self.bus.packets.clear()
        self.assertTrue(self.handler.set_profile_velocity_percentage(50))
        self.assertEqual([('sync_write', cfg.ADDR_PROFILE_VELOCITY, [2, 4])], self.bus.packets)
        self.assertEqual([15, 125, 20, 150], [self.bus.get(servo_id, cfg.ADDR_PROFILE_VELOCITY, 4)
                                              for servo_id in range(1, 5)])
        self.assertEqual(0.5, self.handler.speed_factor)

    def test_the_speed_percentage_of_time_based_servos_only_is_not_written(self):
        self.assertTrue(self.handler.set_profile_velocity_percentage(25, [1, 3]))
        self.assertEqual([], self.bus.packets)
        self.assertEqual(0.25, self.handler.speed_factor)
        # velocity-based servos only: the factor of the time-based ones stays
        self.assertTrue(self.handler.set_profile_velocity_percentage(80, [2]))
        self.assertEqual(0.25, self.handler.speed_factor)

    def test_the_speed_percentage_is_still_checked(self):
        for percentage in (0, -10, 100.1):
            with self.assertRaises(ValueError):
                self.handler.set_profile_velocity_percentage(percentage, [1])
        self.assertEqual((1.0, []), (self.handler.speed_factor, self.bus.packets))

    def test_the_speed_percentage_changes_the_time_of_a_far_goal(self):
        servo = self.servos[0]
        self.handler.move_servo_to_pos(1, 1000)
        self.handler.set_profile_velocity_percentage(50, [1])
        self.handler.move_servo_to_pos(1, 2000)
        self.assertEqual(servo.goal_timing(1000, 0.5)[0], self.bus.get(1, cfg.ADDR_PROFILE_VELOCITY, 4))
        self.handler.set_profile_velocity_percentage(100, [1])
        self.handler.move_servo_to_pos(1, 3000)
        self.assertEqual(servo.goal_timing(1000)[0], self.bus.get(1, cfg.ADDR_PROFILE_VELOCITY, 4))
        self.assertLess(servo.goal_timing(1000)[0], servo.goal_timing(1000, 0.5)[0])

    # feedforward

    def test_feedforward_is_written_in_one_packet_as_second_and_first_gain(self):
        self.assertTrue(self.handler.set_feedforward())
        self.assertEqual([('sync_write', cfg.ADDR_FEEDFORWARD_2ND_GAIN, [1, 2, 3, 4])], self.bus.packets)
        # little endian, the 2nd gain 0 and then the 1st
        self.assertEqual([0, 0, 40, 0], self.raw(1, cfg.ADDR_FEEDFORWARD_2ND_GAIN, 4))
        self.assertEqual([0, 0, 0xFF, 0x3F], self.raw(2, cfg.ADDR_FEEDFORWARD_2ND_GAIN, 4))
        self.assertEqual([0, 0, 0, 0], self.raw(3, cfg.ADDR_FEEDFORWARD_2ND_GAIN, 4))
        self.assertEqual({1: (40, 0), 2: (cfg.MAX_GAIN, 0), 3: (0, 0), 4: (0, 0)}, self.handler.read_feedforward())

    def test_feedforward_of_some_servos(self):
        self.assertTrue(self.handler.set_feedforward([1]))
        self.assertEqual([('sync_write', cfg.ADDR_FEEDFORWARD_2ND_GAIN, [1])], self.bus.packets)

    def test_single_servo_feedforward(self):
        self.assertTrue(self.handler.set_feedforward_single_servo(3, 100))
        self.assertEqual([('sync_write', cfg.ADDR_FEEDFORWARD_2ND_GAIN, [3])], self.bus.packets)
        self.assertEqual([0, 0, 100, 0], self.raw(3, cfg.ADDR_FEEDFORWARD_2ND_GAIN, 4))
        self.assertTrue(self.handler.set_feedforward_single_servo(3, 100, 7))
        self.assertEqual([7, 0, 100, 0], self.raw(3, cfg.ADDR_FEEDFORWARD_2ND_GAIN, 4))
        self.assertEqual({3: (100, 7)}, self.handler.read_feedforward([3]))
        self.assertEqual(0, self.servos[2].ff1, 'the configured value stays')
        self.assertTrue(self.handler.set_feedforward([3]))
        self.assertEqual({3: (0, 0)}, self.handler.read_feedforward([3]))

    def test_feedforward_outside_the_firmware_range_is_refused(self):
        for first, second in ((-1, 0), (cfg.MAX_GAIN + 1, 0), (0, -1), (0, cfg.MAX_GAIN + 1)):
            with self.assertRaises(ValueError):
                self.handler.set_feedforward_single_servo(3, first, second)
        self.assertEqual([], self.bus.packets)
        self.assertTrue(self.handler.set_feedforward_single_servo(3, cfg.MAX_GAIN, cfg.MAX_GAIN))

    def test_feedforward_read_fails_when_a_servo_does_not_answer(self):
        self.bus.unresponsive = {2}
        self.assertIsNone(self.handler.read_feedforward())

    # goals

    def test_a_velocity_based_servo_gets_the_goal_only(self):
        self.servos[1].target_position = 0x0204
        self.assertTrue(self.handler.move_to_angles([2]))
        self.assertEqual([('sync_write', cfg.ADDR_GOAL_POSITION, [2])], self.bus.packets)
        self.assertEqual([0x04, 0x02, 0, 0], self.raw(2, cfg.ADDR_GOAL_POSITION, 4))
        self.assertEqual([0] * 8, self.raw(2, cfg.ADDR_PROFILE_ACCELERATION, 8), 'the profile is left alone')
        self.assertEqual({2: 0x204}, self.handler.last_goal)
        self.assertTrue(self.handler.move_servo_to_pos(4, 1500))
        self.assertEqual([0xDC, 0x05, 0, 0], self.raw(4, cfg.ADDR_GOAL_POSITION, 4))
        self.assertEqual([0] * 8, self.raw(4, cfg.ADDR_PROFILE_ACCELERATION, 8))

    def test_the_first_goal_of_a_time_based_servo_counts_as_a_full_turn_away(self):
        self.servos[0].target_position = 2000
        self.assertTrue(self.handler.move_to_angles([1]))
        self.assertEqual([('sync_write', cfg.ADDR_PROFILE_ACCELERATION, [1])], self.bus.packets)
        time_ms, acceleration_ms = self.servos[0].goal_timing(4096)
        self.assertGreater(time_ms, self.servos[0].move_time)
        self.assertEqual(self.packet_bytes(acceleration_ms, time_ms, 2000),
                         self.raw(1, cfg.ADDR_PROFILE_ACCELERATION, 12))
        self.assertEqual({1: 2000}, self.handler.last_goal)

    def test_a_small_step_takes_the_move_time_and_a_far_goal_longer(self):
        servo = self.servos[0]
        self.handler.move_servo_to_pos(1, 2000)
        self.bus.packets.clear()
        servo.target_position = 2020
        self.assertTrue(self.handler.move_to_angles([1]))
        self.assertEqual((0, 15, 2020), self.goal_packet(1))
        self.assertEqual([('sync_write', cfg.ADDR_PROFILE_ACCELERATION, [1])], self.bus.packets)

        servo.target_position = 1000  # 1020 positions back
        self.assertTrue(self.handler.move_to_angles([1]))
        time_ms, acceleration_ms = servo.goal_timing(1020)
        self.assertGreater(time_ms, 15)
        self.assertEqual((acceleration_ms, time_ms, 1000), self.goal_packet(1))
        self.assertEqual({1: 1000}, self.handler.last_goal)

    def test_the_distance_is_to_the_previous_goal_and_not_to_the_servo(self):
        self.handler.move_servo_to_pos(1, 2000)
        self.bus.set_position(1, 0)  # far from where the servo is, which doesn't matter
        self.servos[0].target_position = 2010
        self.handler.move_to_angles([1])
        self.assertEqual((0, 15, 2010), self.goal_packet(1))

    def test_a_goal_that_was_not_sent_is_not_the_last_goal(self):
        self.handler.move_servo_to_pos(1, 2000)
        with mock.patch.object(dxl.GroupSyncWrite, 'txPacket', return_value=dxl.COMM_TX_FAIL), \
                self.assertLogs(servo_handler_module.logger, 'WARNING'):
            self.assertFalse(self.handler.move_servo_to_pos(1, 3000))
            self.assertFalse(self.handler.move_servo_to_pos(2, 3000))
        self.assertEqual({1: 2000}, self.handler.last_goal)

    def test_time_based_goals_are_clamped_like_the_others(self):
        self.servos[0].target_position = 5000
        with self.assertLogs(servo_handler_module.logger, 'WARNING'):
            self.handler.move_to_angles([1])
        self.assertEqual(cfg.MAX_POSITION, self.goal_packet(1)[2])
        self.assertEqual({1: cfg.MAX_POSITION}, self.handler.last_goal)

    def test_a_time_based_raw_position_is_not_clamped(self):
        self.assertTrue(self.handler.move_servo_to_pos(1, -50))
        self.assertEqual(-50, self.goal_packet(1)[2])
        self.assertEqual({1: -50}, self.handler.last_goal)

    def test_servos_of_both_kinds_are_written_with_two_packets_and_the_time_based_ones_first(self):
        for servo, target in zip(self.servos, (1000, 1100, 1200, 1300)):
            servo.target_position = target
        self.assertTrue(self.handler.move_to_angles())
        self.assertEqual([('sync_write', cfg.ADDR_PROFILE_ACCELERATION, [1, 3]),
                          ('sync_write', cfg.ADDR_GOAL_POSITION, [2, 4])], self.bus.packets)
        self.assertEqual([1000, 1100, 1200, 1300], [self.bus.get(servo_id, cfg.ADDR_GOAL_POSITION, 4)
                                                    for servo_id in range(1, 5)])
        self.assertEqual({1: 1000, 2: 1100, 3: 1200, 4: 1300}, self.handler.last_goal)
        self.assertEqual(self.servos[2].goal_timing(4096)[0], self.bus.get(3, cfg.ADDR_PROFILE_VELOCITY, 4))

    def test_a_mixed_call_is_only_true_when_everything_was_sent(self):
        results = [dxl.COMM_SUCCESS, dxl.COMM_TX_FAIL]  # the time-based packet goes first
        with mock.patch.object(dxl.GroupSyncWrite, 'txPacket', side_effect=results), \
                self.assertLogs(servo_handler_module.logger, 'WARNING'):
            self.assertFalse(self.handler.move_to_angles([1, 2]))
        self.assertEqual({1: 0}, self.handler.last_goal)

    def test_hold_current_position_sends_the_move_time_and_no_acceleration_time(self):
        for servo_id, position in ((1, 1500), (2, 2500), (3, 3000), (4, 1100)):
            self.bus.set_position(servo_id, position)
        self.handler.move_servo_to_pos(1, 100)  # the servo is far from its last goal
        self.bus.packets.clear()
        self.assertTrue(self.handler.hold_current_position())
        self.assertEqual([('sync_read', cfg.ADDR_PRESENT_POSITION, [1, 2]),
                          ('sync_read', cfg.ADDR_PRESENT_POSITION, [3, 4]),
                          ('sync_write', cfg.ADDR_PROFILE_ACCELERATION, [1, 3]),
                          ('sync_write', cfg.ADDR_GOAL_POSITION, [2, 4])], self.bus.packets)
        self.assertEqual((0, 15, 1500), self.goal_packet(1))
        self.assertEqual((0, 20, 3000), self.goal_packet(3))
        self.assertEqual([2500, 1100], [self.bus.get(servo_id, cfg.ADDR_GOAL_POSITION, 4) for servo_id in (2, 4)])
        self.assertEqual([1500, 2500, 3000, 1100], [servo.target_position for servo in self.servos])
        self.assertEqual({1: 1500, 2: 2500, 3: 3000, 4: 1100}, self.handler.last_goal)

    def test_the_step_of_the_gain_tuning_has_no_profile(self):
        self.handler.set_profile_velocity_and_acceleration()
        self.bus.packets.clear()
        self.assertTrue(self.handler.step_servo_to_pos(1, 3000))
        self.assertEqual([('sync_write', cfg.ADDR_PROFILE_ACCELERATION, [1])], self.bus.packets)
        self.assertEqual(self.packet_bytes(0, 0, 3000), self.raw(1, cfg.ADDR_PROFILE_ACCELERATION, 12))
        self.assertEqual({1: 3000}, self.handler.last_goal)
        # the same for a velocity-based servo, and not clamped
        self.assertTrue(self.handler.step_servo_to_pos(2, -100))
        self.assertEqual((0, 0, -100), self.goal_packet(2))
        self.assertEqual({1: 3000, 2: -100}, self.handler.last_goal)
        # the profile registers stay 0 until they are restored
        self.assertEqual({1: (0, 0), 2: (0, 0)}, self.handler.read_profile([1, 2]))
        self.assertTrue(self.handler.set_profile_velocity_and_acceleration([1, 2]))
        self.assertEqual({1: (15, 0), 2: (250, 100)}, self.handler.read_profile([1, 2]))

    def test_the_goal_after_a_step_is_timed_from_the_step(self):
        self.handler.step_servo_to_pos(1, 3000)
        self.servos[0].target_position = 3010
        self.handler.move_to_angles([1])
        self.assertEqual((0, 15, 3010), self.goal_packet(1))


if __name__ == '__main__':
    unittest.main()
