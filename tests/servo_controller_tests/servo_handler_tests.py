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
from src.robot_controllers.dynamixel_robot.servo_handler import ServoHandler
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


if __name__ == '__main__':
    unittest.main()
