import types
import unittest
from unittest import mock

import dynamixel_sdk as dxl
import serial

from src.robot_controllers.dynamixel_robot import dynamixel_utils
from src.robot_controllers.dynamixel_robot import dynamixel_x_config as cfg
from src.robot_controllers.dynamixel_robot.dynamixel_utils import to_bytes, to_signed, DynamixelError

U2D2 = (0x0403, 0x6014)


class TestConversions(unittest.TestCase):

    def test_to_bytes_is_little_endian(self):
        self.assertEqual([0x04, 0x03, 0x02, 0x01], to_bytes(0x01020304, 4))
        self.assertEqual([0x34, 0x12], to_bytes(0x1234, 2))
        self.assertEqual([0x07], to_bytes(7, 1))

    def test_to_bytes_of_negative_values_is_twos_complement(self):
        self.assertEqual([0xFF] * 4, to_bytes(-1, 4))
        self.assertEqual([0xFF, 0xFF], to_bytes(-1, 2))
        self.assertEqual([0x00, 0x80], to_bytes(-32768, 2))
        self.assertEqual([0x00, 0x00, 0x00, 0x80], to_bytes(-2 ** 31, 4))
        self.assertEqual([0xEC, 0xFF, 0xFF, 0xFF], to_bytes(-20, 4))

    def test_to_bytes_accepts_numpy_and_float_values(self):
        import numpy as np
        self.assertEqual([0x00, 0x08, 0, 0], to_bytes(np.int64(2048), 4))
        self.assertEqual([2, 0, 0, 0], to_bytes(2.7, 4))

    def test_to_signed(self):
        self.assertEqual(0, to_signed(0, 4))
        self.assertEqual(2 ** 31 - 1, to_signed(0x7FFFFFFF, 4))
        self.assertEqual(-2 ** 31, to_signed(0x80000000, 4))
        self.assertEqual(-1, to_signed(0xFFFFFFFF, 4))
        self.assertEqual(32767, to_signed(0x7FFF, 2))
        self.assertEqual(-32768, to_signed(0x8000, 2))
        self.assertEqual(-1, to_signed(0xFFFF, 2))
        self.assertEqual(-1, to_signed(0xFF, 1))
        self.assertEqual(127, to_signed(0x7F, 1))

    def test_round_trip(self):
        for length in (1, 2, 4):
            bits = 8 * length
            for value in (-(1 << (bits - 1)), -1, 0, 1, (1 << (bits - 1)) - 1):
                self.assertEqual(value, to_signed(int.from_bytes(bytes(to_bytes(value, length)), 'little'), length))


class FakePacketHandler:

    def __init__(self, comm_result=dxl.COMM_SUCCESS, error=0):
        self.comm_result = comm_result
        self.error = error
        self.writes = []

    def getTxRxResult(self, result):
        return 'comm result %d' % result

    def getRxPacketError(self, error):
        return 'packet error %d' % error

    def write1ByteTxRx(self, port, servo_id, address, value):
        self.writes.append((servo_id, address, value))
        return self.comm_result, self.error

    def read1ByteTxRx(self, port, servo_id, address):
        return 3, self.comm_result, self.error


class TestSingleServoAccess(unittest.TestCase):

    def test_set_torque_writes_the_torque_register(self):
        packet_handler = FakePacketHandler()
        self.assertTrue(dynamixel_utils.set_torque(packet_handler, None, cfg, 5, True))
        self.assertTrue(dynamixel_utils.set_torque(packet_handler, None, cfg, 5, False))
        self.assertEqual([(5, cfg.ADDR_TORQUE_ENABLE, cfg.TORQUE_ENABLE), (5, cfg.ADDR_TORQUE_ENABLE, cfg.TORQUE_DISABLE)],
                         packet_handler.writes)

    def test_communication_and_servo_errors_are_failures(self):
        for packet_handler in (FakePacketHandler(comm_result=dxl.COMM_RX_TIMEOUT), FakePacketHandler(error=0x07)):
            with self.assertLogs(dynamixel_utils.logger, 'WARNING') as logs:
                self.assertFalse(dynamixel_utils.set_torque(packet_handler, None, cfg, 5, True))
                self.assertFalse(dynamixel_utils.write_1_byte(packet_handler, None, cfg.ADDR_OPERATING_MODE, 5, 3))
                self.assertIsNone(dynamixel_utils.read_1_byte(packet_handler, None, cfg.ADDR_OPERATING_MODE, 5))
            self.assertTrue(all('[ID:005]' in line for line in logs.output))

    def test_read_1_byte(self):
        self.assertEqual(3, dynamixel_utils.read_1_byte(FakePacketHandler(), None, cfg.ADDR_OPERATING_MODE, 5))


def comport(device, vid=None, pid=None):
    return types.SimpleNamespace(device=device, vid=vid, pid=pid)


class TestFindDynamixelPort(unittest.TestCase):

    def patch_comports(self, ports):
        patcher = mock.patch.object(dynamixel_utils.serial.tools.list_ports, 'comports', return_value=ports)
        patcher.start()
        self.addCleanup(patcher.stop)

    def patch_ping(self, **kwargs):
        patcher = mock.patch.object(dynamixel_utils, 'ping_servos', **kwargs)
        ping = patcher.start()
        self.addCleanup(patcher.stop)
        return ping

    def test_only_dynamixel_adapters_are_listed(self):
        # other serial ports are never opened, opening the port of an arduino resets it
        self.patch_comports([comport('FAKE3', 0x2341, 0x0043), comport('FAKE9', *U2D2), comport('FAKE1'),
                             comport('FAKE2', *U2D2)])
        self.assertEqual([('FAKE2', 'U2D2'), ('FAKE9', 'U2D2')], dynamixel_utils.list_dynamixel_adapters())

    def test_no_adapter_raises_without_opening_any_port(self):
        self.patch_comports([comport('FAKE3', 0x2341, 0x0043)])
        ping = self.patch_ping()
        with self.assertRaisesRegex(DynamixelError, 'no dynamixel usb adapter found.*FAKE3'):
            dynamixel_utils.find_dynamixel_port(cfg, [1, 2])
        ping.assert_not_called()

    def test_the_adapter_on_which_the_servos_respond_is_used(self):
        self.patch_comports([comport('FAKE1', *U2D2), comport('FAKE2', *U2D2)])
        ping = self.patch_ping(side_effect=lambda port, config, servo_ids: {1: 1020} if port == 'FAKE2' else {})
        self.assertEqual('FAKE2', dynamixel_utils.find_dynamixel_port(cfg, [1, 2]))
        self.assertEqual(['FAKE1', 'FAKE2'], [call.args[0] for call in ping.call_args_list])

    def test_adapter_in_use_is_skipped(self):
        self.patch_comports([comport('FAKE1', *U2D2), comport('FAKE2', *U2D2)])

        def ping(port, config, servo_ids):
            if port == 'FAKE1':
                raise serial.SerialException('access denied')
            return {1: 1020}

        self.patch_ping(side_effect=ping)
        self.assertEqual('FAKE2', dynamixel_utils.find_dynamixel_port(cfg, [1]))

    def test_raises_when_no_servo_responds(self):
        self.patch_comports([comport('FAKE1', *U2D2)])
        self.patch_ping(return_value={})
        with self.assertRaisesRegex(DynamixelError, 'none of the servos'):
            dynamixel_utils.find_dynamixel_port(cfg, [1])

    def test_setup_finds_the_port_when_none_is_given(self):
        port_handler = mock.Mock()
        with mock.patch.object(dynamixel_utils, 'find_dynamixel_port', return_value='FAKE2') as find, \
                mock.patch.object(dynamixel_utils, 'get_port_handler', return_value=port_handler) as get_port:
            handlers = dynamixel_utils.setup_dynamixel_handlers(None, cfg, [1, 2])
        find.assert_called_once_with(cfg, [1, 2])
        get_port.assert_called_once_with('FAKE2', cfg.BAUDRATE)
        self.assertIs(port_handler, handlers[0])
        self.assertEqual(cfg.PROTOCOL_VERSION, handlers[1].getProtocolVersion())

    def test_setup_with_a_port_does_not_search(self):
        with mock.patch.object(dynamixel_utils, 'find_dynamixel_port') as find, \
                mock.patch.object(dynamixel_utils, 'get_port_handler', return_value=mock.Mock()) as get_port:
            dynamixel_utils.setup_dynamixel_handlers('FAKE7', cfg, [1])
        find.assert_not_called()
        get_port.assert_called_once_with('FAKE7', cfg.BAUDRATE)


class TestPingServos(unittest.TestCase):

    def ping_servos(self, ping):
        port_handler = mock.Mock()
        packet_handler = mock.Mock(ping=mock.Mock(side_effect=ping))
        with mock.patch.object(dynamixel_utils, 'get_port_handler', return_value=port_handler), \
                mock.patch.object(dynamixel_utils.dynamixel, 'PacketHandler', return_value=packet_handler):
            try:
                return dynamixel_utils.ping_servos('FAKE1', cfg, [1, 2, 3])
            finally:
                self.port_handler = port_handler

    def test_only_responding_servos_without_error_are_returned(self):
        results = {1: (1020, dxl.COMM_SUCCESS, 0), 2: (0, dxl.COMM_RX_TIMEOUT, 0), 3: (1200, dxl.COMM_SUCCESS, 0x80)}
        self.assertEqual({1: 1020}, self.ping_servos(lambda port, servo_id: results[servo_id]))
        self.port_handler.closePort.assert_called_once_with()

    def test_port_is_closed_when_the_ping_fails(self):
        def ping(port, servo_id):
            raise serial.SerialException('unplugged')

        with self.assertRaises(serial.SerialException):
            self.ping_servos(ping)
        self.port_handler.closePort.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
