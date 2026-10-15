import os
import sys
import tempfile
import threading
import time
import types
import unittest
from contextlib import contextmanager
from unittest import mock

import dynamixel_sdk as dxl
import serial

from src.robot_controllers.dynamixel_robot import dynamixel_utils
from src.robot_controllers.dynamixel_robot.dynamixel_utils import BlockingPortHandler

SERVO_ID = 1
MODEL_NUMBER = 1020


@contextmanager
def loopback_serial():
    """let the sdk open pyserial's loopback port, everything written to it can be read back"""
    with mock.patch.object(dxl.port_handler.serial, 'Serial', lambda port, **kwargs: serial.serial_for_url(port, **kwargs)):
        yield


class CountingPortHandler(BlockingPortHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reads = 0

    def readPort(self, length):
        self.reads += 1
        return super().readPort(length)


def ping_status_packet(packet_handler):
    packet = [0xFF, 0xFF, 0xFD, 0x00, SERVO_ID, 0x07, 0x00, 0x55, 0x00,
              MODEL_NUMBER & 0xFF, MODEL_NUMBER >> 8, 45, 0, 0]
    crc = packet_handler.updateCRC(0, packet, len(packet) - 2)
    packet[-2:] = [crc & 0xFF, crc >> 8]
    return bytes(packet)


class TestBlockingPortHandler(unittest.TestCase):

    def open_port(self, latency_timer):
        with loopback_serial():
            port_handler = CountingPortHandler('loop://', latency_timer)
            self.assertTrue(port_handler.openPort())
        self.addCleanup(port_handler.closePort)
        return port_handler

    def test_read_timeout_is_set_when_opening_the_port(self):
        port_handler = self.open_port(latency_timer=1)
        self.assertEqual(BlockingPortHandler.READ_TIMEOUT, port_handler.ser.timeout)

    def test_missing_response_times_out_without_busy_waiting(self):
        port_handler = self.open_port(latency_timer=1)
        packet_handler = dxl.PacketHandler(2.0)

        start = time.monotonic()
        _, result, _ = packet_handler.ping(port_handler, SERVO_ID)
        elapsed = time.monotonic() - start

        # the loopback echoes the ping packet, so the sdk reports the leftover bytes as a corrupt packet
        self.assertIn(result, (dxl.COMM_RX_TIMEOUT, dxl.COMM_RX_CORRUPT))
        self.assertLess(elapsed, 0.1)
        # the echo of the ping packet is discarded byte by byte, after that every read blocks for READ_TIMEOUT.
        # the sdk port handler polls thousands of times in the same time
        self.assertLess(port_handler.reads, 100)

    def test_response_is_received(self):
        port_handler = self.open_port(latency_timer=50)
        packet_handler = dxl.PacketHandler(2.0)

        def respond():
            time.sleep(0.01)
            port_handler.ser.write(ping_status_packet(packet_handler))

        responder = threading.Thread(target=respond)
        responder.start()
        model_number, result, error = packet_handler.ping(port_handler, SERVO_ID)
        responder.join()

        self.assertEqual((MODEL_NUMBER, dxl.COMM_SUCCESS, 0), (model_number, result, error))
        self.assertLess(port_handler.reads, 100)

    def test_packet_timeout_uses_the_latency_timer(self):
        port_handler = BlockingPortHandler('COM1', latency_timer=1)
        port_handler.tx_time_per_byte = 0.01
        port_handler.setPacketTimeout(100)
        self.assertAlmostEqual(BlockingPortHandler.MIN_PACKET_TIMEOUT, port_handler.packet_timeout)

        port_handler.latency_timer = 16
        port_handler.setPacketTimeout(100)
        self.assertAlmostEqual(0.01 * 100 + 16 * 2.0 + 2.0, port_handler.packet_timeout)

    def test_packet_timeout_clock_has_sub_millisecond_resolution(self):
        # time.monotonic only has a resolution of 15.6ms on windows, which made packets time out too early
        port_handler = BlockingPortHandler('COM1', latency_timer=1)
        start = port_handler.getCurrentTime()
        while port_handler.getCurrentTime() == start:
            pass
        self.assertLess(port_handler.getCurrentTime() - start, 1.0)

    def test_packet_timeout_ignores_wall_clock_changes(self):
        port_handler = BlockingPortHandler('COM1', latency_timer=1)
        port_handler.setPacketTimeoutMillis(1000)
        with mock.patch('time.time', return_value=0.0):
            self.assertFalse(port_handler.isPacketTimeout())


class TestLatencyTimer(unittest.TestCase):

    def test_linux_latency_timer(self):
        with tempfile.TemporaryDirectory() as sysfs_root:
            os.mkdir(os.path.join(sysfs_root, 'ttyUSB0'))
            with open(os.path.join(sysfs_root, 'ttyUSB0', 'latency_timer'), 'w') as latency_file:
                latency_file.write('16\n')
            self.assertEqual(16, dynamixel_utils._get_linux_latency_timer('/dev/ttyUSB0', sysfs_root))

    def test_windows_latency_timer(self):
        registry = {
            'VID_0403+PID_6014+FT1': {'0000': {'Device Parameters': {'PortName': 'COM3', 'LatencyTimer': 16}}},
            'VID_0403+PID_6014+FT2': {'0000': {'Device Parameters': {'PortName': 'COM9', 'LatencyTimer': 1}}},
        }
        with mock.patch.dict(sys.modules, {'winreg': fake_winreg(registry)}):
            self.assertEqual(1, dynamixel_utils._get_windows_latency_timer('com9'))
            self.assertEqual(16, dynamixel_utils._get_windows_latency_timer('COM3'))
            self.assertIsNone(dynamixel_utils._get_windows_latency_timer('COM4'))

    def test_unknown_latency_timer_falls_back_to_the_sdk_default(self):
        with mock.patch.object(dynamixel_utils, 'get_latency_timer', return_value=None), \
                mock.patch.object(BlockingPortHandler, 'openPort', return_value=True), \
                mock.patch.object(BlockingPortHandler, 'setBaudRate', return_value=True):
            port_handler = dynamixel_utils.get_port_handler('COM9', 1000000)
        self.assertEqual(dxl.port_handler.LATENCY_TIMER, port_handler.latency_timer)


def fake_winreg(registry):
    """a minimal winreg replacement where keys are nested dicts and values are the leaves"""

    @contextmanager
    def open_key(key, sub_key):
        for name in sub_key.split('\\'):
            if not isinstance(key, dict) or name not in key:
                raise OSError("key not found: %s" % sub_key)
            key = key[name]
        yield key

    def enum_key(key, index):
        sub_keys = [name for name, value in key.items() if isinstance(value, dict)]
        if index >= len(sub_keys):
            raise OSError("no more keys")
        return sub_keys[index]

    def query_value(key, name):
        if name not in key:
            raise OSError("value not found: %s" % name)
        return key[name], 4

    root = {'SYSTEM': {'CurrentControlSet': {'Enum': {'FTDIBUS': registry}}}}
    return types.SimpleNamespace(HKEY_LOCAL_MACHINE=root, OpenKey=open_key, EnumKey=enum_key,
                                 QueryValueEx=query_value)


if __name__ == '__main__':
    unittest.main()
