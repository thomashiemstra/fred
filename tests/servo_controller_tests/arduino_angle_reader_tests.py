import threading
import time
import unittest
from unittest import mock

import serial

from src.robot_controllers.dynamixel_robot.arduino import ArduinoAngleReader as arduino_module
from src.robot_controllers.dynamixel_robot.arduino.ArduinoAngleReader import ArduinoAngleReader


class ArduinoAngleReaderTests(unittest.TestCase):

    def setUp(self):
        # pyserial's loopback port instead of the arduino, a timeout so a read without data returns
        self.port = serial.serial_for_url('loop://', timeout=0.01)
        patcher = mock.patch.object(arduino_module.serial, 'Serial', return_value=self.port)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.reader = ArduinoAngleReader()

    def stop_within(self, seconds):
        stopper = threading.Thread(target=self.reader.stop, daemon=True)
        stopper.start()
        stopper.join(seconds)
        return not stopper.is_alive()

    def test_the_angle_the_arduino_sends_is_read(self):
        self.reader.start()
        self.port.write(b'512\r\n')
        deadline = time.monotonic() + 2
        while self.reader.get_val() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(512, self.reader.get_val())
        self.assertTrue(self.stop_within(2))

    def test_stop_ends_the_reading_thread(self):
        self.reader.start()
        self.assertTrue(self.stop_within(2), 'stop() did not return')
        self.assertFalse(self.reader._thread.is_alive())

    def test_stop_before_start_does_nothing(self):
        self.assertTrue(self.stop_within(2))

    def test_lines_that_are_no_number_are_skipped(self):
        self.reader.start()
        self.port.write(b'garbage\r\n300\r\n')
        deadline = time.monotonic() + 2
        while self.reader.get_val() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(300, self.reader.get_val())
        self.assertTrue(self.stop_within(2))


if __name__ == '__main__':
    unittest.main()
