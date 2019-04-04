import unittest
from src.dynamixel_robot.servo_handler import ServoHandler
from src.dynamixel_robot import dynamixel_x_config
from numpy import pi
import math
from src.dynamixel_robot.servo import Servo


class TestServoHandler(unittest.TestCase):

    def setUp(self):
        servo1 = Servo(0, 4095, 0, 2 * pi)
        servo2 = Servo(1024, 3072, 0, pi)
        servo3 = Servo(0, 4095, -pi, pi)
        # if a servo is mounted backwards it's min will correspond to the max angle and vice versa
        servo4 = Servo(3072, 1024, -pi/2, pi/2)
        servos = {1: servo1, 2: servo2, 3: servo3, 4: servo4}
        self.servo_handler = ServoHandler(servos, dynamixel_x_config, None, None, None, None)

    def test_convert_angle_1(self):
        expected_position = math.floor(4095 / 2)
        self.servo_handler.set_angle(1, pi)
        self.assertEqual(expected_position, self.servo_handler.get_servo(1).target_position)

    def test_convert_angle_2(self):
        expected_position = 1024 + (3072 - 1024) / 4
        self.servo_handler.set_angle(2, pi/4)
        self.assertEqual(expected_position, self.servo_handler.get_servo(2).target_position)

    def test_convert_angle_3(self):
        expected_position = 1280
        self.servo_handler.set_angle(2, pi/8)
        self.assertEqual(expected_position, self.servo_handler.get_servo(2).target_position)

    def test_convert_angle_4(self):
        expected_position = math.floor(3*4095 / 4)
        self.servo_handler.set_angle(3, pi/2)
        self.assertEqual(expected_position, self.servo_handler.get_servo(3).target_position)

    def test_convert_angle_5(self):
        expected_position = math.ceil(4095 / 2)
        self.servo_handler.set_angle(4, 0)
        self.assertEqual(expected_position, self.servo_handler.get_servo(4).target_position)

    def test_convert_angle_6(self):
        expected_position = math.floor(1024 + (3072 - 1024)/4)
        self.servo_handler.set_angle(4, pi/4)
        self.assertEqual(expected_position, self.servo_handler.get_servo(4).target_position)

    def test_convert_angle_7(self):
        expected_position = math.floor(1024 + 3*(3072 - 1024) / 4)
        self.servo_handler.set_angle(4, - pi/4)
        self.assertEqual(expected_position, self.servo_handler.get_servo(4).target_position)

    def test_convert_position_1(self):
        angle = self.servo_handler.get_angle(1, math.floor(4095/2))
        self.assertAlmostEqual(pi, angle, delta=0.01)

    def test_convert_position_2(self):
        angle = self.servo_handler.get_angle(1, math.floor(4095/4))
        self.assertAlmostEqual(pi/2, angle, delta=0.01)

    def test_convert_position_3(self):
        angle = self.servo_handler.get_angle(2, math.floor(1024 + (3072-1024)/4))
        self.assertAlmostEqual(pi/4, angle, delta=0.01)

    def test_convert_position_4(self):
        angle = self.servo_handler.get_angle(3, math.floor(4095/2))
        self.assertAlmostEqual(0, angle, delta=0.01)

    def test_convert_position_5(self):
        angle = self.servo_handler.get_angle(3, math.floor(3*4095/4))
        self.assertAlmostEqual(pi/2, angle, delta=0.01)

    def test_convert_position_6(self):
        angle = self.servo_handler.get_angle(3, math.floor(3*4095/4))
        self.assertAlmostEqual(pi/2, angle, delta=0.01)

    def test_convert_position_7(self):
        angle = self.servo_handler.get_angle(4, math.ceil(1024 + 3*(3072-1024)/4))
        self.assertAlmostEqual(-pi/4, angle, delta=0.01)

    def test_convert_position_8(self):
        angle = self.servo_handler.get_angle(4, math.ceil(1024 + (3072-1024)/2))
        self.assertAlmostEqual(0, angle, delta=0.01)

    def test_convert_position_9(self):
        angle = self.servo_handler.get_angle(4, 3072)
        self.assertAlmostEqual(-pi/2, angle, delta=0.01)

