import unittest

from src.global_constants import EXTENDED_POSITION_CONTROL
from src.robot_controllers.dynamixel_robot.servo import Servo, Servo2, Servo3, ServoWithOffsetFunction
from numpy import pi


class ServoTest(unittest.TestCase):

    def test_get_angle_from_position_1(self):
        # given
        servo = Servo(0, 100, 0, 100, EXTENDED_POSITION_CONTROL)

        # when
        angle = servo.get_angle_from_position(50)

        # then
        self.assertEqual(50, angle, "got a wrong angle, it should maps 1 to 1 from 50 to 50")

    def test_get_angle_from_position_2(self):
        # given
        servo = Servo(0, 100, 0, 200, EXTENDED_POSITION_CONTROL)

        # when
        angle = servo.get_angle_from_position(100)

        # then
        self.assertEqual(200, angle, "got a wrong angle, it should maps 1 to 2 from 100 to 200")

    def test_get_angle_from_position_3(self):
        # given
        servo = Servo(0, 50, 0, 100, EXTENDED_POSITION_CONTROL)

        # when
        angle = servo.get_angle_from_position(25)

        # then
        self.assertEqual(50, angle, "got a wrong angle, it should map 1 to 2 from 25 to 50")

    def test_set_target_position_from_angle_1(self):
        # given
        servo = Servo(0, 100, 0, 100, EXTENDED_POSITION_CONTROL)

        # when
        servo.set_target_position_from_angle(50)

        # then
        target = servo.target_position
        self.assertEqual(50, target, "got a wrong target, it should maps 1 to 1 from 50 to 50")

    def test_set_target_position_from_angle_2(self):
        # given
        servo = Servo(0, 50, 0, 100, EXTENDED_POSITION_CONTROL)

        # when
        servo.set_target_position_from_angle(40)

        # then
        target = servo.target_position
        self.assertEqual(20, target, "got a wrong target")

    def test_set_target_position_from_angle_3(self):
        # given
        servo = Servo(0, 100, 0, 200, EXTENDED_POSITION_CONTROL)

        # when
        servo.set_target_position_from_angle(50)

        # then
        target = servo.target_position
        self.assertEqual(25, target, "got a wrong target")

    def test_constant_offset_angle_conversion(self):
        # given
        servo = Servo(0, 100, 0, pi, EXTENDED_POSITION_CONTROL, offset=10)

        # when
        angle = pi/2
        servo.set_target_position_from_angle(angle)

        # then
        self.assertEqual(60, servo.target_position)

    def test_constant_offset_angle_conversion_back_to_angle(self):
        # given
        servo = Servo(0, 100, 0, 100, EXTENDED_POSITION_CONTROL, offset=10)

        # when
        angle = 30
        servo.set_target_position_from_angle(angle)

        # then
        servo_target = servo.target_position
        self.assertEqual(angle, servo.get_angle_from_position(servo_target))


class ServoWithOffsetFunctionTest(unittest.TestCase):

    def test_offset_function(self):
        # given
        def offset_function(all_angles):
            return all_angles[0]
        servo = ServoWithOffsetFunction(0, 100, 0, pi, EXTENDED_POSITION_CONTROL, offset_function, offset_function)

        # when
        servo.set_target_position_from_angle(pi/2, [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # then
        target = servo.target_position
        self.assertEqual(40, target, "got a wrong target")

    def test_no_function(self):
        self.assertRaises(ValueError, ServoWithOffsetFunction, 0, 1, 0, 1, EXTENDED_POSITION_CONTROL, None, None)


class Servo2Tests(unittest.TestCase):

    def test_1(self):
        # given
        offset = {"0.0": {"0.0": -10}}
        servo2 = Servo2(0, 100, 0, 100, EXTENDED_POSITION_CONTROL, dynamic_offsets=offset)

        # when
        angle2 = 0.0
        angle3 = 0.0
        servo2.set_target_position_from_angle(angle2, [0.0, 0.0, angle2, angle3, 0.0, 0.0, 0.0])

        # then
        target = servo2.target_position
        self.assertEqual(10, target, "got a wrong target")

    def test_2(self):
        # given
        offset = {"0.2": {"0.0": -10}}
        servo2 = Servo2(0, 100, 0, pi, EXTENDED_POSITION_CONTROL, dynamic_offsets=offset)

        # when
        angle2 = 0.25 * pi
        angle3 = 0.0
        servo2.set_target_position_from_angle(angle2, [0.0, 0.0, angle2, angle3, 0.0, 0.0, 0.0])

        # then
        target = servo2.target_position
        self.assertEqual(35, target, "got a wrong target")

    def test_3(self):
        # given
        offset = {"0.0": {"0.0": -10}, "0.1": {"0.0": -5}}
        servo2 = Servo2(0, 100, 0, pi, EXTENDED_POSITION_CONTROL, dynamic_offsets=offset)

        # when
        angle2 = 0.25 * pi
        angle3 = 0.0
        servo2.set_target_position_from_angle(angle2, [0.0, 0.0, angle2, angle3, 0.0, 0.0, 0.0])

        # then
        target = servo2.target_position
        self.assertEqual(30, target, "got a wrong target")

    def test_4(self):
        # given
        offset = {"0.1": {"0.0": -5}}
        servo2 = Servo2(0, 100, 0, pi, EXTENDED_POSITION_CONTROL, dynamic_offsets=offset)

        # when
        angle2 = 0.0
        angle3 = 0.0
        servo2.set_target_position_from_angle(angle2, [0.0, 0.0, angle2, angle3, 0.0, 0.0, 0.0])

        # then
        target = servo2.target_position
        self.assertEqual(5, target, "got a wrong target")


class Servo3Test(unittest.TestCase):

    def test_1(self):
        # given
        offset = {"0.4": 1, "0.5": 10, "0.6": 2}
        servo3 = Servo3(0, 100, 0, pi, EXTENDED_POSITION_CONTROL, dynamic_offsets=offset)

        # when
        angle2 = 0.0
        angle3 = 0.5 * pi
        servo3.set_target_position_from_angle(angle3, [0.0, 0.0, angle2, angle3, 0.0, 0.0, 0.0])

        # then
        target = servo3.target_position
        self.assertEqual(40, target, "got a wrong target")

    def test_2(self):
        # given
        offset = {"0.4": 1, "0.5": 10, "0.6": 2}
        servo3 = Servo3(0, 100, 0, pi, EXTENDED_POSITION_CONTROL, dynamic_offsets=offset)

        # when
        angle2 = 0.2 * pi
        angle3 = 0.1 * pi
        servo3.set_target_position_from_angle(angle3, [0.0, 0.0, angle2, angle3, 0.0, 0.0, 0.0])

        # then
        target = servo3.target_position
        self.assertEqual(9, target, "got a wrong target")

    def test_3(self):
        # given
        offset = {"0.4": 1, "0.5": 10, "0.6": 2}
        servo3 = Servo3(0, 100, 0, pi, EXTENDED_POSITION_CONTROL, dynamic_offsets=offset)

        # when
        angle2 = 0.5 * pi
        angle3 = 0.5 * pi
        servo3.set_target_position_from_angle(angle3, [0.0, 0.0, angle2, angle3, 0.0, 0.0, 0.0])

        # then
        target = servo3.target_position
        self.assertEqual(48, target, "got a wrong target")