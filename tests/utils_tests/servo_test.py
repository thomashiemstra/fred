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

    def test_reversed_servo_with_offset_round_trip(self):
        # mounted the other way around: the smallest angle is at the highest position
        servo = Servo(3072, 1024, -pi / 2, pi / 2, EXTENDED_POSITION_CONTROL, offset=-100)
        servo.set_target_position_from_angle(-pi / 2)
        self.assertEqual(3072 - 100, servo.target_position)
        servo.set_target_position_from_angle(pi / 2)
        self.assertEqual(1024 - 100, servo.target_position)
        servo.set_target_position_from_angle(pi / 4)
        self.assertEqual(1536 - 100, servo.target_position)
        self.assertAlmostEqual(pi / 4, servo.get_angle_from_position(servo.target_position), places=3)

    def test_angle_outside_the_range_goes_to_the_end_of_the_range(self):
        servo = Servo(1024, 3072, 0, pi, EXTENDED_POSITION_CONTROL, offset=10)
        servo.set_target_position_from_angle(2 * pi)
        self.assertEqual(3072 + 10, servo.target_position)
        servo.set_target_position_from_angle(-pi)
        self.assertEqual(1024 + 10, servo.target_position)

        reversed_servo = Servo(3072, 1024, -pi / 2, pi / 2, EXTENDED_POSITION_CONTROL)
        reversed_servo.set_target_position_from_angle(pi)
        self.assertEqual(1024, reversed_servo.target_position)

    def test_position_outside_the_range_gives_the_angle_at_the_end_of_the_range(self):
        servo = Servo(1024, 3072, 0, pi, EXTENDED_POSITION_CONTROL, offset=10)
        self.assertEqual(pi, servo.get_angle_from_position(4000))
        self.assertEqual(0, servo.get_angle_from_position(-20))
        reversed_servo = Servo(3072, 1024, -pi / 2, pi / 2, EXTENDED_POSITION_CONTROL)
        self.assertEqual(pi / 2, reversed_servo.get_angle_from_position(0))
        self.assertEqual(-pi / 2, reversed_servo.get_angle_from_position(4000))

    def test_target_position_is_a_whole_number(self):
        servo = Servo(0, 4095, 0, pi, EXTENDED_POSITION_CONTROL)
        servo.set_target_position_from_angle(1.0)
        self.assertIsInstance(servo.target_position, int)
        self.assertEqual(round(4095 / pi), servo.target_position)

    def test_clamp_position_includes_the_offset(self):
        servo = Servo(1024, 3072, 0, pi, EXTENDED_POSITION_CONTROL, offset=50)
        self.assertEqual((1074, 3122, 2000), (servo.clamp_position(0), servo.clamp_position(5000),
                                              servo.clamp_position(2000)))
        self.assertIsInstance(servo.clamp_position(2000), int)

    def test_clamp_position_of_reversed_servo(self):
        servo = Servo(3072, 1024, -pi / 2, pi / 2, EXTENDED_POSITION_CONTROL, offset=-24)
        self.assertEqual((1000, 3048, 2000), (servo.clamp_position(0), servo.clamp_position(5000),
                                              servo.clamp_position(2000)))


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

    def test_direction_of_movement_selects_the_offset_function(self):
        servo = ServoWithOffsetFunction(0, 100, 0, pi, EXTENDED_POSITION_CONTROL, lambda angles: 10,
                                        lambda angles: -10)
        all_angles = [0.0] * 7
        servo.set_target_position_from_angle(pi / 2, all_angles)
        self.assertEqual(50 - 10, servo.target_position)
        servo.set_target_position_from_angle(pi / 4, all_angles)
        self.assertEqual(25 + 10, servo.target_position)
        # standing still keeps the last direction
        servo.set_target_position_from_angle(pi / 4, all_angles)
        self.assertEqual(25 + 10, servo.target_position)
        # without the other angles there is no compensation
        servo.set_target_position_from_angle(pi / 2)
        self.assertEqual(50, servo.target_position)

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

class ServoCompensationTest(unittest.TestCase):
    """a Servo with a static load compensation adds its positions to the target"""

    def setUp(self):
        from src.robot_controllers.dynamixel_robot.static_compensation import StaticCompensation, UP, DOWN
        self.UP, self.DOWN = UP, DOWN
        self.compensation = StaticCompensation(3, {'sin_phi': 20.0}, 60, up={'sin_phi': 24.0}, down={'sin_phi': 16.0})
        self.servo = Servo(1024, 3072, 0, pi, EXTENDED_POSITION_CONTROL, offset=10, compensation=self.compensation)
        self.angles = [0.0, 0.0, 0.0, 0.5 * pi, 0.0, 0.0, 0.0]  # phi = pi/2, sin(phi) = 1

    def test_compensation_is_added_to_the_target(self):
        self.servo.set_target_position_from_angle(0.5 * pi, self.angles)
        self.assertEqual(2048 + 10 + 20, self.servo.target_position)
        self.assertEqual(20, self.servo.compensation_positions)

    def test_no_compensation_without_all_angles_or_without_a_compensation(self):
        self.servo.set_target_position_from_angle(0.5 * pi)
        self.assertEqual(2058, self.servo.target_position)
        self.assertEqual(0, self.servo.compensation_positions)
        plain = Servo(1024, 3072, 0, pi, EXTENDED_POSITION_CONTROL, offset=10)
        plain.set_target_position_from_angle(0.5 * pi, self.angles)
        self.assertEqual(2058, plain.target_position)
        self.assertEqual(0, plain.compensation_positions)

    def test_direction_is_the_last_change_of_the_commanded_angle(self):
        self.servo.set_target_position_from_angle(0.5 * pi, self.angles)
        self.assertEqual(20, self.servo.compensation_positions, 'no direction yet: the mean terms')
        self.servo.set_target_position_from_angle(0.5 * pi + 0.01, self.angles)
        self.assertEqual(24, self.servo.compensation_positions, 'moving up')
        self.servo.set_target_position_from_angle(0.5 * pi + 0.01, self.angles)
        self.assertEqual(24, self.servo.compensation_positions, 'the same angle keeps the direction')
        self.servo.set_target_position_from_angle(0.5 * pi, self.angles)
        self.assertEqual(16, self.servo.compensation_positions, 'moving down')

    def test_clamp_target_position_keeps_the_compensation_at_the_end_of_the_range(self):
        self.servo.target_position = 3200
        self.servo.compensation_positions = 30
        self.assertEqual(3072 + 10 + 30, self.servo.clamp_target_position())
        self.assertEqual(3072 + 10, self.servo.clamp_position(self.servo.target_position))
        self.servo.target_position = 900
        self.servo.compensation_positions = -30
        self.assertEqual(1024 + 10 - 30, self.servo.clamp_target_position())
        self.servo.compensation_positions = 0
        self.assertEqual(1024 + 10, self.servo.clamp_target_position())

    def test_direction_only_turns_around_beyond_the_deadband(self):
        # 3 positions of this servo are 3 * pi / 2048 radians
        deadband = self.servo.direction_deadband
        self.assertAlmostEqual(3 * pi / 2048, deadband)
        self.servo.set_target_position_from_angle(1.0, self.angles)
        self.servo.set_target_position_from_angle(1.1, self.angles)
        self.assertEqual(24, self.servo.compensation_positions, 'moving up')
        # dithering around the furthest point within the deadband keeps the direction
        for angle in (1.1 - deadband / 2, 1.1, 1.1 - deadband * 0.9, 1.1 - deadband / 3):
            self.servo.set_target_position_from_angle(angle, self.angles)
            self.assertEqual(24, self.servo.compensation_positions, 'still up at {}'.format(angle))
        # coming back further than the deadband turns it around
        self.servo.set_target_position_from_angle(1.1 - 2 * deadband, self.angles)
        self.assertEqual(16, self.servo.compensation_positions, 'moving down')
        self.servo.set_target_position_from_angle(1.1 - 3 * deadband, self.angles)
        self.assertEqual(16, self.servo.compensation_positions, 'further down')
        self.servo.set_target_position_from_angle(1.1 - 3 * deadband + deadband / 2, self.angles)
        self.assertEqual(16, self.servo.compensation_positions, 'a small step back keeps down')
        self.servo.set_target_position_from_angle(1.1, self.angles)
        self.assertEqual(24, self.servo.compensation_positions, 'up again')

    def test_no_direction_until_the_angle_moved_beyond_the_deadband(self):
        self.servo.set_target_position_from_angle(1.0, self.angles)
        self.servo.set_target_position_from_angle(1.0 + self.servo.direction_deadband / 2, self.angles)
        self.assertEqual(20, self.servo.compensation_positions, 'the mean terms, no direction yet')
