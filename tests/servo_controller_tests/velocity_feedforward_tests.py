import unittest

import numpy as np

from src.robot_controllers.dynamixel_robot.velocity_feedforward import FRICTION_FULL_SPEED, VelocityFeedforward

TEN_DEGREES_PER_SECOND = float(np.radians(10.0))


def degrees_per_second(degrees):
    """the commanded velocity of a joint in rad/s"""
    return float(np.radians(degrees))


class DriveTests(unittest.TestCase):
    """the pwm a motor needs for a commanded velocity"""

    def test_the_drive_is_the_coefficient_per_degree_per_second(self):
        # 14.2 pwm per degree/s at 10 degrees/s
        self.assertAlmostEqual(142.0, VelocityFeedforward(14.2, 30).drive(TEN_DEGREES_PER_SECOND))

    def test_the_velocity_is_in_radians_per_second(self):
        # one radian per second is 57.2958 degrees/s
        self.assertAlmostEqual(2.0 * 57.29577951308232, VelocityFeedforward(2.0, 30).drive(1.0))

    def test_the_drive_has_the_sign_of_the_velocity(self):
        feedforward = VelocityFeedforward(14.2, 30)
        self.assertAlmostEqual(-142.0, feedforward.drive(-TEN_DEGREES_PER_SECOND))
        self.assertEqual(0.0, feedforward.drive(0.0))

    def test_the_drive_is_not_clipped(self):
        # only the positions are limited
        self.assertAlmostEqual(14.2 * 1000, VelocityFeedforward(14.2, 1).drive(float(np.radians(1000.0))))

    def test_a_coefficient_of_0_gives_no_drive(self):
        self.assertEqual(0.0, VelocityFeedforward(0.0, 30).drive(3.0))


class FrictionDriveTests(unittest.TestCase):
    """the drive that keeps the joint sliding: whole from FRICTION_FULL_SPEED, in proportion to the speed below it"""

    def test_the_full_speed_is_1_degree_per_second(self):
        self.assertEqual(1.0, FRICTION_FULL_SPEED)

    def test_above_the_full_speed_the_drive_is_the_coefficient_term_plus_the_whole_friction(self):
        # 1.45 * 10 + 27 = 41.5 pwm
        feedforward = VelocityFeedforward(1.45, 30, 27.0)
        self.assertAlmostEqual(41.5, feedforward.drive(degrees_per_second(10.0)))
        self.assertAlmostEqual(-41.5, feedforward.drive(degrees_per_second(-10.0)))

    def test_at_the_full_speed_the_friction_is_whole(self):
        feedforward = VelocityFeedforward(1.45, 30, 27.0)
        self.assertAlmostEqual(1.45 + 27.0, feedforward.drive(degrees_per_second(FRICTION_FULL_SPEED)))
        self.assertAlmostEqual(-(1.45 + 27.0), feedforward.drive(degrees_per_second(-FRICTION_FULL_SPEED)))

    def test_below_the_full_speed_the_friction_is_in_proportion_to_the_speed(self):
        feedforward = VelocityFeedforward(1.45, 30, 27.0)
        # 0.5 degrees/s: 1.45 * 0.5 + 27 * 0.5 = 0.725 + 13.5
        self.assertAlmostEqual(0.725 + 13.5, feedforward.drive(degrees_per_second(0.5)))
        self.assertAlmostEqual(-(0.725 + 13.5), feedforward.drive(degrees_per_second(-0.5)))
        # 0.1 degrees/s: 0.145 + 2.7
        self.assertAlmostEqual(0.145 + 2.7, feedforward.drive(degrees_per_second(0.1)))

    def test_the_friction_has_no_jump_through_zero_speed(self):
        feedforward = VelocityFeedforward(0.0, 30, 27.0)
        self.assertEqual(0.0, feedforward.drive(0.0))
        step = 1e-6
        self.assertAlmostEqual(0.0, feedforward.drive(degrees_per_second(step)), delta=1e-3)
        self.assertAlmostEqual(0.0, feedforward.drive(degrees_per_second(-step)), delta=1e-3)
        # a speed that wobbles around zero moves the drive by a fraction of the friction, not by twice the whole
        self.assertLess(abs(feedforward.drive(degrees_per_second(0.05)) - feedforward.drive(degrees_per_second(-0.05))),
                        0.1 * 27.0 * 2)

    def test_the_friction_is_the_same_at_every_speed_above_the_full_speed(self):
        with_friction, without = VelocityFeedforward(1.45, 30, 27.0), VelocityFeedforward(1.45, 30)
        for degrees in (1.0, 2.0, 10.0, 30.0, 100.0):
            for sign in (1, -1):
                velocity = degrees_per_second(sign * degrees)
                self.assertAlmostEqual(sign * 27.0, with_friction.drive(velocity) - without.drive(velocity),
                                       msg=degrees)

    def test_the_friction_alone_gives_a_drive_without_a_coefficient(self):
        feedforward = VelocityFeedforward(0.0, 30, 27.0)
        self.assertAlmostEqual(27.0, feedforward.drive(degrees_per_second(5.0)))
        self.assertAlmostEqual(-27.0, feedforward.drive(degrees_per_second(-5.0)))

    def test_without_friction_every_number_is_what_it_was(self):
        plain = VelocityFeedforward(14.2, 30)
        explicit = VelocityFeedforward(14.2, 30, 0.0)
        for degrees in (-30.0, -10.0, -0.5, 0.0, 0.3, 1.0, 10.0, 200.0):
            velocity = degrees_per_second(degrees)
            self.assertEqual(14.2 * float(np.degrees(velocity)), plain.drive(velocity), msg=degrees)
            self.assertEqual(plain.drive(velocity), explicit.drive(velocity))
            self.assertEqual(plain.positions(velocity, 2000), explicit.positions(velocity, 2000))

    def test_the_positions_for_the_drive_with_friction(self):
        # 41.5 pwm at P 780 (6.09375 pwm per position): 6.81 positions
        feedforward = VelocityFeedforward(1.45, 30, 27.0)
        self.assertAlmostEqual(41.5 / (780 / 128), feedforward.positions(degrees_per_second(10.0), 780))
        self.assertAlmostEqual(6.81, feedforward.positions(degrees_per_second(10.0), 780), places=2)
        self.assertAlmostEqual(-6.81, feedforward.positions(degrees_per_second(-10.0), 780), places=2)
        self.assertAlmostEqual((0.725 + 13.5) / (780 / 128), feedforward.positions(degrees_per_second(0.5), 780))

    def test_the_positions_with_friction_are_clipped_and_zero_without_a_p(self):
        feedforward = VelocityFeedforward(1.45, 5, 27.0)
        self.assertEqual(5.0, feedforward.positions(degrees_per_second(10.0), 780))
        self.assertEqual(-5.0, feedforward.positions(degrees_per_second(-10.0), 780))
        self.assertEqual(0.0, feedforward.positions(degrees_per_second(10.0), 0))
        # the friction alone can be clipped as well
        self.assertEqual(2.0, VelocityFeedforward(0.0, 2, 27.0).positions(degrees_per_second(2.0), 780))

    def test_a_joint_at_rest_gets_no_positions_whatever_the_friction(self):
        self.assertEqual(0.0, VelocityFeedforward(1.45, 30, 500.0).positions(0.0, 780))


class PositionsTests(unittest.TestCase):
    """the positions to add to the target for a commanded velocity"""

    def test_positions_are_the_drive_divided_by_the_pwm_per_position(self):
        # 142 pwm at P 2000, which gives 2000 / 128 = 15.625 pwm per position of error: 9.0880 positions
        positions = VelocityFeedforward(14.2, 30).positions(TEN_DEGREES_PER_SECOND, 2000)
        self.assertAlmostEqual(142.0 / 15.625, positions)
        self.assertAlmostEqual(9.088, positions, places=3)

    def test_positions_are_not_rounded(self):
        positions = VelocityFeedforward(14.2, 30).positions(TEN_DEGREES_PER_SECOND, 2000)
        self.assertIsInstance(positions, float)
        self.assertNotEqual(round(positions), positions)

    def test_positions_have_the_sign_of_the_velocity(self):
        feedforward = VelocityFeedforward(14.2, 30)
        self.assertAlmostEqual(-9.088, feedforward.positions(-TEN_DEGREES_PER_SECOND, 2000), places=3)
        self.assertEqual(0.0, feedforward.positions(0.0, 2000))

    def test_the_same_drive_is_fewer_positions_at_a_higher_p(self):
        feedforward = VelocityFeedforward(14.2, 30)
        self.assertAlmostEqual(2 * feedforward.positions(TEN_DEGREES_PER_SECOND, 2000),
                               feedforward.positions(TEN_DEGREES_PER_SECOND, 1000))
        self.assertAlmostEqual(0.5 * feedforward.positions(TEN_DEGREES_PER_SECOND, 2000),
                               feedforward.positions(TEN_DEGREES_PER_SECOND, 4000))

    def test_positions_are_clipped_to_max_positions_in_both_directions(self):
        feedforward = VelocityFeedforward(14.2, 5)
        self.assertEqual(5.0, feedforward.positions(TEN_DEGREES_PER_SECOND, 2000))
        self.assertEqual(-5.0, feedforward.positions(-TEN_DEGREES_PER_SECOND, 2000))
        # below the limit nothing changes
        self.assertAlmostEqual(4.0, feedforward.positions(TEN_DEGREES_PER_SECOND * 4.0 / 9.088, 2000), places=3)

    def test_without_a_positive_p_there_are_no_positions(self):
        feedforward = VelocityFeedforward(14.2, 30)
        self.assertEqual(0.0, feedforward.positions(TEN_DEGREES_PER_SECOND, 0))
        self.assertEqual(0.0, feedforward.positions(TEN_DEGREES_PER_SECOND, -100))
        self.assertIsInstance(feedforward.positions(TEN_DEGREES_PER_SECOND, 0), float)


class ConstructionTests(unittest.TestCase):

    def test_the_values_are_kept(self):
        feedforward = VelocityFeedforward(14, 30)
        self.assertEqual(14.0, feedforward.pwm_per_degree_per_second)
        self.assertEqual(30, feedforward.max_positions)

    def test_a_negative_coefficient_is_refused(self):
        with self.assertRaises(ValueError):
            VelocityFeedforward(-0.1, 30)

    def test_max_positions_has_to_be_positive(self):
        for max_positions in (0, -5):
            with self.assertRaises(ValueError, msg=str(max_positions)):
                VelocityFeedforward(14.2, max_positions)

    def test_max_positions_has_to_be_a_whole_number(self):
        for max_positions in (2.5, True):
            with self.assertRaises(ValueError, msg=str(max_positions)):
                VelocityFeedforward(14.2, max_positions)  # type: ignore[arg-type]

    def test_the_friction_is_kept_and_defaults_to_0(self):
        self.assertEqual(27.0, VelocityFeedforward(14, 30, 27).pwm_for_friction)
        self.assertIsInstance(VelocityFeedforward(14, 30, 27).pwm_for_friction, float)
        self.assertEqual(0.0, VelocityFeedforward(14, 30).pwm_for_friction)
        self.assertEqual(0.0, VelocityFeedforward(14, 30, 0).pwm_for_friction)

    def test_a_negative_or_nan_friction_is_refused(self):
        for friction in (-0.1, -27.0, float('-inf'), float('nan')):
            with self.assertRaises(ValueError, msg=str(friction)):
                VelocityFeedforward(14.2, 30, friction)

    def test_equal_and_printable(self):
        self.assertEqual(VelocityFeedforward(14.2, 30), VelocityFeedforward(14.2, 30))
        self.assertNotEqual(VelocityFeedforward(14.2, 30), VelocityFeedforward(14.3, 30))
        self.assertNotEqual(VelocityFeedforward(14.2, 30), VelocityFeedforward(14.2, 31))
        self.assertNotEqual(VelocityFeedforward(14.2, 30), 'something else')
        self.assertEqual('VelocityFeedforward(pwm_per_degree_per_second=14.2, max_positions=30, pwm_for_friction=0.0)',
                         repr(VelocityFeedforward(14.2, 30)))
        self.assertEqual(VelocityFeedforward(14.2, 30),
                         VelocityFeedforward(pwm_per_degree_per_second=14.2, max_positions=30))

    def test_the_friction_is_in_the_equality_and_the_representation(self):
        self.assertEqual(VelocityFeedforward(14.2, 30, 27.0), VelocityFeedforward(14.2, 30, 27))
        self.assertNotEqual(VelocityFeedforward(14.2, 30, 27.0), VelocityFeedforward(14.2, 30, 26.0))
        self.assertNotEqual(VelocityFeedforward(14.2, 30, 27.0), VelocityFeedforward(14.2, 30))
        self.assertEqual(VelocityFeedforward(14.2, 30), VelocityFeedforward(14.2, 30, 0.0))
        self.assertEqual('VelocityFeedforward(pwm_per_degree_per_second=14.2, max_positions=30, pwm_for_friction=27.0)',
                         repr(VelocityFeedforward(14.2, 30, 27.0)))
        self.assertEqual(VelocityFeedforward(14.2, 30, 27.0),
                         VelocityFeedforward(pwm_per_degree_per_second=14.2, max_positions=30, pwm_for_friction=27.0))


if __name__ == '__main__':
    unittest.main()
