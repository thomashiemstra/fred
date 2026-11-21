import unittest

import numpy as np

from src.robot_controllers.dynamixel_robot.velocity_feedforward import VelocityFeedforward

TEN_DEGREES_PER_SECOND = float(np.radians(10.0))


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

    def test_equal_and_printable(self):
        self.assertEqual(VelocityFeedforward(14.2, 30), VelocityFeedforward(14.2, 30))
        self.assertNotEqual(VelocityFeedforward(14.2, 30), VelocityFeedforward(14.3, 30))
        self.assertNotEqual(VelocityFeedforward(14.2, 30), VelocityFeedforward(14.2, 31))
        self.assertNotEqual(VelocityFeedforward(14.2, 30), 'something else')
        self.assertEqual('VelocityFeedforward(pwm_per_degree_per_second=14.2, max_positions=30)',
                         repr(VelocityFeedforward(14.2, 30)))
        self.assertEqual(VelocityFeedforward(14.2, 30),
                         VelocityFeedforward(pwm_per_degree_per_second=14.2, max_positions=30))


if __name__ == '__main__':
    unittest.main()
