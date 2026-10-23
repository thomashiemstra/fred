import unittest

import numpy as np
from numpy import pi

from src.kinematics.kinematics import forward_position_kinematics
from src.kinematics.kinematics_utils import RobotConfig
from src.robot_controllers.dynamixel_robot.static_compensation import SpringGeometry, StaticCompensation, \
    term_values, term_names, regressor_row, UP, DOWN, SPRING_TERMS

config = RobotConfig(d1=15.7, a2=20, d4=22, d6=12)


def random_angles(rng):
    angles = np.zeros(7)
    angles[1:7] = rng.uniform(-pi / 2, pi / 2, 6)
    return angles


def height_gradient(angles, joint, point, h=1e-6):
    """the change of the height of a point of the arm (index of forward_position_kinematics) per radian of a joint"""
    up, down = angles.copy(), angles.copy()
    up[joint] += h
    down[joint] -= h
    return (forward_position_kinematics(up, config)[point][2] - forward_position_kinematics(down, config)[point][2]) / (2 * h)


class BasisTests(unittest.TestCase):
    """the terms are the derivatives of the heights of the links to the joint angles, which is the gravity load"""

    def test_terms_match_the_forward_kinematics(self):
        rng = np.random.default_rng(1)
        wrist_centre, tip = 3, 4  # p4 and p6 of forward_position_kinematics
        for _ in range(100):
            angles = random_angles(rng)
            values = {joint: term_values(joint, angles) for joint in (2, 3, 4, 5)}
            # a mass at the wrist centre: lever a2 on joint 2, d4 along the upper arm on joint 3, nothing on 4 and 5
            self.assertAlmostEqual(config.a2 * values[2]['cos_q2'] + config.d4 * values[2]['sin_phi'],
                                   height_gradient(angles, 2, wrist_centre), places=6)
            self.assertAlmostEqual(config.d4 * values[3]['sin_phi'], height_gradient(angles, 3, wrist_centre), places=6)
            self.assertAlmostEqual(0.0, height_gradient(angles, 5, wrist_centre), places=6)
            # a mass at the tip adds the gripper terms, d6 from the wrist centre
            self.assertAlmostEqual(config.a2 * values[2]['cos_q2'] + config.d4 * values[2]['sin_phi']
                                   + config.d6 * values[2]['wrist_reach'], height_gradient(angles, 2, tip), places=6)
            self.assertAlmostEqual(config.d4 * values[3]['sin_phi'] + config.d6 * values[3]['wrist_reach'],
                                   height_gradient(angles, 3, tip), places=6)
            self.assertAlmostEqual(config.d6 * values[5]['wrist_pitch'], height_gradient(angles, 5, tip), places=6)
            self.assertAlmostEqual(config.d6 * values[4]['wrist_roll'], -height_gradient(angles, 4, tip), places=6)

    def test_perpendicular_terms(self):
        # a centre of mass beside the link: sin_q2 and cos_phi are the gradients of the perpendicular offsets
        angles = np.array([0, 0, 0.3, 1.2, 0, 0, 0])
        values = term_values(2, angles)
        self.assertAlmostEqual(np.sin(0.3), values['sin_q2'])
        self.assertAlmostEqual(np.cos(1.5), values['cos_phi'])
        self.assertAlmostEqual(np.cos(1.5), term_values(3, angles)['cos_phi'])

    def test_term_names_and_rows(self):
        self.assertEqual(('constant', 'cos_q2', 'sin_q2', 'sin_phi', 'cos_phi', 'wrist_reach'), term_names(2),
                         'the spring terms are not in the default model since the spring was removed')
        self.assertEqual(('constant', 'cos_q2', 'sin_q2', 'sin_phi', 'cos_phi', 'wrist_reach') + SPRING_TERMS,
                         term_names(2, spring=True))
        self.assertEqual(('constant', 'wrist_pitch'), term_names(5))
        self.assertEqual(('constant', 'wrist_pitch'), term_names(5, spring=True), 'the spring loads joints 2 and 3')
        with self.assertRaises(ValueError):
            term_names(1)
        angles = np.array([0, 1.0, 0.2, 1.0, 0.1, 0.4, 0])
        row = regressor_row(3, angles)
        self.assertEqual((4,), row.shape)  # no spring: the spring terms are left out
        self.assertEqual(1.0, row[0])
        self.assertEqual(('constant', 'sin_phi', 'cos_phi', 'wrist_reach'), tuple(term_values(3, angles)))
        self.assertEqual([np.cos(1.2)], list(regressor_row(3, angles, names=('cos_phi',))))
        with self.assertRaises(ValueError):
            regressor_row(3, angles, names=('spring_dl',))

    def test_spring_terms(self):
        spring = SpringGeometry(base_anchor=(-4.0, -6.0), arm_anchor=(2.0, 3.0), a2=20.0)
        angles = np.array([0, 0, 0.3, 1.2, 0, 0, 0])
        length, d_q2, d_q3 = spring.length_and_gradient(0.3, 1.2)
        self.assertGreater(length, 0)
        self.assertAlmostEqual(d_q2, term_values(2, angles, spring)['spring_dl'])
        self.assertAlmostEqual(d_q3, term_values(3, angles, spring)['spring_dl'])
        self.assertEqual(7, len(regressor_row(2, angles, spring)))
        self.assertEqual(5, len(regressor_row(3, angles, spring)))
        self.assertNotIn('spring_dl', term_values(5, angles, spring))

    def test_a_spring_of_zero_free_length_pulls_like_a_mass(self):
        # k L dL/dq is a combination of the gravity terms, so it is no term of its own: the regressor with it would
        # be singular. With dL/dq alone the regressor stays well conditioned.
        spring = SpringGeometry(base_anchor=(-4.0, -6.0), arm_anchor=(2.0, 3.0), a2=20.0)
        rng = np.random.default_rng(0)
        rows_gravity, rows_with_spring, zero_free_length = [], [], []
        for _ in range(200):
            angles = random_angles(rng)
            rows_gravity.append(regressor_row(2, angles))
            rows_with_spring.append(regressor_row(2, angles, spring))
            length, d_q2, _ = spring.length_and_gradient(angles[2], angles[3])
            zero_free_length.append(length * d_q2)
        coefficients, residual, _, _ = np.linalg.lstsq(np.array(rows_gravity), np.array(zero_free_length), rcond=None)
        self.assertLess(np.abs(np.array(rows_gravity) @ coefficients - zero_free_length).max(), 1e-6)
        self.assertLess(np.linalg.cond(np.array(rows_with_spring)), 500)

    def test_spring_length_geometry(self):
        # the anchor on the upper arm sits at the elbow when both anchor offsets are 0: the length is then the
        # distance between the base anchor and the elbow
        spring = SpringGeometry(base_anchor=(0.0, 0.0), arm_anchor=(0.0, 0.0), a2=20.0)
        self.assertAlmostEqual(20.0, spring.length(0.3, 1.2))
        # the elbow and the anchor along the upper arm: at phi = pi/2 the upper arm points radially outward
        spring = SpringGeometry(base_anchor=(0.0, 0.0), arm_anchor=(5.0, 0.0), a2=20.0)
        self.assertAlmostEqual(25.0, spring.length(0.0, pi / 2))
        self.assertAlmostEqual(np.hypot(20.0, 5.0), spring.length(0.0, 0.0))  # upper arm hanging down
        with self.assertRaises(ValueError):
            SpringGeometry((1, 2, 3), (0, 0), 20.0)


class StaticCompensationTests(unittest.TestCase):

    def test_positions_are_the_weighted_terms_rounded(self):
        compensation = StaticCompensation(2, {'cos_q2': 10.0, 'sin_phi': 4.0}, max_positions=60)
        angles = np.array([0, 0, 0.0, pi / 2, 0, 0, 0])  # lower arm horizontal, upper arm in line
        self.assertEqual(14, compensation.positions(angles))
        angles[2] = pi / 2  # straight up: cos 0, sin(pi) 0
        self.assertEqual(0, compensation.positions(angles))
        angles[2] = pi / 3
        self.assertEqual(int(round(10 * np.cos(pi / 3) + 4 * np.sin(pi / 3 + pi / 2))), compensation.positions(angles))

    def test_positions_are_clipped(self):
        compensation = StaticCompensation(3, {'sin_phi': 100.0}, max_positions=30)
        angles = np.array([0, 0, 0.0, pi / 2, 0, 0, 0])
        self.assertEqual(30, compensation.positions(angles))
        angles[3] = -pi / 2
        self.assertEqual(-30, compensation.positions(angles))

    def test_directions(self):
        compensation = StaticCompensation(3, {'sin_phi': 10.0}, 60, up={'sin_phi': 14.0}, down={'sin_phi': 6.0})
        angles = np.array([0, 0, 0.0, pi / 2, 0, 0, 0])
        self.assertEqual(10, compensation.positions(angles))
        self.assertEqual(14, compensation.positions(angles, UP))
        self.assertEqual(6, compensation.positions(angles, DOWN))
        # without a directional fit the direction doesn't matter
        self.assertEqual(10, StaticCompensation(3, {'sin_phi': 10.0}, 60).positions(angles, UP))

    def test_validation(self):
        with self.assertRaises(ValueError):
            StaticCompensation(2, {'nope': 1.0}, 60)
        with self.assertRaises(ValueError):
            StaticCompensation(1, {}, 60)
        with self.assertRaises(ValueError):
            StaticCompensation(2, {'cos_q2': 1.0}, 0)
        with self.assertRaises(ValueError):
            StaticCompensation(2, {'cos_q2': 1.0}, 60, up={'cos_q2': 1.0})
        with self.assertRaises(ValueError):
            StaticCompensation(2, {'spring_dl': 1.0}, 60)  # no spring geometry
        spring = SpringGeometry((-4.0, -6.0), (2.0, 3.0), 20.0)
        compensation = StaticCompensation(2, {'spring_dl': 1.0}, 60, spring=spring)
        self.assertIsInstance(compensation.positions(np.array([0, 0, 0.3, 1.0, 0, 0, 0])), int)

    def test_empty_terms_give_zero(self):
        self.assertEqual(0, StaticCompensation(2, {}, 60).positions(np.zeros(7)))


if __name__ == '__main__':
    unittest.main()
