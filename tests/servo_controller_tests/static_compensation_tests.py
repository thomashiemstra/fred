import inspect
import unittest

import numpy as np
from numpy import pi

from src.kinematics.kinematics import forward_position_kinematics
from src.kinematics.kinematics_utils import RobotConfig
from src.robot_controllers.dynamixel_robot import static_compensation
from src.robot_controllers.dynamixel_robot.static_compensation import StaticCompensation, JOINT_DRIVES, \
    TERMS_OF_JOINT, joint_2_drive, joint_3_drive, joint_4_drive, joint_5_drive, term_values, term_names, \
    regressor_row

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
        self.assertEqual(('constant', 'cos_q2', 'sin_q2', 'sin_phi', 'cos_phi', 'wrist_reach'), term_names(2))
        self.assertEqual(('constant', 'wrist_pitch'), term_names(5))
        for joint in (1, 6, 7):
            with self.assertRaises(ValueError):
                term_names(joint)
            with self.assertRaises(ValueError):
                term_values(joint, np.zeros(7))
        angles = np.array([0, 1.0, 0.2, 1.0, 0.1, 0.4, 0])
        row = regressor_row(3, angles)
        self.assertEqual((4,), row.shape)
        self.assertEqual(1.0, row[0])
        self.assertEqual(('constant', 'sin_phi', 'cos_phi', 'wrist_reach'), tuple(term_values(3, angles)))
        self.assertEqual([np.cos(1.2)], list(regressor_row(3, angles, names=('cos_phi',))))
        with self.assertRaisesRegex(ValueError, 'no terms'):
            regressor_row(3, angles, names=('spring_dl',))  # the term of the removed spring model
        with self.assertRaisesRegex(ValueError, 'no terms'):
            regressor_row(3, angles, names=('wrist_pitch',))  # a term of another joint


def old_term_values(joint, angles):
    """the generic terms of the model before the explicit formulas (without the spring), the reference of the tests"""
    q2, q3, q4, q5 = (float(angles[i]) for i in (2, 3, 4, 5))
    phi = q2 + q3
    values = {
        'constant': 1.0,
        'cos_q2': np.cos(q2),
        'sin_q2': np.sin(q2),
        'sin_phi': np.sin(phi),
        'cos_phi': np.cos(phi),
        'wrist_reach': np.sin(phi) * np.cos(q5) + np.cos(phi) * np.cos(q4) * np.sin(q5),
        'wrist_pitch': np.cos(phi) * np.sin(q5) + np.cos(q4) * np.cos(q5) * np.sin(phi),
        'wrist_roll': np.sin(q4) * np.sin(q5) * np.sin(phi),
    }
    names = {2: ('constant', 'cos_q2', 'sin_q2', 'sin_phi', 'cos_phi', 'wrist_reach'),
             3: ('constant', 'sin_phi', 'cos_phi', 'wrist_reach'),
             4: ('constant', 'wrist_roll'),
             5: ('constant', 'wrist_pitch')}[joint]
    return {name: values[name] for name in names}


def old_drive(joint, angles, coefficients):
    """the holding drive as the generic sum over the terms computed it"""
    if not coefficients:
        return 0.0
    values = old_term_values(joint, angles)
    return float(sum(coefficient * values[name] for name, coefficient in coefficients.items()))


class FormulaTests(unittest.TestCase):
    """the explicit formula of a joint is the whole model: it gives the numbers the generic terms gave"""

    # the terms of every joint as the yaml checks read them, written out so that the table and the formulas can't drift
    TERMS = {2: ('constant', 'cos_q2', 'sin_q2', 'sin_phi', 'cos_phi', 'wrist_reach'),
             3: ('constant', 'sin_phi', 'cos_phi', 'wrist_reach'),
             4: ('constant', 'wrist_roll'),
             5: ('constant', 'wrist_pitch')}
    FORMULAS = {2: joint_2_drive, 3: joint_3_drive, 4: joint_4_drive, 5: joint_5_drive}

    def test_the_table_of_terms_and_the_formulas_agree(self):
        self.assertEqual(self.TERMS, TERMS_OF_JOINT)
        self.assertEqual(self.FORMULAS, JOINT_DRIVES)
        for joint, formula in self.FORMULAS.items():
            parameters = inspect.signature(formula).parameters
            self.assertEqual(['q2', 'q3', 'q4', 'q5'], list(parameters)[:4], 'joint {}'.format(joint))
            self.assertEqual(self.TERMS[joint], tuple(list(parameters)[4:]), 'joint {}'.format(joint))
            for name in list(parameters)[4:]:
                self.assertEqual(0.0, parameters[name].default, 'joint {} {}'.format(joint, name))
            for name in self.TERMS[joint]:  # accepts exactly those keywords
                self.assertIsInstance(formula(0.1, 0.2, 0.3, 0.4, **{name: 1.0}), float)
            with self.assertRaises(TypeError):
                formula(0.1, 0.2, 0.3, 0.4, spring_dl=1.0)
            self.assertEqual(0.0, formula(0.1, 0.2, 0.3, 0.4))  # no coefficients, no drive

    def test_the_formulas_are_the_old_generic_terms_at_random_poses(self):
        rng = np.random.default_rng(3)
        for _ in range(300):
            angles = random_angles(rng)
            for joint in (2, 3, 4, 5):
                old = old_term_values(joint, angles)
                new = term_values(joint, angles)
                self.assertEqual(tuple(old), tuple(new))
                for name in old:
                    self.assertAlmostEqual(old[name], new[name], places=12, msg='joint {} {}'.format(joint, name))
                self.assertTrue(np.allclose(regressor_row(joint, angles), [old[name] for name in old], atol=1e-12))
                names = list(old)[1:][::-1]  # a selection in another order
                self.assertTrue(np.allclose(regressor_row(joint, angles, names=names), [old[name] for name in names],
                                            atol=1e-12))

    def test_drive_and_positions_are_the_old_generic_sum_with_random_coefficients(self):
        rng = np.random.default_rng(4)
        for _ in range(300):
            angles = random_angles(rng)
            for joint, names in self.TERMS.items():
                mean = {name: float(rng.uniform(-100, 100)) for name in names}
                compensation = StaticCompensation(joint, mean, max_positions=40)
                p = float(rng.uniform(300, 6000))
                reference = old_drive(joint, angles, mean)
                self.assertAlmostEqual(reference, compensation.drive(angles), places=9)
                in_positions = reference / (p / 128)
                # a drive within the rounding noise of half a position can round either way
                if abs(abs(in_positions % 1.0) - 0.5) > 1e-6:
                    self.assertEqual(int(np.clip(np.rint(in_positions), -40, 40)),
                                     compensation.positions(angles, p))
                # coefficients of a subset of the terms
                subset = {name: coefficient for name, coefficient in mean.items() if name != 'constant'}
                self.assertAlmostEqual(old_drive(joint, angles, subset),
                                       StaticCompensation(joint, subset, 40).drive(angles), places=9)

    def test_the_block_on_the_robot_for_joint_3(self):
        terms = {'constant': 3.904, 'sin_phi': -108.714, 'cos_phi': 6.748, 'wrist_reach': -8.721}
        compensation = StaticCompensation(3, terms, max_positions=15)
        p = 4710

        def pose(q2_degrees, q3_degrees):
            angles = np.zeros(7)
            angles[2], angles[3] = np.radians(q2_degrees), np.radians(q3_degrees)
            return angles

        self.assertAlmostEqual(-113.531, compensation.drive(pose(90, 0)), places=3)
        self.assertEqual(-3, compensation.positions(pose(90, 0), p))
        self.assertEqual(-2, compensation.positions(pose(90, 57), p))
        self.assertEqual(-1, compensation.positions(pose(90, -57), p))
        self.assertEqual(-3, compensation.positions(pose(60, 30), p))

    def test_each_formula_written_out(self):
        q2, q3, q4, q5 = 0.7, 0.4, -0.5, 0.9
        phi = q2 + q3
        reach = np.sin(phi) * np.cos(q5) + np.cos(phi) * np.cos(q4) * np.sin(q5)
        self.assertAlmostEqual(
            3 + 2 * np.cos(q2) - 1.5 * np.sin(q2) + 4 * np.sin(phi) + 0.5 * np.cos(phi) - 6 * reach,
            joint_2_drive(q2, q3, q4, q5, constant=3, cos_q2=2, sin_q2=-1.5, sin_phi=4, cos_phi=0.5, wrist_reach=-6))
        self.assertAlmostEqual(3 + 4 * np.sin(phi) + 0.5 * np.cos(phi) - 6 * reach,
                               joint_3_drive(q2, q3, q4, q5, constant=3, sin_phi=4, cos_phi=0.5, wrist_reach=-6))
        self.assertAlmostEqual(-2 + 7 * np.sin(q4) * np.sin(q5) * np.sin(phi),
                               joint_4_drive(q2, q3, q4, q5, constant=-2, wrist_roll=7))
        self.assertAlmostEqual(1 + 8 * (np.cos(phi) * np.sin(q5) + np.cos(q4) * np.cos(q5) * np.sin(phi)),
                               joint_5_drive(q2, q3, q4, q5, constant=1, wrist_pitch=8))

    def test_the_spring_model_is_gone(self):
        for name in ('SpringGeometry', 'SPRING_TERMS', 'SPRING_JOINTS'):
            self.assertFalse(hasattr(static_compensation, name), name)
        with self.assertRaises(TypeError):
            StaticCompensation(2, {}, 60, spring=None)  # type: ignore[call-arg]


class StaticCompensationTests(unittest.TestCase):
    """the coefficients are the holding drive in pwm units, the positions follow from it with the P gain"""

    P = 128  # one pwm unit per position: positions = drive

    def test_drive_is_the_weighted_terms_not_rounded(self):
        compensation = StaticCompensation(2, {'cos_q2': 10.0, 'sin_phi': 4.5}, max_positions=60)
        angles = np.array([0, 0, 0.0, pi / 2, 0, 0, 0])  # lower arm horizontal, upper arm in line
        self.assertAlmostEqual(14.5, compensation.drive(angles))
        angles[2] = pi / 2  # straight up: cos 0, sin(pi) 0
        self.assertAlmostEqual(0.0, compensation.drive(angles))
        angles[2] = pi / 3
        self.assertAlmostEqual(10 * np.cos(pi / 3) + 4.5 * np.sin(pi / 3 + pi / 2), compensation.drive(angles))
        self.assertIsInstance(compensation.drive(angles), float)

    def test_drive_without_coefficients_is_zero(self):
        drive = StaticCompensation(2, {}, 60).drive(np.zeros(7))
        self.assertEqual(0.0, drive)
        self.assertIsInstance(drive, float)

    def test_positions_are_the_drive_converted_with_the_p_gain_and_rounded(self):
        compensation = StaticCompensation(2, {'cos_q2': 10.0, 'sin_phi': 4.0}, max_positions=60)
        angles = np.array([0, 0, 0.0, pi / 2, 0, 0, 0])
        self.assertEqual(14, compensation.positions(angles, self.P))
        angles[2] = pi / 2
        self.assertEqual(0, compensation.positions(angles, self.P))
        angles[2] = pi / 3
        self.assertEqual(int(round(10 * np.cos(pi / 3) + 4 * np.sin(pi / 3 + pi / 2))),
                         compensation.positions(angles, self.P))
        self.assertIsInstance(compensation.positions(angles, self.P), int)

    def test_twice_the_p_gain_gives_half_the_positions(self):
        compensation = StaticCompensation(3, {'sin_phi': 640.0}, max_positions=100)
        angles = np.array([0, 0, 0.0, pi / 2, 0, 0, 0])  # sin(phi) = 1: a drive of 640 pwm
        self.assertEqual(80, compensation.positions(angles, 1024))  # 640 * 128 / 1024
        self.assertEqual(40, compensation.positions(angles, 2048))
        self.assertEqual(20, compensation.positions(angles, 4096))
        self.assertEqual(10, compensation.positions(angles, 8192))

    def test_the_positions_scale_with_the_inverse_of_the_p_gain(self):
        compensation = StaticCompensation(3, {'sin_phi': 1000.0}, max_positions=1000)
        angles = np.array([0, 0, 0.0, pi / 2, 0, 0, 0])
        for p in (500, 1000, 1500, 2500):
            self.assertEqual(int(np.rint(1000 * 128 / p)), compensation.positions(angles, p), 'P {}'.format(p))
        self.assertEqual(256, compensation.positions(angles, 500))
        self.assertEqual(128, compensation.positions(angles, 1000))

    def test_positions_are_clipped_in_positions(self):
        compensation = StaticCompensation(3, {'sin_phi': 100.0}, max_positions=30)
        angles = np.array([0, 0, 0.0, pi / 2, 0, 0, 0])
        self.assertEqual(30, compensation.positions(angles, self.P))
        angles[3] = -pi / 2
        self.assertEqual(-30, compensation.positions(angles, self.P))
        # the clip is on the positions, so a drive that is too big at a low P is fine at a high P
        angles[3] = pi / 2
        self.assertEqual(30, compensation.positions(angles, 256))  # 50 positions
        self.assertEqual(25, compensation.positions(angles, 512))

    def test_no_positions_without_a_positive_p_gain(self):
        compensation = StaticCompensation(3, {'sin_phi': 100.0}, max_positions=30)
        angles = np.array([0, 0, 0.0, pi / 2, 0, 0, 0])
        self.assertEqual(0, compensation.positions(angles, 0))
        self.assertEqual(0, compensation.positions(angles, -1000))
        self.assertEqual(0, StaticCompensation(3, {}, 30).positions(angles, 1000))

    def test_the_compensation_is_for_gravity_only_and_has_no_direction(self):
        angles = np.array([0, 0, 0.0, pi / 2, 0, 0, 0])
        for name in ('up', 'down'):
            with self.assertRaises(TypeError, msg=name):
                StaticCompensation(3, {'sin_phi': 10.0}, 60, **{name: {'sin_phi': 14.0}})  # type: ignore[arg-type]
        compensation = StaticCompensation(3, {'sin_phi': 10.0}, 60)
        with self.assertRaises(TypeError):
            compensation.drive(angles, 'up')  # type: ignore[call-arg]
        with self.assertRaises(TypeError):
            compensation.positions(angles, self.P, 'up')  # type: ignore[call-arg]
        self.assertFalse(hasattr(compensation, 'directional'))
        self.assertFalse(hasattr(compensation, 'coefficients'))
        for name in ('UP', 'DOWN', 'DIRECTIONS'):
            self.assertFalse(hasattr(static_compensation, name), name)

    def test_equality_and_repr_have_the_terms_and_the_clip_only(self):
        self.assertEqual(StaticCompensation(3, {'sin_phi': 10.0}, 60), StaticCompensation(3, {'sin_phi': 10.0}, 60))
        self.assertNotEqual(StaticCompensation(3, {'sin_phi': 10.0}, 60), StaticCompensation(3, {'sin_phi': 11.0}, 60))
        self.assertNotEqual(StaticCompensation(3, {'sin_phi': 10.0}, 60), StaticCompensation(3, {'sin_phi': 10.0}, 50))
        self.assertEqual("StaticCompensation(joint=3, terms={'sin_phi': 10.0}, max_positions=60)",
                         repr(StaticCompensation(3, {'sin_phi': 10.0}, 60)))

    def test_validation(self):
        with self.assertRaises(ValueError):
            StaticCompensation(2, {'nope': 1.0}, 60)
        with self.assertRaises(ValueError):
            StaticCompensation(1, {}, 60)
        with self.assertRaises(ValueError):
            StaticCompensation(2, {'cos_q2': 1.0}, 0)
        with self.assertRaisesRegex(ValueError, 'no terms'):
            StaticCompensation(2, {'spring_dl': 1.0}, 60)  # the term of the removed spring model
        with self.assertRaisesRegex(ValueError, 'no terms'):
            StaticCompensation(3, {'cos_q2': 1.0}, 60)  # cos_q2 is a term of joint 2

    def test_empty_terms_give_zero(self):
        self.assertEqual(0, StaticCompensation(2, {}, 60).positions(np.zeros(7), self.P))

    def test_the_conversion_factor_is_shared_with_the_gain_tuning(self):
        from src.robot_controllers.dynamixel_robot import dynamixel_x_config
        from src.robot_controllers.dynamixel_robot.calibration import gain_tuning
        from src.robot_controllers.dynamixel_robot.calibration import fit_static_load
        self.assertEqual(1.0 / 128.0, dynamixel_x_config.PWM_PER_POSITION_PER_P)
        self.assertIs(dynamixel_x_config.PWM_PER_POSITION_PER_P, gain_tuning.PWM_PER_POSITION_PER_P)
        self.assertIs(dynamixel_x_config.PWM_PER_POSITION_PER_P, fit_static_load.PWM_PER_POSITION_PER_P)


if __name__ == '__main__':
    unittest.main()
