import contextlib
import csv
import io
import os
import tempfile
import unittest
from math import pi

import numpy as np

from src.robot_controllers.dynamixel_robot.calibration import fit_static_load
from src.robot_controllers.dynamixel_robot.calibration.load_directions import DOWN, UP
from src.robot_controllers.dynamixel_robot.static_compensation import regressor_row

FIELDS = ['time', 'pose', 'direction', 'servo', 'angle', 'goal_position', 'compensation', 'present_position', 'error',
          'current_raw', 'current', 'current_unit', 'temperature', 'voltage', 'mode', 'speed', 'samples', 'p', 'i', 'd',
          'pwm']

TRUE = {2: {'constant': 1.0, 'cos_q2': -12.0, 'sin_q2': 1.5, 'sin_phi': -5.0, 'cos_phi': 0.8, 'wrist_reach': -2.0},
        3: {'constant': -2.0, 'sin_phi': 30.0, 'cos_phi': -3.0, 'wrist_reach': 6.0},
        5: {'constant': 0.0, 'wrist_pitch': 8.0}}
FRICTION = 4.0  # pwm units
SPEED_DRIVE = 6.0  # pwm units: what the motor needs to turn at the sweep speed, which the lead supplies


def row(pose, direction, servo, angle, p, error, compensation, lead=None):
    """a row like the recorders write, with a lead column only when lead is not None"""
    result = {'time': '1.0', 'pose': str(pose), 'direction': direction, 'servo': str(servo),
              'angle': '{:.6f}'.format(angle), 'goal_position': '2048', 'compensation': str(compensation),
              'present_position': '', 'error': '{:.6f}'.format(error), 'current_raw': '', 'current': '',
              'current_unit': '', 'temperature': '', 'voltage': '', 'mode': 'sweep', 'speed': '5', 'samples': '5',
              'p': '{:g}'.format(p), 'i': '0', 'd': '0', 'pwm': ''}
    if lead is not None:
        result['lead'] = lead
    return result


def sweep_rows(count, with_lead, lead_column='value', noise=0.0, seed=0):
    """
    the rows of a sweep of joints 2, 3 and 5 from the true gravity model: the drive of a direction is the static load
    plus or minus the drive of friction and of the speed. The servo drives with P / 128 * (goal sent - present), the
    goal sent is the trajectory plus the compensation plus the lead, and the error is measured against the trajectory
    alone, so error = compensation + lead - drive * 128 / P. The lead supplies the drive of the speed: lead * P / 128 =
    sign * SPEED_DRIVE.
    :param with_lead: record with the lead (the errors are then smaller by the lead in the direction of motion)
    :param lead_column: 'value' for the lead as it is, 'blank' for an empty lead column, 'absent' for no column
    """
    rng = np.random.default_rng(seed)
    compensations = np.random.default_rng(99)
    rows = []
    for pose in range(count):
        angles = np.zeros(7)
        angles[1] = pi / 2
        angles[2], angles[3] = rng.uniform(0.2, 2.9), rng.uniform(-1.3, 1.3)
        angles[4], angles[5] = rng.uniform(-1.0, 1.0), rng.uniform(-1.5, 1.5)
        p = float(rng.choice((1000.0, 2000.0)))
        for direction, sign in ((UP, 1), (DOWN, -1)):
            for servo in (2, 3, 5):
                names = tuple(TRUE[servo])
                static = float(regressor_row(servo, angles, names) @ np.array([TRUE[servo][name] for name in names]))
                drive = static + sign * (FRICTION + SPEED_DRIVE) + rng.normal(0, noise)
                added = int(compensations.integers(-12, 13))
                lead = sign * SPEED_DRIVE * 128 / p if with_lead else 0.0
                error = added + lead - drive * 128 / p
                cell = {'value': '{:.6f}'.format(lead), 'blank': '', 'absent': None}[lead_column]
                rows.append(row(pose, direction, servo, angles[servo], p, error, added, cell))
            # the joints that are not fitted still give their angles to the pose
            for servo in (1, 4, 6):
                still = {'value': '0', 'blank': '', 'absent': None}[lead_column]
                rows.append(row(pose, direction, servo, angles[servo], p, 0.0, 0, still))
    return rows


class DriveTests(unittest.TestCase):

    def test_the_drive_takes_the_lead_out_of_the_error(self):
        # the goal sent was 2 + 3 = 5 positions ahead of the trajectory and the joint rests 1 position below the
        # trajectory: goal sent - present = 6 positions, at P 1280 that is 6 * 1280 / 128 = 60 pwm, upwards
        self.assertAlmostEqual(60.0, fit_static_load._drive(row(0, UP, 2, 0.0, 1280, -1.0, 2, lead='3.0'), 1280.0))

    def test_without_a_lead_the_drive_is_what_it_was(self):
        for lead in (None, '', '  ', '0', '0.0'):
            r = row(0, UP, 2, 0.0, 1280, -1.0, 2, lead=lead)
            self.assertEqual(-(-1.0 - 2.0) * 1280.0 / 128, fit_static_load._drive(r, 1280.0), repr(lead))

    def test_a_lead_that_is_not_a_number_is_an_error(self):
        with self.assertRaises(ValueError):
            fit_static_load._drive(row(0, UP, 2, 0.0, 1280, -1.0, 2, lead='fast'), 1280.0)

    def test_a_lead_has_opposite_signs_and_so_adds_to_the_drive_of_both_directions(self):
        # moving up the lead is positive and raises the drive, moving down it is negative and lowers it
        up = fit_static_load._drive(row(0, UP, 2, 0.0, 1280, 0.0, 0, lead='2.0'), 1280.0)
        down = fit_static_load._drive(row(0, DOWN, 2, 0.0, 1280, 0.0, 0, lead='-2.0'), 1280.0)
        self.assertAlmostEqual(20.0, up)
        self.assertAlmostEqual(-20.0, down)

    def test_the_servos_with_a_lead(self):
        rows = [row(0, UP, 2, 0.0, 1000, 0.0, 0, lead='1.0'), row(0, UP, 3, 0.0, 1000, 0.0, 0, lead='0'),
                row(0, UP, 5, 0.0, 1000, 0.0, 0, lead=''), row(0, UP, 1, 0.0, 1000, 0.0, 0),
                row(0, DOWN, 3, 0.0, 1000, 0.0, 0, lead='-0.5')]
        self.assertEqual({2, 3}, fit_static_load.servos_with_lead(rows))
        # a row without an error does not count
        blank = row(0, UP, 4, 0.0, 1000, 0.0, 0, lead='1.0')
        blank['error'] = ''
        self.assertEqual(set(), fit_static_load.servos_with_lead([blank]))


class SweepWithLeadTests(unittest.TestCase):

    def samples(self, rows):
        return fit_static_load.samples_from_rows(rows)

    def test_the_static_load_and_the_half_difference_are_the_same_with_and_without_a_lead(self):
        with_lead = self.samples(sweep_rows(20, with_lead=True))
        without = self.samples(sweep_rows(20, with_lead=False))
        self.assertEqual(20, len(with_lead))
        for a, b in zip(with_lead, without):
            for servo in (2, 3, 5):
                self.assertAlmostEqual(b.static[servo], a.static[servo], places=3)
                self.assertAlmostEqual(b.hysteresis[servo], a.hysteresis[servo], places=3)
                # all of the drive that does not come from gravity: friction and the drive of the speed
                self.assertAlmostEqual(FRICTION + SPEED_DRIVE, a.hysteresis[servo], places=3)

    def test_the_fitted_coefficients_are_the_same_with_and_without_a_lead(self):
        rows_with, rows_without = sweep_rows(40, True, noise=0.7), sweep_rows(40, False, noise=0.7)
        samples_with, samples_without = self.samples(rows_with), self.samples(rows_without)
        for joint in (2, 3, 5):
            fit_with = fit_static_load.fit_joint(joint, samples_with, 0.2, np.random.default_rng(0))
            fit_without = fit_static_load.fit_joint(joint, samples_without, 0.2, np.random.default_rng(0))
            for name, value in fit_without.coefficients.items():
                self.assertAlmostEqual(value, fit_with.coefficients[name], places=3, msg='{} {}'.format(joint, name))
            self.assertAlmostEqual(fit_without.rms_train, fit_with.rms_train, places=3)

    def test_the_fit_recovers_the_true_coefficients_from_a_sweep_with_a_lead(self):
        samples = self.samples(sweep_rows(40, True))
        for joint in (2, 3, 5):
            fit = fit_static_load.fit_joint(joint, samples, 0.0)
            for name, value in TRUE[joint].items():
                self.assertAlmostEqual(value, fit.coefficients[name], places=2, msg='{} {}'.format(joint, name))

    def test_a_drive_without_the_lead_term_would_lose_the_speed_drive_from_the_half_difference(self):
        # what the old formula gave for the rows with a lead: only the friction is left of the half difference
        rows = sweep_rows(5, with_lead=True)
        by_key = {}
        for r in rows:
            if r['servo'] == '5':
                drive = -(float(r['error']) - float(r['compensation'])) * float(r['p']) / 128
                by_key.setdefault(r['pose'], {})[r['direction']] = drive
        for drives in by_key.values():
            self.assertAlmostEqual(FRICTION, (drives[UP] - drives[DOWN]) / 2, places=3)
        for sample in self.samples(rows):
            self.assertAlmostEqual(FRICTION + SPEED_DRIVE, sample.hysteresis[5], places=3)

    def test_recordings_without_a_lead_column_give_the_numbers_they_gave_before(self):
        absent = sweep_rows(30, with_lead=False, lead_column='absent', noise=0.5)
        for column in ('blank', 'value'):  # a blank lead and a lead of 0 mean the same as no column
            other = sweep_rows(30, with_lead=False, lead_column=column, noise=0.5)
            if column == 'value':
                self.assertTrue(all(float(r['lead']) == 0.0 for r in other))
            samples_absent, samples_other = self.samples(absent), self.samples(other)
            for a, b in zip(samples_absent, samples_other):
                self.assertEqual(a.static, b.static)
                self.assertEqual(a.hysteresis, b.hysteresis)
            for joint in (2, 3, 5):
                fit_absent = fit_static_load.fit_joint(joint, samples_absent, 0.2, np.random.default_rng(0))
                fit_other = fit_static_load.fit_joint(joint, samples_other, 0.2, np.random.default_rng(0))
                self.assertEqual(fit_absent.coefficients, fit_other.coefficients)
                self.assertEqual(fit_absent.rms_train, fit_other.rms_train)
                self.assertEqual(fit_absent.rms_holdout, fit_other.rms_holdout)

    def test_the_numbers_of_rows_without_the_column_are_those_of_the_old_formula(self):
        rows = sweep_rows(10, with_lead=False, lead_column='absent', noise=0.5)
        self.assertTrue(all('lead' not in r for r in rows))
        expected: dict[tuple[str, str], float] = {}
        for r in rows:
            expected[(r['pose'], r['direction'] + r['servo'])] = (
                -(float(r['error']) - float(r['compensation'])) * float(r['p']) * (1 / 128))
        for r in rows:
            self.assertEqual(expected[(r['pose'], r['direction'] + r['servo'])],
                             fit_static_load._drive(r, float(r['p'])))


class MainTests(unittest.TestCase):

    def run_main(self, rows, *argv):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'sweep.csv')
            with open(path, 'w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=FIELDS + ['lead'], extrasaction='ignore', restval='')
                writer.writeheader()
                writer.writerows(rows)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = fit_static_load.main([path, '--joints', '5', *argv])
        self.assertEqual(0, result)
        return output.getvalue()

    def friction_line(self, text):
        return next(line for line in text.splitlines() if 'up/down difference' in line)

    def test_the_report_says_friction_and_speed_drive_when_a_lead_was_active(self):
        text = self.run_main(sweep_rows(20, with_lead=True))
        line = self.friction_line(text)
        self.assertTrue(line.startswith('  friction and speed drive (half the up/down difference, not compensated): '),
                        line)
        self.assertIn('mean 10.0 pwm', line)  # friction 4 + speed 6, whichever way it was recorded

    def test_the_report_is_unchanged_without_a_lead(self):
        text = self.run_main(sweep_rows(20, with_lead=False, lead_column='absent'))
        line = self.friction_line(text)
        self.assertTrue(line.startswith('  friction (half the up/down difference, not compensated): '), line)
        # a lead column of zeros, a blank one and no column print the same report
        for column in ('value', 'blank'):
            self.assertEqual(text.replace('sweep.csv', ''), self.run_main(
                sweep_rows(20, with_lead=False, lead_column=column)).replace('sweep.csv', ''))

    def test_the_whole_report_is_the_same_with_and_without_a_lead_except_for_the_friction_wording(self):
        with_lead = self.run_main(sweep_rows(20, with_lead=True)).splitlines()
        without = self.run_main(sweep_rows(20, with_lead=False)).splitlines()
        # the friction line is the one that differs in wording, the numbers of every line are the same
        self.assertEqual(len(with_lead), len(without))
        for a, b in zip(with_lead, without):
            if 'up/down difference' in a:
                self.assertEqual(a.replace('friction and speed drive', 'friction'), b)
            else:
                self.assertEqual(a, b)


if __name__ == '__main__':
    unittest.main()
