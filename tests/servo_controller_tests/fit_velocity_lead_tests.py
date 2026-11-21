import contextlib
import csv
import io
import os
import tempfile
import unittest

from src.robot_controllers.dynamixel_robot.calibration import gain_tuning
from src.robot_controllers.dynamixel_robot.calibration.fit_velocity_lead import (
    LAG_TOLERANCE, GroupKey, SpeedStatistics, analyse_group, describe, describe_fit, fit_group, fit_lines, group_p,
    group_rows, intended_line, is_flat, lead_meaning, main, speed_statistics, table, yaml_block)
from src.robot_controllers.dynamixel_robot.calibration.gain_tuning import Sample, SweepSample, cruise_statistics

COLUMNS = ['joint', 'pose', 'speed', 'time', 'direction', 'cruise', 'goal', 'position', 'position_trajectory',
           'velocity', 'velocity_trajectory', 'pwm', 'current', 'voltage', 'temperature', 'p', 'lead', 'lead_gain']
SPEEDS = (5.0, 10.0, 15.0)


def sweep_rows(joint=1, k=10.0, friction=20.0, gravity=0.0, p=2000.0, lead_gain=0.0, speeds=SPEEDS, count=6,
               jitter=0.0, pwm_scale=1.0, file='a.csv', cruise=1, directions=(1, -1)):
    """
    the cruise rows of a sweep of a joint at every speed in both directions, for a servo that obeys the model: the
    drive in the direction of the move is friction + k * speed, gravity adds the same drive in both directions of the
    world (so it flips against the direction), and the servo drives with P / 128 * (trajectory - present) where the
    trajectory includes the lead, so the error is the drive times 128 / P.
    :param lead_gain: the coefficient of the lead in pwm per degree/s, 0 for rows without a lead
    :param jitter: alternate rows have the error and the pwm this far above and below the model (a std of the jitter)
    :param pwm_scale: the measured pwm is this times the drive that makes the error
    """
    rows = []
    for speed in speeds:
        for direction in directions:
            for i in range(count):
                wobble = jitter * (1 if i % 2 == 0 else -1)
                drive = direction * (friction + k * speed) + gravity
                error = drive * 128.0 / p + wobble  # trajectory - position, signed like the position
                rows.append({'file': file, 'joint': joint, 'pose': 1, 'speed': speed, 'time': 0.1 * i,
                             'direction': direction, 'cruise': cruise, 'goal': 1000, 'position': 1000,
                             'position_trajectory': 1000 + error, 'velocity': 0, 'velocity_trajectory': 0,
                             'pwm': drive * pwm_scale + wobble, 'current': 0, 'voltage': 11.5, 'temperature': 30,
                             'p': p, 'lead': direction * lead_gain * speed * 128.0 / p, 'lead_gain': lead_gain})
    return rows


def blank(rows, column):
    """the rows with the cells of a column empty"""
    return [dict(row, **{column: ''}) for row in rows]


def as_text(rows):
    """the rows the way read_rows gives them: every value a string"""
    return [{key: str(value) for key, value in row.items()} for row in rows]


def fit_of(rows, joint=1, lead_gain=0.0):
    return analyse_group(GroupKey('a.csv', joint, lead_gain), as_text(rows))


def write_csv(path, rows, without=()):
    """write the rows, without the columns named (a recording that lacks them)"""
    columns = [column for column in COLUMNS if column not in without]
    with open(path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def run_main(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = main(list(argv))
    return code, out.getvalue()


class FitTests(unittest.TestCase):

    def test_both_routes_recover_the_coefficient_and_the_ratio_is_one(self):
        result = fit_of(sweep_rows(k=10.0, friction=20.0, p=2000.0))
        fit = result.fit
        self.assertEqual(result.problem, '')
        self.assertAlmostEqual(fit.k_pwm, 10.0, places=6)
        self.assertAlmostEqual(fit.k_lag, 10.0, places=6)
        self.assertAlmostEqual(fit.ratio, 1.0, places=6)
        self.assertAlmostEqual(fit.coefficient, fit.k_lag, places=9)

    def test_friction_is_found_in_pwm_and_in_positions(self):
        fit = fit_of(sweep_rows(k=10.0, friction=20.0, p=2000.0)).fit
        self.assertAlmostEqual(fit.friction_pwm, 20.0, places=6)
        self.assertAlmostEqual(fit.friction_positions, 20.0 * 128 / 2000.0, places=6)
        self.assertAlmostEqual(fit.slope, 10.0 * 128 / 2000.0, places=6)

    def test_gravity_cancels_in_both_fits(self):
        without = fit_of(sweep_rows(gravity=0.0)).fit
        with_gravity = fit_of(sweep_rows(gravity=40.0)).fit
        for name in ('k_pwm', 'k_lag', 'friction_pwm', 'friction_positions', 'slope'):
            self.assertAlmostEqual(getattr(without, name), getattr(with_gravity, name), places=6, msg=name)

    def test_the_statistics_are_in_the_direction_of_the_move(self):
        statistics, notes = speed_statistics(as_text(sweep_rows(k=10.0, friction=20.0, p=2000.0)))
        self.assertEqual(notes, [])
        self.assertEqual([s.speed for s in statistics], [5.0, 10.0, 15.0])
        self.assertAlmostEqual(statistics[1].pwm, 20.0 + 10.0 * 10.0, places=6)
        self.assertAlmostEqual(statistics[1].error, (20.0 + 100.0) * 128 / 2000.0, places=6)
        self.assertEqual(statistics[1].samples, 12)

    def test_the_signs_agree_with_cruise_statistics_of_the_feedforward_stage(self):
        rows = sweep_rows(speeds=(10.0,), gravity=25.0, jitter=1.5, count=10)
        sweep = [SweepSample(Sample(r['time'], 0, r['goal'], r['position'], r['position_trajectory'], 0, 0, r['pwm'],
                                    0, 0, 11.5, 30), r['direction'], True, 0) for r in rows]
        stage = cruise_statistics(sweep)
        mine = speed_statistics(as_text(rows))[0][0]
        self.assertAlmostEqual(mine.error, stage.lag, places=6)
        self.assertAlmostEqual(mine.pwm, stage.pwm, places=6)
        self.assertAlmostEqual(mine.error_std, stage.lag_std, places=6)
        self.assertAlmostEqual(mine.pwm_std, stage.pwm_std, places=6)

    def test_the_stds_are_given_and_the_means_do_not_move_with_them(self):
        steady = speed_statistics(as_text(sweep_rows()))[0][1]
        rough = speed_statistics(as_text(sweep_rows(jitter=2.0)))[0][1]
        self.assertAlmostEqual(steady.pwm, rough.pwm, places=6)
        self.assertAlmostEqual(steady.error, rough.error, places=6)
        self.assertAlmostEqual(steady.pwm_std, 0.0, places=6)
        self.assertAlmostEqual(rough.pwm_std, 2.0, places=6)
        self.assertAlmostEqual(rough.error_std, 2.0, places=6)


class LeadTests(unittest.TestCase):

    def test_a_lead_does_not_change_the_fitted_coefficient(self):
        plain = fit_of(sweep_rows(k=12.0)).fit
        led = fit_of(sweep_rows(k=12.0, lead_gain=12.0), lead_gain=12.0).fit
        self.assertAlmostEqual(led.k_pwm, plain.k_pwm, places=6)
        self.assertAlmostEqual(led.k_lag, plain.k_lag, places=6)
        self.assertAlmostEqual(led.k_lag, 12.0, places=6)

    def test_the_lag_behind_the_intended_trajectory_is_flat_with_the_right_lead(self):
        fit = fit_of(sweep_rows(k=12.0, lead_gain=12.0), lead_gain=12.0).fit
        self.assertAlmostEqual(fit.intended_change, 0.0, places=6)
        statistics, _ = speed_statistics(as_text(sweep_rows(k=12.0, friction=20.0, lead_gain=12.0)))
        for s in statistics:
            self.assertAlmostEqual(s.intended, 20.0 * 128 / 2000.0, places=6)
            self.assertGreater(s.lead, 0)
        text = describe(fit_of(sweep_rows(k=12.0, lead_gain=12.0), lead_gain=12.0), 30)
        self.assertIn('flat', text)
        self.assertIn('lead gain 12 pwm per degree/s', text)

    def test_without_a_lead_the_intended_lag_is_the_error_and_grows_with_the_speed(self):
        statistics, _ = speed_statistics(as_text(sweep_rows(k=12.0)))
        for s in statistics:
            self.assertEqual(s.lead, 0.0)
            self.assertEqual(s.intended, s.error)
        fit = fit_of(sweep_rows(k=12.0)).fit
        self.assertAlmostEqual(fit.intended_change, 12.0 * 128 / 2000.0 * 10.0, places=6)
        text = describe(fit_of(sweep_rows(k=12.0)), 30)
        self.assertIn('more than 1 position', text)
        self.assertIn('no lead', text)

    def test_a_lead_that_is_too_small_leaves_a_slope_in_the_intended_lag(self):
        fit = fit_of(sweep_rows(k=12.0, lead_gain=6.0), lead_gain=6.0).fit
        self.assertAlmostEqual(fit.intended_change, 6.0 * 128 / 2000.0 * 10.0, places=6)

    def test_a_blank_lead_cell_counts_as_no_lead(self):
        rows = blank(sweep_rows(lead_gain=5.0), 'lead')
        self.assertEqual({s.lead for s in speed_statistics(as_text(rows))[0]}, {0.0})

    def test_the_slope_that_is_left_in_the_intended_lag_is_the_coefficient_the_lead_lacks(self):
        # a lead of 6 where the joint needs 12 leaves 6 pwm per degree/s, one of 20 has 8 too many, the right one none
        for lead_gain, lacking in ((6.0, 6.0), (20.0, -8.0), (12.0, 0.0), (0.0, 12.0)):
            with self.subTest(lead_gain=lead_gain):
                fit = fit_of(sweep_rows(k=12.0, lead_gain=lead_gain), lead_gain=lead_gain).fit
                self.assertAlmostEqual(fit.leftover_coefficient, lacking, places=6)
                self.assertAlmostEqual(fit.intended_slope * 2000.0 / 128.0, lacking, places=6)
                self.assertAlmostEqual(lead_gain + fit.leftover_coefficient, 12.0, places=6)

    def test_without_a_p_there_is_no_leftover_coefficient(self):
        fit = fit_of(blank(sweep_rows(k=12.0, lead_gain=6.0), 'p'), lead_gain=6.0).fit
        self.assertIsNone(fit.leftover_coefficient)

    def test_flat_is_a_change_within_the_tolerance_of_the_feedforward_stage(self):
        self.assertEqual(LAG_TOLERANCE, gain_tuning.FF_LAG_TOLERANCE)
        self.assertTrue(is_flat(fit_of(sweep_rows(k=12.0, lead_gain=12.0), lead_gain=12.0).fit))
        self.assertFalse(is_flat(fit_of(sweep_rows(k=12.0)).fit))
        # a lead short by 1.5 pwm per degree/s changes the lag by 1.5 * 128 / 2000 positions per degree/s, 10 degrees/s
        # of speed apart (5 to 15) 0.96 positions, 1.6 short is 1.02
        close = fit_of(sweep_rows(k=12.0, lead_gain=10.5), lead_gain=10.5).fit
        self.assertAlmostEqual(0.96, close.intended_change, places=6)
        self.assertTrue(is_flat(close))
        self.assertFalse(is_flat(fit_of(sweep_rows(k=12.0, lead_gain=10.4), lead_gain=10.4).fit))

    def test_describe_fit_is_the_two_fits_the_lag_line_and_the_yaml_block(self):
        fit = fit_of(sweep_rows(k=12.0, lead_gain=6.0), lead_gain=6.0).fit
        text = describe_fit(fit, 30)
        parts = fit_lines(fit) + [intended_line(fit)]
        self.assertTrue(text.startswith('\n'.join(parts)), 'the pieces the feedforward stage prints are these')
        self.assertIn('pwm_per_degree_per_second: 12.00', text)
        self.assertEqual(1, text.count('ratio k_pwm / k_lag'))


class IgnoredRowsTests(unittest.TestCase):

    def test_rows_that_are_not_cruise_are_ignored(self):
        rows = as_text(sweep_rows(k=10.0) + sweep_rows(k=500.0, friction=300.0, gravity=200.0, cruise=0))
        fit = analyse_group(GroupKey('a.csv', 1, 0.0), rows).fit
        self.assertAlmostEqual(fit.k_pwm, 10.0, places=6)
        self.assertAlmostEqual(fit.k_lag, 10.0, places=6)
        statistics, _ = speed_statistics(rows)
        self.assertEqual(statistics[0].samples, 12)


class GainTests(unittest.TestCase):

    def test_the_p_column_is_the_p_of_the_group(self):
        result = fit_of(sweep_rows(p=2500.0))
        self.assertEqual(result.p, 2500.0)
        self.assertAlmostEqual(result.fit.k_lag, 10.0, places=6)
        self.assertAlmostEqual(result.fit.ratio, 1.0, places=6)

    def test_a_blank_p_column_is_no_p(self):
        rows = as_text(blank(sweep_rows(p=2500.0), 'p'))
        self.assertEqual(group_p(rows), (None, ''))

    def test_a_p_that_changes_within_a_group_is_averaged_with_a_remark(self):
        rows = as_text(sweep_rows(p=2000.0, count=2))[:6] + as_text(sweep_rows(p=3000.0, count=2))[6:]
        p, remark = group_p(rows)
        self.assertTrue(2000.0 < p < 3000.0)
        self.assertIn('2000, 3000', remark)

    def test_without_any_p_only_the_pwm_result_is_given(self):
        result = fit_of(blank(sweep_rows(k=10.0), 'p'))
        fit = result.fit
        self.assertIsNone(result.p)
        self.assertIsNone(fit.k_lag)
        self.assertIsNone(fit.ratio)
        self.assertAlmostEqual(fit.coefficient, 10.0, places=6)
        text = describe(result, 30)
        self.assertIn('needs the P', text)
        self.assertIn('P not known', text)
        self.assertIn('pwm_per_degree_per_second: 10.00', text)
        self.assertNotIn('ratio', text)
        self.assertNotIn('the lead is', text)

    def test_the_ratio_is_off_and_warned_about_when_the_pwm_does_not_follow_the_error(self):
        result = fit_of(sweep_rows(k=10.0, pwm_scale=0.5))
        self.assertAlmostEqual(result.fit.ratio, 0.5, places=6)
        self.assertIn('warning: the measured pwm and the lag disagree', describe(result, 30))
        # the value for the yaml is still the one from the lag
        self.assertAlmostEqual(result.fit.coefficient, 10.0, places=6)

    def test_a_ratio_within_15_percent_is_not_warned_about(self):
        result = fit_of(sweep_rows(k=10.0, pwm_scale=1.1))
        self.assertAlmostEqual(result.fit.ratio, 1.1, places=6)
        self.assertNotIn('warning', describe(result, 30))


class SpeedsTests(unittest.TestCase):

    def test_a_speed_with_one_direction_is_left_out_with_a_line(self):
        rows = sweep_rows(speeds=(5.0, 10.0)) + sweep_rows(speeds=(15.0,), directions=(1,))
        statistics, notes = speed_statistics(as_text(rows))
        self.assertEqual([s.speed for s in statistics], [5.0, 10.0])
        self.assertEqual(len(notes), 1)
        self.assertIn('speed 15 degrees/s left out', notes[0])
        self.assertIn('6 cruise samples going up and 0 going down', notes[0])

    def test_a_direction_with_too_few_cruise_samples_leaves_the_speed_out(self):
        rows = sweep_rows(speeds=(5.0, 10.0)) + sweep_rows(speeds=(15.0,), count=2)
        statistics, notes = speed_statistics(as_text(rows))
        self.assertEqual([s.speed for s in statistics], [5.0, 10.0])
        self.assertIn('2 cruise samples going up and 2 going down', notes[0])

    def test_the_minimum_is_the_one_of_the_feedforward_stage(self):
        from src.robot_controllers.dynamixel_robot.calibration.gain_tuning import MIN_CRUISE_SAMPLES
        rows = as_text(sweep_rows(speeds=(10.0,), count=MIN_CRUISE_SAMPLES))
        self.assertEqual(len(speed_statistics(rows)[0]), 1)
        rows = as_text(sweep_rows(speeds=(10.0,), count=MIN_CRUISE_SAMPLES - 1))
        self.assertEqual(len(speed_statistics(rows)[0]), 0)

    def test_the_fit_goes_on_with_the_speeds_that_remain(self):
        rows = sweep_rows(speeds=(5.0, 10.0)) + sweep_rows(speeds=(15.0,), directions=(-1,))
        result = fit_of(rows)
        self.assertEqual(result.problem, '')
        self.assertEqual(len(result.notes), 1)
        self.assertAlmostEqual(result.fit.k_pwm, 10.0, places=6)

    def test_fewer_than_two_speeds_is_said_and_not_fitted(self):
        result = fit_of(sweep_rows(speeds=(10.0,)))
        self.assertIsNone(result.fit)
        self.assertIn('at least two are needed', result.problem)
        self.assertIn('no fit: ', describe(result, 30))

    def test_two_speeds_are_enough_and_one_left_over_is_not(self):
        two = fit_of(sweep_rows(speeds=(5.0, 15.0)))
        self.assertAlmostEqual(two.fit.k_pwm, 10.0, places=6)
        one_left = fit_of(sweep_rows(speeds=(5.0,)) + sweep_rows(speeds=(10.0,), directions=(1,)))
        self.assertIsNone(one_left.fit)

    def test_fit_group_raises_for_one_speed(self):
        statistics = [SpeedStatistics(10.0, 12, 100.0, 5.0, 0.0, 1.0, 1.0)]
        with self.assertRaises(ValueError):
            fit_group(1, 0.0, statistics, 2000.0)

    def test_no_cruise_rows_at_all_gives_no_statistics_and_a_note_per_speed(self):
        statistics, notes = speed_statistics(as_text(sweep_rows(cruise=0)))
        self.assertEqual(statistics, [])
        self.assertEqual(len(notes), 3)


class GroupingTests(unittest.TestCase):

    def test_rows_are_grouped_by_file_joint_and_lead_gain(self):
        rows = as_text(sweep_rows(joint=2, file='b.csv', lead_gain=8.0, count=1)
                       + sweep_rows(joint=2, file='b.csv', count=1)
                       + sweep_rows(joint=1, file='a.csv', count=1)
                       + sweep_rows(joint=3, file='a.csv', count=1)
                       + sweep_rows(joint=1, file='a.csv', lead_gain=4.0, count=1))
        groups = group_rows(rows)
        # the files in the order they were given (a.csv came after b.csv in the rows), then the joints, the rows
        # without a lead before those with one
        self.assertEqual(list(groups), [GroupKey('b.csv', 2, 0.0), GroupKey('b.csv', 2, 8.0),
                                        GroupKey('a.csv', 1, 0.0), GroupKey('a.csv', 1, 4.0),
                                        GroupKey('a.csv', 3, 0.0)])
        self.assertEqual({key: len(members) for key, members in groups.items()}[GroupKey('a.csv', 1, 4.0)], 6)

    def test_a_recording_with_and_without_a_lead_is_two_fits_that_agree(self):
        rows = as_text(sweep_rows(k=12.0) + sweep_rows(k=12.0, lead_gain=12.0))
        groups = group_rows(rows)
        self.assertEqual(len(groups), 2)
        fits = [analyse_group(key, members).fit for key, members in groups.items()]
        self.assertAlmostEqual(fits[0].k_lag, fits[1].k_lag, places=6)
        self.assertAlmostEqual(fits[0].intended_change, 12.0 * 128 / 2000.0 * 10.0, places=6)
        self.assertAlmostEqual(fits[1].intended_change, 0.0, places=6)

    def test_the_same_joint_in_two_files_is_two_groups(self):
        rows = as_text(sweep_rows(file='a.csv', count=1) + sweep_rows(file='b.csv', count=1))
        self.assertEqual(len(group_rows(rows)), 2)

    def test_the_lead_gain_of_a_blank_cell_is_0(self):
        rows = as_text(blank(sweep_rows(count=1, lead_gain=3.0), 'lead_gain'))
        self.assertEqual(list(group_rows(rows)), [GroupKey('a.csv', 1, 0.0)])


class TextTests(unittest.TestCase):

    def test_the_yaml_block(self):
        self.assertEqual(yaml_block(2, 14.2857, 30),
                         '  2:\n'
                         '    velocity_feedforward:\n'
                         '      max_positions: 30\n'
                         '      pwm_per_degree_per_second: 14.29\n')

    def test_the_lead_in_positions_at_10_degrees_per_second_and_the_time_in_ms(self):
        # 10 pwm per degree/s at P 1280 is 1.0 position per degree/s
        text = lead_meaning(10.0, 1280.0)
        self.assertIn('at P 1280', text)
        self.assertIn('10.0 positions at 10 degrees/s', text)
        self.assertIn('88 ms', text)  # 1 position per degree/s = 360 / 4096 * 1000 ms = 87.9
        # the numbers of joint 1 on 18 November 2026: 59 ms of travel at P 2000
        self.assertIn('59 ms', lead_meaning(10.49, 2000.0))

    def test_the_report_has_the_table_the_fits_the_ratio_the_friction_and_the_yaml(self):
        text = describe(fit_of(sweep_rows(k=10.0, friction=20.0)), 25)
        for expected in ('a.csv: joint 1, no lead, P 2000', 'samples', 'intended',
                         'pwm = 20.0 + 10.000 * speed', 'error = 1.28 + 0.6400 * speed', '56.2 ms of travel',
                         'ratio k_pwm / k_lag = 1.00', 'friction part', '20.0 pwm, 1.28 positions',
                         'max_positions: 25', 'pwm_per_degree_per_second: 10.00',
                         'the lead is 6.4 positions at 10 degrees/s'):
            self.assertIn(expected, text)

    def test_the_table_has_a_row_per_speed(self):
        statistics, _ = speed_statistics(as_text(sweep_rows()))
        lines = table(statistics).splitlines()
        self.assertEqual(len(lines), 2 + 3)
        self.assertTrue(lines[2].split()[0] == '5' and lines[4].split()[0] == '15')

    def test_the_notes_of_left_out_speeds_are_in_the_report(self):
        rows = sweep_rows(speeds=(5.0, 10.0)) + sweep_rows(speeds=(15.0,), directions=(1,))
        self.assertIn('note: speed 15 degrees/s left out', describe(fit_of(rows), 30))

    def test_the_source_of_the_yaml_value_is_named(self):
        self.assertIn('from the lag', describe(fit_of(sweep_rows()), 30))
        self.assertIn('from the measured pwm', describe(fit_of(blank(sweep_rows(), 'p')), 30))


class MainTests(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def path(self, name):
        return os.path.join(self.directory.name, name)

    def test_a_missing_file_gives_1_and_a_message(self):
        code, output = run_main(self.path('missing.csv'))
        self.assertEqual(code, 1)
        self.assertIn('missing.csv', output)
        self.assertNotIn('Traceback', output)

    def test_a_file_that_is_not_a_recording_gives_1(self):
        path = self.path('other.csv')
        with open(path, 'w', newline='') as csv_file:
            csv_file.write('a,b\n1,2\n')
        code, output = run_main(path)
        self.assertEqual(code, 1)
        self.assertIn('no column', output)

    def test_a_row_with_a_value_that_is_not_a_number_gives_1(self):
        path = self.path('bad.csv')
        rows = sweep_rows(count=1)
        rows[0]['pwm'] = 'x'
        write_csv(path, rows)
        code, output = run_main(path)
        self.assertEqual(code, 1)
        self.assertIn('pwm', output)

    def test_nothing_to_fit_gives_1(self):
        path = self.path('one_speed.csv')
        write_csv(path, sweep_rows(speeds=(10.0,)))
        code, output = run_main(path)
        self.assertEqual(code, 1)
        self.assertIn('no fit', output)
        self.assertIn('nothing could be fitted', output)

    def test_a_file_without_a_column_the_stage_logs_since_21_november_is_refused_naming_it(self):
        for column in ('lead_gain', 'lead', 'p'):
            with self.subTest(column=column):
                path = self.path('without_{}.csv'.format(column))
                write_csv(path, sweep_rows(lead_gain=5.0), without=(column,))
                code, output = run_main(path)
                self.assertEqual(1, code)
                self.assertEqual('{}: no column {}, the recording was made before the stage logged the gain, the lead '
                                 'and its coefficient (21 November 2026): record the joint again with '
                                 'tune_servo_gains.py --ff-only\n'.format(path, column), output)
                self.assertNotIn('Traceback', output)

    def test_a_recording_of_before_21_november_lacks_all_three_columns_and_is_refused(self):
        path = self.path('old.csv')
        write_csv(path, sweep_rows(), without=('p', 'lead', 'lead_gain'))
        code, output = run_main(path)
        self.assertEqual(1, code)
        self.assertIn('{}: no column p, lead, lead_gain, the recording was made before'.format(path), output)
        self.assertNotIn('Traceback', output)

    def test_one_refused_file_among_good_ones_refuses_the_run(self):
        good, old = self.path('good.csv'), self.path('old.csv')
        write_csv(good, sweep_rows())
        write_csv(old, sweep_rows(), without=('lead_gain',))
        code, output = run_main(good, old)
        self.assertEqual(1, code)
        self.assertIn('old.csv: no column lead_gain', output)
        self.assertNotIn('pwm_per_degree_per_second', output)

    def test_the_other_columns_of_a_recording_are_ignored(self):
        path = self.path('extra.csv')
        columns = COLUMNS + ['note']
        with open(path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(dict(row, note='x') for row in sweep_rows())
        code, output = run_main(path)
        self.assertEqual(0, code)
        self.assertIn('pwm_per_degree_per_second: 10.00', output)

    def test_end_to_end_on_a_written_csv(self):
        path = self.path('tune_joint1_ff.csv')
        write_csv(path, sweep_rows(k=10.0, friction=20.0, gravity=30.0, p=2000.0))
        code, output = run_main(path, '--max-positions', '25')
        self.assertEqual(code, 0)
        self.assertIn('{}: {} rows\n'.format(path, len(sweep_rows())), output)
        self.assertIn('joint 1, no lead, P 2000', output)
        self.assertIn('ratio k_pwm / k_lag = 1.00', output)
        self.assertIn('max_positions: 25', output)
        self.assertIn('pwm_per_degree_per_second: 10.00', output)

    def test_the_option_for_the_p_is_gone(self):
        path = self.path('a.csv')
        write_csv(path, sweep_rows())
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            run_main(path, '--p', '1=2000')
        self.assertEqual(2, raised.exception.code)

    def test_end_to_end_with_two_gains_of_lead(self):
        path = self.path('lead.csv')
        write_csv(path, sweep_rows(k=12.0, p=2500.0, lead_gain=0.0) + sweep_rows(k=12.0, p=2500.0, lead_gain=12.0))
        code, output = run_main(path)
        self.assertEqual(code, 0)
        self.assertIn('no lead, P 2500', output)
        self.assertIn('lead gain 12 pwm per degree/s, P 2500', output)
        self.assertLess(output.index('no lead'), output.index('lead gain 12'))
        self.assertEqual(output.count('pwm_per_degree_per_second: 12.00'), 2)
        self.assertIn('flat', output)

    def test_end_to_end_with_blank_p_cells_prints_only_the_pwm_result_and_returns_0(self):
        path = self.path('nop.csv')
        write_csv(path, blank(sweep_rows(k=10.0), 'p'))
        code, output = run_main(path)
        self.assertEqual(code, 0)
        self.assertIn('P not known', output)
        self.assertIn('needs the P', output)
        self.assertIn('pwm_per_degree_per_second: 10.00', output)

    def test_one_file_that_can_be_fitted_is_enough_for_0(self):
        good, bad = self.path('good.csv'), self.path('one.csv')
        write_csv(good, sweep_rows(joint=2))
        write_csv(bad, sweep_rows(speeds=(10.0,)))
        code, output = run_main(good, bad)
        self.assertEqual(code, 0)
        self.assertIn('no fit', output)


if __name__ == '__main__':
    unittest.main()
