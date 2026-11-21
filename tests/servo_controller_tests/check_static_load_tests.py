import contextlib
import csv
import io
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from src.robot_controllers.dynamixel_robot.calibration import check_static_load
from src.robot_controllers.dynamixel_robot.calibration.check_static_load import PoseJoint
from src.robot_controllers.dynamixel_robot.calibration.load_directions import DOWN, UP

FIELDS = ['time', 'pose', 'direction', 'servo', 'angle', 'goal_position', 'compensation', 'present_position', 'error',
          'current_raw', 'current', 'current_unit', 'temperature', 'voltage', 'mode', 'speed', 'samples', 'p', 'i', 'd',
          'pwm']


def row(pose, direction, servo, angle, error, compensation=0, p=1000, mode='static', speed='', file='a.csv'):
    """a row of the csv of the recorders, error None for a servo that was not measured; angle in degrees"""
    return {'time': '1.0', 'pose': str(pose), 'direction': direction, 'servo': str(servo),
            'angle': '{:.6f}'.format(np.radians(angle)), 'goal_position': '2048',
            'compensation': '' if error is None else str(compensation), 'present_position': '',
            'error': '' if error is None else str(error), 'current_raw': '', 'current': '', 'current_unit': '',
            'temperature': '', 'voltage': '', 'mode': mode, 'speed': str(speed), 'samples': '5',
            'p': '' if error is None else str(p), 'i': '0', 'd': '0', 'pwm': '', 'file': file}


def pose_rows(pose, angle, error_up, error_down, joint=2, **kwargs):
    """the rows of both directions of one joint at one pose"""
    return [row(pose, UP, joint, angle, error_up, **kwargs), row(pose, DOWN, joint, angle, error_down, **kwargs)]


def pose_joint(angle, error_up, error_down, compensation=0.0, joint=2, p=1000.0, mode='static', speed=None):
    """a PoseJoint at the given angle of its joint in degrees, the other joints at 0"""
    angles = np.zeros(7)
    angles[joint] = np.radians(angle)
    return PoseJoint(joint, 'a.csv', 0, angles, error_up, error_down, compensation, compensation, p, mode, speed)


class PoseJointTests(unittest.TestCase):

    def test_sag_uncompensated_and_friction_of_a_pose(self):
        # arriving from below the joint rests 3.0 below its goal, from above 1.0: the sag is -2.0 -/+ 1.0 friction
        # for a compensation of -1.5 the sag without it is -2.0 - (-1.5)
        pose = pose_joint(30, error_up=-3.0, error_down=-1.0, compensation=-1.5)
        self.assertAlmostEqual(-2.0, pose.sag)
        self.assertAlmostEqual(-0.5, pose.uncompensated)
        self.assertAlmostEqual(1.0, pose.friction)

    def test_friction_is_half_the_difference_whichever_direction_is_larger(self):
        self.assertAlmostEqual(1.5, pose_joint(0, 2.0, -1.0).friction)
        self.assertAlmostEqual(1.5, pose_joint(0, -1.0, 2.0).friction)

    def test_the_compensation_is_the_mean_of_the_two_directions(self):
        pose = pose_joint(0, 1.0, 1.0)
        pose.compensation_up, pose.compensation_down = 2.0, 3.0
        self.assertAlmostEqual(2.5, pose.compensation)
        self.assertAlmostEqual(1.0 - 2.5, pose.uncompensated)


class GroupingTests(unittest.TestCase):

    def test_both_directions_are_needed(self):
        rows = pose_rows(0, 30, -2.0, -1.0) + [row(1, UP, 2, 40, -2.0)] + [row(2, DOWN, 2, 50, -1.0)]
        poses = check_static_load.poses_from_rows(rows)
        self.assertEqual([0], [pose.pose for pose in poses[2]])
        self.assertAlmostEqual(-2.0, poses[2][0].error_up)
        self.assertAlmostEqual(-1.0, poses[2][0].error_down)
        self.assertAlmostEqual(np.radians(30), poses[2][0].angles[2], places=5)

    def test_a_blank_error_only_gives_the_angle_of_its_joint(self):
        rows = pose_rows(0, 30, -2.0, -1.0) + pose_rows(0, 10, None, None, joint=1) + pose_rows(0, 5, 1.0, 1.0, joint=3)
        poses = check_static_load.poses_from_rows(rows)
        self.assertEqual({2, 3}, set(poses))
        self.assertAlmostEqual(np.radians(10), poses[2][0].angles[1], places=5)
        self.assertAlmostEqual(np.radians(5), poses[2][0].angles[3], places=5)
        self.assertTrue(np.isnan(poses[2][0].angles[4]), 'a joint with no row at the pose has no angle')
        # a joint measured in one direction only has no pose
        rows = pose_rows(0, 30, -2.0, -1.0) + [row(0, UP, 3, 5, 1.0), row(0, DOWN, 3, 5, None)]
        self.assertEqual({2}, set(check_static_load.poses_from_rows(rows)))

    def test_the_same_pose_index_in_two_files_is_two_poses(self):
        rows = pose_rows(0, 30, -2.0, -1.0, file='a.csv') + pose_rows(0, 60, -4.0, -3.0, file='b.csv')
        poses = check_static_load.poses_from_rows(rows)[2]
        self.assertEqual(2, len(poses))
        self.assertEqual(['a.csv', 'b.csv'], [pose.file for pose in poses])
        self.assertAlmostEqual(-1.5, poses[0].sag)
        self.assertAlmostEqual(-3.5, poses[1].sag)
        # the directions of one file are not mixed with the other file's
        rows = [row(0, UP, 2, 30, -2.0, file='a.csv'), row(0, DOWN, 2, 30, -1.0, file='b.csv')]
        self.assertEqual({}, check_static_load.poses_from_rows(rows))

    def test_the_gripper_is_ignored(self):
        rows = pose_rows(0, 30, -2.0, -1.0) + pose_rows(0, 10, 5.0, 5.0, joint=7)
        self.assertEqual({2}, set(check_static_load.poses_from_rows(rows)))

    def test_compensation_p_mode_and_speed_are_taken_from_the_rows(self):
        rows = pose_rows(0, 30, -2.0, -1.0, compensation=-1.5, p=2000, mode='sweep', speed=5)
        pose = check_static_load.poses_from_rows(rows)[2][0]
        self.assertEqual((-1.5, -1.5, 2000.0, 'sweep', 5.0), (pose.compensation_up, pose.compensation_down, pose.p,
                                                              pose.mode, pose.speed))
        pose = check_static_load.poses_from_rows(pose_rows(0, 30, -2.0, -1.0))[2][0]
        self.assertEqual(('static', None), (pose.mode, pose.speed))

    def test_a_number_that_can_not_be_read_names_the_file_and_the_pose(self):
        rows = pose_rows(3, 30, -2.0, -1.0, file='old.csv')
        rows[1]['error'] = 'oops'
        with self.assertRaisesRegex(ValueError, r"old\.csv: pose 3 servo 2 has error 'oops'"):
            check_static_load.poses_from_rows(rows)
        del rows[1]['pose']
        with self.assertRaisesRegex(ValueError, 'has no column pose'):
            check_static_load.poses_from_rows(rows)

    def test_a_missing_p_gives_none(self):
        rows = pose_rows(0, 30, -2.0, -1.0)
        for r in rows:
            r['p'] = ''
        self.assertIsNone(check_static_load.poses_from_rows(rows)[2][0].p)


class SummaryTests(unittest.TestCase):

    def setUp(self):
        # sags -2.0, 0.0, +1.0 and compensations -3.0, -1.0, 0.0: the sags without the compensation are 1.0, 1.0, 1.0
        self.poses = [pose_joint(10, -3.0, -1.0, -3.0), pose_joint(20, 1.0, -1.0, -1.0, p=2000.0, mode='sweep'),
                      pose_joint(30, 2.0, 0.0, 0.0)]

    def test_the_statistics_of_a_list(self):
        stats = check_static_load.stats_of([-2.0, 0.0, 1.0, 3.0])
        self.assertAlmostEqual(0.5, stats.mean)
        self.assertAlmostEqual(np.sqrt(14 / 4), stats.rms)
        self.assertEqual((-2.0, 3.0, 3.0), (stats.low, stats.high, stats.largest))
        self.assertEqual(2.0, check_static_load.stats_of([-2.0, 1.0]).largest, 'the largest is absolute')

    def test_the_summary(self):
        summary = check_static_load.summarise(2, self.poses)
        self.assertEqual(3, summary.count)
        self.assertEqual((1000.0, 2000.0), summary.gains)
        self.assertEqual(('static', 'sweep'), summary.modes)
        self.assertAlmostEqual(-1.0 / 3, summary.sag.mean)
        self.assertAlmostEqual(np.sqrt(5 / 3), summary.sag.rms)
        self.assertEqual((-2.0, 1.0, 2.0), (summary.sag.low, summary.sag.high, summary.sag.largest))
        self.assertAlmostEqual(1.0, summary.uncompensated.mean)
        self.assertAlmostEqual(1.0, summary.uncompensated.rms)
        self.assertEqual((-3.0, 0.0, -4.0 / 3), (summary.compensation.low, summary.compensation.high,
                                                 summary.compensation.mean))
        self.assertAlmostEqual(0.0, summary.up.mean)
        self.assertEqual((-3.0, 2.0), (summary.up.low, summary.up.high))
        self.assertAlmostEqual(-2.0 / 3, summary.down.mean)
        self.assertEqual(3.0, summary.largest_error)
        self.assertAlmostEqual(1.0, summary.friction_mean)  # 1.0, 1.0, 1.0
        self.assertAlmostEqual(1.0, summary.friction_max)
        self.assertAlmostEqual(100 * (1 - np.sqrt(5 / 3)), summary.rms_reduction)

    def test_the_degrees_are_the_positions_times_360_over_4096(self):
        text = check_static_load.describe(check_static_load.summarise(2, [pose_joint(0, 4.0, 4.0)]))
        self.assertIn('rms 0.352 degrees, largest absolute value 4.00 positions (0.352 degrees)', text)

    def test_the_comparison_line(self):
        poses = [pose_joint(10, -2.8, -2.8, 0.0), pose_joint(20, -2.8, -2.8, 0.0)]
        for pose in poses:
            pose.error_up = pose.error_down = 0.26
            pose.compensation_up = pose.compensation_down = -2.58  # without: 0.26 + 2.58 = 2.84
        text = check_static_load.describe(check_static_load.summarise(2, poses))
        self.assertIn('rms 2.84 without, 0.26 with: 91% less', text)
        self.assertIn('compensation added: from -2.6 to -2.6 positions, mean -2.6', text)
        self.assertIn('sag without the compensation', text)

    def test_a_compensation_that_makes_it_worse_says_more(self):
        text = check_static_load.describe(check_static_load.summarise(2, [pose_joint(0, 3.0, 3.0, 1.0)]))
        self.assertIn('rms 2.00 without, 3.00 with: 50% more', text)

    def test_without_a_compensation_the_comparison_is_left_out(self):
        summary = check_static_load.summarise(2, [pose_joint(0, 1.0, 2.0), pose_joint(10, 0.0, 1.0)])
        self.assertIsNone(summary.compensation)
        self.assertIsNone(summary.rms_reduction)
        text = check_static_load.describe(summary)
        self.assertIn('no compensation was active', text)
        self.assertNotIn('without', text)
        self.assertNotIn('compensation added', text)
        # one row with a compensation is enough for the comparison
        poses = [pose_joint(0, 1.0, 2.0), pose_joint(10, 0.0, 1.0, 0.5)]
        self.assertIsNotNone(check_static_load.summarise(2, poses).compensation)

    def test_a_perfect_joint_has_no_reduction_to_report(self):
        summary = check_static_load.summarise(2, [pose_joint(0, 0.0, 0.0, 0.0)])
        self.assertIsNone(summary.rms_reduction)
        self.assertIn('the sag stays within 1.0 positions at all 1 poses',
                      check_static_load.describe(summary))

    def test_summarising_nothing_is_an_error(self):
        with self.assertRaises(ValueError):
            check_static_load.summarise(2, [])

    def test_the_header_gives_the_gains_and_the_modes(self):
        text = check_static_load.describe(check_static_load.summarise(2, self.poses))
        self.assertTrue(text.startswith('joint 2: 3 poses with an error in both directions, P 1000, 2000, mode '
                                        'static and sweep'), text)

    def test_a_sweep_says_that_the_lag_is_in_the_errors(self):
        poses = [pose_joint(10, 1.0, -1.0, mode='sweep', speed=5.0),
                 pose_joint(20, 1.0, -1.0, mode='sweep', speed=10.0)]
        text = check_static_load.describe(check_static_load.summarise(2, poses))
        self.assertIn('sweep (5, 10 degrees/s): the errors of the two directions include the lag of the speed, '
                      'their mean is the sag', text)
        self.assertNotIn('warning', text)
        self.assertNotIn('lag of the speed', check_static_load.describe(
            check_static_load.summarise(2, [pose_joint(10, 1.0, -1.0)])))

    def test_mixed_modes_warn(self):
        text = check_static_load.describe(check_static_load.summarise(2, self.poses))
        self.assertIn('warning: the rows are of both modes (static and sweep), the statistics mix them', text)
        self.assertNotIn('warning', check_static_load.describe(
            check_static_load.summarise(2, [pose_joint(10, 1.0, -1.0), pose_joint(20, 1.0, -1.0)])))


class LimitTests(unittest.TestCase):

    def poses(self):
        # sags 0.5, -2.0, 1.0 (not over: only above), 3.0, -1.5
        return [pose_joint(10, 0.5, 0.5), pose_joint(20, -3.0, -1.0), pose_joint(30, 1.0, 1.0),
                pose_joint(40, 2.0, 4.0), pose_joint(50, -1.0, -2.0)]

    def test_the_poses_over_the_limit_are_the_largest_first(self):
        summary = check_static_load.summarise(2, self.poses())
        self.assertEqual([3.0, -2.0, -1.5], [pose.sag for pose in summary.over])
        summary = check_static_load.summarise(2, self.poses(), limit=2.5)
        self.assertEqual([3.0], [pose.sag for pose in summary.over])

    def test_the_verdict_when_poses_are_over(self):
        text = check_static_load.describe(check_static_load.summarise(2, self.poses()))
        self.assertIn('3 of 5 poses are over 1.0 positions', text)
        self.assertNotIn('the sag stays within', text)
        lines = text.splitlines()
        heading = '  poses over 1.0 positions, largest sag first (angles of joints 1 to 6 in degrees):'
        listed = lines[lines.index(heading) + 2:]
        self.assertEqual(['+3.00', '-2.00', '-1.50'], [line.split()[-1] for line in listed[:3]])
        # the line of a pose has the angles in whole degrees, the compensation, both errors and the sag
        self.assertEqual(['0', '40', '0', '0', '0', '0', '+0.0', '+2.00', '+4.00', '+3.00'], listed[0].split())

    def test_the_verdict_when_all_poses_are_within_the_limit(self):
        text = check_static_load.describe(check_static_load.summarise(2, self.poses(), limit=3.0))
        self.assertIn('the sag stays within 3.0 positions at all 5 poses', text)
        self.assertNotIn('poses over', text)
        self.assertNotIn('are over', text)

    def test_the_list_is_cut_at_worst_and_the_rest_is_counted(self):
        summary = check_static_load.summarise(2, self.poses())
        text = check_static_load.describe(summary, worst=2)
        self.assertEqual(2, sum(1 for line in text.splitlines() if line.startswith('    ') and line.split()[-1] in
                                ('+3.00', '-2.00', '-1.50')))
        self.assertIn('    and 1 more', text)
        self.assertIn('3 of 5 poses are over 1.0 positions', text)
        self.assertNotIn('and 1 more', check_static_load.describe(summary, worst=3))
        self.assertIn('and 3 more', check_static_load.describe(summary, worst=0))

    def test_the_listed_angles_are_whole_degrees_of_joints_1_to_6(self):
        angles = np.radians([0, 10.4, -20.6, 30.0, 0.0, 179.5, -89.5])
        pose = PoseJoint(2, 'a.csv', 0, angles, 3.0, 3.0, 0.0, 0.0, 1000.0, 'static')
        text = check_static_load.describe(check_static_load.summarise(2, [pose]))
        self.assertIn('10  -21   30    0  180  -90', text)


class TableTests(unittest.TestCase):

    def lines(self, poses, joint=2):
        return check_static_load.table(joint, poses).splitlines()

    def test_the_table_is_sorted_by_the_angle_of_the_joint_then_by_the_others(self):
        poses = []
        for angle, other in ((30, 5), (10, 7), (30, -5), (20, 0)):
            pose = pose_joint(angle, 1.0, 1.0)
            pose.angles[3] = np.radians(other)
            poses.append(pose)
        lines = self.lines(poses)
        self.assertEqual(2 + 4, len(lines))
        self.assertEqual(['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'comp', 'up', 'down', 'sag'], lines[1].split())
        self.assertEqual([('10.0', '7.0'), ('20.0', '0.0'), ('30.0', '-5.0'), ('30.0', '5.0')],
                         [tuple(line.split()[1:3]) for line in lines[2:]])

    def test_the_columns_are_aligned_and_the_speed_is_shown_only_for_a_sweep(self):
        poses = [pose_joint(10, 1.0, -3.0, -1.0), pose_joint(20, 2.0, 0.0, 0.5)]
        lines = self.lines(poses)
        self.assertEqual(1, len({len(line) for line in lines[1:]}), 'one width for header and rows')
        self.assertNotIn('deg/s', lines[1])
        self.assertEqual(['0.0', '10.0', '0.0', '0.0', '0.0', '0.0', '-1.0', '+1.00', '-3.00', '-1.00'],
                         lines[2].split())
        sweep = [pose_joint(10, 1.0, -3.0, speed=5.0, mode='sweep'), pose_joint(20, 2.0, 0.0, speed=10.0, mode='sweep')]
        lines = self.lines(sweep)
        self.assertIn('deg/s', lines[1])
        self.assertEqual(['5', '+0.0', '+1.00', '-3.00', '-1.00'], lines[2].split()[6:])

    def test_a_joint_sorts_by_its_own_angle(self):
        poses = [pose_joint(50, 1.0, 1.0, joint=3), pose_joint(-20, 1.0, 1.0, joint=3)]
        poses[0].angles[1] = -1.0  # a smaller q1 would sort first if the joint were not used
        lines = self.lines(poses, joint=3)
        self.assertEqual(['-20.0', '50.0'], [line.split()[2] for line in lines[2:]])


class ReportTests(unittest.TestCase):

    def test_a_joint_without_data_gets_a_line(self):
        poses = check_static_load.poses_from_rows(pose_rows(0, 30, -2.0, -1.0))
        text = check_static_load.report(poses, [2, 5])
        self.assertIn('joint 2: 1 poses', text)
        self.assertIn('joint 5: no poses with an error in both directions', text)

    def test_the_table_follows_the_report_of_its_joint_only_when_asked(self):
        poses = check_static_load.poses_from_rows(pose_rows(0, 30, -2.0, -1.0))
        self.assertNotIn('every pose', check_static_load.report(poses, [2]))
        self.assertIn('every pose', check_static_load.report(poses, [2], with_table=True))


def lead_rows(pose, angle, error_up, error_down, lead, joint=2, **kwargs):
    """the rows of a sweep pose recorded with a velocity lead: +lead positions arriving from below, -lead from above"""
    rows = pose_rows(pose, angle, error_up, error_down, joint=joint, mode='sweep', speed=5, **kwargs)
    rows[0]['lead'], rows[1]['lead'] = str(lead), str(-lead)
    return rows


# the report of three poses written by the script before it knew the lead, joint 2, P 2000, compensation -1 / 1 / 2
REPORT_WITHOUT_LEAD = """\
joint 2: 3 poses with an error in both directions, P 2000, mode static
  sag that is left (mean of up and down, with the compensation as recorded):
    mean +1.50, rms 2.53, lowest -1.00, highest +4.00 positions
    rms 0.223 degrees, largest absolute value 4.00 positions (0.352 degrees)
  sag without the compensation (sag less compensation):
    mean +0.83, rms 1.19, lowest +0.00, highest +2.00 positions
    rms 0.105 degrees, largest absolute value 2.00 positions (0.176 degrees)
  rms 1.19 without, 2.53 with: 113% more
  compensation added: from -1.0 to +2.0 positions, mean +0.7
  arriving from below (up): mean +0.50, from -2.00 to +3.00 positions
  arriving from above (down): mean +2.50, from +0.00 to +5.00 positions
  largest absolute error of either direction: 5.00 positions (0.439 degrees)
  friction band (half the up/down difference, not compensated): mean 1.00, largest 1.00 positions
  poses over 1.0 positions, largest sag first (angles of joints 1 to 6 in degrees):
                         q1 .. q6    comp       up     down      sag
       -   70    -    -    -    -    +2.0    +3.00    +5.00    +4.00
       -   40    -    -    -    -    +1.0    +0.50    +2.50    +1.50
  2 of 3 poses are over 1.0 positions

joint 2: every pose, by the angle of the joint (angles in degrees, errors in positions)
       q1      q2      q3      q4      q5      q6    comp       up     down      sag
        -    10.0       -       -       -       -    -1.0    -2.00    -0.00    -1.00
        -    40.0       -       -       -       -    +1.0    +0.50    +2.50    +1.50
        -    70.0       -       -       -       -    +2.0    +3.00    +5.00    +4.00"""


class LeadTests(unittest.TestCase):

    def lead_poses(self):
        """two sweep poses of joint 2 at P 1000 with hand-made numbers, errors as recorded with the lead:
        pose 0: lead 3, errors -1.0 and +1.0, so without the lead -1 - 3 = -4.0 and +1 + 3 = +4.0
        pose 1: lead 1, errors -0.5 and +1.5, so without the lead -1.5 and +2.5"""
        rows = lead_rows(0, 10, -1.0, 1.0, 3.0) + lead_rows(1, 40, -0.5, 1.5, 1.0)
        return check_static_load.poses_from_rows(rows)

    def test_the_lead_of_both_directions_is_read(self):
        first, second = self.lead_poses()[2]
        self.assertEqual((3.0, -3.0), (first.lead_up, first.lead_down))
        self.assertEqual((1.0, -1.0), (second.lead_up, second.lead_down))
        pose = first
        self.assertAlmostEqual(3.0, pose.lead, msg='the size of the lead is the mean of the absolute values')

    def test_a_missing_or_blank_lead_is_0(self):
        # no column at all
        pose = check_static_load.poses_from_rows(pose_rows(0, 30, -2.0, -1.0))[2][0]
        self.assertEqual((0.0, 0.0), (pose.lead_up, pose.lead_down))
        # a blank cell, and a column that only one direction has filled
        rows = pose_rows(0, 30, -2.0, -1.0)
        rows[0]['lead'], rows[1]['lead'] = '', '  '
        pose = check_static_load.poses_from_rows(rows)[2][0]
        self.assertEqual((0.0, 0.0), (pose.lead_up, pose.lead_down))
        rows[0]['lead'] = '2.5'
        pose = check_static_load.poses_from_rows(rows)[2][0]
        self.assertEqual((2.5, 0.0), (pose.lead_up, pose.lead_down))

    def test_a_lead_that_is_not_a_number_names_the_file_and_the_pose(self):
        rows = lead_rows(3, 30, -2.0, -1.0, 1.0, file='new.csv')
        rows[1]['lead'] = 'fast'
        with self.assertRaisesRegex(ValueError, r"new\.csv: pose 3 servo 2 has lead 'fast'"):
            check_static_load.poses_from_rows(rows)

    def test_the_sag_and_the_sag_without_the_compensation_do_not_depend_on_the_lead(self):
        # the lead is not part of the error any more and cancels in the mean of the two directions
        with_lead = self.lead_poses()[2]
        without = check_static_load.poses_from_rows(
            pose_rows(0, 10, -1.0, 1.0, mode='sweep') + pose_rows(1, 40, -0.5, 1.5, mode='sweep'))[2]
        self.assertEqual([pose.sag for pose in without], [pose.sag for pose in with_lead])
        self.assertEqual([pose.uncompensated for pose in without], [pose.uncompensated for pose in with_lead])
        self.assertEqual([pose.friction for pose in without], [pose.friction for pose in with_lead])

    def test_without_the_lead_the_joint_trails_the_trajectory_by_the_lead_more(self):
        # the goal that was sent is the trajectory plus the lead. Moving up (lead +3) the joint rests 1.0 below the
        # trajectory with the lead, so 4.0 below it without; moving down (lead -3) it rests 1.0 above, so 4.0 above
        pose = self.lead_poses()[2][0]
        self.assertAlmostEqual(-4.0, pose.error_up_without_lead)
        self.assertAlmostEqual(4.0, pose.error_down_without_lead)
        self.assertAlmostEqual(4.0, pose.friction_without_lead)
        self.assertAlmostEqual(1.0, pose.friction, msg='with the lead the half difference is smaller')
        # the P term works on goal sent - present = -(error - compensation - lead) and has the same drive either way:
        # error - lead without the lead plus the lead added to the goal gives the error that was recorded
        self.assertAlmostEqual(pose.error_up, pose.error_up_without_lead + pose.lead_up)
        self.assertAlmostEqual(pose.error_down, pose.error_down_without_lead + pose.lead_down)

    def test_the_summary_has_no_lead_when_it_was_0_in_every_row(self):
        self.assertIsNone(check_static_load.summarise(2, [pose_joint(0, 1.0, 2.0), pose_joint(10, 0.0, 1.0)]).lead)

    def test_the_summary_of_the_lead(self):
        lead = check_static_load.summarise(2, self.lead_poses()[2]).lead
        self.assertAlmostEqual(2.0, lead.mean)  # (3 + 1) / 2
        self.assertAlmostEqual(3.0, lead.largest)
        self.assertAlmostEqual(-2.75, lead.up_without.mean)  # (-4.0 + -1.5) / 2
        self.assertEqual((-4.0, -1.5), (lead.up_without.low, lead.up_without.high))
        self.assertAlmostEqual(3.25, lead.down_without.mean)  # (4.0 + 2.5) / 2
        self.assertEqual((2.5, 4.0), (lead.down_without.low, lead.down_without.high))
        self.assertAlmostEqual(3.0, lead.friction_mean)  # (4.0 + 2.0) / 2
        self.assertAlmostEqual(4.0, lead.friction_max)
        self.assertAlmostEqual(100 * (1 - 1.0 / 3.0), lead.friction_reduction)  # 1.0 with the lead

    def test_a_lead_in_one_direction_only_is_found(self):
        pose = pose_joint(10, 1.0, -1.0)
        pose.lead_down = -2.0
        self.assertIsNotNone(check_static_load.summarise(2, [pose]).lead)

    def test_the_lead_lines_are_in_the_report_only_when_a_lead_was_active(self):
        text = check_static_load.describe(check_static_load.summarise(2, self.lead_poses()[2]))
        self.assertIn('  velocity lead was active: mean absolute lead 2.00, largest 3.00 positions', text)
        self.assertIn('without the lead (error less lead, the joint would trail by the lead more):', text)
        self.assertIn('    up: mean -2.75, from -4.00 to -1.50; down: mean +3.25, from +2.50 to +4.00 positions', text)
        self.assertIn('    friction band: mean 3.00, largest 4.00 positions', text)
        self.assertIn('    the lead made the mean friction band 3.00 -> 1.00 positions, 67% smaller', text)
        # the lead line is under the header lines, before the numbers; the lines about the errors follow the friction
        lines = text.splitlines()
        self.assertLess(lines.index('  velocity lead was active: mean absolute lead 2.00, largest 3.00 positions'),
                        next(i for i, line in enumerate(lines) if line.startswith('  sag that is left')))
        self.assertLess(next(i for i, line in enumerate(lines) if line.startswith('  friction band')),
                        next(i for i, line in enumerate(lines) if line.startswith('  without the lead')))
        # without a lead none of it is there
        text = check_static_load.describe(check_static_load.summarise(2, [pose_joint(10, 1.0, -1.0, mode='sweep')]))
        self.assertNotIn('lead', text)

    def test_the_lead_lines_belong_to_the_joint_that_had_a_lead(self):
        rows = lead_rows(0, 10, -1.0, 1.0, 3.0, joint=2) + pose_rows(0, 20, 1.0, 1.0, joint=3, mode='sweep')
        text = check_static_load.report(check_static_load.poses_from_rows(rows), [2, 3])
        joint_2, joint_3 = text.split('\n\njoint 3')
        self.assertIn('velocity lead was active', joint_2)
        self.assertNotIn('lead', joint_3)

    def test_a_lead_that_makes_the_friction_band_larger_says_larger(self):
        # errors +3.0 / -3.0 with a lead of 1: without it +2.0 / -2.0, so the band went from 2.0 to 3.0
        poses = check_static_load.poses_from_rows(lead_rows(0, 10, 3.0, -3.0, 1.0))[2]
        text = check_static_load.describe(check_static_load.summarise(2, poses))
        self.assertIn('the lead made the mean friction band 2.00 -> 3.00 positions, 50% larger', text)

    def test_without_a_friction_band_left_there_is_no_percentage(self):
        # without the lead both directions would have rested at 0.0: all of the band is the lead
        poses = check_static_load.poses_from_rows(lead_rows(0, 10, 1.0, -1.0, 1.0))[2]
        text = check_static_load.describe(check_static_load.summarise(2, poses))
        self.assertIn('    friction band: mean 0.00, largest 0.00 positions', text)
        self.assertNotIn('the lead made', text)

    def rows_without_a_lead(self):
        rows = []
        for pose, (angle, up, down, comp) in enumerate([(10, -2.0, -0.0, -1.0), (40, 0.5, 2.5, 1.0),
                                                        (70, 3.0, 5.0, 2.0)]):
            rows += pose_rows(pose, angle, up, down, joint=2, compensation=comp, p=2000)
        return rows

    def test_the_report_without_a_lead_is_unchanged(self):
        rows = self.rows_without_a_lead()
        text = check_static_load.report(check_static_load.poses_from_rows(rows), [2], with_table=True)
        self.assertEqual(REPORT_WITHOUT_LEAD, text, 'rows without a lead column')
        for value in ('0', '0.0', ''):
            for r in rows:
                r['lead'] = value
            text = check_static_load.report(check_static_load.poses_from_rows(rows), [2], with_table=True)
            self.assertEqual(REPORT_WITHOUT_LEAD, text, 'a lead column of {!r}'.format(value))

    def test_the_table_has_a_lead_column_only_with_a_lead(self):
        without = check_static_load.table(2, check_static_load.poses_from_rows(self.rows_without_a_lead())[2])
        self.assertNotIn('lead', without)
        lines = check_static_load.table(2, self.lead_poses()[2]).splitlines()
        self.assertEqual(['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'deg/s', 'comp', 'lead', 'up', 'down', 'sag'],
                         lines[1].split())
        self.assertEqual(1, len({len(line) for line in lines[1:]}), 'one width for header and rows')
        # the size of the lead of the pose, the errors as recorded
        self.assertEqual(['10.0', '-', '-', '-', '-', '5', '+0.0', '3.0', '-1.00', '+1.00', '+0.00'],
                         lines[2].split()[1:])
        self.assertEqual(['5', '+0.0', '1.0', '-0.50', '+1.50', '+0.50'], lines[3].split()[6:])

    def test_the_lead_column_of_the_table_is_per_joint(self):
        rows = lead_rows(0, 10, -1.0, 1.0, 3.0, joint=2) + pose_rows(0, 20, 1.0, 1.0, joint=3, mode='sweep')
        poses = check_static_load.poses_from_rows(rows)
        self.assertIn('lead', check_static_load.table(2, poses[2]))
        self.assertNotIn('lead', check_static_load.table(3, poses[3]))

    def test_main_prints_the_lead_and_the_table_column(self):
        fields = FIELDS + ['lead']
        rows = lead_rows(0, 10, -1.0, 1.0, 3.0) + lead_rows(1, 40, -0.5, 1.5, 1.0)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'sweep.csv')
            with open(path, 'w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(rows)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(0, check_static_load.main([path, '--table']))
            text = output.getvalue()
        self.assertIn('velocity lead was active: mean absolute lead 2.00, largest 3.00 positions', text)
        self.assertIn('the lead made the mean friction band 3.00 -> 1.00 positions, 67% smaller', text)
        self.assertIn('comp    lead', text)


class MainTests(unittest.TestCase):

    def run_main(self, *argv):
        """returns (result, printed text)"""
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = check_static_load.main(list(argv))
        return result, output.getvalue()

    def write(self, directory, name, rows):
        path = os.path.join(directory, name)
        with open(path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=FIELDS, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_a_missing_file_returns_1_with_a_message(self):
        result, output = self.run_main(os.path.join(tempfile.gettempdir(), 'no_such_recording_1234.csv'))
        self.assertEqual(1, result)
        self.assertIn('can not read the recordings', output)
        self.assertNotIn('Traceback', output)

    def test_a_csv_with_a_bad_number_returns_1(self):
        rows = pose_rows(0, 30, -2.0, -1.0)
        rows[0]['error'] = 'x'
        with tempfile.TemporaryDirectory() as directory:
            result, output = self.run_main(self.write(directory, 'bad.csv', rows))
        self.assertEqual(1, result)
        self.assertIn("pose 0 servo 2 has error 'x'", output)

    def test_no_data_returns_1(self):
        rows = [row(0, UP, 2, 30, -2.0), row(0, DOWN, 2, 30, None), row(0, UP, 1, 0, None)]
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, 'one_direction.csv', rows)
            result, output = self.run_main(path)
            self.assertEqual(1, result)
            self.assertIn('no poses with an error in both directions for any arm joint', output)
            # asked for a joint that has no data
            result, output = self.run_main(path, '--joints', '2')
            self.assertEqual(1, result)
            self.assertIn('for joints 2', output)
            # an empty file
            result, _ = self.run_main(self.write(directory, 'empty.csv', []))
            self.assertEqual(1, result)

    def test_end_to_end(self):
        rows = []
        for pose, (angle, up, down, comp) in enumerate([(10, -2.0, -0.0, -1.0), (40, 0.5, 2.5, 1.0),
                                                        (70, 3.0, 5.0, 2.0)]):
            rows += pose_rows(pose, angle, up, down, joint=2, compensation=comp, p=2000)
            rows += pose_rows(pose, 5 * pose, None, None, joint=1)
            rows += pose_rows(pose, 3 * pose, 1.0, 1.0, joint=3, compensation=0)
            rows += pose_rows(pose, 0, 9.0, 9.0, joint=7)
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, 'static_load.csv', rows)
            result, output = self.run_main(path)
            self.assertEqual(0, result)
            # joints 2 and 3 have data, 1 has no errors, 7 is not an arm joint
            self.assertEqual(['joint 2:', 'joint 3:'],
                             [line[:8] for line in output.splitlines() if line.startswith('joint')])
            self.assertIn('joint 2: 3 poses with an error in both directions, P 2000, mode static', output)
            # sags -1.0, 1.5, 4.0; without the compensation -0.0, 0.5, 2.0
            self.assertIn('mean +1.50, rms 2.53, lowest -1.00, highest +4.00 positions', output)
            self.assertIn('mean +0.83, rms 1.19, lowest +0.00, highest +2.00 positions', output)
            self.assertIn('rms 1.19 without, 2.53 with: 113% more', output)
            self.assertIn('2 of 3 poses are over 1.0 positions', output)
            self.assertIn('no compensation was active', output)  # joint 3
            self.assertNotIn('every pose', output)
            # the options
            result, output = self.run_main(path, '--joints', '2', '--limit', '3.5', '--worst', '1', '--table')
            self.assertEqual(0, result)
            self.assertNotIn('joint 3', output)
            self.assertIn('1 of 3 poses are over 3.5 positions', output)
            self.assertIn('every pose, by the angle of the joint', output)
            result, output = self.run_main(path, '--joints', '4', '2')
            self.assertEqual(0, result)
            self.assertLess(output.index('joint 4: no poses'), output.index('joint 2: 3 poses'))

    def test_two_files_are_pooled(self):
        with tempfile.TemporaryDirectory() as directory:
            a = self.write(directory, 'a.csv', pose_rows(0, 10, 1.0, 1.0, file='a'))
            b = self.write(directory, 'b.csv', pose_rows(0, 20, 3.0, 3.0, file='b'))
            result, output = self.run_main(a, b, '--joints', '2')
        self.assertEqual(0, result)
        self.assertIn('joint 2: 2 poses', output)
        self.assertIn('mean +2.00', output)

    def test_main_does_not_open_a_port(self):
        # the script only reads csv files: it must not import the recorders or the robot
        import inspect
        source = inspect.getsource(check_static_load)
        for name in ('get_robot', 'record_static_load', 'record_load_sweep', 'serial'):
            self.assertNotIn('import ' + name, source)
            self.assertNotIn('get_robot(', source)
        with mock.patch('builtins.print'):
            self.assertEqual(1, check_static_load.main(['no_such_file.csv']))


if __name__ == '__main__':
    unittest.main()
