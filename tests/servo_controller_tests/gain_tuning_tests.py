import argparse
import contextlib
import csv
import dataclasses
import io
import itertools
import os
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
from numpy import pi

from src.global_constants import POSITION_CONTROL_MODE
from src.robot_controllers.dynamixel_robot import dynamixel_x_config as cfg
from src.robot_controllers.dynamixel_robot.calibration import gain_tuning, tune_servo_gains
import math

from src.robot_controllers.dynamixel_robot.calibration.gain_tuning import (
    FF_CRUISE_HOLD, FF_LAG_TOLERANCE, FF_RAMP, MIN_BIN_SAMPLES, SWING_MIN_LEG, AbortLimits, BinSide, Criteria,
    CruiseStatistics, LineFit, RestMetrics, Sample, StepMetrics, SweepBin, SweepSample, SweepStep, SwingPair,
    SwingSample, SwingSide, SwingStatistics, abort_reason, bin_sweep, choose_d, choose_p, cruise_statistics, d_ladder,
    environmental_abort, evaluate, ff_sweep, fit_line, lag_growth, p_values, p_values_down, pwm_clamp,
    pwm_per_position, recommended_swing_p, rest_metrics, rougher_speeds, step_metrics, step_size, swing_change,
    swing_changed, swing_legs, swing_spread, swing_statistics, tracking_metrics)
from src.robot_controllers.dynamixel_robot.calibration import fit_velocity_lead
from src.robot_controllers.dynamixel_robot.calibration.tune_servo_gains import (
    FF_SAMPLE_FIELDS, SAMPLE_FIELDS, SWING_SAMPLE_FIELDS, TRIAL_FIELDS, PoseResult, Settings, StopFlag, Sweep,
    SwingResult, SwingRun, TuningSession, compared_d, d_candidates, d_grid, d_progress, feedforward_refusal,
    grid_is_done, joint_knee_at, joint_p_at, knee_trial, parse_pose, position_sign, recommend, stream_sweep,
    swept_d_values, swing_gains, swing_refusal)
from src.robot_controllers.dynamixel_robot.velocity_feedforward import VelocityFeedforward
from src.robot_controllers.dynamixel_robot.servo import Servo
from src.robot_controllers.dynamixel_robot.servo_handler import ServoState
from src.utils.movement import step_velocities
from tests.servo_controller_tests.fit_velocity_lead_tests import as_text, sweep_rows
from tests.servo_controller_tests.static_load_scripts_tests import FakeRobot

DT = 0.002


def samples(positions, dt=DT, goal=None, pwm=None, trajectory=None, voltage=11.8, temperature=30, start=0.0,
            moving_status=1):
    """samples at dt intervals from the positions, the goal and pwm default to the last position and 0"""
    rows = []
    for index, position in enumerate(positions):
        rows.append(Sample(time=start + index * dt, tick=int(1000 * (start + index * dt)) % 32768,
                           goal=positions[-1] if goal is None else goal, position=int(round(position)),
                           position_trajectory=(positions[-1] if goal is None else goal) if trajectory is None
                           else trajectory, velocity=0, velocity_trajectory=0,
                           pwm=0 if pwm is None else pwm, current=0, moving_status=moving_status, voltage=voltage,
                           temperature=temperature))
    return rows


class StepSizingTests(unittest.TestCase):

    def test_pwm_clamp(self):
        self.assertEqual(702, pwm_clamp(234))
        self.assertEqual(702, pwm_clamp(-234))
        self.assertEqual(442, pwm_clamp(10), 'at least half the limit')
        self.assertEqual(885, pwm_clamp(500), 'never above the limit')

    def test_step_size(self):
        # P 1500 is 11.7 pwm per position: (0.8 * 702 - 234) / 11.7 = 28 positions
        self.assertEqual((27, False), step_size(1500, 234, 702))
        self.assertEqual((60, False), step_size(800, 0, 885), 'capped at the maximum')
        self.assertEqual((8, True), step_size(16000, 500, 702), 'raised to the minimum and flagged')
        self.assertEqual((14, False), step_size(1500, 234, 702, pwm_per_position=23.4), 'a measured slope')
        with self.assertRaises(ValueError):
            step_size(0, 0, 885)

    def test_sweep_values(self):
        self.assertEqual([1500, 1950, 2540], p_values(1500, 1.3, 2.0))
        self.assertEqual([800], p_values(800, 1.3, 1.0))
        self.assertEqual([16000], p_values(16000, 1.3, 2.0), 'capped at the firmware range')
        self.assertEqual([1150, 890, 680], p_values_down(1500, 1.3))
        self.assertEqual([], p_values_down(100, 1.3))
        self.assertEqual([0, 250, 400, 640, 1020], d_ladder(250, 1.6, 1500))
        self.assertEqual([0, 300, 600], d_ladder(300, 2.0, 1000), 'up to the maximum')
        with self.assertRaises(ValueError):
            d_ladder(0, 1.6)
        with self.assertRaises(ValueError):
            d_ladder(250, 1.0)
        with self.assertRaises(ValueError):
            p_values(1500, 1.0, 2.0)


class MetricsTests(unittest.TestCase):

    def test_rest_metrics(self):
        rows = samples([2000, 2001, 2000, 2000, 2001] * 20, pwm=100, temperature=30, goal=2020)
        rows[-1] = rows[-1]._replace(temperature=32, voltage=11.1)
        rest = rest_metrics(rows)
        self.assertEqual(1, rest.position_pp)
        self.assertAlmostEqual(19.6, rest.error_mean)
        self.assertEqual(100, rest.pwm_mean)
        self.assertEqual(0, rest.pwm_std)
        self.assertAlmostEqual(0, rest.pwm_residual_std)
        self.assertEqual(2, rest.temperature_rise)
        self.assertEqual(11.1, rest.voltage_min)
        self.assertAlmostEqual(DT, rest.period)
        self.assertEqual(100, rest.samples)
        with self.assertRaises(ValueError):
            rest_metrics(rows[:1])

    def test_pwm_that_follows_an_encoder_flicker_is_not_noise(self):
        # joint 6 on 23 October: the position flickers between two counts and the pwm steps with it by P / 128
        p = 2240
        positions = [2055, 2056] * 400
        rows = samples(positions, goal=2075)
        rows = [s._replace(pwm=int(round(p / 128 * (s.goal - s.position)))) for s in rows]
        rest = rest_metrics(rows, p)
        self.assertEqual(1, rest.position_pp)
        self.assertAlmostEqual(8.5, rest.pwm_std, delta=0.5, msg='half the step of 17.5 units')
        self.assertAlmostEqual(p / 128, rest.pwm_slope, delta=0.6)
        self.assertLess(rest.pwm_residual_std, 1.0, 'the position explains the pwm')
        # the yaml P of 1720 with the same flicker gave the baseline
        baseline = rest_metrics([s._replace(pwm=int(round(1720 / 128 * (s.goal - s.position)))) for s in rows],
                                1720)
        self.assertEqual(([], []), evaluate(rest, [step()], baseline, p_ratio=p / 1720))
        # without the position the slope falls back to P / 128 and the residual is the whole std
        still = rest_metrics(samples([2055] * 100, goal=2075, pwm=350), p)
        self.assertEqual(p / 128, still.pwm_slope)
        self.assertEqual(0, still.pwm_residual_std)
        self.assertIsNone(rest_metrics(samples([2055] * 100, goal=2075, pwm=350)).pwm_slope)

    def test_pwm_noise_the_position_does_not_explain_is_a_warning(self):
        p = 2240
        rng = np.random.default_rng(3)
        rows = samples([2055, 2056] * 400, goal=2075)
        rows = [s._replace(pwm=int(round(p / 128 * (s.goal - s.position) + rng.normal(0, 12))))
                for s in rows]
        rest = rest_metrics(rows, p)
        self.assertGreater(rest.pwm_residual_std, 10)
        baseline = rest_metrics(samples([2055] * 800, goal=2075, pwm=270), 1720)
        self.assertEqual(([], ['pwm_noise']), evaluate(rest, [step()], baseline, p_ratio=p / 1720))

    def test_first_order_step_has_no_overshoot(self):
        times = np.arange(0, 0.8, DT)
        before = samples([2000] * 5, start=-5 * DT, goal=2000)
        response = 2027 - 27 * np.exp(-times / 0.03)
        rows = before + samples(response, goal=2030, pwm=35)
        m = step_metrics(rows, 0.0, 2000, 2030, 702)
        self.assertEqual(30, m.step)
        self.assertEqual(2000, m.start)
        self.assertAlmostEqual(2027, m.final, delta=0.5)
        self.assertTrue(m.moved)
        self.assertEqual(0, m.overshoot)
        self.assertEqual(0, m.ring_count)
        self.assertAlmostEqual(0.08, m.settle_time, delta=0.01)
        self.assertAlmostEqual(0.066, m.rise_time, delta=0.01)
        self.assertAlmostEqual(-3, m.error, delta=0.5)
        self.assertEqual(35, m.pwm_final)
        self.assertEqual(0, m.saturated_fraction)

    def test_damped_oscillation_counts_the_overshoot_and_the_rings(self):
        times = np.arange(0, 0.8, DT)
        response = 2030 - 30 * np.exp(-times / 0.05) * np.cos(2 * pi * 8 * times)
        rows = samples([2000] * 5, start=-5 * DT, goal=2000) + samples(response, goal=2030)
        m = step_metrics(rows, 0.0, 2000, 2030, 702)
        self.assertAlmostEqual(8.6, m.overshoot, delta=1.0)
        self.assertEqual(1, m.ring_count)
        self.assertGreater(m.settle_time, 0.1)
        self.assertTrue(m.moved)
        # the same the other way around
        rows = samples([2060] * 5, start=-5 * DT, goal=2060) + samples(4060 - response, goal=2030)
        m = step_metrics(rows, 0.0, 2060, 2030, 702)
        self.assertEqual(-30, m.step)
        self.assertAlmostEqual(8.6, m.overshoot, delta=1.0)
        self.assertEqual(1, m.ring_count)

    def test_stuck_and_unsettled_responses(self):
        rows = samples([2000] * 100, goal=2008)
        m = step_metrics(rows, 0.0, 2000, 2008, 702)
        self.assertFalse(m.moved)
        self.assertEqual(0, m.overshoot)
        self.assertIsNone(m.rise_time)
        ramp = samples(2000 + np.arange(400) * 0.2, goal=2100)
        m = step_metrics(ramp, 0.0, 2000, 2100, 702)
        self.assertIsNone(m.settle_time, 'still moving at the end')
        with self.assertRaises(ValueError):
            step_metrics(rows[:3], 0.0, 2000, 2008, 702)

    def test_saturated_fraction(self):
        rows = samples([2000] * 10, pwm=700) + samples([2000] * 10, pwm=100, start=10 * DT)
        self.assertEqual(0.5, step_metrics(rows, 0.0, 1990, 2000, 702).saturated_fraction)

    def test_tracking_metrics(self):
        rows = [Sample(time=i * 0.01, tick=0, goal=2000 + 10 * i, position=2000 + 10 * i - 6,
                       position_trajectory=2000 + 10 * i - 2, velocity=0, velocity_trajectory=0, pwm=0, current=0,
                       moving_status=0, voltage=11.8, temperature=30) for i in range(20)]
        rows.append(rows[-1]._replace(moving_status=cfg.MOVING_STATUS_FOLLOWING_ERROR))
        t = tracking_metrics(rows)
        self.assertEqual(4, t.peak_trajectory_error)
        self.assertAlmostEqual(4, t.rms_trajectory_error)
        self.assertEqual(6, t.peak_goal_error)
        self.assertEqual(6, t.final_error)
        self.assertTrue(t.following_error)

    def test_pwm_per_position(self):
        self.assertAlmostEqual(11.7, pwm_per_position([(0, 0), (10, 117), (20, 234), (-10, -117)]))
        self.assertIsNone(pwm_per_position([(0, 0), (1, 10), (2, 20)]), 'too little spread')
        self.assertIsNone(pwm_per_position([(0, 0), (10, 117)]), 'too few points')


def rest(pp=1, pwm_std=1.0, pwm_mean=200.0, temperature_rise=0.0, pwm_residual_std=None):
    """rest metrics with pwm noise that the position doesn't explain, unless a residual is given"""
    return RestMetrics(duration=3.0, samples=1500, position_pp=pp, position_std=0.3, pwm_mean=pwm_mean,
                       pwm_std=pwm_std, temperature_rise=temperature_rise, voltage_min=11.7, period=DT,
                       pwm_residual_std=pwm_std if pwm_residual_std is None else pwm_residual_std)


def step(overshoot=0.0, ring_count=0, settle_time=0.1, moved=True, saturated=0.0, size=28):
    return StepMetrics(step=size, start=2000, final=2000 + size - 3, moved=moved, overshoot=overshoot,
                       ring_count=ring_count, settle_time=settle_time, rise_time=0.05, error=-3,
                       pwm_final=35, saturated_fraction=saturated)


class EvaluationTests(unittest.TestCase):

    def test_a_clean_trial_passes(self):
        self.assertEqual(([], []), evaluate(rest(), [step(), step(size=-28)], rest()))

    def test_flags(self):
        baseline = rest()
        self.assertEqual((['overshoot'], []), evaluate(rest(), [step(overshoot=3.0)], baseline))
        self.assertEqual(([], []), evaluate(rest(), [step(overshoot=1.9)], baseline), 'below 2 positions')
        self.assertEqual(([], []), evaluate(rest(), [step(overshoot=5.0, size=60)], baseline), 'below 10 %')
        self.assertEqual((['ringing'], []), evaluate(rest(), [step(ring_count=2)], baseline))
        self.assertEqual((['settle'], []), evaluate(rest(), [step(settle_time=None)], baseline))
        self.assertEqual((['settle'], []), evaluate(rest(), [step(settle_time=0.6)], baseline))
        self.assertEqual((['ripple'], []), evaluate(rest(pp=3), [step()], baseline))
        self.assertEqual(([], []), evaluate(rest(pp=3), [step()], rest(pp=2)), 'twice the baseline is allowed')
        self.assertEqual(([], ['pwm_noise']), evaluate(rest(pwm_std=6.0), [step()], baseline),
                         'a warning, not a flag: the trial passes')
        self.assertEqual(([], []), evaluate(rest(pwm_std=4.0), [step()], baseline), 'below the floor')
        self.assertEqual(([], []), evaluate(rest(pwm_std=12.0), [step()], rest(pwm_std=10.0)))

    def test_warnings(self):
        self.assertEqual(([], ['saturation']), evaluate(rest(), [step(saturated=0.5)], rest()))
        self.assertEqual(([], ['stiction']), evaluate(rest(), [step(moved=False), step()], rest()))
        self.assertEqual((['stuck'], ['stiction']), evaluate(rest(), [step(moved=False)], rest()),
                         'nothing moved, the trial says nothing')
        self.assertEqual(([], ['heating']), evaluate(rest(temperature_rise=4), [step()], rest()))
        criteria = Criteria(max_temperature_rise=5)
        self.assertEqual(([], []), evaluate(rest(temperature_rise=4), [step()], rest(), criteria))
        self.assertEqual(([], ['heating']), evaluate(rest(), [step()], rest(), temperature_rise=4.0),
                         'the rise over the whole trial counts when given')

    def test_the_pwm_noise_warning_does_not_hide_a_flag(self):
        # joint 3 on 13 November 2026: the noise follows where the joint parks at an encoder edge, not the gain
        self.assertEqual((['overshoot'], ['pwm_noise']),
                         evaluate(rest(pwm_std=6.0), [step(overshoot=3.0)], rest()))
        self.assertEqual((['ripple'], ['pwm_noise']), evaluate(rest(pp=3, pwm_std=6.0), [step()], rest()))

    def test_pwm_noise_is_compared_per_unit_of_p(self):
        # a dither of one position gives a pwm std that scales with P: the same dither at 1.69 times the P passes
        baseline = rest(pwm_std=4.0)
        self.assertEqual(([], []), evaluate(rest(pwm_std=6.7), [step()], baseline, p_ratio=1.69))
        self.assertEqual(([], ['pwm_noise']), evaluate(rest(pwm_std=6.7), [step()], baseline))
        self.assertEqual(([], ['pwm_noise']), evaluate(rest(pwm_std=13.0), [step()], baseline, p_ratio=1.69),
                         'twice the dither at the higher P is noise')


class AbortTests(unittest.TestCase):

    def check(self, rows, step=28, baseline_pp=1, start_voltage=11.8, last_goal_time=0.0, clamp=702,
              limits=AbortLimits(), sag=0.0, streaming=False):
        return abort_reason(rows, clamp, step, baseline_pp, start_voltage, last_goal_time, limits, sag=sag,
                            streaming=streaming)

    def test_nothing_wrong(self):
        self.assertIsNone(self.check([]))
        self.assertIsNone(self.check(samples([2000] * 400, pwm=200, trajectory=1980)))

    def test_temperature_and_voltage(self):
        limit = int(AbortLimits().max_temperature)
        self.assertEqual('temperature {}C'.format(limit), self.check(samples([2000] * 10, temperature=limit)))
        self.assertIsNone(self.check(samples([2000] * 10, temperature=limit),
                                     limits=AbortLimits(max_temperature=limit + 3)))
        drop = AbortLimits().max_voltage_drop
        self.assertIn('voltage', self.check(samples([2000] * 10, voltage=11.8 - drop - 0.1)))
        self.assertIsNone(self.check(samples([2000] * 10, voltage=11.8 - drop + 0.1)))

    def test_position_jump(self):
        self.assertIn('jumped', self.check(samples([2000, 2000, 2060])))
        self.assertIsNone(self.check(samples([2000, 2000, 2040])))

    def test_divergence_after_the_grace_period(self):
        rows = samples([2000] * 400, trajectory=2100)
        self.assertIn('divergence', self.check(rows))
        self.assertIsNone(self.check(rows[:200]), 'still in the grace period')
        self.assertIsNone(self.check(samples([2000] * 400, trajectory=2030)), 'the sag is not divergence')
        self.assertIsNone(self.check(samples([2000] * 400, trajectory=2100), step=40), 'within 3 steps')
        self.assertIsNone(self.check(rows, sag=45), 'within 2.5 times the sag expected at these gains')
        self.assertIsNone(self.check(samples([2000] * 400, trajectory=2300), sag=math.inf), 'rule off')
        self.assertIn('divergence', self.check(samples([2000] * 400, trajectory=2300), sag=45))

    def test_divergence_during_a_streamed_move(self):
        lagging = samples([2000] * 10, trajectory=2100)
        self.assertIsNone(self.check(lagging, streaming=True), 'a lag of 100 positions is allowed during a move')
        self.assertIn('during the move', self.check(samples([2000] * 10, trajectory=2130), streaming=True))
        self.assertIsNone(self.check(samples([2000] * 10, trajectory=2130), streaming=True, sag=60))
        self.assertIsNone(self.check(samples([2000, 2010] * 200), streaming=True), 'no oscillation rule')

    def test_environmental_reasons(self):
        for rows in (samples([2000] * 10, temperature=int(AbortLimits().max_temperature)),
                     samples([2000] * 10, voltage=11.8 - AbortLimits().max_voltage_drop - 0.1),
                     samples([2000, 2000, 2060])):
            self.assertTrue(environmental_abort(self.check(rows)))
        for rows in (samples([2000] * 400, trajectory=2100), samples([2000, 2010] * 200),
                     samples([2000] * 100, pwm=-650)):
            self.assertFalse(environmental_abort(self.check(rows)))

    def test_oscillation_at_rest(self):
        rows = samples([2000, 2010] * 200)
        self.assertIn('oscillation', self.check(rows))
        self.assertEqual('oscillation, 10 positions back and forth at rest', self.check(rows))
        self.assertIsNone(self.check(samples([2000, 2002] * 200)), 'quantization')
        self.assertIsNone(self.check(samples([2000, 2010] * 200, ), baseline_pp=4), 'three times the baseline')
        self.assertIsNone(self.check(samples([2000, 2010] * 100), last_goal_time=0.0), 'before the grace period')

    # the oscillation rule: the position has to go both up and down by more than the threshold (6 positions here)
    # within the last second, never counting the grace of 0.5 s after the goal write

    @staticmethod
    def ms(positions):
        """samples 1 ms apart, the goal at the last position"""
        return samples(positions, dt=0.001)

    def first_oscillation(self, rows, first=500, **kwargs):
        """the time of the first check, made every 5 samples as the session does, that reports oscillation"""
        for end in range(first, len(rows) + 1, 5):
            reason = self.check(rows[:end], **kwargs)
            if reason is not None and reason.startswith('oscillation'):
                return rows[end - 1].time
        return None

    def test_a_creep_in_one_direction_is_not_oscillation(self):
        # joint 1 on 17 November 2026: still 8, 4 and 1 positions from the goal at 300, 350 and 400 ms after a
        # step of 60, which is 7 positions in 0.2 s, all in one direction
        for sign in (1, -1):
            creep = [2000 + sign * 9 * i / 200 for i in range(200)]
            rows = self.ms([2000] * 500 + creep)
            self.assertIsNone(self.check(rows), 'a creep of 9 positions in 0.2 s and still going')
            self.assertIsNone(self.first_oscillation(rows))
            # the same creep over the whole second after the grace, from further away
            long_creep = [2000 + sign * 30 * (1 - i / 1000) ** 2 for i in range(1000)]
            self.assertIsNone(self.first_oscillation(self.ms([2000] * 500 + long_creep)))

    def test_a_creep_with_a_small_spring_back_is_not_oscillation(self):
        up_9 = [2000 + 9 * i / 100 for i in range(100)]
        back_4 = [2009 - 4 * i / 50 for i in range(50)]
        up_5 = [2005 + 5 * i / 50 for i in range(50)]
        rows = self.ms([2000] * 500 + up_9 + back_4 + up_5)
        self.assertIsNone(self.check(rows))
        self.assertIsNone(self.first_oscillation(rows))
        # the reverse: down 9, up 4, down 5
        mirrored = self.ms([4000 - x.position for x in rows])
        self.assertIsNone(self.first_oscillation(mirrored))

    def test_a_sustained_swing_is_found_soon_after_the_grace(self):
        for hz, amplitude, within in ((20, 9, 0.15), (2, 10, 1.0)):
            rows = self.ms([2000 + amplitude * math.sin(2 * math.pi * hz * i / 1000) for i in range(2500)])
            found = self.first_oscillation(rows)
            self.assertIsNotNone(found, '{} Hz'.format(hz))
            self.assertLessEqual(found, 0.5 + within, '{} Hz'.format(hz))

    def test_a_swing_inside_the_grace_followed_by_a_still_position_is_not_oscillation(self):
        overshoot = [2000 + 40 * math.sin(math.pi * i / 400) for i in range(400)]  # up 40 and back in 0.4 s
        rows = self.ms(overshoot + [2000] * 600)
        self.assertIsNone(self.first_oscillation(rows, first=400))
        # a swing that is still going at the end of the grace counts from there on
        late = self.ms([2000 + 10 * math.sin(2 * math.pi * 10 * i / 1000) for i in range(800)])
        self.assertIsNotNone(self.first_oscillation(late))

    def test_both_directions_have_to_exceed_the_threshold(self):
        # a rise above the threshold with a fall of 3 positions
        rows = self.ms([2000] * 510 + [2012] * 100 + [2009] * 100)
        self.assertIsNone(self.check(rows), 'up 12, back 3')
        rows = self.ms([2000] * 510 + [1988] * 100 + [1991] * 100)
        self.assertIsNone(self.check(rows), 'down 12, back 3')
        # up 12 and back 8 is oscillation of 8 positions
        rows = self.ms([2000] * 510 + [2012] * 100 + [2004] * 100)
        self.assertEqual('oscillation, 8 positions back and forth at rest', self.check(rows))
        # and down 12 and up 8
        rows = self.ms([2000] * 510 + [1988] * 100 + [1996] * 100)
        self.assertEqual('oscillation, 8 positions back and forth at rest', self.check(rows))

    def test_old_samples_are_not_in_the_window(self):
        swing = [2000 + 10 * math.sin(2 * math.pi * 10 * i / 1000) for i in range(300)]  # t 0.6 to 0.9 after the grace
        rows = self.ms([2000] * 600 + swing + [2000] * 1200)
        self.assertIsNotNone(self.check(rows[:900]), 'the swing is in the window')
        self.assertIsNone(self.check(rows), 'more than a second ago')
        self.assertIsNone(self.check(rows[:2000]), 'the swing ended 1.1 s before this sample')
        narrow = AbortLimits(oscillation_window=0.2)
        self.assertIsNone(self.check(rows[:1300], limits=narrow), 'a window of 0.2 s holds a still position only')
        self.assertIsNotNone(self.check(rows[:1300]), 'the default window still sees the swing')
        self.assertEqual(1.0, AbortLimits().oscillation_window)

    def test_the_streaming_case_has_no_oscillation_rule(self):
        rows = self.ms([2000 + 10 * math.sin(2 * math.pi * 10 * i / 1000) for i in range(1500)])
        self.assertIsNone(self.check(rows, streaming=True))
        self.assertIsNotNone(self.check(rows))

    def test_the_oscillation_rule_is_cheap(self):
        # the session calls it every 5 samples with all the samples of the trial, up to 15,000 of them
        rows = self.ms([2000 + (i % 3) for i in range(15000)])
        began = time.perf_counter()
        for _ in range(100):
            self.check(rows)
        self.assertLess(time.perf_counter() - began, 0.5, 'it looks at the last second only')

    def test_pwm_stuck_at_the_clamp(self):
        rows = samples([2000] * 100, pwm=-650)
        self.assertIn('clamp', self.check(rows))
        self.assertIsNone(self.check(samples([2000] * 100, pwm=600)), 'below 90 % of the clamp')
        self.assertIsNone(self.check(samples([2000] * 40, pwm=650)), 'not yet long enough')
        mixed = samples([2000] * 60, pwm=650)
        mixed[-1] = mixed[-1]._replace(pwm=100)
        self.assertIsNone(self.check(mixed), 'not the whole window')


def knee_pose(number, by_d, phase='p'):
    """a pose result from {d: (knee P, recommended P, overshoot of the knee trial, flags of the knee trial)}; the
    knee trial fails with the flags, a trial of the comparison of the D candidates at the same P and D (the largest
    overshoot there is) and a trial at another D with the same P come first and must not be taken for it"""
    trials, sweeps = [], {}
    for d, (knee_p, recommended, overshoot, flags) in by_d.items():
        passing = ChoiceTests.trial(p=recommended or 100)._replace(d=d)
        failing = ChoiceTests.trial(flags, p=knee_p)._replace(d=d, steps=(step(overshoot=overshoot),), phase=phase)
        noise = ChoiceTests.trial(['overshoot'], p=knee_p)._replace(d=d, steps=(step(overshoot=50.0),), phase='d')
        other_d = ChoiceTests.trial(['overshoot'], p=knee_p)._replace(d=d + 1, steps=(step(overshoot=60.0),))
        trials += [noise, other_d, passing, failing]
        sweeps[d] = Sweep(knee_p, recommended, recommended)
    return PoseResult(number, None, None, trials, sweeps, None, False)


class ChoiceTests(unittest.TestCase):

    def test_choose_p(self):
        self.assertEqual((1650, 2540), choose_p([(1500, True), (1950, True), (2540, False)], 1.3))
        self.assertEqual((1950, None), choose_p([(1500, True), (1950, True), (2540, True)], 1.3))
        self.assertEqual((None, 1500), choose_p([(1500, False)], 1.3))
        self.assertEqual((980, 1500), choose_p([(1500, False), (1150, True)], 1.3))
        self.assertEqual((None, None), choose_p([], 1.3))

    def test_choose_d(self):
        # (d, passed, overshoot, rings, pwm residual std, mean |error|)
        results = [(0, True, 5.0, 1, 3.0, 2.0), (200, True, 2.0, 0, 3.5, 2.0), (400, True, 1.0, 0, 6.0, 2.0),
                   (800, False, 0.5, 0, 9.0, 2.0)]
        self.assertEqual(200, choose_d(results), 'the least overshoot without pwm noise')
        self.assertEqual(400, choose_d(results, pwm_std_floor=6.0), 'the floor admits the noisier one')
        self.assertEqual(0, choose_d([(0, True, 1.0, 0, 3.0, 2.0), (200, True, 1.0, 0, 3.0, 2.0)]),
                         'the smallest on a tie')
        self.assertIsNone(choose_d([(0, False, 1.0, 0, 3.0, 2.0)]))
        self.assertIsNone(choose_d([]))

    def test_choose_d_on_the_resting_error(self):
        # joint 6 on 23 October: every D passed with an overshoot of a position or less, D 0 rested 4.4 positions
        # from the goal on average, the damped joint 1.1
        self.assertEqual(650, choose_d([(0, True, 1.0, 0, 3.0, 4.4), (650, True, 0.7, 0, 3.5, 1.4)]),
                         'the same overshoot within noise, 3 positions less resting error')
        self.assertEqual(0, choose_d([(0, True, 1.0, 0, 3.0, 2.0), (650, True, 0.7, 0, 3.5, 1.5)]),
                         'half a position is noise')
        self.assertEqual(0, choose_d([(0, True, 1.0, 0, 3.0, 4.4), (650, True, 1.0, 0, 6.0, 1.4)]),
                         'less resting error but pwm noise above the limit')
        self.assertEqual(0, choose_d([(0, True, 0.5, 0, 3.0, 4.4), (650, True, 1.8, 0, 3.5, 1.4)]),
                         'less resting error but more overshoot by more than a position')
        self.assertEqual(320, choose_d([(0, True, 1.0, 1, 3.0, 2.0), (320, True, 1.0, 0, 3.0, 2.0)]),
                         'removes a ring')
        self.assertEqual(1100, choose_d([(0, True, 1.0, 0, 3.0, 4.4), (320, True, 0.8, 0, 3.0, 3.2),
                                         (650, True, 0.6, 0, 3.5, 2.0), (1100, True, 0.9, 0, 4.0, 1.1),
                                         (1300, True, 0.5, 0, 4.4, 1.2)]),
                         'the least resting error of the ones that qualify')
        # D 0 failed: the reference is the smallest D that passed, a larger one has to improve on that
        failed_at_0 = [(0, False, 4.0, 2, 3.0, 4.0), (320, True, 1.0, 0, 3.5, 3.0), (650, True, 0.8, 0, 4.0, 2.5)]
        self.assertEqual(320, choose_d(failed_at_0), '650 gains only half a position over 320')
        failed_at_0[2] = (650, True, 0.8, 0, 4.0, 1.8)
        self.assertEqual(650, choose_d(failed_at_0))
        self.assertEqual(320, choose_d(failed_at_0, min_improvement=1.5))

    def test_grid_is_done(self):
        self.assertFalse(grid_is_done([1650], 2))
        self.assertFalse(grid_is_done([1650, None], 2))
        self.assertTrue(grid_is_done([1650, None, None], 2))
        self.assertTrue(grid_is_done([1650, 1950, 1650, 1270], 2), 'lower than the best, not than the previous')
        self.assertFalse(grid_is_done([1650, 1270, 1950], 2), 'a better D starts over')
        self.assertFalse(grid_is_done([1650, 1650], 2))
        self.assertTrue(grid_is_done([1650, 1650, 1650], 2), 'the same P twice gains nothing either')
        # joint 2 on 25 October: nothing passed at D 0, then 1430 from D 250 on
        self.assertFalse(grid_is_done([None, 1430], 2))
        self.assertFalse(grid_is_done([None, 1430, 1430], 2))
        self.assertTrue(grid_is_done([None, 1430, 1430, 1430], 2))
        self.assertTrue(grid_is_done([None, None], 2), 'nothing passes at the first two D gains either')
        self.assertFalse(grid_is_done([1650, None, None], 3))

    def test_no_knees_is_the_rule_of_the_p_of_the_joint_alone(self):
        for joint_ps in ([1650], [1650, None, None], [1650, 1950, 1650, 1270], [1650, 1270, 1950], [1650, 1650, 1650],
                         [None, 1430, 1430, 1430], [None, None]):
            for patience in (1, 2, 3):
                self.assertEqual(grid_is_done(joint_ps, patience),
                                 grid_is_done(joint_ps, patience, [None] * len(joint_ps)), joint_ps)
        with self.assertRaises(ValueError):
            grid_is_done([1650, 1650], 2, [None])

    def test_a_falling_overshoot_at_the_knee_is_progress(self):
        # joint 4 on 18 November 2026: the knee stayed at P 1010 from D 250 to 640, the overshoot fell, the limit is 6
        joint_ps = [510, 660, 660, 660]
        knees = [(1, 780, 8.0), (1, 1010, 11.7), (1, 1010, 10.0), (1, 1010, 8.7)]
        self.assertTrue(grid_is_done(joint_ps, 2), 'without the knees the grid stops after D 640')
        self.assertFalse(grid_is_done(joint_ps, 2, knees))
        self.assertEqual(['p', 'p', 'overshoot', 'overshoot'], [p.kind for p in d_progress(joint_ps, knees)])
        self.assertEqual([None, None, 11.7, 10.0], [p.from_overshoot for p in d_progress(joint_ps, knees)])
        # the same, with the overshoot falling less than the threshold
        knees = [(1, 780, 8.0), (1, 1010, 11.7), (1, 1010, 11.5), (1, 1010, 11.3)]
        self.assertTrue(grid_is_done(joint_ps, 2, knees))
        self.assertEqual(['p', 'p', None, None], [p.kind for p in d_progress(joint_ps, knees)])
        self.assertFalse(grid_is_done(joint_ps, 3, knees))

    def test_the_drop_has_to_reach_the_larger_of_a_position_and_a_tenth(self):
        def done(first, second):
            return grid_is_done([660, 660], 1, [(1, 1010, first), (1, 1010, second)])

        self.assertTrue(done(11.7, 11.0), '0.7 is below the 1.17 that a tenth of 11.7 asks')
        self.assertTrue(done(11.7, 10.6))
        self.assertFalse(done(11.7, 10.53), 'exactly a tenth')
        self.assertFalse(done(11.7, 9.0))
        self.assertTrue(done(5.0, 4.1), 'below a position')
        self.assertFalse(done(5.0, 4.0), 'exactly a position')
        self.assertTrue(done(5.0, 5.0))
        self.assertTrue(done(5.0, 6.0), 'a rise')

    def test_the_drop_is_measured_against_the_lowest_overshoot_so_far(self):
        joint_ps = [660, 660, 660]
        # 11.0 is 1.5 below the 12.5 before it, but not 1.17 below the 11.7 of the first D
        self.assertTrue(grid_is_done(joint_ps, 2, [(1, 1010, 11.7), (1, 1010, 12.5), (1, 1010, 11.0)]))
        self.assertTrue(grid_is_done(joint_ps, 2, [(1, 1010, 11.7), (1, 1010, 11.2), (1, 1010, 10.6)]),
                        'every drop is below the threshold of the lowest so far')
        self.assertFalse(grid_is_done(joint_ps, 2, [(1, 1010, 11.7), (1, 1010, 12.5), (1, 1010, 10.0)]))

    def test_a_knee_at_another_p_or_pose_is_compared_from_then_on(self):
        joint_ps = [660, 660, 660, 660]
        # the first knee of a pose and P is no progress, it only starts the record
        knees = [(1, 1010, 11.7), (1, 1300, 9.0), (1, 1300, 7.0), (1, 1300, 6.0)]
        self.assertEqual(['p', None, 'overshoot', 'overshoot'], [p.kind for p in d_progress(joint_ps, knees)])
        self.assertFalse(grid_is_done(joint_ps, 2, knees))
        self.assertTrue(grid_is_done(joint_ps[:2], 1, knees[:2]))
        knees = [(1, 1010, 11.7), (2, 1010, 5.0), (2, 1010, 3.5), (1, 1010, 10.0)]
        self.assertEqual(['p', None, 'overshoot', 'overshoot'], [p.kind for p in d_progress(joint_ps, knees)])
        self.assertEqual([None, None, 5.0, 11.7], [p.from_overshoot for p in d_progress(joint_ps, knees)],
                         'both records are kept')
        # a knee that is no progress still lowers its record
        knees = [(1, 1010, 11.7), (1, 1010, 11.2), (1, 1010, 10.0), (1, 1010, 9.0)]
        self.assertEqual(['p', None, 'overshoot', 'overshoot'], [p.kind for p in d_progress(joint_ps, knees)])
        self.assertEqual([None, None, 11.2, 10.0], [p.from_overshoot for p in d_progress(joint_ps, knees)])

    def test_a_higher_p_of_the_joint_starts_a_new_record(self):
        joint_ps = [660, 660, 990, 990, 990]
        knees = [(1, 1010, 11.7), (1, 1010, 11.5), (1, 1500, 6.0), (1, 1010, 5.0), (1, 1010, 3.0)]
        progress = d_progress(joint_ps, knees)
        self.assertEqual(['p', None, 'p', None, 'overshoot'], [p.kind for p in progress])
        self.assertEqual(5.0, progress[4].from_overshoot, 'the 11.5 of before the rise of P is forgotten')

    def test_a_knee_without_the_overshoot_flag_is_no_progress(self):
        joint_ps = [660, 660, 660]
        self.assertTrue(grid_is_done(joint_ps, 2, [(1, 1010, 11.7), None, None]))
        self.assertTrue(grid_is_done(joint_ps, 2, [None, None, None]))
        self.assertFalse(grid_is_done(joint_ps, 2, [(1, 1010, 11.7), (1, 1010, 8.0), None]))
        self.assertTrue(grid_is_done(joint_ps, 1, [(1, 1010, 11.7), (1, 1010, 8.0), None]))

    def test_a_lower_p_of_the_joint_is_no_progress_whatever_the_overshoot(self):
        knees = [(1, 1010, 11.7), (1, 1010, 3.0), (1, 1010, 1.0)]
        self.assertTrue(grid_is_done([660, 510, 510], 2, knees))
        self.assertEqual(['p', None, None], [p.kind for p in d_progress([660, 510, 510], knees)])
        self.assertTrue(grid_is_done([660, 660, None], 2, [(1, 1010, 11.7), (1, 1010, 11.6), None]))
        self.assertTrue(grid_is_done([660, 510], 1, [(1, 1010, 11.7), (1, 1010, 3.0)]))

    def test_the_knee_of_the_joint_is_the_one_of_the_limiting_pose(self):
        results = [knee_pose(1, {0: (1300, 850, 4.0, ('overshoot',)), 250: (2000, 1300, 4.0, ('overshoot',))}),
                   knee_pose(2, {0: (1010, 660, 8.0, ('overshoot',)), 250: (1500, 980, 9.5, ('overshoot',))})]
        self.assertEqual((2, 1010, 8.0), joint_knee_at(results, 0), 'the smallest recommendation limits the joint')
        self.assertEqual((2, 1500, 9.5), joint_knee_at(results, 250))
        self.assertIsNone(joint_knee_at(results, 400), 'not swept at this D')
        self.assertEqual(660, joint_p_at(results, 0))

    def test_poses_that_tie_are_told_apart_by_their_overshoot(self):
        results = [knee_pose(1, {0: (1010, 660, 8.0, ('overshoot',))}),
                   knee_pose(2, {0: (1010, 660, 11.7, ('overshoot',))}),
                   knee_pose(3, {0: (1010, 660, 9.0, ('overshoot',))})]
        self.assertEqual((2, 1010, 11.7), joint_knee_at(results, 0))
        # the tie is on the overshoot of the knee trial, not on whether it is flagged
        results = [knee_pose(1, {0: (1010, 660, 8.0, ('overshoot',))}),
                   knee_pose(2, {0: (1300, 660, 12.0, ('ringing',))})]
        self.assertIsNone(joint_knee_at(results, 0), 'the limiting pose is the one with the larger overshoot')
        # a pose without a knee ranks after the ones that have one
        results = [knee_pose(1, {0: (None, 660, 0.0, ())}), knee_pose(2, {0: (1010, 660, 8.0, ('overshoot',))})]
        self.assertEqual((2, 1010, 8.0), joint_knee_at(results, 0))

    def test_no_knee_without_the_overshoot_flag_or_a_knee(self):
        flagged = (1010, 660, 8.0, ('overshoot',))
        for flags in (('ringing',), ('settle',), ('ripple',), (), ('aborted', 'ringing')):
            results = [knee_pose(1, {0: (1010, 660, 8.0, flags)})]
            self.assertIsNone(joint_knee_at(results, 0), flags)
        both = knee_pose(1, {0: (1010, 660, 8.0, ('ringing', 'overshoot'))})
        self.assertEqual((1, 1010, 8.0), joint_knee_at([both], 0), 'one flag among others is enough')
        self.assertEqual((1, 1010, 8.0), joint_knee_at([knee_pose(1, {0: flagged})], 0))
        every_p_passed = knee_pose(1, {0: (None, 1950, 0.0, ())})
        self.assertIsNone(joint_knee_at([every_p_passed], 0), 'every P passed')
        no_p = knee_pose(1, {0: (1010, None, 8.0, ('overshoot',))})
        self.assertIsNone(joint_knee_at([no_p], 0), 'no P of the joint')
        self.assertIsNone(joint_knee_at([], 0))

    def test_the_knee_trial_is_one_of_the_p_sweep_at_that_d(self):
        result = knee_pose(1, {0: (1010, 660, 8.0, ('overshoot',)), 250: (1010, 660, 6.5, ('overshoot',))})
        trial = knee_trial(result, 250)
        self.assertEqual(('p', 1010, 250, 6.5), (trial.phase, trial.p, trial.d, trial.max_overshoot))
        self.assertEqual(8.0, knee_trial(result, 0).max_overshoot, 'the other D gains and the comparison are not it')
        self.assertIsNone(knee_trial(result, 400))
        self.assertIsNone(knee_trial(knee_pose(1, {0: (None, 660, 0.0, ())}), 0))
        down = knee_pose(1, {0: (1010, None, 8.0, ('overshoot',))}, phase='p_down')
        self.assertEqual(8.0, knee_trial(down, 0).max_overshoot, 'a sweep that went down has its knee there')

    @staticmethod
    def trial(flags=(), aborted=None, p=1000):
        return tune_servo_gains.TrialResult(
            phase='p', p=p, d=0, clamp=442, step=10, step_limited=False, rest=None, steps=(), tracking=(),
            pwm_slope=None, flags=tuple(flags), warnings=(), aborted=aborted, temperature_start=30, temperature_end=30,
            voltage_min=11.8)

    def test_a_trial_with_only_the_pwm_noise_warning_passes(self):
        flags, warnings = evaluate(rest(pwm_std=6.0), [step()], rest())
        result = self.trial(flags)._replace(warnings=tuple(warnings))
        self.assertEqual(('pwm_noise',), result.warnings)
        self.assertTrue(result.passed)
        self.assertFalse(result.sluggish)
        # a sweep over such trials has no knee
        recommended, knee = choose_p([(1500, result.passed), (1950, result.passed)], 1.3)
        self.assertIsNone(knee)
        self.assertEqual(1500, recommended)

    def test_a_trial_that_only_settles_too_slowly_is_sluggish(self):
        self.assertTrue(self.trial(['settle']).sluggish)
        self.assertTrue(self.trial(['stuck']).sluggish)
        self.assertFalse(self.trial(['overshoot', 'settle']).sluggish, 'it overshoots as well')
        self.assertFalse(self.trial(['aborted'], aborted='oscillation, 8 positions back and forth at rest').sluggish)
        self.assertFalse(self.trial().sluggish, 'it passed')

    def test_d_grid(self):
        settings = Settings(d_start=250, d_factor=1.6, d_max=1500)
        self.assertEqual([0, 250, 400, 500, 640, 1020], d_grid(settings, 500), 'the yaml D joins the ladder')
        self.assertEqual([0, 250, 400, 640, 1020], d_grid(settings, 0))
        self.assertEqual([500], d_grid(settings._replace(tune_d=False), 500))
        self.assertEqual([400, 0], d_grid(settings._replace(d_values=(400, 0, 400)), 500), 'as given, in order')

    def test_the_default_d_grid_ends_at_1640(self):
        self.assertEqual(2000, tune_servo_gains.DEFAULT_D_MAX)
        self.assertEqual(tune_servo_gains.DEFAULT_D_MAX, Settings().d_max)
        self.assertEqual([0, 250, 400, 640, 1020, 1640], d_grid(Settings(), 0))
        self.assertEqual([0, 250, 400, 550, 640, 1020, 1640], d_grid(Settings(), 550), 'the yaml D joins the ladder')
        self.assertEqual([0, 250, 400, 640, 1020, 1640], d_grid(Settings(), 640), 'already on it')
        self.assertEqual([0, 250, 400, 640, 1020, 1640, 2500], d_grid(Settings(), 2500), 'a yaml D above the cap')
        # --d-max raises it up to the register
        raised = d_grid(Settings(d_max=cfg.MAX_GAIN), 0)
        self.assertEqual([0, 250, 400, 640, 1020, 1640, 2620, 4190, 6710, 10740], raised)
        # --d-values is not limited by the default
        self.assertEqual([0, 5000, cfg.MAX_GAIN], d_grid(Settings(d_values=(0, 5000, cfg.MAX_GAIN)), 0))

    def test_the_d_max_option_defaults_to_the_constant(self):
        captured = []

        def fake_robot():
            raise tune_servo_gains.DynamixelError('no robot in the tests')

        def capture(d_start, d_factor, d_max):
            captured.append(d_max)
            return [0]

        with mock.patch.object(tune_servo_gains, 'get_robot', fake_robot),                 mock.patch.object(tune_servo_gains, 'd_ladder', capture),                 mock.patch('builtins.print'):
            tune_servo_gains.main(['--joint', '3'])
            tune_servo_gains.main(['--joint', '3', '--d-max', '16383'])
        self.assertEqual([tune_servo_gains.DEFAULT_D_MAX, cfg.MAX_GAIN], captured)

    def test_the_max_step_option(self):
        def no_robot():
            raise AssertionError('the robot was touched')

        with mock.patch.object(tune_servo_gains, 'get_robot', no_robot), mock.patch('builtins.print') as shown:
            self.assertEqual(1, tune_servo_gains.main(['--joint', '1', '--max-step', '5']))
            self.assertEqual(1, tune_servo_gains.main(['--joint', '1', '--max-step', '-20']))
        self.assertIn('--max-step', ' '.join(str(call.args[0]) for call in shown.call_args_list))

        def no_robot_found():
            raise tune_servo_gains.DynamixelError('no robot in the tests')

        # 8 is the smallest step and is accepted: main gets as far as the robot
        with mock.patch.object(tune_servo_gains, 'get_robot', no_robot_found), mock.patch('builtins.print'):
            self.assertEqual(1, tune_servo_gains.main(['--joint', '1', '--max-step', '8']))

    def test_the_max_step_option_defaults_to_none(self):
        captured = []
        original = argparse.ArgumentParser.parse_args

        def capture(parser, argv=None, namespace=None):
            args = original(parser, argv, namespace)
            captured.append(args)
            return args

        def no_robot_found():
            raise tune_servo_gains.DynamixelError('no robot in the tests')

        with mock.patch.object(argparse.ArgumentParser, 'parse_args', capture),                 mock.patch.object(tune_servo_gains, 'get_robot', no_robot_found), mock.patch('builtins.print'):
            tune_servo_gains.main(['--joint', '3'])
            tune_servo_gains.main(['--joint', '3', '--max-step', '20'])
        self.assertEqual([None, 20], [args.max_step for args in captured])

    @staticmethod
    def pose_result(number, recommended_by_d):
        sweeps = {d: Sweep(None if p is None else round(p / 0.65), p, p) for d, p in recommended_by_d.items()}
        return PoseResult(number, None, None, [], sweeps, None, False)

    def test_the_d_with_the_highest_p_of_the_joint_wins(self):
        results = [self.pose_result(1, {0: 1650, 250: 1950, 400: 1950, 640: 1950}),
                   self.pose_result(2, {0: 1650, 250: 1650, 400: 1950})]  # the run stopped at pose 2 of D 640
        self.assertEqual([0, 250, 400], swept_d_values(results))
        self.assertEqual([1650, 1650, 1950, None], [joint_p_at(results, d) for d in (0, 250, 400, 640)])
        self.assertEqual([400], d_candidates(results), 'one step of the sweep higher counts')
        self.assertIsNone(compared_d(results))
        self.assertEqual((1950, 400), recommend(results))

    def test_d_gains_that_share_the_best_p_without_a_comparison(self):
        results = [self.pose_result(1, {0: 1270, 250: 1950, 400: 1950}),
                   self.pose_result(2, {0: None, 250: 1950, 400: 1950})]
        self.assertEqual([None, 1950, 1950], [joint_p_at(results, d) for d in (0, 250, 400)])
        self.assertEqual([250, 400], d_candidates(results))
        self.assertEqual((1950, 250), recommend(results), 'not compared (the run stopped): the smallest')
        self.assertEqual((None, None), recommend([self.pose_result(1, {0: None})]))
        self.assertEqual((None, None), recommend([]))

    def test_parse_pose(self):
        pose = parse_pose('elbow_out')
        self.assertEqual(0, pose[0])
        np.testing.assert_allclose(np.radians([90, 90, 0, 0, 0, 0]), pose[1:])
        np.testing.assert_allclose(np.radians([90, 10, 20, 30, 40, 50]), parse_pose('90,10,20,30,40,50')[1:])
        for bad in ('90,10', 'nowhere', '1,2,3,4,5,x'):
            with self.assertRaises(Exception):
                parse_pose(bad)


SIM_DAMPING = 73.0


class SimulatedServo:
    """a second order position loop with a constant load: pwm = P / 128 * error, clamped to the goal pwm, the
    damping falls as P rises, so the step overshoots more and more; D adds damping and pwm noise, and narrows the
    friction band the servo comes to rest in (on the robot the damped joint arrives at the setpoint instead of running
    into the far side of the band, joint 6 on 23 October)"""

    FORCE_PER_PWM = 213.0
    DAMPING = SIM_DAMPING
    DAMPING_PER_D = 0.078
    NOISE_PER_D = 0.02
    MAX_VELOCITY = 2000.0  # positions per second, bounds the amplitude of an unstable loop
    FRICTION = 6000.0  # force units, about 28 pwm or 2.4 positions of error at P 1500, at D = 0
    FRICTION_D = 600.0  # the friction is FRICTION / (1 + D / FRICTION_D)

    def __init__(self, position, load=50000.0, resonance_above=None, resonance_slope=0.1, damping=SIM_DAMPING,
                 dt=DT, trajectory_time=None):
        """
        :param resonance_above: above this P gain the damping drops by resonance_slope per unit of P, so the loop
            rings and, with a large enough slope, oscillates with an amplitude bounded by the velocity limit
        :param trajectory_time: seconds the trajectory takes to reach a new goal, a linear ramp as the time-based
            profile makes with the move time; the loop then follows the trajectory. None: the trajectory is the goal
        """
        self.position = float(position)
        self.damping = damping
        self.resonance_above = resonance_above
        self.resonance_slope = resonance_slope
        self.velocity = 0.0
        self.p, self.i, self.d = 0, 0, 0
        self.goal_pwm = cfg.MAX_PWM
        self.load = load
        self.dt = dt
        self.now = 0.0
        self.trajectory_time = trajectory_time
        self._goal = position
        self._ramp_from = float(position)
        self._ramp_start = 0.0
        self.pwm = 0.0
        self.temperature = 30
        self.rng = np.random.default_rng(1)

    @property
    def goal(self):
        return self._goal

    @goal.setter
    def goal(self, value):
        self._ramp_from = self.trajectory()
        self._ramp_start = self.now
        self._goal = value

    def jump_to_goal(self):
        """a step: the trajectory is at the goal at once"""
        self._ramp_from = float(self._goal)

    def trajectory(self):
        if self.trajectory_time is None:
            return self._goal
        fraction = min(1.0, (self.now - self._ramp_start) / self.trajectory_time)
        return self._ramp_from + (self._goal - self._ramp_from) * fraction

    def advance(self, seconds):
        steps = max(1, int(round(seconds / self.dt)))
        friction = self.FRICTION / (1.0 + self.d / self.FRICTION_D)
        for _ in range(steps):
            error = self.trajectory() - self.position
            self.pwm = self.p / 128.0 * error - self.d / 16.0 * self.velocity * 0.0059
            self.pwm = max(-self.goal_pwm, min(self.goal_pwm, self.pwm))
            damping = self.damping + self.DAMPING_PER_D * self.d
            if self.resonance_above is not None and self.p > self.resonance_above:
                damping -= self.resonance_slope * (self.p - self.resonance_above)
            net = self.pwm * self.FORCE_PER_PWM - damping * self.velocity - self.load
            # coulomb friction: it sticks until the net force exceeds the friction, and stops at a zero crossing
            if self.velocity == 0.0:
                acceleration = 0.0 if abs(net) <= friction else net - friction * np.sign(net)
            else:
                acceleration = net - friction * np.sign(self.velocity)
            velocity = self.velocity + acceleration * self.dt
            if velocity * self.velocity < 0:
                velocity = 0.0
            self.velocity = max(-self.MAX_VELOCITY, min(self.MAX_VELOCITY, velocity))
            self.position += self.velocity * self.dt
            self.now += self.dt

    def state(self):
        noise = self.rng.normal(0, 1.0 + self.NOISE_PER_D * self.d)
        return ServoState(tick=int(1000 * self.now) % 32768, moving=0, moving_status=1, pwm=int(self.pwm + noise),
                          current=0, velocity=int(self.velocity), position=int(round(self.position)),
                          velocity_trajectory=0, position_trajectory=int(round(self.trajectory())), voltage=11.8,
                          temperature=self.temperature)


class FakeTuningHandler:
    """the handler methods the session uses, with the simulated servo under test"""

    def __init__(self, servo_config, joint, sim):
        self.servos = servo_config.servos
        self.joint = joint
        self.sim = sim
        self.profiles = {servo_id: servo.profile_registers() for servo_id, servo in self.servos.items()}
        self.goal_pwm = {servo_id: cfg.MAX_PWM for servo_id in self.servos}
        self.gains = {servo_id: (servo.p, servo.i, servo.d) for servo_id, servo in self.servos.items()}
        self.gain_writes = []
        self.goal_pwm_writes = []
        self.goal_writes = []  # ('move' or 'step', servo id, position)
        self.failed_reads = 0
        self.sim.p, self.sim.i, self.sim.d = self.gains[joint]
        self.fail_reads = False
        self.before_read = None  # called before every state read, for the tests to disturb the servo
        self.temperatures = {servo_id: 30 for servo_id in self.servos}

    def get_servo(self, servo_id):
        return self.servos[servo_id]

    def read_goal_pwm(self, servo_ids):
        return None if self.fail_reads else {servo_id: self.goal_pwm[servo_id] for servo_id in servo_ids}

    def read_temperatures(self, servo_ids):
        return None if self.fail_reads else {servo_id: self.temperatures[servo_id] for servo_id in servo_ids}

    def set_goal_pwm(self, servo_id, value):
        self.goal_pwm_writes.append((servo_id, value))
        self.goal_pwm[servo_id] = value
        if servo_id == self.joint:
            self.sim.goal_pwm = abs(value)
        return True

    def read_pid(self, servo_ids):
        return None if self.fail_reads else {servo_id: self.gains[servo_id] for servo_id in servo_ids}

    def set_pid_single_servo(self, servo_id, p, i, d):
        self.gains[servo_id] = (p, i, d)
        self.gain_writes.append((servo_id, p, i, d))
        if servo_id == self.joint:
            self.sim.p, self.sim.i, self.sim.d = p, i, d
        return True

    def set_pid(self, servo_ids=None):
        for servo_id in servo_ids or self.servos:
            servo = self.servos[servo_id]
            self.set_pid_single_servo(servo_id, servo.p, servo.i, servo.d)
        return True

    def set_profile(self, servo_id, velocity, acceleration):
        self.profiles[servo_id] = (velocity, acceleration)
        return True

    def read_profile(self, servo_ids):
        return {servo_id: self.profiles[servo_id] for servo_id in servo_ids}

    def read_state(self, servo_ids):
        if self.fail_reads:
            return None
        if self.before_read is not None:
            self.before_read()
        self.sim.advance(self.sim.dt)
        return {servo_id: self.sim.state() for servo_id in servo_ids}

    def move_servo_to_pos(self, servo_id, position):
        self.goal_writes.append(('move', servo_id, position))
        if servo_id == self.joint:
            self.sim.goal = position
        return True

    def step_servo_to_pos(self, servo_id, position):
        """a step in either drive mode: the profile registers go to 0, the goal is in the same packet"""
        self.goal_writes.append(('step', servo_id, position))
        self.profiles[servo_id] = (0, 0)
        if servo_id == self.joint:
            self.sim.goal = position
            self.sim.jump_to_goal()
        return True


class FakeTuningRobot(FakeRobot):

    def __init__(self, angles, joint, sim, resonance_by_pose=None, move_time=None):
        """
        :param resonance_by_pose: optional function of the commanded angles giving the resonance_above of the
            simulated servo there, so the knee differs per pose
        :param move_time: ms of the time-based profile of the joint under test, None for the velocity-based profile
        """
        super().__init__(angles)
        self.joint = joint
        self.sim = sim
        self.resonance_by_pose = resonance_by_pose
        servo = self.servo_config.servos[joint]
        servo.move_time = move_time
        self.servo_handler = FakeTuningHandler(self.servo_config, joint, sim)
        self.enabled = False

    def enable_servos(self):
        self.enabled = True
        return True

    def get_servo_temperatures(self):
        return {servo_id: 30 for servo_id in range(1, 7)}

    def move_servos(self, angles, velocities=None):
        super().move_servos(angles, velocities)
        self.sim.goal = self.servo_handler.get_servo(self.joint).target_position
        if self.resonance_by_pose is not None:
            self.sim.resonance_above = self.resonance_by_pose(angles)


def session_for(joint=3, resonance_above=None, resonance_slope=0.1, settings=None, writers=False,
                resonance_by_pose=None, load=50000.0, move_time=None, trajectory_time=None):
    pose = parse_pose('elbow_out')
    sim = SimulatedServo(2048, load=load, resonance_above=resonance_above, resonance_slope=resonance_slope,
                         trajectory_time=trajectory_time)
    robot = FakeTuningRobot(pose, joint, sim, resonance_by_pose, move_time)
    servo = robot.servo_handler.get_servo(joint)
    servo.set_target_position_from_angle(pose[joint], pose)
    sim.position = sim.goal = servo.target_position
    # the yaml D only, unless the test is about the D grid
    settings = settings or Settings(repeats=1, hold=0.5, rest=1.0, baseline_rest=1.0, move_degrees=0.0,
                                    tune_d=False)
    sample_file, trial_file = io.StringIO(), io.StringIO()
    sample_writer = trial_writer = None
    if writers:
        sample_writer = csv.DictWriter(sample_file, fieldnames=SAMPLE_FIELDS)
        trial_writer = csv.DictWriter(trial_file, fieldnames=TRIAL_FIELDS)
    session = TuningSession(robot, joint, settings, sample_writer, trial_writer, clock=lambda: sim.now,
                            sleep=sim.advance, stop=StopFlag())
    return session, robot, sim, sample_file, trial_file


class SessionTests(unittest.TestCase):
    """the sweep on the simulated servo; the planned moves between poses run without sleeping"""

    def patch_movement_time(self, sim):
        """execute_move keeps time with the simulation: every step advances it by its period"""
        for name, replacement in (('sleep', sim.advance), ('perf_counter', lambda: sim.now)):
            patcher = mock.patch('src.utils.movement.' + name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def session(self, **kwargs):
        session, robot, sim, sample_file, trial_file = session_for(**kwargs)
        self.patch_movement_time(sim)
        return session, robot, sim, sample_file, trial_file

    def test_the_sweep_finds_the_knee_and_restores_the_servo(self):
        session, robot, sim, _, _ = self.session(resonance_above=2000, writers=False)
        servo = robot.servo_handler.get_servo(3)
        results = session.run([parse_pose('elbow_out'), parse_pose('low')])
        self.assertEqual(2, len(results))
        self.assertEqual([1, 2], [r.number for r in results])
        extended = results[0]
        p_trials = [t for t in extended.trials if t.phase == 'p']
        self.assertEqual([1500, 1950, 2540], [t.p for t in p_trials])
        self.assertEqual([True, True, False], [t.passed for t in p_trials])
        self.assertIn('overshoot', p_trials[-1].flags)
        self.assertEqual({0: Sweep(2540, 1650, 1950)}, extended.sweeps, 'swept at the yaml D only')
        # the sag of the simulated load is 20 positions at P 1500 and the slope is P / 128
        self.assertAlmostEqual(-20, p_trials[0].steps[0].error, delta=2)
        self.assertAlmostEqual(1500 / 128, p_trials[0].pwm_slope, delta=1.0)
        self.assertTrue(all(m.moved for t in p_trials for m in t.steps))
        self.assertEqual({0}, {t.d for r in results for t in r.trials})
        self.assertEqual((1650, 0), recommend(results))
        # the servo is back at the yaml gains, profile and goal pwm
        self.assertEqual((servo.p, servo.i, servo.d), robot.servo_handler.gains[3])
        self.assertEqual((servo.profile_velocity, servo.profile_acceleration), robot.servo_handler.profiles[3])
        self.assertEqual(cfg.MAX_PWM, robot.servo_handler.goal_pwm[3])
        self.assertTrue(all(write[0] == 3 for write in robot.servo_handler.gain_writes), 'only the joint under test')
        # the goal pwm clamp: 3 times the holding pwm of about 234, written before the gains, the limit at the end
        writes = robot.servo_handler.goal_pwm_writes
        self.assertTrue(all(write[0] == 3 for write in writes))
        clamps = [value for _, value in writes if value != cfg.MAX_PWM]
        # the baseline sits on one side of the friction band, the clamp is 3 times what it holds there
        self.assertAlmostEqual(3 * abs(extended.baseline.pwm_mean), clamps[0], delta=30)
        self.assertGreater(max(clamps), clamps[0], 'widened once a rest needed more pwm')
        self.assertEqual((3, cfg.MAX_PWM), writes[-1])

    def test_an_unstable_trial_is_aborted_and_the_gains_restored_at_once(self):
        session, robot, sim, _, trial_file = self.session(resonance_above=1800, resonance_slope=0.8, writers=True)
        results = session.run([parse_pose('elbow_out')])
        trials = results[0].trials
        aborted = [t for t in trials if t.aborted is not None]
        self.assertEqual(1, len(aborted))
        self.assertEqual(1950, aborted[0].p)
        self.assertEqual('aborted', aborted[0].flags[0])
        self.assertEqual(1950, results[0].sweeps[0].knee_p)
        self.assertEqual(1270, results[0].sweeps[0].recommended_p)
        # the gains went back to the yaml right after the abort, before anything else was written
        writes = robot.servo_handler.gain_writes
        index = writes.index((3, 1950, 0, 0))
        self.assertEqual((3, 1500, 0, 0), writes[index + 1])
        self.assertEqual((1500, 0, 0), robot.servo_handler.gains[3])
        rows = list(csv.DictReader(io.StringIO(trial_file.getvalue()), fieldnames=TRIAL_FIELDS))
        self.assertEqual(len(trials), len(rows))
        self.assertTrue(rows[1]['aborted'])

    def test_the_arm_gets_time_to_settle_after_an_abort(self):
        self.assertEqual(1.5, tune_servo_gains.ABORT_SETTLE)
        session, robot, sim, _, _ = self.session(resonance_above=1800, resonance_slope=0.8)
        sleeps = []
        original = session.sleep

        def recording(seconds):
            sleeps.append((seconds, robot.servo_handler.gains[3]))
            original(seconds)
        session.sleep = recording
        session.run([parse_pose('elbow_out')])
        settles = [gains for seconds, gains in sleeps if seconds == tune_servo_gains.ABORT_SETTLE]
        self.assertEqual([(1500, 0, 0)], settles, 'once, after the abort, with the yaml gains already back')
        self.assertEqual(1, [seconds for seconds, _ in sleeps].count(tune_servo_gains.ABORT_SETTLE))

    def test_abort_restores_writes_the_goal_and_then_waits(self):
        session, robot, sim, _, _ = self.session()
        order = []
        with mock.patch.object(session, 'restore', side_effect=lambda: order.append('restore') or True),                 mock.patch.object(session, 'write_goal', side_effect=lambda goal: order.append('goal') or goal),                 mock.patch.object(session, 'sleep', side_effect=lambda seconds: order.append(('sleep', seconds))),                 mock.patch('builtins.print'):
            self.assertTrue(session.abort('oscillation'))
        self.assertEqual(['restore', 'goal', ('sleep', tune_servo_gains.ABORT_SETTLE)], order)

    def steps_of(self, joint, settings=None, load=50000.0):
        """the step sizes the session asks step_size for, and the step of the one trial it runs"""
        settings = settings or Settings(repeats=1, hold=0.5, rest=1.0, baseline_rest=1.0, move_degrees=0.0,
                                        max_factor=1.0, tune_d=False)
        session, robot, sim, _, _ = self.session(joint=joint, load=load, settings=settings)
        with mock.patch.object(tune_servo_gains, 'step_size', wraps=step_size) as sized:
            results = session.run([parse_pose('elbow_out')])
        return session, sized, results[0].trials

    def test_the_base_and_the_shoulder_take_smaller_steps(self):
        self.assertEqual({1: 15, 2: 15}, tune_servo_gains.MAX_STEP)
        self.assertEqual(60, tune_servo_gains.DEFAULT_MAX_STEP)
        self.assertIsNone(Settings().max_step)
        for joint in (1, 2):
            session, sized, trials = self.steps_of(joint, load=0.0)
            self.assertEqual(15, session.max_step)
            self.assertTrue(sized.called)
            self.assertEqual({15}, {call.kwargs['maximum'] for call in sized.call_args_list})
            self.assertTrue(trials)
            self.assertLessEqual(max(t.step for t in trials), 15)
            self.assertEqual(15, trials[0].step, 'the cap is what limits it at these gains')

    def test_the_other_joints_still_step_by_60(self):
        for joint in (4, 6):
            session, sized, trials = self.steps_of(joint, load=0.0)
            self.assertEqual(60, session.max_step)
            self.assertEqual({60}, {call.kwargs['maximum'] for call in sized.call_args_list})
            self.assertEqual(60, trials[0].step)

    def test_max_step_overrides_the_table(self):
        settings = Settings(repeats=1, hold=0.5, rest=1.0, baseline_rest=1.0, move_degrees=0.0, max_factor=1.0,
                            tune_d=False, max_step=30)
        for joint in (1, 6):
            session, sized, trials = self.steps_of(joint, settings=settings, load=0.0)
            self.assertEqual(30, session.max_step)
            self.assertEqual({30}, {call.kwargs['maximum'] for call in sized.call_args_list})
            self.assertEqual(30, trials[0].step)

    def test_a_failing_yaml_gain_is_swept_down(self):
        session, robot, sim, sample_file, _ = self.session(resonance_above=1300, resonance_slope=0.25, writers=True)
        results = session.run([parse_pose('elbow_out')])
        phases = [(t.phase, t.p, t.passed) for t in results[0].trials]
        self.assertEqual(('p', 1500, False), phases[0])
        self.assertEqual(('p_down', 1150, True), phases[1])
        self.assertEqual(Sweep(1500, 980, 1150), results[0].sweeps[0])
        rows = list(csv.DictReader(io.StringIO(sample_file.getvalue()), fieldnames=SAMPLE_FIELDS))
        self.assertEqual({'baseline', 'rest', 'step'}, {row['segment'] for row in rows})

    def test_a_sluggish_start_is_not_swept_down(self):
        # joint 2 at the low pose on 25 October: P 1000 at D 0 didn't settle, and every lower P oscillated
        session, robot, sim, _, _ = self.session()
        scripted = {1000: ChoiceTests.trial(['settle'], p=1000)}
        with mock.patch.object(session, 'run_trial', side_effect=lambda pose, phase, p, d, baseline: scripted[p]):
            trials, sweep = session.run_sweep(parse_pose('low'), 0, 1000, None)
        self.assertEqual([1000], [t.p for t in trials])
        self.assertEqual(tune_servo_gains.Sweep(1000, None, None), sweep)
        # a start that overshoots is too stiff: down from there
        scripted = {1000: ChoiceTests.trial(['overshoot'], p=1000), 770: ChoiceTests.trial(p=770)}
        with mock.patch.object(session, 'run_trial', side_effect=lambda pose, phase, p, d, baseline: scripted[p]):
            trials, sweep = session.run_sweep(parse_pose('low'), 0, 1000, None)
        self.assertEqual([1000, 770], [t.p for t in trials])
        self.assertEqual(tune_servo_gains.Sweep(1000, 650, 770), sweep)

    def test_the_move_test_streams_a_planned_move(self):
        settings = Settings(repeats=1, hold=0.5, rest=1.0, baseline_rest=1.0, move_degrees=8.0, max_factor=1.0,
                            tune_d=False)
        session, robot, sim, _, _ = self.session(settings=settings)
        results = session.run([parse_pose('elbow_out')])
        trial = results[0].trials[0]
        self.assertEqual(2, len(trial.tracking), 'away and back')
        self.assertGreater(trial.tracking[0].peak_trajectory_error, 0)
        self.assertGreater(len(robot.moves), 10, 'the move was streamed through move_servos')
        self.assertEqual(sim.goal, robot.servo_handler.get_servo(3).target_position, 'back at the pose')

    def test_a_stop_request_restores_and_raises(self):
        session, robot, sim, _, _ = self.session()
        session.stop.requested = True
        with self.assertRaises(tune_servo_gains.Stop):
            session.run([parse_pose('elbow_out')])
        self.assertTrue(session.restore())
        self.assertEqual((1500, 0, 0), robot.servo_handler.gains[3])

    def test_a_failed_read_back_stops(self):
        session, robot, sim, _, _ = self.session()
        robot.servo_handler.fail_reads = True
        with self.assertRaises(tune_servo_gains.Stop):
            session.run([parse_pose('elbow_out')])
        self.assertFalse(session.restore(), 'the restore reports that it could not verify')

    def test_the_p_of_the_joint_is_the_smallest_of_the_poses(self):
        # the 'low' pose resonates earlier than 'elbow_out', so its knee and recommendation are lower
        session, robot, sim, _, _ = self.session(
            resonance_by_pose=lambda angles: 2000 if angles[2] > np.radians(75) else 1600)
        results = session.run([parse_pose('elbow_out'), parse_pose('low')])
        self.assertEqual(1650, results[0].sweeps[0].recommended_p)
        self.assertEqual(1270, results[1].sweeps[0].recommended_p)
        self.assertEqual(1270, joint_p_at(results, 0))
        self.assertEqual((1270, 0), recommend(results))

    def grid_settings(self, **kwargs):
        return Settings(repeats=1, hold=0.5, rest=1.0, baseline_rest=1.0, move_degrees=0.0, **kwargs)

    def test_damping_that_raises_the_knee_wins(self):
        # D adds damping to the simulated loop, so the knee of 2540 at D 0 moves above the top of the sweep
        session, robot, sim, _, trial_file = self.session(resonance_above=2000, writers=True,
                                                          settings=self.grid_settings(d_values=(0, 400, 800)))
        sim.NOISE_PER_D = 0.003  # quiet enough for D 800
        results = session.run([parse_pose('elbow_out'), parse_pose('low')])
        for r in results:
            self.assertEqual({0: Sweep(2540, 1650, 1950), 400: Sweep(None, 1950, 2540),
                              800: Sweep(None, 1950, 2540)}, r.sweeps)
            # the sweep at D 400 starts at the P that passed last at D 0
            self.assertEqual(('p', 1950, 400), next((t.phase, t.p, t.d) for t in r.trials if t.d == 400))
        self.assertEqual([0, 400, 800], swept_d_values(results))
        self.assertEqual({0: 1650, 400: 1950, 800: 1950}, {d: joint_p_at(results, d) for d in (0, 400, 800)})
        # 400 and 800 share the best P: compared at it, the smaller one wins as 800 is no better
        self.assertEqual([400, 800], d_candidates(results))
        for r in results:
            compared = [(t.p, t.d) for t in r.trials if t.phase == 'd']
            self.assertEqual([(1950, 400), (1950, 800)], compared)
            self.assertEqual(400, r.recommended_d)
        self.assertEqual((1950, 400), recommend(results))
        # D by D: every pose at D 0 before any at D 400, so a run that stops keeps whole D gains
        rows = list(csv.DictReader(io.StringIO(trial_file.getvalue()), fieldnames=TRIAL_FIELDS))
        order = [(int(row['d']), int(row['pose'])) for row in rows if row['phase'] != 'd']
        self.assertEqual([(0, 1), (0, 2), (400, 1), (400, 2), (800, 1), (800, 2)], list(dict.fromkeys(order)))

    def test_d_that_adds_pwm_noise_is_a_warning_and_is_held_back(self):
        # until 13 November 2026 the noise failed the trial: D 250 was already noisy at every P, so no P passed there
        # and the grid ended. Now the trials pass with the warning, damping raises the knee as it would without the
        # noise, and the comparison of the candidates is what holds the noisier D gains back
        session, robot, sim, _, trial_file = self.session(resonance_above=2000, writers=True,
                                                          settings=self.grid_settings())
        sim.NOISE_PER_D = 0.05
        results = session.run([parse_pose('elbow_out'), parse_pose('low')])
        # D 250, 400 and 640 pass up to the top of the sweep: two D gains in a row that don't raise the P end the grid
        self.assertEqual([0, 250, 400, 640], swept_d_values(results))
        self.assertEqual([1650, 1950, 1950, 1950], [joint_p_at(results, d) for d in (0, 250, 400, 640)])
        for r in results:
            noisy = [t for t in r.trials if t.d > 0]
            self.assertTrue(noisy)
            self.assertTrue(all(t.passed and t.flags == () for t in noisy), 'the noise alone fails nothing')
            self.assertTrue(all('pwm_noise' in t.warnings for t in noisy))
            self.assertTrue(all(t.warnings == () for t in r.trials if t.d == 0))
            self.assertEqual(Sweep(None, 1950, 2540), r.sweeps[250], 'no knee')
            self.assertTrue(r.d_tested)
            self.assertEqual(250, r.recommended_d, 'the larger D gains add noise and gain nothing')
        self.assertEqual([250, 400, 640], d_candidates(results))
        self.assertEqual((1950, 250), recommend(results))
        rows = list(csv.DictReader(io.StringIO(trial_file.getvalue()), fieldnames=TRIAL_FIELDS))
        noisy_rows = [row for row in rows if row['phase'] != 'd' and int(row['d']) > 0]
        self.assertTrue(noisy_rows)
        self.assertTrue(all(row['flags'] == '' and row['warnings'] == 'pwm_noise' for row in noisy_rows))

    def test_the_grid_goes_on_while_the_overshoot_at_the_knee_falls(self):
        # joint 4 on 18 November 2026: the knee stayed at P 1010 from D 250 to 640 while the overshoot fell from 11.7 to
        # 8.7, and the grid stopped before D 1020. The sweeps are scripted: the simulated servo can't be made to keep
        # its knee in one step of P while its overshoot falls with D
        session, robot, sim, _, _ = self.session(settings=self.grid_settings(d_values=(0, 250, 400, 640, 1020, 1640,
                                                                                        2620)))
        scripted = {0: (780, 510, 8.0), 250: (1010, 660, 11.7), 400: (1010, 660, 10.0), 640: (1010, 660, 8.7),
                    1020: (1300, 850, 6.5), 1640: (1300, 850, 6.4), 2620: (1300, 850, 6.4)}

        def sweep_pose(index, pose, d):
            knee_p, recommended, overshoot = scripted[d]
            result = knee_pose(index, {d: (knee_p, recommended, overshoot, ('overshoot',))})
            result = result._replace(pose=pose)
            if session.results:
                previous = session.results[0]
                result = previous._replace(trials=previous.trials + result.trials, sweeps={**previous.sweeps,
                                                                                           **result.sweeps})
            session.results[:] = [result]
            return True

        with mock.patch.object(session, 'sweep_pose', side_effect=sweep_pose), \
                mock.patch.object(session, 'compare_d') as compared, mock.patch('builtins.print') as shown:
            results = session.run([parse_pose('elbow_out')])
        output = '\n'.join(str(call.args[0]) for call in shown.call_args_list if call.args)
        self.assertEqual([0, 250, 400, 640, 1020, 1640, 2620], swept_d_values(results))
        self.assertEqual([510, 660, 660, 660, 850, 850, 850], [joint_p_at(results, d) for d in scripted])
        self.assertIn('D 400: the overshoot at the knee (pose 1, p=1010) fell from 11.7 to 10.0 positions, D still '
                      'helps', output)
        self.assertIn('D 640: the overshoot at the knee (pose 1, p=1010) fell from 10.0 to 8.7 positions, D still '
                      'helps', output)
        self.assertNotIn('D 250: the overshoot', output, 'the P of the joint rose there')
        self.assertNotIn('D 1020: the overshoot', output)
        self.assertEqual(1, output.count('the grid stops'))
        self.assertIn('the last 2 D gains neither raised the P of the joint nor lowered the overshoot at its knee: '
                      'the grid stops', output)
        compared.assert_called_once_with([1020, 1640, 2620], 850)
        # the rule of the P of the joint alone stops at D 640, before D 1020 is tried
        self.assertTrue(grid_is_done([joint_p_at(results, d) for d in (0, 250, 400, 640)], 2))

    def test_the_grid_stops_when_the_overshoot_at_the_knee_does_not_fall(self):
        session, robot, sim, _, _ = self.session(settings=self.grid_settings(d_values=(0, 250, 400, 640, 1020)))
        scripted = {0: (780, 510, 8.0), 250: (1010, 660, 11.7), 400: (1010, 660, 11.5), 640: (1010, 660, 11.3),
                    1020: (1010, 660, 5.0)}

        def sweep_pose(index, pose, d):
            knee_p, recommended, overshoot = scripted[d]
            result = knee_pose(index, {d: (knee_p, recommended, overshoot, ('overshoot',))})
            session.results[:] = [result._replace(sweeps={**(session.results[0].sweeps if session.results else {}),
                                                          **result.sweeps})]
            return True

        with mock.patch.object(session, 'sweep_pose', side_effect=sweep_pose), \
                mock.patch.object(session, 'compare_d'), mock.patch('builtins.print') as shown:
            results = session.run([parse_pose('elbow_out')])
        output = '\n'.join(str(call.args[0]) for call in shown.call_args_list if call.args)
        self.assertEqual([0, 250, 400, 640], swept_d_values(results), 'stopped before D 1020')
        self.assertNotIn('still helps', output)
        self.assertIn('neither raised the P of the joint nor lowered the overshoot at its knee', output)

    def test_damping_is_chosen_for_the_friction_band_it_lands_in(self):
        # an unloaded roll like joint 6: nothing overshoots and every D passes up to the top of the sweep, so the
        # D gains share the P of the joint; D narrows the band the steps come to rest in
        session, robot, sim, _, trial_file = self.session(joint=6, load=0.0, writers=True,
                                                          settings=self.grid_settings(d_values=(0, 200, 400)))
        sim.NOISE_PER_D = 0.005  # so the largest D of the grid passes the pwm noise rule as well
        results = session.run([parse_pose('elbow_out'), parse_pose('low')])
        self.assertEqual([0, 200, 400], d_candidates(results))
        for r in results:
            self.assertEqual({Sweep(None, 780, 1010)}, set(r.sweeps.values()))
            d_trials = [t for t in r.trials if t.phase == 'd']
            self.assertEqual([0, 200, 400], [t.d for t in d_trials])
            self.assertEqual({780}, {t.p for t in d_trials})
            self.assertTrue(all(t.passed for t in d_trials))
            self.assertLessEqual(max(t.max_overshoot for t in d_trials), 1.0, 'no D wins on overshoot')
            self.assertGreaterEqual(d_trials[0].mean_abs_error - d_trials[-1].mean_abs_error, 1.5)
            self.assertEqual(400, r.recommended_d)
        self.assertEqual((780, 400), recommend(results))
        rows = list(csv.DictReader(io.StringIO(trial_file.getvalue()), fieldnames=TRIAL_FIELDS))
        self.assertEqual('{:.1f}'.format(results[-1].trials[-1].mean_abs_error), rows[-1]['mean_abs_error'])

    def clamps_of(self, session):
        """the clamps the trials of the session set, in order"""
        clamps = []
        original = session.set_clamp

        def recording(clamp):
            clamps.append(clamp)
            original(clamp)
        session.set_clamp = recording
        return clamps

    def test_the_unloaded_joints_run_at_the_full_clamp(self):
        settings = Settings(repeats=1, hold=0.5, rest=1.0, baseline_rest=1.0, move_degrees=0.0, max_factor=1.0,
                            tune_d=False)
        self.assertIn(6, tune_servo_gains.UNLOADED_JOINTS)
        session, robot, sim, _, _ = self.session(joint=6, load=0.0, settings=settings)
        clamps = self.clamps_of(session)
        results = session.run([parse_pose('elbow_out')])
        self.assertEqual(cfg.MAX_PWM, clamps[0])
        self.assertEqual({cfg.MAX_PWM}, {t.clamp for t in results[0].trials})
        self.assertTrue(results[0].trials[0].passed)
        self.assertEqual({(6, cfg.MAX_PWM)}, set(robot.servo_handler.goal_pwm_writes), 'never below the limit')
        # a loaded joint keeps 3 times its holding pwm
        session, robot, sim, _, _ = self.session(joint=3, settings=settings)
        clamps = self.clamps_of(session)
        results = session.run([parse_pose('elbow_out')])
        self.assertAlmostEqual(3 * abs(results[0].baseline.pwm_mean), clamps[0], delta=30)
        self.assertLess(clamps[0], cfg.MAX_PWM)
        # --pwm-clamp still overrides on every joint
        session, robot, sim, _, _ = self.session(joint=6, load=0.0, settings=settings._replace(clamp=600))
        clamps = self.clamps_of(session)
        results = session.run([parse_pose('elbow_out')])
        self.assertEqual(600, clamps[0])
        self.assertEqual({600}, {t.clamp for t in results[0].trials})

    def test_an_environmental_abort_stops_the_run_and_keeps_the_finished_poses(self):
        session, robot, sim, _, trial_file = self.session(writers=True)
        # the servo gets too warm during the first trial of the second pose
        limit = int(AbortLimits().max_temperature)
        robot.servo_handler.before_read = lambda: setattr(sim, 'temperature',
                                                          limit if session.trial_index >= 6 else 30)
        with self.assertRaises(tune_servo_gains.Stop) as raised:
            session.run([parse_pose('elbow_out'), parse_pose('low')])
        self.assertIn('temperature {}C'.format(limit), str(raised.exception))
        self.assertEqual([1], [r.number for r in session.results], 'the finished pose is kept')
        self.assertEqual((1500, 0, 0), robot.servo_handler.gains[3], 'restored at the abort')
        rows = list(csv.DictReader(io.StringIO(trial_file.getvalue()), fieldnames=TRIAL_FIELDS))
        self.assertEqual('temperature {}C'.format(limit), rows[-1]['aborted'])

    def test_a_warm_servo_is_not_tried(self):
        session, robot, sim, _, _ = self.session()
        # warmer than the abort limit less the allowed rise of a trial, or than the warning of the robot
        robot.servo_handler.temperatures[3] = int(AbortLimits().max_temperature - Criteria().max_temperature_rise) + 1
        with self.assertRaises(tune_servo_gains.Stop) as raised:
            session.run([parse_pose('elbow_out')])
        self.assertIn('cool', str(raised.exception))

    def test_a_leftover_clamp_is_reset_at_the_start(self):
        session, robot, sim, _, _ = self.session(settings=Settings(repeats=1, hold=0.5, rest=1.0, baseline_rest=1.0,
                                                                   move_degrees=0.0, max_factor=1.0, tune_d=False))
        robot.servo_handler.goal_pwm[3] = 300
        sim.goal_pwm = 300
        session.run([parse_pose('elbow_out')])
        self.assertEqual((3, cfg.MAX_PWM), robot.servo_handler.goal_pwm_writes[0])
        self.assertEqual(cfg.MAX_PWM, session.original_goal_pwm)

    def test_a_flaky_read_is_retried(self):
        session, robot, sim, _, _ = self.session(settings=Settings(repeats=1, hold=0.5, rest=1.0, baseline_rest=1.0,
                                                                   move_degrees=0.0, max_factor=1.0, tune_d=False))
        handler = robot.servo_handler
        original = handler.read_state
        calls = []

        def flaky(servo_ids):
            calls.append(1)
            return None if len(calls) % 50 == 0 else original(servo_ids)
        handler.read_state = flaky
        results = session.run([parse_pose('elbow_out')])
        self.assertEqual(1, len(results))
        self.assertGreater(session.read_failures, 0)

    def test_a_pose_the_planner_refuses_is_skipped(self):
        session, robot, sim, _, _ = self.session()
        results = session.run([parse_pose('90,0,0,0,0,0'), parse_pose('elbow_out')])
        self.assertEqual(1, len(results))


class FinishTests(unittest.TestCase):

    def test_a_restored_servo_lets_the_arm_lower_itself(self):
        robot = FakeRobot([0, pi / 2, 0.4, 0.8, 0, 0, 0])
        with mock.patch.object(tune_servo_gains.record_static_load, 'lower_and_disable_torque') as lower, \
                mock.patch.object(tune_servo_gains.record_static_load, 'disable_torque_on_request') as prompt:
            tune_servo_gains.finish(robot, True)
        lower.assert_called_once_with(robot)
        prompt.assert_not_called()

    def test_a_servo_without_its_yaml_gains_gets_the_prompt_and_no_move(self):
        robot = FakeRobot([0, pi / 2, 0.4, 0.8, 0, 0, 0])
        with mock.patch.object(tune_servo_gains.record_static_load, 'lower_and_disable_torque') as lower, \
                mock.patch.object(tune_servo_gains.record_static_load, 'disable_torque_on_request') as prompt, \
                mock.patch('builtins.input', return_value='') as read, mock.patch('builtins.print') as shown:
            tune_servo_gains.finish(robot, False)
        lower.assert_not_called()
        prompt.assert_called_once_with(robot)
        read.assert_not_called()  # the prompt itself is patched out
        self.assertEqual([], robot.moves)
        self.assertIn('not lowered', shown.call_args_list[0].args[0])


class TimeBasedStepTests(unittest.TestCase):
    """the P and D stages for a servo with a move_time: its profile registers are (move_time, 0) between the steps,
    and the steps are written with step_servo_to_pos, which leaves them at 0"""

    def session(self, move_time=20, settings=None):
        settings = settings or Settings(repeats=1, hold=0.5, rest=1.0, baseline_rest=1.0, move_degrees=0.0,
                                        max_factor=1.0, tune_d=False)
        session, robot, sim, _, _ = session_for(joint=4, load=0.0, settings=settings, move_time=move_time)
        for name, replacement in (('sleep', sim.advance), ('perf_counter', lambda: sim.now)):
            patcher = mock.patch('src.utils.movement.' + name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        return session, robot, sim

    def test_the_steps_are_steps_and_the_profile_registers_come_back(self):
        session, robot, sim = self.session()
        handler = robot.servo_handler
        self.assertEqual((20, 0), handler.profiles[4])
        results = session.run([parse_pose('elbow_out')])
        self.assertEqual(1, len(results))
        steps = [write for write in handler.goal_writes if write[0] == 'step']
        self.assertEqual(4, len(steps), 'up, back, down, back, once')
        self.assertTrue(all(write[1] == 4 for write in steps))
        self.assertTrue([write for write in handler.goal_writes if write[0] == 'move'],
                        'the goals that are not steps follow the profile')
        servo = handler.get_servo(4)
        self.assertEqual((20, 0), handler.profiles[4], 'the move time, not the yaml velocity and acceleration')
        self.assertNotEqual((servo.profile_velocity, servo.profile_acceleration), handler.profiles[4])

    def test_the_streamed_move_runs_with_the_move_time(self):
        settings = Settings(repeats=1, hold=0.5, rest=1.0, baseline_rest=1.0, move_degrees=8.0, max_factor=1.0,
                            tune_d=False)
        session, robot, sim = self.session(settings=settings)
        seen = []
        original = robot.move_servos

        def recording(angles, velocities=None):
            seen.append(robot.servo_handler.profiles[4])
            original(angles, velocities)
        robot.move_servos = recording
        session.run([parse_pose('elbow_out')])
        self.assertGreater(len(seen), 100)
        self.assertEqual({(20, 0)}, set(seen), 'the steps left (0, 0), the profile is back before the moves')

    def test_the_restore_writes_and_verifies_the_move_time(self):
        session, robot, sim = self.session()
        handler = robot.servo_handler
        handler.profiles[4] = (0, 0)
        self.assertTrue(session.restore())
        self.assertEqual((20, 0), handler.profiles[4])
        # a profile that doesn't take is found by the read back; the yaml velocity and acceleration are not it
        handler.set_profile = lambda servo_id, velocity, acceleration: True
        handler.profiles[4] = (300, 100)
        with self.assertLogs(tune_servo_gains.logger, 'ERROR'):
            self.assertFalse(session.restore())

    def test_the_restore_puts_the_velocity_lead_of_the_servo_back_once_it_was_noted(self):
        session, robot, sim = self.session()
        servo = robot.servo_handler.get_servo(4)
        own = VelocityFeedforward(3.0, 20)
        servo.velocity_feedforward = own
        session.restore()
        self.assertIs(own, servo.velocity_feedforward, 'nothing was noted yet, the restore leaves the servo alone')
        servo.velocity_feedforward = None
        session.restore()
        self.assertIsNone(servo.velocity_feedforward)
        servo.velocity_feedforward = own
        session.snapshot()
        servo.velocity_feedforward = VelocityFeedforward(9.0, 5)
        self.assertTrue(session.restore())
        self.assertIs(own, servo.velocity_feedforward)

    def test_a_servo_without_move_time_is_unchanged(self):
        session, robot, sim = self.session(move_time=None)
        handler = robot.servo_handler
        servo = handler.get_servo(4)
        handler.profiles[4] = (0, 0)
        self.assertTrue(session.restore())
        self.assertEqual((servo.profile_velocity, servo.profile_acceleration), handler.profiles[4])
        session.run([parse_pose('elbow_out')])
        self.assertEqual((servo.profile_velocity, servo.profile_acceleration), handler.profiles[4])
        self.assertEqual(4, sum(1 for write in handler.goal_writes if write[0] == 'step'),
                         'steps go through step_servo_to_pos in either drive mode')


class SweepTests(unittest.TestCase):
    """the sweep of the feedforward stage"""
    DT = 0.01

    def test_the_sweep_starts_and_ends_at_the_pose_within_the_amplitude(self):
        for speed in (10, 20, 30, 100):  # at 100 the legs of 15 degrees are triangles
            steps = ff_sweep(15, speed, self.DT)
            offsets = [s.offset for s in steps]
            self.assertEqual(0.0, steps[0].time)
            self.assertEqual(0.0, offsets[0])
            self.assertAlmostEqual(0.0, offsets[-1], places=9)
            self.assertLessEqual(max(offsets), 15 + 1e-9)
            self.assertGreaterEqual(min(offsets), -15 - 1e-9)
            self.assertAlmostEqual(15, max(offsets), delta=0.01)
            self.assertAlmostEqual(-15, min(offsets), delta=0.01)
            if speed < 70:  # five legs, each its distance over the speed plus the time of the ramps
                self.assertAlmostEqual(8 * 15 / speed + 5 * FF_RAMP, steps[-1].time, places=9)
            np.testing.assert_allclose(np.diff([s.time for s in steps])[:-1], self.DT)
            legs = [direction for direction, _ in itertools.groupby(s.direction for s in steps)]
            self.assertEqual([1, -1, 1, -1, 1], legs, 'to +A, -A, +A, -A and back to the pose')

    def test_the_change_per_step_matches_the_speed_in_the_cruise_parts(self):
        steps = ff_sweep(15, 20, self.DT)
        offsets = np.array([s.offset for s in steps])
        per_second = np.abs(np.diff(offsets)) / self.DT  # per_second[k - 1] is the speed up to step k
        cruise = [k for k, s in enumerate(steps) if s.cruise]
        self.assertGreater(len(cruise), 200)
        for k in cruise:
            self.assertAlmostEqual(20, per_second[k - 1], places=6)
        self.assertLess(per_second.max(), 20 + 1e-6, 'never faster than the speed')
        # the ramp is linear: the speed grows by speed / ramp per second over the first 0.2 s of the first leg
        np.testing.assert_allclose(per_second[:20], 20 * (np.arange(20) + 0.5) * self.DT / FF_RAMP, rtol=1e-6)

    def test_cruise_only_where_the_speed_has_been_full_for_the_hold_time_and_still_is(self):
        steps = ff_sweep(15, 20, self.DT)
        offsets = np.array([s.offset for s in steps])
        full = np.concatenate([[False], np.abs(np.diff(offsets)) / self.DT > 20 - 1e-6])  # the speed up to step k
        hold_steps = round(FF_CRUISE_HOLD / self.DT) - 1
        for k, step in enumerate(steps):
            if step.cruise:
                self.assertTrue(all(full[k - hold_steps:k + 1]), 'full speed for the hold time up to step {}'.format(k))
                if steps[k + 1].cruise:  # the last step of a cruise is where the ramp down begins
                    self.assertTrue(full[k + 1], 'and still at full speed after it')
        # the first leg (15 degrees, from 0 to 0.95 s) reaches its speed at 0.2 s and cruises until 0.75 s
        first_leg = [s.time for s in steps if s.cruise and s.time < 0.95]
        self.assertAlmostEqual(FF_RAMP + FF_CRUISE_HOLD, first_leg[0], delta=1e-6)
        self.assertAlmostEqual(0.75, first_leg[-1], delta=1e-6)
        # the second leg (30 degrees, from 0.95 to 2.65 s) cruises from 1.15 s until 2.45 s
        second_leg = [s.time for s in steps if s.cruise and 0.95 < s.time < 2.65]
        self.assertAlmostEqual(0.95 + FF_RAMP + FF_CRUISE_HOLD, second_leg[0], delta=self.DT)
        self.assertAlmostEqual(2.45, second_leg[-1], delta=self.DT)
        self.assertEqual({-1}, {s.direction for s in steps if s.cruise and 0.95 < s.time < 2.65})

    def test_a_leg_too_short_to_reach_the_speed_is_a_triangle_without_cruise(self):
        steps = ff_sweep(1.0, 30, self.DT)
        self.assertFalse(any(s.cruise for s in steps))
        offsets = np.array([s.offset for s in steps])
        self.assertAlmostEqual(1.0, offsets.max(), delta=0.05)
        self.assertAlmostEqual(-1.0, offsets.min(), delta=0.05)
        peak = np.abs(np.diff(offsets)).max() / self.DT
        self.assertAlmostEqual(math.sqrt(2.0 * 30 / FF_RAMP), peak, delta=1.0, msg='the legs of 2 degrees peak below 30')
        # a leg that reaches the speed and doesn't hold it for the hold time has no cruise steps either: the legs of
        # 2.1 degrees at 10 degrees/s cruise for 0.01 s, those of 4.2 degrees for 0.22 s
        steps = ff_sweep(2.1, 10, self.DT)
        cruise_legs = [list(group) for flag, group in itertools.groupby(steps, key=lambda s: s.cruise) if flag]
        self.assertEqual(3, len(cruise_legs), 'only the three legs of 4.2 degrees')
        self.assertTrue(all(5 <= len(leg) <= 9 for leg in cruise_legs))

    def test_bad_arguments(self):
        for arguments in ((0, 10, 0.01), (10, 0, 0.01), (10, 10, 0)):
            with self.assertRaises(ValueError):
                ff_sweep(*arguments)
        for round_trips in (0, -1):
            with self.assertRaises(ValueError):
                ff_sweep(15, 10, self.DT, round_trips=round_trips)

    def test_two_round_trips_are_the_default_and_unchanged(self):
        self.assertEqual(ff_sweep(15, 20, self.DT), ff_sweep(15, 20, self.DT, round_trips=2))

    def test_the_joint_goes_to_both_ends_as_often_as_the_round_trips(self):
        for round_trips in (1, 2, 3):
            steps = ff_sweep(15, 20, self.DT, round_trips=round_trips)
            legs = [direction for direction, _ in itertools.groupby(s.direction for s in steps)]
            self.assertEqual([1, -1] * round_trips + [1], legs, 'to +A and -A each time, and back to the pose')
            numbers = [leg for leg, _ in itertools.groupby(s.leg for s in steps)]
            self.assertEqual(list(range(2 * round_trips + 1)), numbers)
            self.assertEqual(0.0, steps[0].offset)
            self.assertAlmostEqual(0.0, steps[-1].offset, places=9)
            self.assertAlmostEqual(15, max(s.offset for s in steps), delta=0.01)
            self.assertAlmostEqual(-15, min(s.offset for s in steps), delta=0.01)
            # 60 degrees per round trip at 20 degrees per second, and the ramps of every leg add their time
            self.assertAlmostEqual(3.0 * round_trips + (2 * round_trips + 1) * FF_RAMP, steps[-1].time, delta=self.DT)

    def test_one_round_trip_cruises_both_ways(self):
        steps = ff_sweep(15, 20, self.DT, round_trips=1)
        self.assertEqual({1, -1}, {s.direction for s in steps if s.cruise})
        # the first and the last leg are half a span, the one between them the whole span
        self.assertEqual([0, 1, 2], sorted({s.leg for s in steps if s.cruise}))


class BinSweepTests(unittest.TestCase):
    """pairing the two directions of a sweep per bin of the angle"""

    @staticmethod
    def passes(angle, direction, error, pwm=0.0, count=MIN_BIN_SAMPLES, spread=1.0):
        """count samples within +-spread degrees of the angle going the direction, as bin_sweep takes them:
        (angles, directions, errors, pwms)"""
        angles = list(angle + np.linspace(-spread, spread, count))
        return angles, [direction] * count, [error] * count, [pwm] * count

    @staticmethod
    def stack(*groups):
        return tuple(sum((list(group[i]) for group in groups), []) for i in range(4))

    def test_the_bins_are_a_grid_of_multiples_of_the_width(self):
        # 12.6 to 14.6 lie in [10, 15), 15.5 to 17.5 in [15, 20): the centres are 12.5 and 17.5
        data = self.stack(self.passes(13.6, 1, 2.0), self.passes(13.6, -1, 4.0), self.passes(16.5, 1, 6.0),
                          self.passes(16.5, -1, 8.0))
        bins = bin_sweep(*data, bin_degrees=5.0)
        self.assertEqual([12.5, 17.5], [b.angle for b in bins])
        self.assertEqual([(2.0, 4.0), (6.0, 8.0)], [(b.up.error, b.down.error) for b in bins])
        # another width is another grid, from 0 again: [10, 20) has all four passes, its centre is 15
        (wide,) = bin_sweep(*data, bin_degrees=10.0)
        self.assertEqual(15.0, wide.angle)
        self.assertEqual((60, 60), (wide.up.samples, wide.down.samples))
        self.assertAlmostEqual(4.0, wide.up.error)
        self.assertAlmostEqual(6.0, wide.down.error)

    def test_sweeps_over_different_spans_share_the_bins(self):
        wide = self.stack(*(self.passes(a, d, 1.0) for a in (-12.5, -2.5, 7.5, 17.5) for d in (1, -1)))
        narrow = self.stack(*(self.passes(a, d, 2.0) for a in (-2.5, 7.5) for d in (1, -1)))
        wide_bins, narrow_bins = bin_sweep(*wide, bin_degrees=5.0), bin_sweep(*narrow, bin_degrees=5.0)
        self.assertEqual([-12.5, -2.5, 7.5, 17.5], [b.angle for b in wide_bins])
        self.assertEqual([-2.5, 7.5], [b.angle for b in narrow_bins])

    def test_negative_angles_and_the_edges_of_the_bins(self):
        # a bin covers [k * width, (k + 1) * width): -5 belongs to [-5, 0), 0 to [0, 5)
        angles = [-5.0, -0.1, 0.0, 4.9, -5.0001]
        index = gain_tuning.bin_index(angles, 5.0)
        self.assertEqual([-1, -1, 0, 0, -2], list(index))
        data = self.stack(self.passes(-2.5, 1, 1.0), self.passes(-2.5, -1, 1.0), self.passes(-7.5, 1, 1.0),
                          self.passes(-7.5, -1, 1.0))
        self.assertEqual([-7.5, -2.5], [b.angle for b in bin_sweep(*data, bin_degrees=5.0)])
        edge = ([-5.0] * 30 + [-0.0001] * 30, [1] * 30 + [-1] * 30, [0.0] * 60, [0.0] * 60)
        self.assertEqual([-2.5], [b.angle for b in bin_sweep(*edge, bin_degrees=5.0)], 'both ends of [-5, 0)')

    def test_a_bin_needs_both_directions_with_enough_samples(self):
        data = self.stack(self.passes(2.5, 1, 1.0), self.passes(2.5, -1, 1.0),  # both: kept
                          self.passes(7.5, 1, 1.0),  # only up
                          self.passes(12.5, -1, 1.0),  # only down
                          self.passes(17.5, 1, 1.0, count=MIN_BIN_SAMPLES - 1),  # too few going up
                          self.passes(17.5, -1, 1.0),
                          self.passes(22.5, 1, 1.0, count=MIN_BIN_SAMPLES),  # exactly the minimum: kept
                          self.passes(22.5, -1, 1.0, count=MIN_BIN_SAMPLES))
        self.assertEqual([2.5, 22.5], [b.angle for b in bin_sweep(*data, bin_degrees=5.0)])
        self.assertEqual(30, MIN_BIN_SAMPLES)
        # the minimum is a parameter
        self.assertEqual([2.5, 17.5, 22.5], [b.angle for b in bin_sweep(*data, bin_degrees=5.0, minimum=29)])
        self.assertEqual([2.5, 22.5], [b.angle for b in bin_sweep(*data, bin_degrees=5.0, minimum=MIN_BIN_SAMPLES)])
        self.assertEqual([], bin_sweep(*data, bin_degrees=5.0, minimum=31))
        # a minimum of 0 does not make a bin of a direction nobody passed in
        self.assertEqual([2.5, 17.5, 22.5], [b.angle for b in bin_sweep(*data, bin_degrees=5.0, minimum=0)])

    def test_the_means_and_the_counts(self):
        rng = np.random.default_rng(0)
        up_errors, down_errors = rng.normal(-3.0, 1.0, 50), rng.normal(5.0, 1.0, 40)
        up_pwms, down_pwms = rng.normal(100.0, 5.0, 50), rng.normal(-60.0, 5.0, 40)
        data = (list(rng.uniform(10, 15, 50)) + list(rng.uniform(10, 15, 40)), [1] * 50 + [-1] * 40,
                list(up_errors) + list(down_errors), list(up_pwms) + list(down_pwms))
        (found,) = bin_sweep(*data, bin_degrees=5.0)
        self.assertEqual(12.5, found.angle)
        self.assertAlmostEqual(up_errors.mean(), found.up.error)
        self.assertAlmostEqual(down_errors.mean(), found.down.error)
        self.assertAlmostEqual(up_pwms.mean(), found.up.pwm)
        self.assertAlmostEqual(down_pwms.mean(), found.down.pwm)
        self.assertEqual((50, 40), (found.up.samples, found.down.samples))
        self.assertAlmostEqual((up_errors.mean() + down_errors.mean()) / 2, found.sag)
        self.assertAlmostEqual((up_errors.mean() - down_errors.mean()) / 2, found.half_difference)
        self.assertIsInstance(found, SweepBin)
        self.assertIsInstance(found.up, BinSide)

    def test_the_bins_are_sorted_by_angle_whatever_the_order_of_the_samples(self):
        groups = [self.passes(a, d, a) for a in (32.5, -7.5, 12.5, 2.5) for d in (-1, 1)]
        bins = bin_sweep(*self.stack(*groups), bin_degrees=5.0)
        self.assertEqual([-7.5, 2.5, 12.5, 32.5], [b.angle for b in bins])
        self.assertEqual([-7.5, 2.5, 12.5, 32.5], [b.up.error for b in bins])

    def test_other_directions_are_left_out_and_bad_arguments_raise(self):
        data = self.stack(self.passes(2.5, 1, 1.0), self.passes(2.5, -1, 1.0), self.passes(2.5, 0, 99.0))
        (found,) = bin_sweep(*data, bin_degrees=5.0)
        self.assertEqual((1.0, 1.0), (found.up.error, found.down.error))
        self.assertEqual([], bin_sweep([], [], [], [], 5.0))
        with self.assertRaises(ValueError):
            bin_sweep([1.0, 2.0], [1], [0.0, 0.0], [0.0, 0.0], 5.0)
        for width in (0, -5.0):
            with self.assertRaises(ValueError):
                bin_sweep(*data, bin_degrees=width)


def cruise_rows(direction, lag, drive, speed, count=20, gravity_lag=0.0, gravity_pwm=0.0, cruise=True, start=0.0,
                lag_wobble=0, leg=0, goal_start=2000):
    """the samples of one direction: the position trails its trajectory by lag positions in the direction of the
    move, the pwm is drive units in that direction, the goal moves at speed positions per millisecond; the gravity
    pulls the position and the pwm the same way for both directions"""
    rows = []
    for i in range(count):
        t = start + i * 0.01
        goal = int(round(goal_start + direction * speed * 1000 * (t - start)))
        wobble = lag_wobble * (1 if i % 2 else -1)
        rows.append(SweepSample(
            Sample(time=t, tick=0, goal=goal, position=int(goal - direction * (lag + wobble) - gravity_lag),
                   position_trajectory=goal, velocity=0, velocity_trajectory=0,
                   pwm=int(direction * drive + gravity_pwm), current=0, moving_status=0, voltage=11.8,
                   temperature=30), direction, cruise, leg))
    return rows


class CruiseStatisticsTests(unittest.TestCase):

    def test_the_lag_is_positive_when_the_servo_trails_in_either_direction(self):
        rows = cruise_rows(1, 6, 100, 0.3) + cruise_rows(-1, 6, 100, 0.3)
        # going up the position is below the trajectory, going down above it
        self.assertEqual(6, rows[0].sample.position_trajectory - rows[0].sample.position)
        self.assertEqual(-6, rows[-1].sample.position_trajectory - rows[-1].sample.position)
        stats = cruise_statistics(rows)
        self.assertAlmostEqual(6, stats.lag)
        self.assertAlmostEqual(100, stats.pwm, msg='the pwm in the direction of the move')
        different = cruise_statistics(cruise_rows(1, 6, 100, 0.3) + cruise_rows(-1, 2, 100, 0.3))
        self.assertAlmostEqual(4, different.lag, msg='the mean of the two directions')

    def test_a_pull_that_has_one_sign_cancels_in_the_mean_of_the_directions(self):
        # gravity: 4 positions and 50 pwm units the same way going up and going down
        plain = cruise_statistics(cruise_rows(1, 6, 100, 0.3) + cruise_rows(-1, 6, 100, 0.3))
        pulled = cruise_statistics(cruise_rows(1, 6, 100, 0.3, gravity_lag=4, gravity_pwm=50) +
                                   cruise_rows(-1, 6, 100, 0.3, gravity_lag=4, gravity_pwm=50))
        self.assertAlmostEqual(plain.lag, pulled.lag)
        self.assertAlmostEqual(plain.pwm, pulled.pwm)
        up_only = cruise_statistics(cruise_rows(1, 6, 100, 0.3, gravity_lag=4, gravity_pwm=50) +
                                    cruise_rows(-1, 6, 100, 0.3))
        self.assertNotAlmostEqual(plain.pwm, up_only.pwm, msg='a pull on one direction only is not cancelled')

    def test_the_speed_comes_from_the_goals(self):
        stats = cruise_statistics(cruise_rows(1, 6, 100, 0.34) + cruise_rows(-1, 6, 100, 0.34))
        self.assertAlmostEqual(0.34, stats.speed, delta=0.002)
        slower = cruise_statistics(cruise_rows(1, 6, 100, 0.2) + cruise_rows(-1, 6, 100, 0.2))
        self.assertAlmostEqual(0.2, slower.speed, delta=0.002, msg='the absolute slope, going down as well')
        self.assertEqual(40, stats.samples)

    def test_the_speed_is_the_slope_within_the_legs(self):
        # two legs going up cover the same goals 2 s apart: one line through both would have a much smaller slope
        up = (cruise_rows(1, 6, 100, 0.3, count=40, leg=0, goal_start=2000) +
              cruise_rows(1, 6, 100, 0.3, count=40, leg=2, start=2.0, goal_start=2000))
        down = cruise_rows(-1, 6, 100, 0.3, count=40, leg=1, start=1.0, goal_start=2100)
        stats = cruise_statistics(up + down)
        self.assertAlmostEqual(0.3, stats.speed, delta=0.002)

    def test_the_std_is_per_direction_and_averaged(self):
        stats = cruise_statistics(cruise_rows(1, 6, 100, 0.3, lag_wobble=1) + cruise_rows(-1, 6, 100, 0.3))
        self.assertAlmostEqual(0.5, stats.lag_std, delta=0.05, msg='1 in one direction, 0 in the other')
        self.assertAlmostEqual(0.0, stats.pwm_std)
        # the means of the two directions differ by a lot, which is no std
        far = cruise_statistics(cruise_rows(1, 12, 100, 0.3) + cruise_rows(-1, 0, 100, 0.3))
        self.assertAlmostEqual(0.0, far.lag_std, delta=0.01)

    def test_only_cruise_samples_count_and_a_direction_needs_some(self):
        rows = cruise_rows(1, 6, 100, 0.3) + cruise_rows(-1, 6, 100, 0.3)
        ramp = cruise_rows(1, 30, 500, 0.1, cruise=False, start=1.0)
        self.assertAlmostEqual(6, cruise_statistics(rows + ramp).lag, msg='the ramp samples are left out')
        with self.assertRaises(ValueError):
            cruise_statistics(cruise_rows(1, 6, 100, 0.3))
        with self.assertRaises(ValueError):
            cruise_statistics(cruise_rows(1, 6, 100, 0.3) + cruise_rows(-1, 6, 100, 0.3, count=2))


class FeedforwardArithmeticTests(unittest.TestCase):

    def test_points_on_a_line_give_its_friction_and_slope(self):
        speeds = [0.1, 0.2, 0.34]
        fit = fit_line(speeds, [3.0 + 20.0 * v for v in speeds])
        self.assertAlmostEqual(3.0, fit.friction)
        self.assertAlmostEqual(20.0, fit.slope)
        noisy = fit_line(speeds, [3.0 + 20.0 * 0.1 + 0.1, 3.0 + 20.0 * 0.2 - 0.1, 3.0 + 20.0 * 0.34 + 0.05])
        self.assertAlmostEqual(20.0, noisy.slope, delta=1.5, msg='a least squares line through all the points')
        with self.assertRaises(ValueError):
            fit_line([0.2, 0.2, 0.2], [1.0, 2.0, 3.0])

    def test_the_joint_4_numbers_give_the_line(self):
        # 18 November 2026, P 780: a lag of 8.0 positions at 145 positions/s and 12.9 at 392, a slope of 20 ms
        fit = fit_line([0.145, 0.392], [8.0, 12.9])
        self.assertAlmostEqual(20, fit.slope, delta=0.5)
        self.assertAlmostEqual(8.0 - 0.145 * fit.slope, fit.friction)

    def test_the_growth_of_the_lag_over_the_speeds(self):
        stats = [CruiseStatistics(speed, 3.0 + 20.0 * speed, 100, 0.5, 2, 40) for speed in (0.1, 0.2, 0.34)]
        self.assertAlmostEqual(20.0 * 0.24, lag_growth(LineFit(3.0, 20.0), stats))
        self.assertAlmostEqual(-1.0, lag_growth(LineFit(3.0, -1.0 / 0.24), stats))

    def test_rougher_speeds(self):
        before, after = [0.5, 1.0], [1.4, 2.5]
        self.assertEqual([1], rougher_speeds(before, after), 'more than a position more at the second speed only')
        self.assertEqual([], rougher_speeds(before, before))
        self.assertEqual([0, 1], rougher_speeds(before, after, limit=0.4), 'in the unit of the stds, pwm units too')
        self.assertEqual([], rougher_speeds([10.0, 12.0], [20.0, 25.0], limit=15.0))


class FeedforwardStageTests(unittest.TestCase):
    """the stage on the simulated servo: its lag grows with the speed (the damping of the loop is a back voltage the
    position error has to pay for) and the velocity lead takes that part away; joint 4 at P 1000"""
    P = 1000
    MOVE_TIME = 20
    # pwm per degree/s the simulated servo needs: the damping force per position/s over the force per pwm unit
    TRUE_K = SIM_DAMPING * 4096 / 360 / SimulatedServo.FORCE_PER_PWM

    def stage(self, velocity_feedforward=None, load=20000.0, reverse=False, move_time=MOVE_TIME, **overrides):
        """the session, robot and simulation of the stage; `velocity_feedforward` is the velocity lead of the servo in
        the yaml"""
        settings = Settings(repeats=1, hold=0.5, rest=1.0, baseline_rest=1.0, move_degrees=0.0, tune_d=False,
                            ff_degrees=10.0)._replace(**overrides)
        session, robot, sim, _, _ = session_for(joint=4, load=load, settings=settings, move_time=move_time,
                                                trajectory_time=0.01)
        robot.servo_handler.get_servo(4).velocity_feedforward = velocity_feedforward
        if reverse:  # the positions fall as the angle rises; the pose is at the middle of the range, the same position
            servo = robot.servo_handler.get_servo(4)
            servo.min_position, servo.max_position = servo.max_position, servo.min_position
        for name, replacement in (('sleep', sim.advance), ('perf_counter', lambda: sim.now)):
            patcher = mock.patch('src.utils.movement.' + name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        return session, robot, sim

    @staticmethod
    def run_stage(session):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = session.run_feedforward(parse_pose('elbow_out'))
        return result, out.getvalue()

    @staticmethod
    def watch_writes(session, robot):
        """note, per sweep of the stage, the speed, the coefficient it was given and every write of the sweep: the
        velocity_feedforward the servo had, the lead it added, its target position and the angles that were written"""
        servo = robot.servo_handler.get_servo(4)
        seen = []
        move_servos = robot.move_servos
        sweep = session.feedforward_sweep

        def marked(pose, speed, lead_gain):
            seen.append({'speed': speed, 'lead_gain': lead_gain, 'writes': []})
            return sweep(pose, speed, lead_gain)

        def noted(angles, velocities=None):
            move_servos(angles, velocities)
            if seen:
                seen[-1]['writes'].append((servo.velocity_feedforward, servo.lead_positions, servo.target_position,
                                           np.array(angles)))
        session.feedforward_sweep = marked
        robot.move_servos = noted
        return seen

    def test_without_a_velocity_feedforward_round_1_finds_the_coefficient_and_round_2_leads_with_it_and_is_flat(self):
        session, robot, sim = self.stage()
        handler = robot.servo_handler
        servo = handler.get_servo(4)
        seen = self.watch_writes(session, robot)
        result, output = self.run_stage(session)

        off, on = result.rounds
        fit = off.analysis.fit
        self.assertEqual((0, 0), (off.lead_gain, off.max_positions))
        self.assertAlmostEqual(self.TRUE_K, fit.coefficient, delta=0.08 * self.TRUE_K, msg='the lag route')
        self.assertAlmostEqual(self.TRUE_K, fit.k_pwm, delta=0.12 * self.TRUE_K, msg='the measured pwm route')
        self.assertAlmostEqual(1.0, fit.ratio, delta=0.15)
        self.assertEqual(self.P, fit.p)
        self.assertGreater(fit.intended_change, 2 * FF_LAG_TOLERANCE, 'without a lead the lag grows with the speed')
        # round 2 leads with the coefficient of round 1, to the 2 decimals of the yaml, and the default clamp
        self.assertEqual(round(fit.coefficient, 2), on.lead_gain)
        self.assertEqual(30, on.max_positions)
        self.assertIn('fitted from the lag', on.origin)
        self.assertLessEqual(abs(on.analysis.fit.intended_change), FF_LAG_TOLERANCE, 'the lag is flat with the lead')
        self.assertLess(abs(on.analysis.fit.intended_change), 0.3 * fit.intended_change)
        self.assertTrue(result.flat)
        self.assertEqual(on.lead_gain, result.recommended)
        self.assertAlmostEqual(self.TRUE_K, on.analysis.fit.coefficient, delta=0.08 * self.TRUE_K,
                               msg='the physics does not change with the lead')
        speeds = [s.speed for s in on.analysis.statistics]
        self.assertEqual([10.0, 20.0, 30.0], speeds)
        self.assertAlmostEqual(self.TRUE_K * 30 * 128 / self.P, on.analysis.statistics[2].lead, delta=1.0)
        self.assertEqual(0.0, off.analysis.statistics[2].lead)

        # round 1 really ran without a lead (the goals are those of the trajectory), round 2 with it
        self.assertEqual([0, 0, 0, on.lead_gain, on.lead_gain, on.lead_gain], [s['lead_gain'] for s in seen])
        self.assertEqual([10.0, 20.0, 30.0] * 2, [s['speed'] for s in seen])
        reference = FakeRobot(parse_pose('elbow_out')).servo_config.servos[4]
        for sweep in seen:
            self.assertGreater(len(sweep['writes']), 300)
            leads = []
            for velocity_feedforward, lead, target, angles in sweep['writes']:
                reference.set_target_position_from_angle(angles[4], angles)
                if sweep['lead_gain'] == 0:
                    self.assertIsNone(velocity_feedforward)
                    self.assertEqual(0, lead)
                    self.assertEqual(reference.target_position, target)
                else:
                    self.assertEqual(VelocityFeedforward(on.lead_gain, 30), velocity_feedforward)
                    self.assertEqual(reference.target_position + lead, target, 'the lead is all that was added')
                leads.append(abs(lead))
            if sweep['lead_gain'] == 0:
                self.assertEqual(0, max(leads))
            else:
                self.assertGreater(max(leads), 0.8 * self.TRUE_K * sweep['speed'] * 128 / self.P)
                self.assertEqual(0, leads[0], 'the sweep starts at rest')
                self.assertEqual(0, leads[-1], 'and ends at rest: the end point does not carry a lead')

        # afterwards the servo has its own (no) velocity_feedforward and its yaml gains
        self.assertIsNone(servo.velocity_feedforward)
        self.assertEqual((self.MOVE_TIME, 0), handler.profiles[4])
        self.assertEqual((1000, 0, 0), handler.gains[4])
        self.assertEqual(cfg.MAX_PWM, handler.goal_pwm[4])

        # only the joint under test moves in the sweeps, within 10 degrees of the pose
        pose = parse_pose('elbow_out')
        moves = np.array(robot.moves)
        others = [1, 2, 3, 5, 6]
        np.testing.assert_allclose(moves[:, others], np.tile(pose[others], (len(moves), 1)))
        self.assertAlmostEqual(np.radians(10), np.abs(moves[:, 4] - pose[4]).max(), places=6)

        # the report
        self.assertIn('round 1, without a lead', output)
        self.assertIn('round 2, with a lead of {:g} pwm per degree/s from the coefficient fitted'.format(on.lead_gain),
                      output)
        self.assertIn('ratio k_pwm / k_lag', output)
        self.assertIn('flat (within 1 position)', output)
        self.assertIn('more than 1 position, the speed dependent part is not taken out', output)
        self.assertIn('for servo_config.yaml, the coefficient of round 2 (from the coefficient fitted', output)
        self.assertIn('      pwm_per_degree_per_second: {:.2f}'.format(on.lead_gain), output)
        self.assertIn('      max_positions: 30', output)
        self.assertNotIn('makes the joint rougher', output)
        self.assertNotIn('pid:', output, 'the stage does not set P or D')

    def test_the_stage_numbers_are_those_of_fit_velocity_lead_on_the_csv_it_wrote(self):
        session, robot, sim = self.stage()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'tune_joint4_ff.csv')
            with open(path, 'w', newline='') as csv_file:
                session.ff_writer = csv.DictWriter(csv_file, fieldnames=FF_SAMPLE_FIELDS)
                session.ff_writer.writeheader()
                result, output = self.run_stage(session)
            with open(path, newline='') as csv_file:
                rows = list(csv.DictReader(csv_file))
            groups = fit_velocity_lead.group_rows(fit_velocity_lead.read_rows([path]))
            from_file = [fit_velocity_lead.analyse_group(key, members) for key, members in groups.items()]
            code, script_output = self.run_script(path)

        # the columns
        self.assertEqual(list(FF_SAMPLE_FIELDS), list(rows[0].keys()))
        self.assertEqual(['p', 'lead', 'lead_gain'], list(FF_SAMPLE_FIELDS[-3:]))
        self.assertEqual(['joint', 'pose', 'speed'], list(FF_SAMPLE_FIELDS[:3]), 'no column between the pose and speed')
        self.assertNotIn('ff1', FF_SAMPLE_FIELDS)
        self.assertEqual({'4'}, {row['joint'] for row in rows})
        self.assertEqual({str(self.P)}, {row['p'] for row in rows})
        self.assertEqual({'10.0', '20.0', '30.0'}, {row['speed'] for row in rows})
        self.assertEqual({'1', '-1'}, {row['direction'] for row in rows})
        self.assertEqual({'0', '1'}, {row['cruise'] for row in rows})
        off, on = result.rounds
        gains = {row['lead_gain'] for row in rows}
        self.assertEqual({'0', '{:g}'.format(on.lead_gain)}, gains)
        for gain in gains:
            mine = [row for row in rows if row['lead_gain'] == gain]
            leads = [int(row['lead']) * int(row['direction']) for row in mine if row['cruise'] == '1']
            if gain == '0':
                self.assertEqual({0}, set(int(row['lead']) for row in mine), 'no lead in round 1')
            else:
                self.assertGreater(len(set(int(row['lead']) for row in mine)), 5, 'the lead changes with the speed')
                self.assertGreaterEqual(min(leads), 0, 'in the direction of the move for every cruise sample')
                self.assertGreater(max(leads), 5)
        slow = [row for row in rows if row['lead_gain'] == '0' and row['speed'] == '10.0']
        steps = ff_sweep(10.0, 10.0, 0.01)
        self.assertGreater(len(slow), 3 * len(steps), 'sampled as often as it fits between two writes')
        seconds = [float(row['time']) for row in slow]
        self.assertEqual(sorted(seconds), seconds)

        # the numbers of the stage are the numbers of fit_velocity_lead on the csv
        self.assertEqual(2, len(from_file))
        for stage_round, analysed in zip(result.rounds, from_file):
            self.assertEqual(stage_round.analysis.statistics, analysed.statistics)
            self.assertEqual(stage_round.analysis.fit, analysed.fit)
            self.assertEqual(stage_round.analysis.p, analysed.p)
            self.assertEqual(stage_round.lead_gain, analysed.key.lead_gain)
        self.assertEqual(off.analysis.fit.coefficient, from_file[0].fit.coefficient)
        self.assertEqual(0, code)
        self.assertEqual(2, script_output.count('pwm_per_degree_per_second: '))
        for stage_round in result.rounds:
            fit = stage_round.analysis.fit
            self.assertIn('{:.2f} pwm per degree/s'.format(fit.k_pwm), script_output)
            self.assertIn('error = {:.2f} + {:.4f} * speed positions'.format(fit.friction_positions, fit.slope),
                          script_output)
        fit = off.analysis.fit  # the stage prints the two fits for the round without a lead
        self.assertIn('error = {:.2f} + {:.4f} * speed positions'.format(fit.friction_positions, fit.slope), output)
        self.assertIn('pwm = {:.1f} + {:.3f} * speed'.format(fit.friction_pwm, fit.k_pwm), output)
        for line in script_output.splitlines():  # every line of the table of a round is in the report of the stage
            if line.startswith('        ') and line.split()[0] in ('10', '20', '30'):
                self.assertIn(line, output)
        self.assertIn('pwm_per_degree_per_second: {:.2f}'.format(off.analysis.fit.coefficient), script_output)
        self.assertIn('pwm_per_degree_per_second: {:.2f}'.format(result.recommended), output)

    @staticmethod
    def run_script(path):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = fit_velocity_lead.main([path])
        return code, out.getvalue()

    def test_a_velocity_feedforward_on_the_servo_gives_its_coefficient_and_clamp_to_round_2(self):
        own = VelocityFeedforward(3.7, 25)  # a bit below the 3.9 the simulation needs: the lag is still flat
        session, robot, sim = self.stage(velocity_feedforward=own)
        servo = robot.servo_handler.get_servo(4)
        seen = self.watch_writes(session, robot)
        result, output = self.run_stage(session)

        off, on = result.rounds
        self.assertEqual(2, len(result.rounds))
        self.assertAlmostEqual(self.TRUE_K, off.analysis.fit.coefficient, delta=0.08 * self.TRUE_K)
        self.assertNotEqual(round(off.analysis.fit.coefficient, 2), 3.7, 'it is not the one that was fitted')
        self.assertEqual((3.7, 25), (on.lead_gain, on.max_positions))
        self.assertIn('velocity_feedforward block of servo_config.yaml', on.origin)
        self.assertTrue(result.flat)
        self.assertEqual(3.7, result.recommended)
        self.assertIn('      max_positions: 25', output)
        self.assertIn('lead of 3.7 pwm per degree/s from the velocity_feedforward block of servo_config.yaml, at most '
                      '25 positions', output)
        # round 1 ran without the velocity_feedforward of the servo, round 2 with it, and the servo has it again
        # afterwards
        for sweep in seen:
            for velocity_feedforward, lead, target, angles in sweep['writes']:
                if sweep['lead_gain'] == 0:
                    self.assertIsNone(velocity_feedforward)
                    self.assertEqual(0, lead)
                else:
                    self.assertEqual(own, velocity_feedforward)
        self.assertEqual(3, sum(1 for sweep in seen if sweep['lead_gain'] == 0))
        self.assertIs(own, servo.velocity_feedforward)

    def test_the_ff_lead_option_wins_over_the_velocity_feedforward_of_the_servo_and_the_fit(self):
        own = VelocityFeedforward(3.0, 25)
        session, robot, sim = self.stage(velocity_feedforward=own, ff_lead=3.8, ff_max_positions=22)
        result, output = self.run_stage(session)
        off, on = result.rounds
        self.assertEqual(2, len(result.rounds))
        self.assertEqual(3.8, on.lead_gain)
        self.assertEqual(25, on.max_positions, 'the clamp of the velocity_feedforward block of the yaml, there is one')
        self.assertIn('--ff-lead', on.origin)
        self.assertIn('from the --ff-lead option', output)
        self.assertEqual(3.8, result.recommended)
        self.assertIs(own, robot.servo_handler.get_servo(4).velocity_feedforward)

    def test_a_coefficient_that_is_too_small_is_corrected_once_and_the_corrected_one_is_right(self):
        session, robot, sim = self.stage(ff_lead=2.0, ff_max_positions=22)
        result, output = self.run_stage(session)
        self.assertEqual(3, len(result.rounds), 'without a lead, the coefficient, the corrected coefficient')
        off, small, corrected = result.rounds
        self.assertEqual((0, 2.0), (off.lead_gain, small.lead_gain))
        self.assertEqual((22, 22), (small.max_positions, corrected.max_positions))
        self.assertGreater(abs(small.analysis.fit.intended_change), FF_LAG_TOLERANCE)
        # the coefficient in force plus the slope that is left times P / 128, to 2 decimals
        fit = small.analysis.fit
        expected = round(2.0 + fit.intended_slope * self.P / 128, 2)
        self.assertEqual(expected, corrected.lead_gain)
        self.assertEqual(expected, tune_servo_gains.corrected_coefficient(small))
        self.assertAlmostEqual(self.TRUE_K, corrected.lead_gain, delta=0.08 * self.TRUE_K)
        self.assertIn('the correction of round 2', corrected.origin)
        self.assertLessEqual(abs(corrected.analysis.fit.intended_change), FF_LAG_TOLERANCE)
        self.assertLess(abs(corrected.analysis.fit.intended_change), abs(small.analysis.fit.intended_change))
        self.assertTrue(result.flat)
        self.assertEqual(corrected.lead_gain, result.recommended)
        self.assertIn('the coefficient should be {:g} instead of 2'.format(corrected.lead_gain), output)
        self.assertIn('round 3, with the corrected lead of {:g} pwm per degree/s'.format(corrected.lead_gain), output)
        self.assertIn('      pwm_per_degree_per_second: {:.2f}'.format(corrected.lead_gain), output)
        self.assertIn('      max_positions: 22', output)

    def test_a_lag_that_stays_not_flat_is_not_corrected_twice_and_the_report_says_so(self):
        # the stage is told the lag is never flat: the correction of round 2 is measured, a round 3 is not
        session, robot, sim = self.stage(ff_lead=2.0)
        with mock.patch.object(tune_servo_gains, 'is_flat', return_value=False):
            result, output = self.run_stage(session)
            flat, recommended = result.flat, result.recommended  # both ask is_flat
        self.assertEqual(3, len(result.rounds), 'never a second correction')
        first, second = result.lead_rounds
        self.assertGreater(second.lead_gain, first.lead_gain)
        self.assertFalse(flat)
        self.assertEqual(tune_servo_gains.corrected_coefficient(second), recommended)
        self.assertAlmostEqual(self.TRUE_K, recommended, delta=0.08 * self.TRUE_K)
        self.assertIn('is NOT flat in round 3', output)
        self.assertIn('{:g} (not measured)'.format(recommended), output)
        self.assertEqual(1, output.count('the coefficient should be'), 'one correction')

    def test_a_lead_at_its_clamp_is_not_chased_with_a_larger_coefficient(self):
        # the clamp of 6 positions cuts the lead at the faster speeds: no coefficient can make the lag flat
        session, robot, sim = self.stage(ff_max_positions=6)
        result, output = self.run_stage(session)
        self.assertEqual(2, len(result.rounds), 'no correction round, it would only chase the clamp')
        (on,) = result.lead_rounds
        self.assertTrue(on.clamped)
        self.assertFalse(result.flat)
        self.assertGreater(abs(on.analysis.fit.intended_change), FF_LAG_TOLERANCE)
        self.assertIsNone(result.recommended)
        self.assertIn('at its clamp of 6 positions', output)
        self.assertIn('no correction is measured, raise --ff-max-positions', output)
        self.assertIn('no coefficient to recommend: the lead was at its clamp of 6 positions in round 2', output)
        self.assertNotIn('velocity_feedforward:', output)
        self.assertNotIn('the coefficient should be', output)

    def test_a_lag_that_does_not_grow_with_the_speed_gives_no_lead_round(self):
        session, robot, sim = self.stage()
        analyse = tune_servo_gains.analyse_group

        def flat_lag(key, rows, *args):
            found = analyse(key, rows, *args)
            return dataclasses.replace(found, fit=dataclasses.replace(found.fit, slope=-0.05))
        with mock.patch.object(tune_servo_gains, 'analyse_group', flat_lag):
            result, output = self.run_stage(session)
        self.assertEqual(1, len(result.rounds), 'nothing to verify')
        self.assertEqual((), result.lead_rounds)
        self.assertIsNone(result.recommended)
        self.assertIsNone(robot.servo_handler.get_servo(4).velocity_feedforward)
        self.assertIn('does not grow with the speed', output)
        self.assertIn('no lead to recommend', output)
        self.assertNotIn('velocity_feedforward:', output)
        # a lead asked for with --ff-lead is measured all the same
        session, robot, sim = self.stage(ff_lead=1.0)
        with mock.patch.object(tune_servo_gains, 'analyse_group', flat_lag):
            result, output = self.run_stage(session)
        self.assertEqual([0, 1.0], [r.lead_gain for r in result.rounds[:2]])

    def test_a_servo_mounted_reversed_gets_the_same_coefficient_and_a_lead_in_the_direction_of_its_position(self):
        # joint 3 of servo_config.yaml has its positions fall as its angle rises: the lag in positions of a joint that
        # trails its trajectory while the angle goes up is negative, and the stage must still see it grow with speed
        session, robot, sim = self.stage(reverse=True)
        servo = robot.servo_handler.get_servo(4)
        self.assertGreater(servo.min_position, servo.max_position)
        buffer = io.StringIO()
        session.ff_writer = csv.DictWriter(buffer, fieldnames=FF_SAMPLE_FIELDS)
        session.ff_writer.writeheader()
        result, output = self.run_stage(session)
        self.assertEqual(-1, position_sign(servo))
        self.assertEqual(2, len(result.rounds))
        self.assertAlmostEqual(self.TRUE_K, result.rounds[0].analysis.fit.coefficient, delta=0.08 * self.TRUE_K)
        self.assertTrue(result.flat)
        self.assertNotIn('does not grow', output)
        # the direction in the csv is the one of the position: the goal rises where it is 1, and the lead is in it
        rows = [row for row in csv.DictReader(io.StringIO(buffer.getvalue())) if row['cruise'] == '1']
        self.assertTrue(rows)
        steps = 0
        for before, after in zip(rows, rows[1:]):
            if before['direction'] == after['direction'] and before['speed'] == after['speed'] and \
                    before['lead_gain'] == after['lead_gain']:
                steps += 1
        self.assertGreater(steps, 1000)
        led = [row for row in rows if row['lead_gain'] != '0']
        self.assertGreater(max(int(row['lead']) * int(row['direction']) for row in led), 5)
        self.assertGreaterEqual(min(int(row['lead']) * int(row['direction']) for row in led), 0)

    def endings(self):
        """the three ways a stage can end early, as a function that raises (or has the stage stop) when `when` says"""
        return {
            'a stop request': lambda session, handler, sim, when: setattr(session.stop, 'requested', when()),
            'an abort rule': lambda session, handler, sim, when: setattr(
                sim, 'temperature', int(AbortLimits().max_temperature) if when() else 30),
            'ctrl+c': lambda session, handler, sim, when: self.interrupt(when)}

    @staticmethod
    def interrupt(when):
        if when():
            raise KeyboardInterrupt

    def test_the_velocity_feedforward_of_the_servo_is_back_whatever_ends_the_stage(self):
        own = VelocityFeedforward(3.7, 25)
        expected = {'a stop request': tune_servo_gains.Stop, 'an abort rule': tune_servo_gains.Stop,
                    'ctrl+c': KeyboardInterrupt}
        for ending, make in self.endings().items():
            for where in ('without a lead', 'with a lead'):
                with self.subTest(ending=ending, where=where):
                    session, robot, sim = self.stage(velocity_feedforward=own)
                    handler = robot.servo_handler
                    servo = handler.get_servo(4)
                    if where == 'without a lead':
                        when = lambda: servo.velocity_feedforward is None
                    else:
                        when = lambda: servo.velocity_feedforward is not None and servo.velocity_feedforward is not own
                    handler.before_read = lambda: make(session, handler, sim, when)
                    with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(expected[ending]):
                        session.run_feedforward(parse_pose('elbow_out'))
                    self.assertIs(own, servo.velocity_feedforward,
                                  'its own velocity_feedforward is back, the lead of the round is gone')
                    session.stop.requested = False
                    handler.before_read = None
                    self.assertTrue(session.restore())
                    self.assertIs(own, servo.velocity_feedforward)
                    self.assertEqual((self.MOVE_TIME, 0), handler.profiles[4])

    def test_a_servo_without_a_velocity_feedforward_has_none_again_after_an_abort_in_a_lead_round(self):
        session, robot, sim = self.stage()
        handler = robot.servo_handler
        servo = handler.get_servo(4)
        warm = int(AbortLimits().max_temperature)
        handler.before_read = lambda: setattr(sim, 'temperature',
                                              warm if servo.velocity_feedforward is not None else 30)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(tune_servo_gains.Stop) as raised:
            session.run_feedforward(parse_pose('elbow_out'))
        self.assertIn('temperature {}C'.format(warm), str(raised.exception))
        self.assertIsNone(servo.velocity_feedforward)
        self.assertEqual(session.pose_goal, sim.goal, 'the goal is back at the pose')
        self.assertEqual((self.MOVE_TIME, 0), handler.profiles[4])

    def test_a_servo_without_move_time_is_skipped(self):
        session, robot, sim = self.stage(move_time=None)
        result, output = self.run_stage(session)
        self.assertIsNone(result)
        self.assertEqual('servo 4 has no move_time in servo_config.yaml, the feedforward stage needs the time-based '
                         'profile\n', output)
        self.assertEqual([], robot.moves, 'nothing moved')

    def test_a_refused_sweep_skips_the_stage_before_anything_moves(self):
        pose = parse_pose('elbow_out')
        for overrides, expected in (({'ff_degrees': 200.0}, 'range'),  # beyond the +-180 degrees of joint 4
                                    ({'ff_degrees': 1.0}, 'cruise'),
                                    ({'ff_speeds': (10.0, 20.0, 900.0)}, 'joint 4 would move')):
            session, robot, sim = self.stage(**overrides)
            reason = feedforward_refusal(robot, 4, pose, session.settings)
            self.assertIsNotNone(reason, overrides)
            self.assertIn(expected, reason)
            result, output = self.run_stage(session)
            self.assertIsNone(result)
            self.assertIn(expected, output)
            self.assertEqual([], robot.moves, 'nothing moved')
        session, robot, sim = self.stage()
        self.assertIsNone(feedforward_refusal(robot, 4, pose, session.settings))


class FeedforwardJointDefaultsTests(unittest.TestCase):
    """the sweeps of the stage per joint: slow and small on the belt joints 1 and 2"""

    def test_the_defaults_per_joint(self):
        settings = Settings()
        for joint in (1, 2):
            self.assertEqual(10.0, settings.ff_degrees_of(joint))
            self.assertEqual((5.0, 10.0, 15.0), settings.ff_speeds_of(joint))
        for joint in (3, 4, 5, 6):
            self.assertEqual(15.0, settings.ff_degrees_of(joint))
            self.assertEqual((10.0, 20.0, 30.0), settings.ff_speeds_of(joint))

    def test_explicit_degrees_and_speeds_override_the_defaults_on_every_joint(self):
        settings = Settings(ff_degrees=12.0, ff_speeds=(4.0, 8.0, 12.0))
        for joint in (1, 2, 3, 6):
            self.assertEqual(12.0, settings.ff_degrees_of(joint))
            self.assertEqual((4.0, 8.0, 12.0), settings.ff_speeds_of(joint))
        # one of the two given: the other stays the default of the joint
        self.assertEqual(10.0, Settings(ff_speeds=(1.0, 2.0, 3.0)).ff_degrees_of(2))
        self.assertEqual((5.0, 10.0, 15.0), Settings(ff_degrees=12.0).ff_speeds_of(2))

    def refusal(self, joint, settings):
        pose = parse_pose('elbow_out')
        sim = SimulatedServo(2048)
        robot = FakeTuningRobot(pose, joint, sim, move_time=20)
        return feedforward_refusal(robot, joint, pose, settings)

    def test_the_default_sweeps_of_the_belt_joints_pass_the_planner(self):
        for joint in (1, 2):
            self.assertIsNone(self.refusal(joint, Settings()), joint)

    def test_a_speed_above_15_degrees_per_second_is_refused_on_joints_1_and_2_only(self):
        for joint in (1, 2):
            reason = self.refusal(joint, Settings(ff_speeds=(10.0, 20.0, 30.0)))
            self.assertEqual('no feedforward stage, joint {} can not be swept faster than 15 degrees/s (it moves '
                             'behind a belt), got 20 30'.format(joint), reason)
            self.assertIsNotNone(self.refusal(joint, Settings(ff_speeds=(5.0, 10.0, 15.1))))
            self.assertIsNone(self.refusal(joint, Settings(ff_speeds=(5.0, 10.0, 15.0))), 'the top itself is allowed')
            self.assertIsNone(self.refusal(joint, Settings(ff_degrees=8.0, ff_speeds=(3.0, 6.0, 12.0))))
        for joint in (3, 4, 5, 6):
            self.assertIsNone(self.refusal(joint, Settings(ff_speeds=(10.0, 20.0, 30.0))), joint)


class FeedforwardReportTests(unittest.TestCase):
    """the report of the stage from rounds made of rows of the model of fit_velocity_lead_tests: joint 4 at P 1000"""
    K = 3.9

    @staticmethod
    def round_of(lead_gain, k=K, max_positions=30, origin='the test', **rows):
        analysis = fit_velocity_lead.analyse_group(
            fit_velocity_lead.GroupKey('', 4, lead_gain),
            as_text(sweep_rows(joint=4, k=k, friction=20.0, p=1000.0, lead_gain=lead_gain, **rows)))
        return tune_servo_gains.FeedforwardRound(lead_gain, max_positions if lead_gain else 0,
                                                 origin if lead_gain else '', analysis)

    def report(self, *rounds):
        servo = Servo(0, 4096, -pi, pi, POSITION_CONTROL_MODE, p=1000, i=0, d=0)
        result = tune_servo_gains.FeedforwardResult((5.0, 10.0, 15.0), tuple(rounds))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            tune_servo_gains.print_feedforward(4, servo, result)
        return result, out.getvalue()

    def test_a_flat_lead_is_the_recommendation_with_the_tables_and_the_fits_of_fit_velocity_lead(self):
        off, on = self.round_of(0), self.round_of(self.K)
        result, output = self.report(off, on)
        self.assertTrue(result.flat)
        self.assertEqual(self.K, result.recommended)
        self.assertEqual(self.K, result.corrected, 'what is left of the slope is 0')
        for expected in ('round 1, without a lead:', 'round 2, with a lead of 3.9 pwm per degree/s from the test, at '
                         'most 30 positions:', 'samples', 'intended', 'err std', 'pwm std',
                         'pwm = 20.0 + 3.900 * speed', 'error = 2.56 + 0.4992 * speed positions',
                         'ratio k_pwm / k_lag = 1.00', 'friction part', 'flat (within 1 position)',
                         'for servo_config.yaml, the coefficient of round 2 (from the test)',
                         '      max_positions: 30', '      pwm_per_degree_per_second: 3.90',
                         'at P 1000 the lead is'):
            self.assertIn(expected, output)
        self.assertEqual(1, output.count('ratio k_pwm / k_lag'), 'the two fits are for the round without a lead')
        self.assertEqual(1, output.count('from the measured pwm'))
        self.assertIn('more than 1 position, the speed dependent part is not taken out', output, 'round 1 is not flat')
        self.assertNotIn('NOT flat', output)
        self.assertIn('does not make the joint rougher', output)
        self.assertNotIn('makes the joint rougher', output)

    def test_a_lead_that_leaves_the_lag_not_flat_says_so_and_gives_the_corrected_value(self):
        off, small = self.round_of(0), self.round_of(1.9)
        result, output = self.report(off, small)
        self.assertFalse(result.flat)
        self.assertEqual(3.9, result.corrected, '1.9 plus the slope that is left, 2.0 pwm per degree/s')
        self.assertEqual(3.9, result.recommended)
        self.assertIn('NOT flat in round 2: it changes by', output)
        self.assertIn('the coefficient would be 3.9 (not measured)', output)
        self.assertIn('      pwm_per_degree_per_second: 3.90', output)
        self.assertNotIn('flat (within 1 position)\nfor servo', output)
        # a lead that is too large is corrected down, and never below 0
        big = self.round_of(9.0)
        self.assertAlmostEqual(self.K, tune_servo_gains.corrected_coefficient(big), places=2)
        # a joint that needs no drive that grows with the speed (a negative slope) gets 0, never a negative lead
        self.assertEqual(0.0, tune_servo_gains.corrected_coefficient(self.round_of(1.0, k=-2.0)))

    def test_without_a_round_with_a_lead_there_is_nothing_to_recommend(self):
        result, output = self.report(self.round_of(0))
        self.assertIsNone(result.recommended)
        self.assertIsNone(result.corrected)
        self.assertFalse(result.flat)
        self.assertIn('no lead to recommend', output)
        self.assertNotIn('velocity_feedforward:', output)

    def test_the_report_warns_when_the_lead_makes_the_joint_rougher(self):
        off = self.round_of(0)
        quiet = self.round_of(self.K, jitter=0.5)
        rough = self.round_of(self.K, jitter=2.0)
        _, output = self.report(off, quiet)
        self.assertNotIn('makes the joint rougher', output)
        _, output = self.report(off, rough)
        # the error std grew by 2 positions, the pwm std by 2 pwm units, which is below the drive of a position (7.81)
        self.assertIn('warning: in round 2 the std of the error grew by more than 1 position at 5, 10, 15 degrees/s, '
                      'the lead makes the joint rougher', output)
        self.assertIn('round 2: the std of the pwm is at most 7.81 pwm units above round 1 at every speed', output)
        very_rough = self.round_of(self.K, jitter=10.0)
        _, output = self.report(off, very_rough)
        self.assertIn('the std of the pwm grew by more than 7.81 pwm units at 5, 10, 15 degrees/s', output)

    def test_a_lead_at_its_clamp_is_pointed_out(self):
        # the lead at 15 degrees/s is 3.9 * 15 * 128 / 1000 = 7.5 positions
        result, output = self.report(self.round_of(0), self.round_of(self.K, max_positions=7))
        self.assertIn('the lead is at its clamp of 7 positions at the fast speeds', output)
        self.assertTrue(result.flat)
        self.assertEqual(self.K, result.recommended, 'flat is flat')
        result, output = self.report(self.round_of(0), self.round_of(self.K, max_positions=30))
        self.assertNotIn('at its clamp', output)
        self.assertFalse(self.round_of(self.K, max_positions=30).clamped)
        self.assertFalse(self.round_of(0).clamped, 'a round without a lead has no clamp')

    def test_a_clamped_lead_that_is_not_flat_gives_no_recommendation(self):
        # a lead of 1.9 where 3.9 is needed is 3.65 positions at 15 degrees/s: at the clamp of 3 and not flat
        result, output = self.report(self.round_of(0), self.round_of(1.9, max_positions=3))
        self.assertTrue(result.lead_rounds[0].clamped)
        self.assertFalse(result.flat)
        self.assertIsNone(result.recommended)
        self.assertIn('no coefficient to recommend: the lead was at its clamp of 3 positions in round 2', output)
        self.assertNotIn('velocity_feedforward:', output)


class StreamSweepTests(unittest.TestCase):
    """the loop that streams a sweep keeps time with the clock, and a late step does not make the next ones hurry"""
    DT = 0.01
    READ_TIME = 0.003  # three reads fit between two writes, then 1 ms of sleep is left

    def stream(self, steps=60, stalled_writes=None, stalled_reads=None, read_time=READ_TIME):
        """
        stream a sweep of steps with distinct angles on a fake clock. stalled_writes maps the number of a step to the
        seconds its write takes, stalled_reads maps it to the seconds of the first read after its write.
        Returns the times the steps were written, the angles written, the sleeps asked for and the samples recorded.
        """
        stalled_writes, stalled_reads = stalled_writes or {}, stalled_reads or {}
        now = [0.0]
        written, sent, sleeps, reads_of_step = [], [], [], {}
        self.velocities = []  # what every write was given as the velocities of the joints
        servo = mock.Mock(target_position=2048, compensation_positions=0, lead_positions=0)

        def clock():
            return now[0]

        def sleep(seconds):
            sleeps.append(seconds)
            now[0] += seconds

        def move_servos(angles, velocities=None):
            written.append(now[0])
            sent.append(angles[0])
            self.velocities.append(None if velocities is None else np.array(velocities))
            servo.lead_positions = 100 + len(written)  # a lead that changes with every write
            now[0] += stalled_writes.get(len(written) - 1, 0.0)

        def read(goal, time_written):
            number = reads_of_step[len(written)] = reads_of_step.get(len(written), 0) + 1
            now[0] += read_time + (stalled_reads.get(len(written) - 1, 0.0) if number == 1 else 0.0)
            return samples([0])[0]

        robot = mock.Mock()
        robot.move_servos.side_effect = move_servos
        steps_of_sweep = [SweepStep(index * self.DT, 0.0, 1, False) for index in range(steps)]
        angles = np.tile(np.arange(steps, dtype=np.float64)[:, np.newaxis], (1, 7))
        recorded = []
        reason = stream_sweep(robot, servo, steps_of_sweep, angles, read, lambda time_written: None, recorded,
                              clock, sleep)
        self.assertIsNone(reason)
        self.angles = angles
        return written, sent, sleeps, recorded

    def test_every_step_is_written_with_its_velocities_and_the_lead_of_every_write_is_noted(self):
        written, sent, sleeps, recorded = self.stream(steps=40)
        expected = step_velocities(self.angles, self.DT)
        self.assertEqual(40, len(self.velocities))
        for step, (given, wanted) in enumerate(zip(self.velocities, expected)):
            np.testing.assert_allclose(given, wanted, err_msg='step {}'.format(step))
        # the angle of the test grows by 1 rad per step: 100 rad/s inside, nothing at the first and the last step
        np.testing.assert_allclose(self.velocities[0], 0.0)
        np.testing.assert_allclose(self.velocities[-1], 0.0)
        np.testing.assert_allclose(self.velocities[20], 100.0)
        # every sample carries the lead that its step's write added, which is not that of the other steps
        by_step = {}
        for sample in recorded:
            by_step.setdefault(sample.step.time, set()).add(sample.lead)
        self.assertEqual(40, len(by_step))
        for index, time_of_step in enumerate(sorted(by_step)):
            self.assertEqual({101 + index}, by_step[time_of_step])
        self.assertEqual(3 * 40, len(recorded))

    def test_without_a_stall_the_steps_follow_the_clock_and_the_servo_is_read_between_them(self):
        written, sent, sleeps, recorded = self.stream()
        np.testing.assert_allclose(written, np.arange(60) * self.DT, atol=1e-9)
        np.testing.assert_allclose(sleeps, 0.001, atol=1e-9)
        self.assertEqual(60, len(sleeps))
        self.assertEqual(3 * 60, len(recorded))
        self.assertEqual(list(range(60)), sent)

    def test_a_step_that_is_a_little_early_does_not_sleep_and_keeps_the_time_of_the_step(self):
        # the last read ends 0.01 ms before the next write is due: no sleep for that, and the time is not reset
        written, _, sleeps, _ = self.stream(steps=5, read_time=0.00333)
        self.assertEqual([], sleeps)
        for index, seconds in enumerate(written):
            self.assertLessEqual(seconds, index * self.DT + 1e-9)
            self.assertGreater(seconds, index * self.DT - 1e-4)

    def check_stall(self, written, sent, sleeps, late, stall):
        gaps = np.diff(written)
        # the next step goes out when the stall is over (after the read it started), then the period is back
        self.assertAlmostEqual(stall + self.READ_TIME, gaps[late], places=9)
        np.testing.assert_allclose(gaps[late + 1:], self.DT, atol=1e-9)
        np.testing.assert_allclose(gaps[:late], self.DT, atol=1e-9)
        self.assertGreaterEqual(gaps.min(), self.DT - 1e-9, 'no two steps are written back to back')
        # what is asked for after the stall is the rest of a period, as before it, not nothing
        self.assertEqual(len(written) - 1, len(sleeps), 'only the late step has no sleep')
        np.testing.assert_allclose(sleeps, 0.001, atol=1e-9)
        self.assertEqual(list(range(len(written))), sent, 'every step is written, in order')

    def test_a_stall_in_a_write_does_not_make_the_next_steps_hurry(self):
        written, sent, sleeps, _ = self.stream(stalled_writes={20: 0.25})
        self.check_stall(written, sent, sleeps, 20, 0.25)

    def test_a_stall_in_a_read_does_not_make_the_next_steps_hurry(self):
        written, sent, sleeps, _ = self.stream(stalled_reads={30: 0.25})
        self.check_stall(written, sent, sleeps, 30, 0.25)

    def test_every_step_is_written_in_order_after_several_stalls(self):
        written, sent, _, recorded = self.stream(stalled_writes={5: 0.25, 6: 0.02, 40: 1.0}, stalled_reads={20: 0.1})
        self.assertEqual(list(range(60)), sent)
        self.assertGreaterEqual(np.diff(written).min(), self.DT - 1e-9)
        self.assertEqual(60, len({sample.step.time for sample in recorded}))


class PositionSignTests(unittest.TestCase):

    def test_the_sign_follows_the_positions_of_the_calibration(self):
        normal = Servo(0, 4096, -pi, pi, POSITION_CONTROL_MODE)
        reversed_ = Servo(4096, 0, -pi, pi, POSITION_CONTROL_MODE)
        self.assertEqual(1, position_sign(normal))
        self.assertEqual(-1, position_sign(reversed_))
        # and so they move: the angle up is the position up or down
        for servo, expected in ((normal, 1), (reversed_, -1)):
            servo.set_target_position_from_angle(0.0)
            middle = servo.target_position
            servo.set_target_position_from_angle(0.3)
            self.assertEqual(expected, np.sign(servo.target_position - middle))

    def test_the_servos_of_the_yaml_are_read_the_same_way(self):
        servos = FakeRobot(parse_pose('elbow_out')).servo_config.servos
        self.assertEqual(-1, position_sign(servos[3]), 'positions: [3072, 1024] for angles: [-90, 90]')
        self.assertEqual(1, position_sign(servos[1]), 'positions: [1024, 3072] for angles: [0, 180]')
        self.assertEqual(1, position_sign(servos[4]))


class FeedforwardOptionTests(unittest.TestCase):
    """main with the options of the stage, with a session that only records what it is asked to do"""

    def run_main(self, argv, move_time=20, stop_in_run=False, joint=4):
        sim = SimulatedServo(2048)
        robot = FakeTuningRobot(parse_pose('elbow_out'), joint, sim, move_time=move_time)
        calls = []
        swing_writers, settings_seen = [], []

        class RecordingSession:
            def __init__(self, robot, joint, settings, sample_writer=None, trial_writer=None, stop=None,
                         ff_writer=None, swing_writer=None):
                calls.append(('session', sample_writer is not None, trial_writer is not None, ff_writer is not None))
                swing_writers.append(swing_writer is not None)
                settings_seen.append(settings)
                self.settings = settings
                self.results = []
                self.read_failures = 0

            def run(self, poses):
                calls.append(('run', len(poses)))
                if stop_in_run:
                    raise tune_servo_gains.Stop('a trial aborted')
                return []

            def run_feedforward(self, pose):
                calls.append(('ff', np.degrees(pose[1:]).round().tolist()))

            def run_swing(self, pose):
                calls.append(('swing', np.degrees(pose[1:]).round().tolist()))

            def restore(self):
                calls.append('restore')
                return True

        out = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(tune_servo_gains, 'get_robot', lambda: robot), \
                mock.patch.object(tune_servo_gains, 'CombinedRobot', FakeTuningRobot), \
                mock.patch.object(tune_servo_gains, 'TuningSession', RecordingSession), \
                mock.patch.object(tune_servo_gains, 'finish') as finish, contextlib.redirect_stdout(out):
            code = tune_servo_gains.main(['--joint', str(joint), '--yes', '--pose', 'elbow_out', '--output-dir',
                                          directory] + argv)
            files = sorted(os.listdir(directory))
        self.recorded = (calls, finish, [name.split('_', 4)[-1] for name in files], robot)
        self.swing_writers = swing_writers
        self.settings = settings_seen
        return code, out.getvalue()

    def test_the_defaults(self):
        settings = Settings()
        self.assertIsNone(settings.ff_speeds, 'per joint')
        self.assertIsNone(settings.ff_degrees, 'per joint')
        self.assertIsNone(settings.ff_lead)
        self.assertEqual(30, settings.ff_max_positions)
        self.assertFalse(hasattr(settings, 'ff_max'), 'the old --ff-max is gone')

    def test_without_the_options_the_stage_does_not_run(self):
        code, output = self.run_main([])
        calls, finish, files, robot = self.recorded
        self.assertEqual(0, code)
        self.assertEqual([('session', True, True, False), ('run', 1), 'restore'], calls)
        self.assertEqual(['samples.csv', 'trials.csv'], files)
        self.assertNotIn('feedforward stage', output)

    def test_ff_runs_the_stage_after_the_p_and_d_tuning(self):
        code, output = self.run_main(['--ff'])
        calls, finish, files, robot = self.recorded
        self.assertEqual(0, code)
        self.assertEqual([('session', True, True, True), ('run', 1), ('ff', [90, 90, 0, 0, 0, 0]), 'restore'], calls)
        self.assertEqual(['ff.csv', 'samples.csv', 'trials.csv'], files)
        finish.assert_called_once()

    def test_ff_only_skips_the_p_and_d_stages(self):
        code, output = self.run_main(['--ff-only', '--ff-degrees', '12', '--ff-speeds', '5', '10', '15',
                                      '--ff-lead', '4.2', '--ff-max-positions', '20'])
        calls, finish, files, robot = self.recorded
        self.assertEqual(0, code)
        self.assertEqual([('session', False, False, True), ('ff', [90, 90, 0, 0, 0, 0]), 'restore'], calls)
        self.assertEqual(['ff.csv'], files)
        self.assertTrue(robot.enabled)
        self.assertIn('sweeps of 12 degrees either way at 5 10 15 degrees/s, first without a lead, then with a '
                      'coefficient of 4.2 pwm per degree/s', output)
        (settings,) = self.settings
        self.assertEqual((12.0, (5.0, 10.0, 15.0), 4.2, 20),
                         (settings.ff_degrees, settings.ff_speeds, settings.ff_lead, settings.ff_max_positions))

    def test_without_the_options_the_sweeps_are_those_of_the_joint_and_the_lead_is_fitted(self):
        code, output = self.run_main(['--ff-only'])
        (settings,) = self.settings
        self.assertEqual(0, code)
        self.assertEqual((None, None, None, 30),
                         (settings.ff_degrees, settings.ff_speeds, settings.ff_lead, settings.ff_max_positions))
        self.assertIn('sweeps of 15 degrees either way at 10 20 30 degrees/s, first without a lead, then with the '
                      'coefficient fitted in the first round', output)

    def test_a_belt_joint_defaults_to_small_and_slow_sweeps_and_is_refused_above_15_degrees_per_second(self):
        for joint in (1, 2):
            code, output = self.run_main(['--ff-only'], joint=joint)
            self.assertEqual(0, code, joint)
            self.assertIn('sweeps of 10 degrees either way at 5 10 15 degrees/s', output)
            code, output = self.run_main(['--ff-only', '--ff-speeds', '10', '20', '30'], joint=joint)
            self.assertEqual(1, code, joint)
            self.assertIn('joint {} can not be swept faster than 15 degrees/s'.format(joint), output)
            self.assertFalse(self.recorded[3].enabled, 'refused before the servos are enabled')
            self.assertEqual([], self.recorded[0])
            self.assertEqual([], self.recorded[2])
            # with --ff the P and D stages still run and the stage is left out
            code, output = self.run_main(['--ff', '--ff-speeds', '10', '20', '30'], joint=joint)
            self.assertEqual(0, code)
            self.assertEqual([('session', True, True, False), ('run', 1), 'restore'], self.recorded[0])
        code, output = self.run_main(['--ff-only', '--ff-speeds', '10', '20', '30'], joint=3)
        self.assertEqual(0, code, 'joint 3 may be swept at 30 degrees/s')

    def test_the_old_ff_max_is_refused_and_not_taken_for_the_abbreviation_of_ff_max_positions(self):
        code, output = self.run_main(['--ff-only', '--ff-max', '4000'])
        self.assertEqual(1, code)
        self.assertIn('--ff-max is gone: the stage measures the velocity lead of the PC', output)
        self.assertIn('--ff-max-positions', output)
        self.assertEqual([], self.recorded[0])
        self.assertFalse(self.recorded[3].enabled)
        code, output = self.run_main(['--ff-only', '--ff-max-positions', '12'])
        self.assertEqual(0, code)
        self.assertEqual(12, self.settings[0].ff_max_positions)

    def test_the_stage_does_not_start_after_a_p_and_d_run_that_stopped(self):
        code, output = self.run_main(['--ff'], stop_in_run=True)
        calls, finish, files, robot = self.recorded
        self.assertEqual(0, code)
        self.assertEqual([('session', True, True, True), ('run', 1), 'restore'], calls)
        self.assertIn('stopped: a trial aborted', output)

    def test_a_servo_without_move_time_skips_the_stage_and_ff_only_ends_before_the_servos_are_enabled(self):
        line = 'servo 4 has no move_time in servo_config.yaml, the feedforward stage needs the time-based profile'
        code, output = self.run_main(['--ff-only'], move_time=None)
        calls, finish, files, robot = self.recorded
        self.assertEqual(1, code)
        self.assertEqual([], calls)
        self.assertFalse(robot.enabled)
        self.assertEqual([], files)
        self.assertEqual(1, output.count(line))
        # with --ff the P and D stages still run
        code, output = self.run_main(['--ff'], move_time=None)
        calls, finish, files, robot = self.recorded
        self.assertEqual(0, code)
        self.assertEqual([('session', True, True, False), ('run', 1), 'restore'], calls)
        self.assertEqual(1, output.count(line))
        self.assertTrue(robot.enabled)
        self.assertEqual(['samples.csv', 'trials.csv'], files)

    def test_fewer_than_three_speeds_and_bad_values_are_refused(self):
        def no_robot():
            raise AssertionError('the robot was touched')

        for argv in (['--ff-speeds', '10', '20'], ['--ff-speeds', '10'], ['--ff-speeds', '10', '10', '20'],
                     ['--ff-speeds', '0', '10', '20'], ['--ff-speeds', '5', '10', 'nan'], ['--ff-degrees', '0'],
                     ['--ff-lead', '0'], ['--ff-lead', '-2'], ['--ff-lead', 'nan'], ['--ff-lead', 'inf'],
                     ['--ff-max-positions', '0'], ['--ff-max-positions', '-3'], ['--ff-max', '4000']):
            out = io.StringIO()
            with mock.patch.object(tune_servo_gains, 'get_robot', no_robot), contextlib.redirect_stdout(out):
                self.assertEqual(1, tune_servo_gains.main(['--joint', '4', '--ff-only'] + argv), argv)
            self.assertIn('--ff-', out.getvalue())
        code, output = self.run_main(['--ff-only', '--ff-speeds', '5', '10', '15', '20'])
        self.assertEqual(0, code, 'four speeds are fine')


def swing_leg_rows(direction, leg, start, duration, sines=((0.0, 6.6),), rate=750.0, trend=0.0, offset=0.0,
                   cruise=True, pwm=0, seed=0):
    """
    the samples of one leg of a sweep, read at irregular intervals of about 1 / rate seconds: the position trails its
    trajectory (2048) by the lag offset + trend * t + the sum of amplitude * sin(2 pi f t) of the sines, in positions,
    t in seconds since the start of the leg. The direction is that of the angle. The positions are not rounded to
    integers: a swing of a position or two that is read in whole positions loses up to 20 % of its amplitude to the
    quantization (1.5 positions read as 0, 1 and -1 give 1.2), unless something else moves the lag through several
    positions, see test_a_swing_read_in_whole_positions.
    """
    rng = np.random.default_rng(seed + 100 * leg + (direction > 0))
    times = start + np.cumsum(rng.uniform(0.2, 1.8, int(duration * rate * 1.2))) / rate
    times = times[times <= start + duration]
    seconds = times - start
    lag = offset + trend * seconds
    for amplitude, frequency in sines:
        lag = lag + amplitude * np.sin(2 * np.pi * frequency * seconds)
    return [SwingSample(Sample(time=float(t), tick=0, goal=2048, position=float(2048 - value),
                               position_trajectory=2048, velocity=0, velocity_trajectory=0, pwm=pwm, current=0,
                               moving_status=0, voltage=11.8, temperature=30), direction, cruise, leg)
            for t, value in zip(times, lag)]


# the legs of ff_sweep(10, 10, 0.01) with two round trips: (direction of the angle, start of the cruise, its length)
SWING_LEGS = ((1, 0.35, 0.65), (-1, 1.55, 1.65), (1, 3.75, 1.65), (-1, 5.95, 1.65), (1, 8.15, 0.65))


def swing_sweep_rows(falling=((1.5, 6.6),), rising=((0.5, 7.2),), **kwargs):
    """a sweep with a swing of the given sines in every cruise leg going each way, and a violent ramp before every leg
    that is not cruise"""
    rows = []
    for leg, (direction, start, duration) in enumerate(SWING_LEGS):
        rows += swing_leg_rows(direction, leg, start, duration, rising if direction > 0 else falling, **kwargs)
        rows += swing_leg_rows(direction, leg, start - 0.3, 0.25, ((20.0, 5.0),), cruise=False, **kwargs)
    return rows


class SwingStatisticsTests(unittest.TestCase):

    def test_the_swing_and_its_frequency_are_recovered_per_direction_of_the_angle(self):
        stats = swing_statistics(swing_sweep_rows())
        self.assertEqual((3, 2), (stats.rising.legs, stats.falling.legs))
        # 1.5 positions at 6.6 Hz with the angle falling, 0.5 at 7.2 Hz rising, within 10 %
        self.assertAlmostEqual(1.5, stats.falling.swing, delta=0.15)
        self.assertAlmostEqual(0.5, stats.rising.swing, delta=0.05)
        # and the frequency within the resolution of the leg, 1 / its length
        self.assertAlmostEqual(6.6, stats.falling.frequency, delta=1 / 1.65)
        self.assertAlmostEqual(7.2, stats.rising.frequency, delta=1 / 0.65)
        self.assertAlmostEqual(1.5 / math.sqrt(2), stats.falling.lag_std, delta=0.15, msg='the std of a sine')
        self.assertAlmostEqual(0.5 / math.sqrt(2), stats.rising.lag_std, delta=0.05)

    def test_a_swing_between_two_bins_is_not_read_low(self):
        # a sine whose frequency falls between two bins of the spectrum of a leg is not scaled down by the window
        for frequency in (6.0, 6.3, 6.6, 6.9, 7.2):
            stats = swing_statistics(swing_sweep_rows(falling=((1.0, frequency),), rising=((1.0, frequency),)))
            self.assertAlmostEqual(1.0, stats.falling.swing, delta=0.05, msg=frequency)

    def test_the_legs_are_not_joined(self):
        rows = (swing_leg_rows(1, 0, 0.0, 1.0, ((1.0, 6.6),)) + swing_leg_rows(1, 2, 3.0, 1.0, ((2.0, 6.6),)) +
                swing_leg_rows(-1, 1, 1.5, 1.0, ((1.0, 6.6),)))
        stats = swing_statistics(rows)
        self.assertEqual(2, stats.rising.legs)
        self.assertAlmostEqual(1.5, stats.rising.swing, delta=0.15, msg='the mean of the legs of 1 and 2 positions')
        self.assertEqual(1, stats.falling.legs)

    def test_a_trend_and_a_constant_lag_are_not_swing(self):
        # the lag of the speed (a constant), the lag growing through the leg, no oscillation: 5 positions + 3 per second
        stats = swing_statistics(swing_sweep_rows(falling=(), rising=(), offset=5.0, trend=3.0))
        self.assertLess(stats.rising.swing, 0.1)
        self.assertLess(stats.falling.swing, 0.1)
        self.assertLess(stats.falling.lag_std, 0.3)
        self.assertAlmostEqual(5.0 + 3.0 * 0.825, abs(stats.falling.lag), delta=0.2, msg='the mean lag is reported')
        # the trend does not hide a swing either
        stats = swing_statistics(swing_sweep_rows(offset=5.0, trend=3.0))
        self.assertAlmostEqual(1.5, stats.falling.swing, delta=0.15)
        self.assertAlmostEqual(0.5, stats.rising.swing, delta=0.05)

    def test_a_component_outside_the_band_is_not_picked(self):
        # 2 positions at 0.5 Hz and 2 at 25 Hz, 0.4 at 8 Hz in the band (a component close to the edge leaks into it,
        # most in the legs of 0.65 s, whose spectrum has a resolution of 1.5 Hz)
        slow_and_fast = ((2.0, 0.5), (2.0, 25.0), (0.4, 8.0))
        stats = swing_statistics(swing_sweep_rows(falling=slow_and_fast, rising=slow_and_fast))
        for side in (stats.rising, stats.falling):
            self.assertAlmostEqual(0.4, side.swing, delta=0.06)
            self.assertAlmostEqual(8.0, side.frequency, delta=1.0)
        # the strongest component in the band wins, not the first
        both = ((0.5, 5.0), (1.2, 11.0))
        stats = swing_statistics(swing_sweep_rows(falling=both, rising=both))
        self.assertAlmostEqual(1.2, stats.falling.swing, delta=0.15)
        self.assertAlmostEqual(11.0, stats.falling.frequency, delta=0.7)
        # another band takes the component the default leaves out
        stats = swing_statistics(swing_sweep_rows(falling=slow_and_fast, rising=slow_and_fast), band=(20.0, 40.0))
        self.assertAlmostEqual(2.0, stats.falling.swing, delta=0.2)
        self.assertAlmostEqual(25.0, stats.falling.frequency, delta=1.0)

    def test_legs_that_are_too_short_or_sampled_too_sparsely_are_left_out(self):
        sweep = swing_sweep_rows()
        # a 0.5 s leg with a huge swing going down, and a 1.65 s leg sampled at 20 Hz going up
        short = swing_leg_rows(-1, 7, 20.0, 0.5, ((9.0, 6.6),))
        sparse = swing_leg_rows(1, 8, 30.0, 1.65, ((9.0, 6.6),), rate=20.0)
        self.assertGreater(len(sparse), 10)
        stats = swing_statistics(sweep + short + sparse)
        self.assertEqual((3, 2), (stats.rising.legs, stats.falling.legs))
        self.assertAlmostEqual(1.5, stats.falling.swing, delta=0.15)
        self.assertAlmostEqual(0.5, stats.rising.swing, delta=0.05)
        # the minimum is a parameter: with 0.4 s the short leg counts
        stats = swing_statistics(sweep + short, minimum_leg=0.4)
        self.assertEqual(3, stats.falling.legs)
        self.assertGreater(stats.falling.swing, 3.0)
        # a leg of just the minimum counts
        exact = swing_leg_rows(1, 9, 40.0, SWING_MIN_LEG + 0.01, ((1.0, 6.6),))
        self.assertEqual(1, swing_statistics(exact + swing_leg_rows(-1, 10, 50.0, 1.0)).rising.legs)

    def test_a_direction_without_a_usable_leg_raises(self):
        rising_only = [row for row in swing_sweep_rows() if row.direction > 0]
        with self.assertRaisesRegex(ValueError, 'falling'):
            swing_statistics(rising_only)
        falling_only = [row for row in swing_sweep_rows() if row.direction < 0]
        with self.assertRaisesRegex(ValueError, 'rising'):
            swing_statistics(falling_only)
        with self.assertRaises(ValueError):
            swing_statistics([])
        too_short = swing_leg_rows(1, 0, 0.0, 0.5) + swing_leg_rows(-1, 1, 1.0, 0.5)
        with self.assertRaisesRegex(ValueError, '0.6 s'):
            swing_statistics(too_short)
        # samples that are not cruise don't make a leg
        ramps = swing_leg_rows(1, 0, 0.0, 1.0, cruise=False) + swing_leg_rows(-1, 1, 2.0, 1.0, cruise=False)
        with self.assertRaises(ValueError):
            swing_statistics(ramps)

    def test_the_direction_is_that_of_the_angle_whatever_the_position_does(self):
        rows = swing_sweep_rows()
        normal = swing_statistics(rows, position_sign=1)
        reversed_ = swing_statistics(rows, position_sign=-1)
        # the swing is the same: the angle falling swings more, in a servo whose positions fall with the angle too
        self.assertEqual(normal.rising.swing, reversed_.rising.swing)
        self.assertEqual(normal.falling.swing, reversed_.falling.swing)
        self.assertGreater(reversed_.falling.swing, 2 * reversed_.rising.swing)
        # the lag and the pwm are signed with the direction of the move in positions: trailing is positive
        trailing = swing_sweep_rows(falling=(), rising=(), offset=4.0, pwm=100)
        up_and_down = swing_statistics(trailing, position_sign=1)
        self.assertAlmostEqual(4.0, up_and_down.rising.lag, delta=0.1, msg='positions rise with the angle: trailing')
        self.assertAlmostEqual(-4.0, up_and_down.falling.lag, delta=0.1, msg='a falling angle, falling positions')
        self.assertEqual(100, up_and_down.rising.pwm)
        self.assertEqual(-100, up_and_down.falling.pwm)
        self.assertEqual(0.0, up_and_down.rising.pwm_std)
        flipped = swing_statistics(trailing, position_sign=-1)
        self.assertAlmostEqual(-4.0, flipped.rising.lag, delta=0.1)
        self.assertAlmostEqual(4.0, flipped.falling.lag, delta=0.1)
        self.assertEqual(-100, flipped.rising.pwm)

    def test_a_swing_read_in_whole_positions(self):
        # the servo reports whole positions: a swing of 0.5 positions on a lag that doesn't move at all would be read as
        # 0, but the lag of a real sweep drifts through several positions in a leg, which dithers it
        rows = [row._replace(sample=row.sample._replace(position=int(round(row.sample.position))))
                for row in swing_sweep_rows(offset=5.0, trend=3.0)]
        stats = swing_statistics(rows)
        self.assertAlmostEqual(1.5, stats.falling.swing, delta=0.15)
        self.assertAlmostEqual(0.5, stats.rising.swing, delta=0.05)

    def test_the_samples_may_come_in_any_order_and_the_band_is_checked(self):
        rows = swing_sweep_rows()
        shuffled = list(rows)
        np.random.default_rng(3).shuffle(shuffled)
        self.assertEqual(swing_statistics(rows), swing_statistics(shuffled))
        for band in ((15.0, 3.0), (0.0, 15.0), (3.0, 300.0)):
            with self.assertRaises(ValueError):
                swing_statistics(rows, band=band)


def swing_statistics_of(rising, falling, legs=3, frequency=6.6):
    """SwingStatistics with the given swings"""
    return SwingStatistics(SwingSide(legs, frequency, rising, 0.5, 2.0, 100.0, 5.0),
                           SwingSide(legs, frequency, falling, 0.5, -2.0, -100.0, 5.0))


class SwingArithmeticTests(unittest.TestCase):

    def test_the_recommendation_is_the_largest_p_within_the_limit_in_both_directions(self):
        tried = [(2000, 0, 2.4, 1.0), (1600, 0, 1.2, 0.8), (1200, 0, 0.9, 0.4), (2000, 0, 0.5, 2.0)]
        self.assertEqual(1200, recommended_swing_p(tried, 0, 1.0))
        self.assertEqual(1600, recommended_swing_p(tried, 0, 1.2), 'at the limit counts')
        self.assertIsNone(recommended_swing_p(tried, 0, 0.3))
        self.assertIsNone(recommended_swing_p([], 0, 1.0))

    def test_a_p_that_is_within_the_limit_in_one_direction_only_is_not_recommended(self):
        tried = [(1500, 0, 0.1, 1.5), (1000, 0, 0.9, 0.9), (1200, 0, 1.5, 0.1)]
        self.assertEqual(1000, recommended_swing_p(tried, 0, 1.0))

    def test_only_the_yaml_d_counts(self):
        tried = [(2000, 0, 2.0, 2.0), (2000, 250, 0.1, 0.1), (1500, 0, 0.5, 0.5), (3000, 400, 0.1, 0.1)]
        self.assertEqual(1500, recommended_swing_p(tried, 0, 1.0))
        self.assertEqual(3000, recommended_swing_p(tried, 400, 1.0))
        self.assertIsNone(recommended_swing_p(tried, 100, 1.0))

    def test_a_p_that_was_run_twice_has_to_be_within_the_limit_both_times(self):
        tried = [(2000, 0, 0.9, 0.9), (1500, 0, 0.5, 0.5), (2000, 0, 1.1, 0.9)]
        self.assertEqual(1500, recommended_swing_p(tried, 0, 1.0), 'the repeat of the yaml P swings too much')
        tried = [(2000, 0, 0.9, 0.9), (1500, 0, 0.5, 0.5), (2000, 0, 0.8, 0.7)]
        self.assertEqual(2000, recommended_swing_p(tried, 0, 1.0))

    def test_the_spread_is_how_much_two_runs_differ(self):
        spread = swing_spread(swing_statistics_of(0.6, 1.9), swing_statistics_of(0.9, 2.4))
        self.assertAlmostEqual(0.3, spread.rising)
        self.assertAlmostEqual(0.5, spread.falling)
        self.assertEqual(swing_spread(swing_statistics_of(0.9, 2.4), swing_statistics_of(0.6, 1.9)), spread)
        same = swing_statistics_of(1.0, 1.0)
        self.assertEqual(SwingPair(0.0, 0.0), swing_spread(same, same))

    def test_the_change_is_against_the_mean_of_the_reference_runs(self):
        reference = [swing_statistics_of(0.6, 1.9), swing_statistics_of(0.9, 2.4)]
        change = swing_change(reference, swing_statistics_of(0.5, 2.5))
        self.assertAlmostEqual(-0.25, change.rising)
        self.assertAlmostEqual(0.35, change.falling)

    def test_a_change_counts_beyond_the_spread_and_the_resolution(self):
        self.assertTrue(swing_changed(0.6, 0.5))
        self.assertTrue(swing_changed(-0.6, 0.5), 'either way')
        self.assertFalse(swing_changed(0.4, 0.5))
        self.assertFalse(swing_changed(0.05, 0.0), 'two runs that happen to agree are no reason to count every change')
        self.assertTrue(swing_changed(0.2, 0.0))

    def test_the_cruise_legs_of_a_sweep_that_are_long_enough(self):
        self.assertEqual({1: 3, -1: 2}, swing_legs(ff_sweep(10, 10, 0.01), 0.01))
        self.assertEqual({1: 0, -1: 0}, swing_legs(ff_sweep(4, 10, 0.01), 0.01), 'cruise of 0.2 s')
        self.assertEqual({1: 0, -1: 0}, swing_legs(ff_sweep(1, 10, 0.01), 0.01), 'no cruise at all')
        # the legs to and from the pose cruise 0.65 s, those between the ends 1.65 s
        self.assertEqual({1: 1, -1: 2}, swing_legs(ff_sweep(10, 10, 0.01), 0.01, minimum_leg=1.0))
        self.assertEqual({1: 3, -1: 2}, swing_legs(ff_sweep(10, 10, 0.01), 0.01, minimum_leg=0.66))
        self.assertEqual({1: 1, -1: 2}, swing_legs(ff_sweep(10, 10, 0.01), 0.01, minimum_leg=0.67))
        self.assertEqual({1: 0, -1: 1}, swing_legs(ff_sweep(10, 10, 0.01, round_trips=1), 0.01, minimum_leg=1.0))


class SwingSettingsTests(unittest.TestCase):

    def test_the_default_settings_are_the_yaml_gains_two_lower_p_gains_and_the_yaml_gains_again(self):
        self.assertEqual([(2000, 0), (1600, 0), (1200, 0), (2000, 0)], swing_gains(Settings(), 2000, 0))
        self.assertEqual([(2500, 250), (2000, 250), (1500, 250), (2500, 250)], swing_gains(Settings(), 2500, 250))
        self.assertEqual([(780, 1020), (620, 1020), (470, 1020), (780, 1020)], swing_gains(Settings(), 780, 1020),
                         'rounded to tens')

    def test_the_extra_d_gains_come_first_then_the_p_gains(self):
        settings = Settings(swing_d=(250, 640))
        self.assertEqual([(2000, 0), (2000, 250), (2000, 640), (1600, 0), (1200, 0), (2000, 0)],
                         swing_gains(settings, 2000, 0))

    def test_given_p_gains_replace_the_default_ones_in_the_order_given(self):
        settings = Settings(swing_p=(2400, 900), swing_d=(100,))
        self.assertEqual([(2000, 0), (2000, 100), (2400, 0), (900, 0), (2000, 0)], swing_gains(settings, 2000, 0))
        self.assertEqual([(2000, 0), (2000, 0)], swing_gains(Settings(swing_p=()), 2000, 0), 'none: only the repeat')

    def test_the_yaml_gains_and_repeated_values_are_run_once(self):
        settings = Settings(swing_p=(2000, 1500, 1500), swing_d=(0, 250, 250))
        self.assertEqual([(2000, 0), (2000, 250), (1500, 0), (2000, 0)], swing_gains(settings, 2000, 0))

    def test_a_default_p_that_is_not_a_gain_is_left_out(self):
        self.assertEqual([(1, 0), (1, 0)], swing_gains(Settings(), 1, 0), '0.8 and 0.6 times 1 round to 0 and 0')
        self.assertEqual([(10, 0), (10, 0)], swing_gains(Settings(), 10, 0), 'and to 10 and 10: the yaml P, once')
        self.assertEqual([(cfg.MAX_GAIN, 0), (cfg.MAX_GAIN, 0)], swing_gains(Settings(swing_p=(cfg.MAX_GAIN + 1,)),
                                                                              cfg.MAX_GAIN, 0))


class SwingStageTests(unittest.TestCase):
    """the stage on the simulated servo: joint 4 at P 1000 and D 0 of the test yaml, a trajectory that follows the goal
    in 10 ms, the settings (1000, 0), (800, 0), (600, 0) and (1000, 0) again by default"""
    YAML = (1000, 0, 0)
    MOVE_TIME = 20
    DEFAULT_SETTINGS = [(1000, 0), (800, 0), (600, 0), (1000, 0)]
    # above P 700 the damping of the simulation falls by 0.25 per unit of P, which leaves the loop at P 1000 with a
    # damping ratio of 0.03: it rings at its own frequency of 6.5 Hz and, kicked by the legs of the sweep, doesn't stop
    RESONANT = {'resonance_above': 700, 'resonance_slope': 0.25}

    def stage(self, move_time=MOVE_TIME, load=20000.0, reverse=False, resonance_above=None, resonance_slope=0.1,
              **overrides):
        settings = Settings(repeats=1, hold=0.5, rest=1.0, baseline_rest=1.0, move_degrees=0.0, tune_d=False,
                            swing_degrees=10.0)._replace(**overrides)
        session, robot, sim, _, _ = session_for(joint=4, load=load, settings=settings, move_time=move_time,
                                                trajectory_time=0.01, resonance_above=resonance_above,
                                                resonance_slope=resonance_slope)
        if reverse:  # the positions fall as the angle rises; the pose is at the middle of the range, the same position
            servo = robot.servo_handler.get_servo(4)
            servo.min_position, servo.max_position = servo.max_position, servo.min_position
        buffer = io.StringIO()
        session.swing_writer = csv.DictWriter(buffer, fieldnames=SWING_SAMPLE_FIELDS)
        session.swing_writer.writeheader()
        for name, replacement in (('sleep', sim.advance), ('perf_counter', lambda: sim.now)):
            patcher = mock.patch('src.utils.movement.' + name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        return session, robot, sim, buffer

    @staticmethod
    def run_stage(session):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = session.run_swing(parse_pose('elbow_out'))
        return result, out.getvalue()

    @staticmethod
    def rows_of(buffer):
        return list(csv.DictReader(io.StringIO(buffer.getvalue())))

    @staticmethod
    def note_settings(session, robot, sim):
        """the (setting, p, d, gains read back from the servo, gains of the simulation) of every sweep, as it starts"""
        seen = []
        sweep = session.swing_sweep

        def noted(pose, setting, p, d):
            seen.append((setting, p, d, robot.servo_handler.gains[4], (sim.p, sim.i, sim.d)))
            return sweep(pose, setting, p, d)
        session.swing_sweep = noted
        return seen

    def test_the_settings_run_in_order_and_the_servo_has_its_yaml_gains_again(self):
        session, robot, sim, buffer = self.stage()
        handler = robot.servo_handler
        seen = self.note_settings(session, robot, sim)
        times, starts = [], []
        move_servos = robot.move_servos

        def timed(angles, velocities=None):
            times.append(sim.now)
            move_servos(angles, velocities)
        robot.move_servos = timed
        sweep = session.swing_sweep

        def marked(pose, setting, p, d):
            starts.append(len(times))
            return sweep(pose, setting, p, d)
        session.swing_sweep = marked

        result, output = self.run_stage(session)

        self.assertEqual(self.DEFAULT_SETTINGS, [(r.p, r.d) for r in result.runs])
        self.assertEqual(1.0, result.limit)
        # every sweep ran with the gains of its setting, read back from the servo as well as in the simulation
        self.assertEqual([(i, p, d, (p, 0, d), (p, 0, d)) for i, (p, d) in enumerate(self.DEFAULT_SETTINGS)], seen)
        # afterwards the servo has its yaml gains again, and everything else as it was
        self.assertEqual(self.YAML, handler.gains[4])
        self.assertEqual((self.MOVE_TIME, 0), handler.profiles[4])
        self.assertEqual(cfg.MAX_PWM, handler.goal_pwm[4])

        # only the joint under test moves in the sweeps, within 10 degrees of the pose
        pose = parse_pose('elbow_out')
        moves = np.array(robot.moves)
        others = [1, 2, 3, 5, 6]
        np.testing.assert_allclose(moves[:, others], np.tile(pose[others], (len(moves), 1)))
        self.assertAlmostEqual(np.radians(10), np.abs(moves[:, 4] - pose[4]).max(), places=6)

        # the writes are on the clock without drift, the servo is read several times between two writes
        steps = ff_sweep(10.0, 10.0, 0.01, round_trips=2)
        for first, last in zip(starts, starts[1:] + [len(times)]):
            written_at = np.array(times[first:last])
            self.assertEqual(len(steps), len(written_at))
            np.testing.assert_allclose(written_at - written_at[0], [s.time for s in steps], atol=0.003)

        # the samples of every setting with its p and d
        rows = self.rows_of(buffer)
        self.assertEqual(list(SWING_SAMPLE_FIELDS), list(rows[0].keys()))
        self.assertEqual({'4'}, {row['joint'] for row in rows})
        self.assertEqual({'1'}, {row['pose'] for row in rows})
        self.assertEqual({'10.0'}, {row['speed'] for row in rows})
        self.assertEqual({'1', '-1'}, {row['direction'] for row in rows})
        self.assertEqual({'0', '1'}, {row['cruise'] for row in rows})
        self.assertEqual({'0', '1', '2', '3', '4'}, {row['leg'] for row in rows})
        self.assertEqual({'0', '1', '2', '3'}, {row['setting'] for row in rows})
        self.assertEqual('lead', list(SWING_SAMPLE_FIELDS)[-1])
        self.assertEqual({'0'}, {row['lead'] for row in rows},
                         'a servo without a velocity_feedforward block gets no lead')
        for setting, (p, d) in enumerate(self.DEFAULT_SETTINGS):
            mine = [row for row in rows if row['setting'] == str(setting)]
            self.assertEqual({(str(p), str(d))}, {(row['p'], row['d']) for row in mine})
            self.assertGreater(len(mine), 3 * len(steps), 'sampled as often as it fits between two writes')
            seconds = [float(row['time']) for row in mine]
            self.assertEqual(sorted(seconds), seconds)

        # the report: a line per setting, the table, the spread and the recommendation
        for p, d in self.DEFAULT_SETTINGS:
            self.assertIn('  p={} d={}: swing '.format(p, d), output)
        self.assertEqual(4, output.count(' Hz with the angle rising, '))
        self.assertIn('swing of joint 4 (yaml p=1000 d=0)', output)
        self.assertIn('how much a repeat differs', output)
        self.assertIn('the yaml P 1000 is fine', output)
        self.assertNotIn('pid:', output, 'the script recommends, it does not write the yaml')

    def test_a_servo_with_a_velocity_feedforward_runs_the_sweeps_with_its_lead_and_the_csv_has_it(self):
        session, robot, sim, buffer = self.stage(swing_p=(800,))
        servo = robot.servo_handler.get_servo(4)
        servo.velocity_feedforward = VelocityFeedforward(3.9, 25)
        self.run_stage(session)
        rows = self.rows_of(buffer)
        leads = [int(row['lead']) * int(row['direction']) for row in rows if row['cruise'] == '1']
        # the angle of joint 4 rises with its position: at 10 degrees/s the lead is 3.9 * 10 * 128 / 1000 = 5 positions
        self.assertAlmostEqual(5.0, float(np.median(leads)), delta=1.0)
        self.assertGreaterEqual(min(leads), 0)
        self.assertEqual(0, int(rows[0]['lead']), 'the sweep starts at rest')
        self.assertEqual(0, int(rows[-1]['lead']), 'and ends at rest')
        self.assertIsNotNone(servo.velocity_feedforward, 'the swing stage never takes it away')

    def test_the_direction_in_the_csv_is_that_of_the_angle_even_for_a_servo_mounted_reversed(self):
        for reverse in (False, True):
            session, robot, sim, buffer = self.stage(reverse=reverse)
            servo = robot.servo_handler.get_servo(4)
            self.assertEqual(-1 if reverse else 1, position_sign(servo))
            self.run_stage(session)
            rows = [row for row in self.rows_of(buffer) if row['cruise'] == '1' and row['setting'] == '0']
            moved = 0
            for before, after in zip(rows, rows[1:]):
                if before['leg'] == after['leg'] and int(after['goal']) != int(before['goal']):
                    moved += 1
                    # the goal is in positions: it rises with the angle, or falls with it when mounted reversed
                    self.assertEqual(int(before['direction']) * position_sign(servo),
                                     int(np.sign(int(after['goal']) - int(before['goal']))))
            self.assertGreater(moved, 300)

    def test_the_lag_is_signed_with_the_direction_of_the_move_in_positions(self):
        # the same simulation, mounted one way and the other: what trails when the position falls is the same
        normal = self.run_stage(self.stage()[0])[0].runs[0].statistics
        flipped = self.run_stage(self.stage(reverse=True)[0])[0].runs[0].statistics
        # the sweep goes to the upper end first: three legs with the angle rising, two falling, whatever the mounting
        for statistics in (normal, flipped):
            self.assertEqual((3, 2), (statistics.rising.legs, statistics.falling.legs))
        self.assertAlmostEqual(normal.rising.lag, flipped.falling.lag, delta=1.5)
        self.assertAlmostEqual(normal.falling.lag, flipped.rising.lag, delta=1.5)
        self.assertGreater(abs(normal.rising.lag - normal.falling.lag), 10, 'the load gives the two directions a lag')

    def test_extra_d_gains_run_at_the_yaml_p_before_the_other_p_gains(self):
        session, robot, sim, buffer = self.stage(swing_d=(250,))
        seen = self.note_settings(session, robot, sim)
        result, output = self.run_stage(session)
        self.assertEqual([(1000, 0), (1000, 250), (800, 0), (600, 0), (1000, 0)], [(r.p, r.d) for r in result.runs])
        self.assertEqual([(i, p, d, (p, 0, d), (p, 0, d)) for i, (p, d) in enumerate(
            [(1000, 0), (1000, 250), (800, 0), (600, 0), (1000, 0)])], seen)
        self.assertEqual({'0', '1', '2', '3', '4'}, {row['setting'] for row in self.rows_of(buffer)})
        self.assertIn('d=250: the swing changes by', output)
        self.assertIn('D made no difference beyond the spread', output)
        self.assertEqual(self.YAML, robot.servo_handler.gains[4])

    def test_given_p_gains_are_tried_with_the_yaml_d(self):
        session, robot, sim, buffer = self.stage(swing_p=(1400, 500))
        result, output = self.run_stage(session)
        self.assertEqual([(1000, 0), (1400, 0), (500, 0), (1000, 0)], [(r.p, r.d) for r in result.runs])
        self.assertNotIn('D made', output, 'no extra D was tried')
        self.assertEqual(self.YAML, robot.servo_handler.gains[4])

    def test_a_p_that_swings_more_is_found_and_the_largest_p_within_the_limit_is_recommended(self):
        session, robot, sim, buffer = self.stage(**self.RESONANT)
        result, output = self.run_stage(session)
        first, lower, lowest, last = (run.statistics for run in result.runs)
        for statistics in (first, last):
            for side in (statistics.rising, statistics.falling):
                self.assertGreater(side.swing, 1.5, 'the yaml P rings')
                self.assertAlmostEqual(6.5, side.frequency, delta=0.7, msg='at the frequency of the loop')
        for statistics in (lower, lowest):
            for side in (statistics.rising, statistics.falling):
                self.assertLess(side.swing, 0.6)
        self.assertEqual(800, tune_servo_gains.recommended_swing_p(
            [(r.p, r.d, r.statistics.rising.swing, r.statistics.falling.swing) for r in result.runs], 0, 1.0))
        self.assertIn('recommended for servo_config.yaml', output)
        self.assertIn('pid: {p: 800, i: 0, d: 0}', output)
        self.assertNotIn('is fine', output)
        self.assertEqual(self.YAML, robot.servo_handler.gains[4])

    def test_without_a_p_within_the_limit_the_report_sends_you_to_the_mechanics(self):
        session, robot, sim, buffer = self.stage(swing_limit=0.2)
        result, output = self.run_stage(session)
        self.assertIn('no P tried swings at most 0.2 positions in both directions at d=0: check the belt or the '
                      'mechanics before lowering P further', output)
        self.assertNotIn('pid:', output)
        self.assertEqual(self.YAML, robot.servo_handler.gains[4])

    def test_an_abort_during_a_sweep_restores_the_gains_and_keeps_the_samples(self):
        session, robot, sim, buffer = self.stage()
        handler = robot.servo_handler
        warm = int(AbortLimits().max_temperature)
        reads = []

        def heat():
            # the servo gets too warm 500 reads into the sweep at the second setting
            if handler.gains[4] == (800, 0, 0):
                reads.append(1)
                sim.temperature = warm if len(reads) > 500 else 30
        handler.before_read = heat
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(tune_servo_gains.Stop) as raised:
            session.run_swing(parse_pose('elbow_out'))
        self.assertEqual('the swing stage was aborted: temperature {}C'.format(warm), str(raised.exception))
        self.assertIn('aborted: temperature', out.getvalue())
        self.assertEqual(self.YAML, handler.gains[4], 'restored at the abort')
        self.assertEqual(self.YAML, (sim.p, sim.i, sim.d))
        self.assertEqual((self.MOVE_TIME, 0), handler.profiles[4])
        self.assertEqual(session.pose_goal, sim.goal, 'the goal is back at the pose')
        self.assertNotIn('swing of joint 4', out.getvalue(), 'no report of a stage that was cut short')
        rows = self.rows_of(buffer)
        self.assertEqual({'0', '1'}, {row['setting'] for row in rows}, 'the samples of the aborted sweep are written')
        aborted = [row for row in rows if row['setting'] == '1']
        self.assertEqual({('800', '0')}, {(row['p'], row['d']) for row in aborted})
        self.assertAlmostEqual(501, len(aborted), delta=5)

    def test_a_stop_request_during_a_sweep_stops_the_stage_and_the_samples_are_written(self):
        session, robot, sim, buffer = self.stage()
        handler = robot.servo_handler
        handler.before_read = lambda: setattr(session.stop, 'requested', handler.gains[4] == (800, 0, 0))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(tune_servo_gains.Stop) as raised:
            session.run_swing(parse_pose('elbow_out'))
        self.assertEqual('interrupted', str(raised.exception))
        self.assertEqual({'0', '1'}, {row['setting'] for row in self.rows_of(buffer)})
        self.assertNotEqual(self.YAML, handler.gains[4], 'main restores in its finally')
        self.assertTrue(session.restore())
        self.assertEqual(self.YAML, handler.gains[4])

    def test_reads_that_keep_failing_stop_the_stage_and_the_samples_are_written(self):
        session, robot, sim, buffer = self.stage()
        handler = robot.servo_handler
        read_state = handler.read_state
        reads = []

        def failing(servo_ids):
            if handler.gains[4] == (800, 0, 0):
                reads.append(1)
                if len(reads) > 300:
                    return None
            return read_state(servo_ids)
        handler.read_state = failing
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(tune_servo_gains.Stop) as raised:
            session.run_swing(parse_pose('elbow_out'))
        self.assertEqual('failed to read servo 4 3 times in a row', str(raised.exception))
        self.assertEqual(3, session.read_failures)
        rows = self.rows_of(buffer)
        self.assertEqual({'0', '1'}, {row['setting'] for row in rows})
        self.assertEqual(300, len([row for row in rows if row['setting'] == '1']))
        self.assertTrue(session.restore())
        self.assertEqual(self.YAML, handler.gains[4])

    def test_a_servo_without_move_time_is_skipped(self):
        session, robot, sim, buffer = self.stage(move_time=None)
        result, output = self.run_stage(session)
        self.assertIsNone(result)
        self.assertEqual('servo 4 has no move_time in servo_config.yaml, the swing stage needs the time-based '
                         'profile\n', output)
        self.assertEqual([], robot.moves, 'nothing moved')
        self.assertEqual([], robot.servo_handler.gain_writes)
        self.assertEqual([], self.rows_of(buffer))

    def test_a_refused_sweep_skips_the_stage_before_anything_moves(self):
        pose = parse_pose('elbow_out')
        for overrides, expected in (({'swing_degrees': 200.0}, 'range'),  # beyond the +-180 degrees of joint 4
                                    ({'swing_degrees': 1.0}, 'no cruise leg of at least 0.6 s going both ways'),
                                    ({'swing_degrees': 4.0}, 'use a larger --swing-degrees or a lower --swing-speed'),
                                    ({'swing_speed': 900.0}, 'joint 4 would move')):
            session, robot, sim, buffer = self.stage(**overrides)
            reason = swing_refusal(robot, 4, pose, session.settings)
            self.assertIsNotNone(reason, overrides)
            self.assertIn(expected, reason)
            result, output = self.run_stage(session)
            self.assertIsNone(result)
            self.assertIn(expected, output)
            self.assertEqual([], robot.moves, 'nothing moved')
            self.assertEqual([], robot.servo_handler.gain_writes)
        session, robot, sim, buffer = self.stage()
        self.assertIsNone(swing_refusal(robot, 4, pose, session.settings))

    def test_a_pose_the_planner_refuses_skips_the_stage(self):
        session, robot, sim, buffer = self.stage()
        with mock.patch.object(session, 'move_to_pose', return_value=False):
            result, output = self.run_stage(session)
        self.assertIsNone(result)
        self.assertIn('no swing stage: the planner refuses the move to the pose', output)
        self.assertEqual([], robot.servo_handler.gain_writes)
        self.assertEqual([], self.rows_of(buffer))

    def test_a_pose_the_workspace_refuses_skips_the_stage(self):
        session, robot, sim, buffer = self.stage()
        reason = swing_refusal(robot, 4, parse_pose('90,0,0,0,0,0'), session.settings)
        self.assertIn('no swing stage at pose 90 0 0 0 0 0 degrees', reason)
        self.assertIn('below the minimum height', reason)


def swing_result(runs, limit=1.0):
    """a SwingResult from (p, d, swing rising, swing falling)"""
    return SwingResult(tuple(SwingRun(p, d, swing_statistics_of(rising, falling)) for p, d, rising, falling in runs),
                       limit)


class PrintSwingTests(unittest.TestCase):
    SERVO = Servo(0, 4096, -pi, pi, POSITION_CONTROL_MODE, p=2500, i=0, d=250)

    def report(self, runs, limit=1.0):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            tune_servo_gains.print_swing(2, self.SERVO, swing_result(runs, limit))
        return out.getvalue()

    def test_the_measurements_of_20_november_recommend_the_p_that_stays_within_the_limit(self):
        # joint 2: the swing with the angle falling is 2 positions at P 2500, 1.2 at 2000 and 0.75 at 1500
        output = self.report([(2500, 250, 0.6, 1.9), (2500, 0, 0.5, 2.2), (2500, 640, 0.6, 2.3), (2000, 250, 0.3, 1.2),
                              (1500, 250, 0.3, 0.75), (2500, 250, 0.9, 2.4)])
        self.assertIn('swing of joint 2 (yaml p=2500 d=250)', output)
        self.assertIn('the two runs at the yaml gains differ by 0.30 positions with the angle rising and 0.50 '
                      'falling: that is how much a repeat differs', output)
        self.assertIn('at the yaml gains the joint swings more with the angle falling: 2.15 against 0.75 positions',
                      output)
        # D 0 and D 640: within the spread of a repeat
        self.assertIn('d=0: the swing changes by -0.25 rising and +0.05 falling from the yaml D, within the spread',
                      output)
        self.assertIn('d=640: the swing changes by -0.15 rising and +0.15 falling from the yaml D, within the spread',
                      output)
        self.assertIn('D made no difference beyond the spread of a repeat (0.30 rising and 0.50 falling)', output)
        self.assertIn('recommended for servo_config.yaml', output)
        self.assertIn('pid: {p: 1500, i: 0, d: 250}', output)
        self.assertNotIn('belt', output)
        self.assertEqual(6, len([line for line in output.splitlines() if line[:5].strip().isdigit()]),
                         'a row of the table per setting')

    def test_a_d_that_changes_the_swing_beyond_the_spread_is_reported(self):
        output = self.report([(2500, 250, 0.6, 1.0), (2500, 800, 0.5, 2.9), (2500, 250, 0.7, 1.1)])
        self.assertIn('d=800: the swing changes by -0.15 rising and +1.85 falling from the yaml D, beyond the spread',
                      output)
        self.assertIn('D made a difference beyond the spread of a repeat', output)

    def test_the_yaml_p_is_fine_when_it_is_within_the_limit(self):
        output = self.report([(2500, 250, 0.4, 0.9), (2000, 250, 0.3, 0.4), (2500, 250, 0.5, 0.8)])
        self.assertIn('the yaml P 2500 is fine: it swings at most 1 positions in both directions at d=250', output)
        self.assertNotIn('pid:', output)
        self.assertNotIn('D made', output, 'no extra D was tried')
        self.assertIn('swings alike in both directions', self.report([(2500, 250, 0.5, 0.52), (2500, 250, 0.5, 0.5)]))

    def test_without_a_p_within_the_limit_the_belt_and_the_mechanics_are_to_be_checked(self):
        output = self.report([(2500, 250, 1.4, 2.9), (2000, 250, 1.1, 2.0), (1500, 250, 1.0, 1.5),
                              (2500, 250, 1.5, 3.0)])
        self.assertIn('no P tried swings at most 1 positions in both directions at d=250: check the belt or the '
                      'mechanics before lowering P further (a slack belt lowers the resonance and raises the swing)',
                      output)
        self.assertNotIn('pid:', output)
        self.assertNotIn('is fine', output)

    def test_the_limit_is_used(self):
        runs = [(2500, 250, 0.4, 2.9), (2000, 250, 0.3, 1.9), (1500, 250, 0.3, 1.5), (2500, 250, 0.5, 3.0)]
        self.assertIn('pid: {p: 2000, i: 0, d: 250}', self.report(runs, limit=2.0))
        self.assertIn('pid: {p: 1500, i: 0, d: 250}', self.report(runs, limit=1.5))
        self.assertIn('belt', self.report(runs, limit=1.0))


class SwingOptionTests(unittest.TestCase):
    """main with the options of the stage, with a session that only records what it is asked to do"""
    run_main = FeedforwardOptionTests.run_main

    def test_the_defaults(self):
        settings = Settings()
        self.assertEqual((10.0, 10.0, None, (), 1.0), (settings.swing_degrees, settings.swing_speed, settings.swing_p,
                                                       settings.swing_d, settings.swing_limit))

    def test_without_the_options_the_stage_does_not_run(self):
        code, output = self.run_main([])
        calls, finish, files, robot = self.recorded
        self.assertEqual(0, code)
        self.assertEqual([('session', True, True, False), ('run', 1), 'restore'], calls)
        self.assertEqual([False], self.swing_writers)
        self.assertEqual(['samples.csv', 'trials.csv'], files)
        self.assertNotIn('swing stage', output)

    def test_swing_runs_the_stage_after_the_p_and_d_tuning(self):
        code, output = self.run_main(['--swing'])
        calls, finish, files, robot = self.recorded
        self.assertEqual(0, code)
        self.assertEqual([('session', True, True, False), ('run', 1), ('swing', [90, 90, 0, 0, 0, 0]), 'restore'],
                         calls)
        self.assertEqual([True], self.swing_writers)
        self.assertEqual(['samples.csv', 'swing.csv', 'trials.csv'], files)
        finish.assert_called_once()
        self.assertIn('swing stage: joint 4 (yaml p=1000 d=0) at pose 90 90 0 0 0 0 degrees, sweeps of 10 degrees '
                      'either way at 10 degrees/s with the gains (p, d) (1000, 0) (800, 0) (600, 0) (1000, 0), '
                      'recommending the largest P that swings at most 1 positions, output', output)
        self.assertIn('swing samples in', output)

    def test_with_ff_and_swing_the_swing_stage_comes_last(self):
        code, output = self.run_main(['--ff', '--swing'])
        calls, finish, files, robot = self.recorded
        self.assertEqual(0, code)
        self.assertEqual([('session', True, True, True), ('run', 1), ('ff', [90, 90, 0, 0, 0, 0]),
                          ('swing', [90, 90, 0, 0, 0, 0]), 'restore'], calls)
        self.assertEqual(['ff.csv', 'samples.csv', 'swing.csv', 'trials.csv'], files)

    def test_swing_only_skips_the_p_and_d_stages(self):
        code, output = self.run_main(['--swing-only', '--swing-degrees', '12', '--swing-speed', '8', '--swing-p', '900',
                                      '700', '--swing-d', '250', '--swing-limit', '0.5'])
        calls, finish, files, robot = self.recorded
        self.assertEqual(0, code)
        self.assertEqual([('session', False, False, False), ('swing', [90, 90, 0, 0, 0, 0]), 'restore'], calls)
        self.assertEqual(['swing.csv'], files)
        self.assertTrue(robot.enabled)
        settings = self.settings[0]
        self.assertEqual((12.0, 8.0, (900, 700), (250,), 0.5), (settings.swing_degrees, settings.swing_speed,
                                                                settings.swing_p, settings.swing_d,
                                                                settings.swing_limit))
        self.assertIn('sweeps of 12 degrees either way at 8 degrees/s with the gains (p, d) (1000, 0) (1000, 250) '
                      '(900, 0) (700, 0) (1000, 0), recommending the largest P that swings at most 0.5 positions',
                      output)
        self.assertIn('joint 4 sweeps up to 12 degrees either way at 8 degrees/s once per setting', output)

    def test_the_stage_does_not_start_after_a_p_and_d_run_that_stopped(self):
        code, output = self.run_main(['--swing'], stop_in_run=True)
        calls, finish, files, robot = self.recorded
        self.assertEqual(0, code)
        self.assertEqual([('session', True, True, False), ('run', 1), 'restore'], calls)
        self.assertIn('stopped: a trial aborted', output)

    def test_a_servo_without_move_time_skips_the_stage_and_swing_only_ends_before_the_servos_are_enabled(self):
        line = 'servo 4 has no move_time in servo_config.yaml, the swing stage needs the time-based profile'
        code, output = self.run_main(['--swing-only'], move_time=None)
        calls, finish, files, robot = self.recorded
        self.assertEqual(1, code)
        self.assertEqual([], calls)
        self.assertFalse(robot.enabled)
        self.assertEqual([], files)
        self.assertEqual(1, output.count(line))
        # with --swing the P and D stages still run
        code, output = self.run_main(['--swing'], move_time=None)
        calls, finish, files, robot = self.recorded
        self.assertEqual(0, code)
        self.assertEqual([('session', True, True, False), ('run', 1), 'restore'], calls)
        self.assertEqual(1, output.count(line))
        self.assertTrue(robot.enabled)
        self.assertEqual(['samples.csv', 'trials.csv'], files)

    def test_a_sweep_without_a_cruise_leg_is_refused_before_the_servos_are_enabled(self):
        code, output = self.run_main(['--swing-only', '--swing-degrees', '3'])
        calls, finish, files, robot = self.recorded
        self.assertEqual(1, code)
        self.assertEqual([], calls)
        self.assertFalse(robot.enabled)
        self.assertIn('use a larger --swing-degrees or a lower --swing-speed', output)

    def test_the_stages_that_run_alone_do_not_run_together(self):
        def no_robot():
            raise AssertionError('the robot was touched')

        for argv in (['--swing-only', '--ff-only'], ['--swing-only', '--ff'], ['--ff-only', '--swing']):
            out = io.StringIO()
            with mock.patch.object(tune_servo_gains, 'get_robot', no_robot), contextlib.redirect_stdout(out):
                self.assertEqual(1, tune_servo_gains.main(['--joint', '4'] + argv), argv)
            self.assertIn('--swing-only runs the swing stage alone' if argv[0] == '--swing-only'
                          else '--ff-only runs the feedforward stage alone', out.getvalue())

    def test_gains_out_of_the_range_of_the_servo_and_bad_values_are_refused(self):
        def no_robot():
            raise AssertionError('the robot was touched')

        for argv in (['--swing-p', '0'], ['--swing-p', str(cfg.MAX_GAIN + 1)], ['--swing-p', '1000', '-5'],
                     ['--swing-d', '-1'], ['--swing-d', str(cfg.MAX_GAIN + 1)], ['--swing-d', '250', '99999'],
                     ['--swing-degrees', '0'], ['--swing-speed', '-1'], ['--swing-limit', '0']):
            out = io.StringIO()
            with mock.patch.object(tune_servo_gains, 'get_robot', no_robot), contextlib.redirect_stdout(out):
                self.assertEqual(1, tune_servo_gains.main(['--joint', '4', '--swing-only'] + argv), argv)
            self.assertIn('--swing-', out.getvalue(), argv)
        code, output = self.run_main(['--swing-only', '--swing-p', '1', str(cfg.MAX_GAIN), '--swing-d', '0',
                                      str(cfg.MAX_GAIN)])
        self.assertEqual(0, code, 'the ends of the range are fine')
