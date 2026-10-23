import csv
import io
import unittest
from unittest import mock

import numpy as np
from numpy import pi

from src.global_constants import POSITION_CONTROL_MODE
from src.robot_controllers.dynamixel_robot import dynamixel_x_config as cfg
from src.robot_controllers.dynamixel_robot.calibration import gain_tuning, tune_servo_gains
import math

from src.robot_controllers.dynamixel_robot.calibration.gain_tuning import (
    AbortLimits, Criteria, RestMetrics, Sample, StepMetrics, abort_reason, choose_d, choose_p, d_values,
    environmental_abort, evaluate, p_values, p_values_down, pwm_clamp, pwm_per_position, rest_metrics, step_metrics,
    step_size, tracking_metrics)
from src.robot_controllers.dynamixel_robot.calibration.tune_servo_gains import (
    SAMPLE_FIELDS, TRIAL_FIELDS, Settings, StopFlag, TuningSession, joint_p_of, parse_pose, recommend)
from src.robot_controllers.dynamixel_robot.servo import Servo
from src.robot_controllers.dynamixel_robot.servo_handler import ServoState
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
        self.assertEqual([0, 200, 400, 800], d_values(1600))
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

    def test_pwm_noise_the_position_does_not_explain_still_fails(self):
        p = 2240
        rng = np.random.default_rng(3)
        rows = samples([2055, 2056] * 400, goal=2075)
        rows = [s._replace(pwm=int(round(p / 128 * (s.goal - s.position) + rng.normal(0, 12))))
                for s in rows]
        rest = rest_metrics(rows, p)
        self.assertGreater(rest.pwm_residual_std, 10)
        baseline = rest_metrics(samples([2055] * 800, goal=2075, pwm=270), 1720)
        self.assertEqual((['pwm_noise'], []), evaluate(rest, [step()], baseline, p_ratio=p / 1720))

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
        self.assertEqual((['pwm_noise'], []), evaluate(rest(pwm_std=6.0), [step()], baseline))
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

    def test_pwm_noise_is_compared_per_unit_of_p(self):
        # a dither of one position gives a pwm std that scales with P: the same dither at 1.69 times the P passes
        baseline = rest(pwm_std=4.0)
        self.assertEqual(([], []), evaluate(rest(pwm_std=6.7), [step()], baseline, p_ratio=1.69))
        self.assertEqual((['pwm_noise'], []), evaluate(rest(pwm_std=6.7), [step()], baseline))
        self.assertEqual((['pwm_noise'], []), evaluate(rest(pwm_std=13.0), [step()], baseline, p_ratio=1.69),
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
        self.assertIn('voltage', self.check(samples([2000] * 10, voltage=11.0)))
        self.assertIsNone(self.check(samples([2000] * 10, voltage=11.2)))

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
                     samples([2000] * 10, voltage=11.0),
                     samples([2000, 2000, 2060])):
            self.assertTrue(environmental_abort(self.check(rows)))
        for rows in (samples([2000] * 400, trajectory=2100), samples([2000, 2010] * 200),
                     samples([2000] * 100, pwm=-650)):
            self.assertFalse(environmental_abort(self.check(rows)))

    def test_oscillation_at_rest(self):
        rows = samples([2000, 2010] * 200)
        self.assertIn('oscillation', self.check(rows))
        self.assertIsNone(self.check(samples([2000, 2002] * 200)), 'quantization')
        self.assertIsNone(self.check(samples([2000, 2010] * 200, ), baseline_pp=4), 'three times the baseline')
        self.assertIsNone(self.check(samples([2000, 2010] * 100), last_goal_time=0.0), 'before the grace period')

    def test_pwm_stuck_at_the_clamp(self):
        rows = samples([2000] * 100, pwm=-650)
        self.assertIn('clamp', self.check(rows))
        self.assertIsNone(self.check(samples([2000] * 100, pwm=600)), 'below 90 % of the clamp')
        self.assertIsNone(self.check(samples([2000] * 40, pwm=650)), 'not yet long enough')
        mixed = samples([2000] * 60, pwm=650)
        mixed[-1] = mixed[-1]._replace(pwm=100)
        self.assertIsNone(self.check(mixed), 'not the whole window')


class ChoiceTests(unittest.TestCase):

    def test_choose_p(self):
        self.assertEqual((1650, 2540), choose_p([(1500, True), (1950, True), (2540, False)], 1.3))
        self.assertEqual((1950, None), choose_p([(1500, True), (1950, True), (2540, True)], 1.3))
        self.assertEqual((None, 1500), choose_p([(1500, False)], 1.3))
        self.assertEqual((980, 1500), choose_p([(1500, False), (1150, True)], 1.3))
        self.assertEqual((None, None), choose_p([], 1.3))

    def test_choose_d(self):
        results = [(0, True, 5.0, 1, 3.0), (200, True, 2.0, 0, 3.5), (400, True, 1.0, 0, 6.0),
                   (800, False, 0.5, 0, 9.0)]
        self.assertEqual(200, choose_d(results), 'the least overshoot without pwm noise')
        self.assertEqual(400, choose_d(results, pwm_std_floor=6.0), 'the floor admits the noisier one')
        self.assertEqual(0, choose_d([(0, True, 1.0, 0, 3.0), (200, True, 1.0, 0, 3.0)]), 'the smallest on a tie')
        self.assertIsNone(choose_d([(0, False, 1.0, 0, 3.0)]))
        self.assertIsNone(choose_d([]))

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
    damping falls as P rises, so the step overshoots more and more; D adds damping and pwm noise"""

    FORCE_PER_PWM = 213.0
    DAMPING = SIM_DAMPING
    DAMPING_PER_D = 0.078
    NOISE_PER_D = 0.02
    MAX_VELOCITY = 2000.0  # positions per second, bounds the amplitude of an unstable loop
    FRICTION = 6000.0  # force units, about 28 pwm or 2.4 positions of error at P 1500

    def __init__(self, position, load=50000.0, resonance_above=None, resonance_slope=0.1, damping=SIM_DAMPING,
                 dt=DT):
        """
        :param resonance_above: above this P gain the damping drops by resonance_slope per unit of P, so the loop
            rings and, with a large enough slope, oscillates with an amplitude bounded by the velocity limit
        """
        self.position = float(position)
        self.damping = damping
        self.resonance_above = resonance_above
        self.resonance_slope = resonance_slope
        self.velocity = 0.0
        self.goal = position
        self.p, self.i, self.d = 0, 0, 0
        self.goal_pwm = cfg.MAX_PWM
        self.load = load
        self.dt = dt
        self.now = 0.0
        self.pwm = 0.0
        self.temperature = 30
        self.rng = np.random.default_rng(1)

    def advance(self, seconds):
        steps = max(1, int(round(seconds / self.dt)))
        for _ in range(steps):
            error = self.goal - self.position
            self.pwm = self.p / 128.0 * error - self.d / 16.0 * self.velocity * 0.0059
            self.pwm = max(-self.goal_pwm, min(self.goal_pwm, self.pwm))
            damping = self.damping + self.DAMPING_PER_D * self.d
            if self.resonance_above is not None and self.p > self.resonance_above:
                damping -= self.resonance_slope * (self.p - self.resonance_above)
            net = self.pwm * self.FORCE_PER_PWM - damping * self.velocity - self.load
            # coulomb friction: it sticks until the net force exceeds the friction, and stops at a zero crossing
            if self.velocity == 0.0:
                acceleration = 0.0 if abs(net) <= self.FRICTION else net - self.FRICTION * np.sign(net)
            else:
                acceleration = net - self.FRICTION * np.sign(self.velocity)
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
                          velocity_trajectory=0, position_trajectory=self.goal, voltage=11.8,
                          temperature=self.temperature)


class FakeTuningHandler:
    """the handler methods the session uses, with the simulated servo under test"""

    def __init__(self, servo_config, joint, sim):
        self.servos = servo_config.servos
        self.joint = joint
        self.sim = sim
        self.profiles = {servo_id: (servo.profile_velocity, servo.profile_acceleration)
                         for servo_id, servo in self.servos.items()}
        self.goal_pwm = {servo_id: cfg.MAX_PWM for servo_id in self.servos}
        self.gains = {servo_id: (servo.p, servo.i, servo.d) for servo_id, servo in self.servos.items()}
        self.gain_writes = []
        self.goal_pwm_writes = []
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
        if servo_id == self.joint:
            self.sim.goal = position
        return True


class FakeTuningRobot(FakeRobot):

    def __init__(self, angles, joint, sim, resonance_by_pose=None):
        """
        :param resonance_by_pose: optional function of the commanded angles giving the resonance_above of the
            simulated servo there, so the knee differs per pose
        """
        super().__init__(angles)
        self.joint = joint
        self.sim = sim
        self.resonance_by_pose = resonance_by_pose
        self.servo_handler = FakeTuningHandler(self.servo_config, joint, sim)
        self.enabled = False

    def move_servos(self, angles):
        super().move_servos(angles)
        self.sim.goal = self.servo_handler.get_servo(self.joint).target_position
        if self.resonance_by_pose is not None:
            self.sim.resonance_above = self.resonance_by_pose(angles)


def session_for(joint=3, resonance_above=None, resonance_slope=0.1, settings=None, writers=False,
                resonance_by_pose=None):
    pose = parse_pose('elbow_out')
    sim = SimulatedServo(2048, resonance_above=resonance_above, resonance_slope=resonance_slope)
    robot = FakeTuningRobot(pose, joint, sim, resonance_by_pose)
    servo = robot.servo_handler.get_servo(joint)
    servo.set_target_position_from_angle(pose[joint], pose)
    sim.position = sim.goal = servo.target_position
    settings = settings or Settings(repeats=1, hold=0.5, rest=1.0, baseline_rest=1.0, move_degrees=0.0)
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
        self.assertEqual(2540, extended.knee_p)
        self.assertEqual(1650, extended.recommended_p)
        # the sag of the simulated load is 20 positions at P 1500 and the slope is P / 128
        self.assertAlmostEqual(-20, p_trials[0].steps[0].error, delta=2)
        self.assertAlmostEqual(1500 / 128, p_trials[0].pwm_slope, delta=1.0)
        self.assertTrue(all(m.moved for t in p_trials for m in t.steps))
        # D: nothing to damp at the recommended P with a yaml D of 0, so only d=0 is tried, at the P of the joint
        d_trials = [t for t in extended.trials if t.phase == 'd']
        self.assertEqual([0], [t.d for t in d_trials])
        self.assertEqual([1650], [t.p for t in d_trials])
        self.assertEqual(0, extended.recommended_d)
        self.assertEqual((1650, 0), recommend(results, servo.d))
        # the trials ran in two phases: the P sweeps of both poses, then the D sweeps
        phases = [(t.phase, r.number) for r in results for t in r.trials]
        self.assertEqual(('p', 1), phases[0])
        self.assertEqual(('d', 2), phases[-1])
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
        self.assertEqual(1950, results[0].knee_p)
        self.assertEqual(1270, results[0].recommended_p)
        # the gains went back to the yaml right after the abort, before anything else was written
        writes = robot.servo_handler.gain_writes
        index = writes.index((3, 1950, 0, 0))
        self.assertEqual((3, 1500, 0, 0), writes[index + 1])
        self.assertEqual((1500, 0, 0), robot.servo_handler.gains[3])
        rows = list(csv.DictReader(io.StringIO(trial_file.getvalue()), fieldnames=TRIAL_FIELDS))
        self.assertEqual(len(trials), len(rows))
        self.assertTrue(rows[1]['aborted'])

    def test_a_failing_yaml_gain_is_swept_down(self):
        session, robot, sim, sample_file, _ = self.session(resonance_above=1300, resonance_slope=0.25, writers=True)
        results = session.run([parse_pose('elbow_out')])
        phases = [(t.phase, t.p, t.passed) for t in results[0].trials if t.phase != 'd']
        self.assertEqual(('p', 1500, False), phases[0])
        self.assertEqual(('p_down', 1150, True), phases[1])
        self.assertEqual(1500, results[0].knee_p)
        self.assertEqual(980, results[0].recommended_p)
        rows = list(csv.DictReader(io.StringIO(sample_file.getvalue()), fieldnames=SAMPLE_FIELDS))
        self.assertEqual({'baseline', 'rest', 'step'}, {row['segment'] for row in rows})

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

    def test_the_d_sweep_runs_at_the_p_of_the_joint(self):
        # the 'low' pose resonates earlier than 'elbow_out', so its knee and recommendation are lower
        session, robot, sim, _, _ = self.session(
            resonance_by_pose=lambda angles: 2000 if angles[2] > np.radians(75) else 1600)
        results = session.run([parse_pose('elbow_out'), parse_pose('low')])
        self.assertEqual(1650, results[0].recommended_p)
        self.assertEqual(1270, results[1].recommended_p)
        self.assertEqual(1270, joint_p_of(results))
        for r in results:
            d_trials = [t for t in r.trials if t.phase == 'd']
            self.assertTrue(d_trials)
            self.assertEqual({1270}, {t.p for t in d_trials}, 'D is tried at the P of the joint at every pose')
        self.assertEqual((1270, 0), recommend(results, 0))

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
