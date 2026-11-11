import math
import unittest

from src.robot_controllers.dynamixel_robot import dynamixel_x_config as cfg
from src.robot_controllers.dynamixel_robot.calibration import wrist_roll_limits
from src.robot_controllers.dynamixel_robot.calibration.wrist_roll_limits import MoveResult, Stop, WristRollTest, \
    profile_time, recommend
from src.robot_controllers.dynamixel_robot.servo_handler import ServoState

DT = 0.002  # seconds per read, like the servo


class FakeServo:
    min_position = 0
    max_position = 4096


class FakeHandler:
    """servo 6 as a current limited mover: it moves NEED mA worth of current at SPEED positions per second toward its
    goal, and doesn't move at all when the goal current is below NEED; the 5 V line drops SAG volts per mA"""

    NEED = 250  # mA the moves take
    SPEED = 3000.0  # positions per second
    HOLDING = 20  # mA at rest
    SAG = 0.001  # volts per mA

    def __init__(self, position=2048, sag=None):
        self.now = 0.0
        self.position = float(position)
        self.goal = position
        self.goal_current = 400
        self.servo = FakeServo()
        self.errors = {}
        self.sag = self.SAG if sag is None else sag
        self.profiles = []

    def get_servo(self, servo_id):
        return self.servo

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds

    def read_state(self, servo_ids):
        self.now += DT
        error = self.goal - self.position
        moving = abs(error) > 0.5
        current = self.HOLDING
        if moving and self.goal_current >= self.NEED:
            step = min(abs(error), self.SPEED * DT)
            self.position += step if error > 0 else -step
            current = self.NEED
        elif moving:
            current = self.goal_current  # pushing, but not enough to move
        voltage = 5.2 - self.sag * current
        status = cfg.MOVING_STATUS_PROFILE_ONGOING if moving else 0
        return {6: ServoState(tick=0, moving=int(moving), moving_status=status, pwm=0, current=current, velocity=0,
                              position=int(round(self.position)), velocity_trajectory=0,
                              position_trajectory=self.goal, voltage=round(voltage, 2), temperature=30)}

    def read_hardware_errors(self, servo_ids):
        return self.errors

    def move_servo_to_pos(self, servo_id, position):
        self.goal = position
        return True

    def set_profile(self, servo_id, velocity, acceleration):
        self.profiles.append((velocity, acceleration))
        return True

    def set_goal_current(self, servo_id, value):
        self.goal_current = value
        return True


class SwingingHandler(FakeHandler):
    """reaches its goal, then swings around it with an amplitude of 115 positions at 4 Hz"""

    def read_state(self, servo_ids):
        states = super().read_state(servo_ids)
        if abs(self.goal - self.position) <= 0.5:
            self.swing_start = getattr(self, 'swing_start', self.now)
            offset = 115 * math.sin(2 * math.pi * 4 * (self.now - self.swing_start))
            states[6] = states[6]._replace(position=int(round(self.goal + offset)), moving_status=0)
        return states


def result(goal_current, arrived=True, peak=250, voltage=5.0, delta=90.0):
    return MoveResult(velocity=200, acceleration=50, goal_current=goal_current, delta=delta, start=2048, target=3072,
                      arrived=arrived, reason=None if arrived else 'stalled', duration=1.0, expected=0.9,
                      peak_current=peak, mean_current=peak / 2, holding_current=20, min_voltage=voltage,
                      start_voltage=5.2, max_temperature=30, final_error=0)


class ProfileTimeTests(unittest.TestCase):

    def test_a_long_move_is_a_trapezoid(self):
        # 200 is 45.8 rpm (0.763 rev/s), 50 is 2.98 rev/s squared: 0.5 rev takes 0.655 s plus 0.256 s of ramps
        self.assertAlmostEqual(0.911, profile_time(180, 200, 50), places=3)
        self.assertAlmostEqual(profile_time(180, 200, 50), profile_time(-180, 200, 50))

    def test_a_short_move_is_a_triangle(self):
        self.assertAlmostEqual(2 * (10 / 360 / 2.98)**0.5, profile_time(10, 200, 50), places=3)

    def test_an_unlimited_profile_has_no_time(self):
        self.assertEqual(0.0, profile_time(90, 0, 50))
        self.assertEqual(4.0, wrist_roll_limits.move_timeout(0.0))


class RecommendTests(unittest.TestCase):

    def test_one_and_a_half_times_what_the_moves_needed(self):
        results = [result(150, arrived=False, peak=150), result(250, peak=240), result(400, peak=250),
                   result(600, peak=250)]
        goal, note = recommend(results)
        self.assertEqual(400, goal, '1.5 * 250 rounded up to 50')
        self.assertIn('at most 250 mA', note)

    def test_at_least_the_lowest_that_worked(self):
        goal, _ = recommend([result(150, arrived=False), result(600, peak=100)])
        self.assertEqual(600, goal)

    def test_capped_where_the_5_volt_line_stays_healthy(self):
        results = [result(250, peak=240), result(400, peak=380, voltage=4.6)]
        goal, note = recommend(results)
        self.assertEqual(250, goal)
        self.assertIn('capped', note)
        self.assertIn('still limited', note, 'the peak was close to the goal current')

    def test_nothing_arrived(self):
        self.assertEqual((None, 'no goal current let every move arrive'), recommend([result(150, arrived=False)]))


class RunTests(unittest.TestCase):

    def run_test(self, handler, deltas=(90.0, -90.0), goal_currents=(150, 400), profiles=((200, 50),)):
        test = WristRollTest(handler, clock=handler.clock, sleep=handler.sleep)
        return test, test.run(deltas, goal_currents, profiles)

    def test_too_little_current_stalls_and_more_arrives(self):
        handler = FakeHandler()
        test, results = self.run_test(handler)
        low = [r for r in results if r.goal_current == 150]
        self.assertTrue(low and not any(r.arrived for r in low))
        self.assertIn('stalled', low[0].reason)
        high = [r for r in results if r.goal_current == 400]
        self.assertEqual(4, len(high), 'out and back for both deltas')
        self.assertTrue(all(r.arrived for r in high))
        self.assertEqual(FakeHandler.NEED, max(r.peak_current for r in high))
        self.assertEqual(2048, handler.goal, 'back at the start')
        self.assertEqual((400, ), recommend(results)[:1])

    def test_a_sagging_5_volt_line_stops_the_move_and_the_higher_currents(self):
        handler = FakeHandler(sag=0.003)  # 250 mA drops it to 4.45 V
        test, results = self.run_test(handler, goal_currents=(400, 600))
        self.assertTrue(results[0].reason.startswith('the voltage dropped'))
        self.assertEqual({400}, {r.goal_current for r in results}, 'not tried higher')

    def test_a_delta_outside_the_yaml_range_is_skipped(self):
        handler = FakeHandler(position=3500)
        test, results = self.run_test(handler, deltas=(90.0, -90.0), goal_currents=(400,))
        self.assertEqual({-90.0, 90.0}, {r.delta for r in results}, 'only -90 out and +90 back')
        self.assertEqual(2, len(results))

    def test_a_hardware_error_ends_the_run(self):
        handler = FakeHandler()
        handler.errors = {6: ['overload']}
        with self.assertRaisesRegex(Stop, 'overload'):
            self.run_test(handler, goal_currents=(400,))

    def test_a_servo_that_swings_around_the_target_did_not_arrive(self):
        # the XC330 at servo 6 with P 1200 and D 0: it passes the target and swings +-115 positions around it
        handler = SwingingHandler()
        test = WristRollTest(handler, clock=handler.clock, sleep=handler.sleep)
        move = test.move(2048, 2560, 200, 50, 400, 45.0)
        self.assertFalse(move.arrived)
        self.assertIn('did not stay', move.reason)

    def test_the_xc330_on_12_volts_has_its_own_limits(self):
        limits = wrist_roll_limits.MODELS[1220]
        self.assertEqual(('XC330-T288', 910), (limits.name, limits.max_current))
        # 11.6 V at rest is healthy on the 12 V bus, where the XL330's limits would call nothing healthy
        goal, note = recommend([result(150, peak=80, voltage=11.6)], limits.healthy_voltage, limits.supply)
        self.assertEqual(150, goal)
        self.assertNotIn('check it', note)
        goal, note = recommend([result(150, peak=80, voltage=10.8)], limits.healthy_voltage, limits.supply)
        self.assertIn('the 12 V supply dropped below 11.0 V', note)

    def test_a_given_range_replaces_the_yaml_range(self):
        # the gripper before its open and closed positions are calibrated
        handler = FakeHandler(position=3999)
        test = WristRollTest(handler, clock=handler.clock, sleep=handler.sleep, servo_id=6,
                             position_range=(3800, 4095))
        results = test.run((-10.0, 20.0), (400,), ((200, 50),))
        self.assertEqual({-10.0, 10.0}, {r.delta for r in results}, '+20 degrees would leave the range')
        with self.assertRaisesRegex(Stop, r'no delta stays inside the range of servo 6 \[0, 4096\]'):
            WristRollTest(handler, clock=handler.clock, sleep=handler.sleep).run((200.0,), (400,), ((200, 50),))

    def test_an_unreachable_start_ends_the_run(self):
        handler = FakeHandler()
        test = WristRollTest(handler, clock=handler.clock, sleep=handler.sleep)
        handler.goal_current = 100  # it can't move at all
        with self.assertRaisesRegex(Stop, 'did not get back to its start'):
            handler.position = 2200
            test.back_to(2048, 200, 50, 90.0)


if __name__ == '__main__':
    unittest.main()
