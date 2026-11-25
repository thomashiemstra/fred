import contextlib
import io
import os
import re
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from src.global_constants import CURRENT_BASED_POSITION_CONTROL_MODE
from src.robot_controllers.dynamixel_robot import dynamixel_x_config as cfg
from src.robot_controllers.dynamixel_robot.calibration import gripper_test
from src.robot_controllers.dynamixel_robot.calibration.gripper_test import CONTACT, NO_CONTACT, NO_MOVE
from src.robot_controllers.dynamixel_robot.servo import Servo
from src.robot_controllers.dynamixel_robot.servo_config import load_servo_config, DEFAULT_SERVO_CONFIG_PATH

GRIPPER = gripper_test.GRIPPER_ID

YAML_LINES = [
    '# the servos of a test robot',
    'temperature_warning: 60  # degrees celsius',
    'temperature_limit: 65',
    '',
    'servos:',
    '  6:',
    '    name: wrist 3',
    '    model: XC330-T288-T',
    '    bus: ttl',
    '    mode: position',
    '    positions: [0, 4096]  # a full turn',
    '    angles: [-180, 180]',
    '    profile: {velocity: 200, acceleration: 50}',
    '    pid: {p: 1150, i: 0, d: 0}',
    '  # NOT CALIBRATED, the positions are those of the old servo',
    '  7:',
    '    name: gripper',
    '    model: XC330-T288-T',
    '    bus: ttl',
    '    mode: current_based_position',
    '    positions: [1024, 2048]',
    '    states: [0, 100]',
    '    goal_current: 150  # mA',
    '    profile: {velocity: 50, acceleration: 50}',
    '    pid: {p: 600, i: 0, d: 400}',
]


def make_servo(positions=(1024, 2048), goal_current=150):
    return Servo(positions[0], positions[1], 0.0, 100.0, CURRENT_BASED_POSITION_CONTROL_MODE, profile_velocity=50,
                 profile_acceleration=50, p=600, i=0, d=400, goal_current=goal_current)


class FakeHandler:
    """
    the handler of a gripper that follows its goal at SPEED positions per second while the torque is on, between its
    mechanical ends (the open end and the closed end, in either order); it draws the goal current register when it
    pushes against an end, WORKING_CURRENT when it moves and IDLE_CURRENT otherwise. Time only passes in sleep(). The
    writes are recorded in calls, in order.

    With a stiffness (mA per position) the closed end is an object: when it blocks the gripper, the current is
    min(goal current, stiffness * (goal - position)). The object gives way give positions per 100 mA of current.
    """
    SPEED = 1500.0  # positions per second
    WORKING_CURRENT = 25
    IDLE_CURRENT = 5

    def __init__(self, open_position=1086, closed_end=1900, open_end=1040, positions=(1024, 2048), torque=False,
                 stiffness=None, give=0.0):
        self.servo = make_servo(positions)
        self.now = 0.0
        self.position = float(open_position)
        self.goal = open_position
        self.lower_end = min(open_end, closed_end)
        self.upper_end = max(open_end, closed_end)
        self.torque = torque
        self.stiffness = stiffness
        self.give = give
        self.goal_current_register = 910  # what the servo holds until it is written, its maximum
        self.goal_current_at_torque_on = None
        self.ignore_goal_current_writes = False
        self.current = self.IDLE_CURRENT
        self.errors: list[str] = []
        self.temperature = 30
        self.failing_reads = 0  # the next reads of the position that fail
        self.on_read = None  # called with the handler at every read of the position, once
        self.calls: list[tuple] = []
        self.goal_log: list[tuple[int, int]] = []  # (goal, goal current register) at every goal write
        self.register_reads: list[tuple[int, int]] = []  # (number of calls so far, value) of every register read
        self.port_handler = object()
        self.packet_handler = SimpleNamespace(read2ByteTxRx=self.read_2_bytes)

    # time
    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        if not self.torque:
            self.current = self.IDLE_CURRENT
            return
        error = self.goal - self.position
        move = max(-self.SPEED * seconds, min(self.SPEED * seconds, error))
        new_position = max(self.lower_end, min(self.upper_end, self.position + move))
        blocked = abs(new_position - self.position - move) > 1e-9 and abs(self.goal - new_position) > 1
        self.position = new_position
        if blocked and self.stiffness is not None:
            self.press_against_object(1 if new_position >= self.upper_end else -1)
        elif blocked:
            self.current = self.goal_current_register
        elif abs(move) > 0.5:
            self.current = self.WORKING_CURRENT
        else:
            self.current = self.IDLE_CURRENT

    def press_against_object(self, direction):
        """
        the gripper is blocked by the object, whose surface is at self.position, and squeezes it in the closing
        direction by give * current / 100 positions: with the distance d between the goal and the surface, the current c
        is stiffness * (d - give * c / 100), at most the goal current register
        """
        surface = self.position
        give = self.give / 100.0
        distance = (self.goal - surface) * direction
        current = min(self.goal_current_register, self.stiffness * distance / (1.0 + self.stiffness * give))
        self.current = round(max(current, 0.0))
        self.position = surface + direction * give * self.current

    # reads
    def get_servo(self, servo_id):
        return self.servo

    def read_current_pos_single_servo(self, servo_id):
        if self.on_read is not None:
            hook, self.on_read = self.on_read, None  # a hook that wants to stay sets itself again
            hook(self)
        if self.failing_reads > 0:
            self.failing_reads -= 1
            return None
        self.servo.current_position = round(self.position)
        return self.servo.current_position

    def read_current_pos(self, servo_ids=None):
        return self.read_current_pos_single_servo(GRIPPER) is not None

    def read_present_current(self, servo_ids=None):
        return {GRIPPER: self.current}

    def read_hardware_errors(self, servo_ids=None):
        return {GRIPPER: list(self.errors)} if self.errors else {}

    def read_temperature_and_voltage(self, servo_ids=None):
        return {GRIPPER: (self.temperature, 12.0)}

    def read_torque_enabled(self, servo_ids=None):
        return {GRIPPER: self.torque}

    def read_2_bytes(self, port_handler, servo_id, address):
        if address == cfg.ADDR_GOAL_CURRENT:
            self.register_reads.append((len(self.calls), self.goal_current_register))
            return self.goal_current_register, 0, 0
        return self.current, 0, 0

    # writes
    def set_configured_operating_mode(self, servo_ids=None):
        self.calls.append(('mode',))
        return True

    def set_profile_velocity_and_acceleration(self, servo_ids=None):
        self.calls.append(('profile',))
        return True

    def set_pid(self, servo_ids=None):
        self.calls.append(('pid',))
        return True

    def set_goal_current(self, servo_id, value):
        self.calls.append(('goal_current', value))
        if not self.ignore_goal_current_writes:
            self.goal_current_register = value
        return True

    def set_configured_goal_current(self, servo_ids=None):
        self.calls.append(('configured_goal_current',))
        self.goal_current_register = self.servo.goal_current
        return True

    def hold_current_position(self, servo_ids=None):
        self.calls.append(('hold',))
        self.goal = round(self.position)
        return True

    def set_torque(self, enable, servo_ids=None):
        self.calls.append(('torque', enable))
        if enable:
            self.goal_current_at_torque_on = self.goal_current_register
        self.torque = enable
        return True

    def move_servo_to_pos(self, servo_id, position):
        self.calls.append(('goal', position))
        self.goal_log.append((position, self.goal_current_register))
        self.goal = position
        return True

    def set_angle(self, servo_id, angle, all_angles=None, velocity=None):
        self.servo.set_target_position_from_angle(angle)

    def move_to_angles(self, servo_ids=None):
        self.calls.append(('goal', self.servo.target_position))
        self.goal = self.servo.target_position
        return True

    # what the tests look at
    def goals(self):
        return [call[1] for call in self.calls if call[0] == 'goal']

    def kinds(self):
        return [call[0] for call in self.calls]


def quietly(function, *args, **kwargs):
    """returns what the function returns and what it printed"""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        result = function(*args, **kwargs)
    return result, out.getvalue()


def no_questions(prompt):
    raise AssertionError('asked: {}'.format(prompt))


def calibrate(handler, **kwargs):
    kwargs.setdefault('yes', True)
    kwargs.setdefault('ask', no_questions)
    return quietly(gripper_test.calibrate, handler, sleep=handler.sleep, clock=handler.clock, **kwargs)


class CalibrateTests(unittest.TestCase):

    def test_contact_is_found_at_the_end_and_reported(self):
        handler = FakeHandler(closed_end=1900)
        code, out = calibrate(handler)
        self.assertEqual(code, 0)
        self.assertIn('positions: [1086, 1900]', out)
        self.assertIn('open position 1086, closed position 1900', out)
        self.assertIn('travel 814 positions, 71.5 degrees', out)
        self.assertIn('peak current while closing 60 mA, current at contact 60 mA', out)

    def test_closed_position_is_close_to_the_end(self):
        handler = FakeHandler(closed_end=1777)
        result = self.find(handler)
        self.assertEqual(result.outcome, CONTACT)
        self.assertLessEqual(abs(result.position - 1777), 3)

    @staticmethod
    def find(handler, direction=1, step=10, max_travel=1500, open_position=1086):
        handler.torque = True
        return gripper_test.find_closed_position(handler, open_position, direction, step, max_travel, handler.sleep,
                                                 handler.clock)

    def test_goal_current_is_written_before_the_torque_goes_on(self):
        handler = FakeHandler()
        calibrate(handler)
        kinds = handler.kinds()
        self.assertLess(handler.calls.index(('goal_current', 60)), kinds.index('torque'))
        self.assertEqual(handler.calls[kinds.index('torque')], ('torque', True))
        self.assertEqual(handler.goal_current_at_torque_on, 60)
        self.assertLess(kinds.index('hold'), kinds.index('torque'))  # the goal is where it stands before it is on

    def test_current_option_is_the_goal_current(self):
        handler = FakeHandler()
        calibrate(handler, current=40)
        self.assertIn(('goal_current', 40), handler.calls)
        self.assertEqual(handler.goal_current_at_torque_on, 40)
        self.assertNotIn(('goal_current', 60), handler.calls)

    def test_default_current_is_below_the_goal_current_of_the_yaml(self):
        self.assertLess(gripper_test.DEFAULT_CALIBRATION_CURRENT, make_servo().goal_current)

    def test_current_above_the_goal_current_of_the_yaml_is_refused_before_anything_is_written(self):
        handler = FakeHandler()
        code, out = calibrate(handler, current=151)
        self.assertEqual(code, 1)
        self.assertEqual(handler.calls, [])
        self.assertIn('gentler than a grip', out)
        self.assertEqual(calibrate(FakeHandler(), current=150)[0], 0)  # the goal current itself is allowed

    def test_current_above_the_limit_of_the_model_is_refused(self):
        handler = FakeHandler()
        handler.servo.goal_current = 1000
        code, out = calibrate(handler, current=950, model_limit=910)
        self.assertEqual(code, 1)
        self.assertEqual(handler.calls, [])

    def test_bad_numbers_are_refused(self):
        for kwargs in ({'current': 0}, {'step': 0}, {'max_travel': 30}, {'max_travel': -5}):
            handler = FakeHandler()
            self.assertEqual(calibrate(handler, **kwargs)[0], 1, kwargs)
            self.assertEqual(handler.calls, [], kwargs)

    def test_pushing_stops_at_contact(self):
        handler = FakeHandler(closed_end=1900)
        calibrate(handler)
        goals = handler.goals()
        highest = max(goals)
        self.assertLess(highest, 1900 + 150)  # it noticed within about half a second, 100 positions of goal
        after = goals[goals.index(highest) + 1]
        self.assertLessEqual(abs(after - 1900), 3)  # the goal went back to where the gripper is

    def test_goal_moves_in_steps_of_the_step_option(self):
        handler = FakeHandler(closed_end=1500)
        calibrate(handler, step=25)
        goals = handler.goals()
        self.assertEqual(goals[:3], [1111, 1136, 1161])

    def test_gripper_is_opened_again_and_the_torque_is_off(self):
        handler = FakeHandler()
        calibrate(handler)
        self.assertFalse(handler.torque)
        self.assertEqual(handler.calls[-1], ('torque', False))
        self.assertLessEqual(abs(handler.position - 1086), gripper_test.ARRIVED)
        self.assertEqual(handler.goal, 1086)
        # the goal goes back to the open position in small steps, never by a jump
        goals = handler.goals()
        back = goals[goals.index(max(goals)) + 1:]
        self.assertTrue(all(abs(b - a) <= 10 for a, b in zip(back, back[1:])))

    def test_open_position_is_where_the_gripper_stands(self):
        handler = FakeHandler(open_position=1234, closed_end=2000, open_end=1200)
        code, out = calibrate(handler)
        self.assertEqual(code, 0)
        self.assertIn('positions: [1234, 2000]', out)

    def test_stop_within_the_first_30_positions_returns_1_and_still_opens_and_switches_off(self):
        handler = FakeHandler(closed_end=1100)  # it can only travel 14 positions
        code, out = calibrate(handler)
        self.assertEqual(code, 1)
        self.assertIn('did not move in that direction', out)
        self.assertNotIn('for servo_config.yaml', out)
        self.assertFalse(handler.torque)
        self.assertEqual(handler.calls[-1], ('torque', False))
        self.assertEqual(handler.goal, 1086)

    def test_stop_within_the_first_30_positions_is_a_no_move_result(self):
        self.assertEqual(self.find(FakeHandler(closed_end=1100)).outcome, NO_MOVE)

    def test_no_contact_within_the_travel_returns_1_and_still_opens_and_switches_off(self):
        handler = FakeHandler(closed_end=4000)
        code, out = calibrate(handler, max_travel=300)
        self.assertEqual(code, 1)
        self.assertIn('did not stop within 300 positions', out)
        self.assertNotIn('for servo_config.yaml', out)
        self.assertLessEqual(max(handler.goals()), 1086 + 300)
        self.assertFalse(handler.torque)
        self.assertEqual(handler.calls[-1], ('torque', False))
        self.assertLessEqual(abs(handler.position - 1086), gripper_test.ARRIVED)

    def test_no_contact_result(self):
        result = self.find(FakeHandler(closed_end=4000), max_travel=200)
        self.assertEqual(result.outcome, NO_CONTACT)
        self.assertIsNone(result.contact_current)

    def test_closing_down_works_on_a_gripper_whose_closed_end_is_lower(self):
        handler = FakeHandler(open_position=1086, closed_end=900, open_end=1100)
        code, out = calibrate(handler, closing='down')
        self.assertEqual(code, 0)
        self.assertIn('positions: [1086, 900]', out)
        self.assertLessEqual(max(handler.goals()), 1086)
        self.assertLessEqual(abs(handler.position - 1086), gripper_test.ARRIVED)
        self.assertFalse(handler.torque)

    def test_closing_up_on_a_gripper_that_closes_down_finds_no_move(self):
        handler = FakeHandler(open_position=1086, closed_end=900, open_end=1100)
        code, out = calibrate(handler, closing='up')
        self.assertEqual(code, 1)
        self.assertFalse(handler.torque)

    def test_default_direction_follows_the_yaml(self):
        up = make_servo((1024, 2048))
        down = make_servo((2048, 1024))
        self.assertEqual(gripper_test.closing_direction(up, None), 1)
        self.assertEqual(gripper_test.closing_direction(down, None), -1)
        self.assertEqual(gripper_test.closing_direction(up, 'down'), -1)
        self.assertEqual(gripper_test.closing_direction(down, 'up'), 1)
        handler = FakeHandler(open_position=1700, closed_end=1500, open_end=1750, positions=(2048, 1024))
        code, out = calibrate(handler)
        self.assertEqual(code, 0)
        self.assertIn('positions: [1700, 1500]', out)

    def test_open_position_without_room_in_the_closing_direction_is_refused(self):
        handler = FakeHandler(open_position=4090, closed_end=4095, open_end=4000)
        code, out = calibrate(handler)
        self.assertEqual(code, 1)
        self.assertIn('--closing down', out)
        self.assertEqual(handler.calls, [])

    def test_goal_never_leaves_the_range_of_the_servo(self):
        handler = FakeHandler(open_position=4000, closed_end=4095, open_end=3900)
        self.find(handler, max_travel=1500, open_position=4000)
        self.assertLessEqual(max(handler.goals()), cfg.MAX_POSITION)

    def test_hardware_error_refuses_before_anything_is_written(self):
        handler = FakeHandler()
        handler.errors = ['overload']
        code, out = calibrate(handler)
        self.assertEqual(code, 1)
        self.assertEqual(handler.calls, [])
        self.assertIn('overload', out)

    def test_torque_already_on_refuses_before_anything_is_written(self):
        handler = FakeHandler(torque=True)
        code, out = calibrate(handler)
        self.assertEqual(code, 1)
        self.assertEqual(handler.calls, [])
        self.assertIn('torque of the gripper is on', out)
        self.assertTrue(handler.torque)  # it is not touched

    def test_failed_read_before_the_start_refuses(self):
        handler = FakeHandler()
        handler.failing_reads = 3
        code, out = calibrate(handler)
        self.assertEqual(code, 1)
        self.assertEqual(handler.calls, [])

    def test_failed_goal_current_write_does_not_enable_the_torque(self):
        handler = FakeHandler()
        handler.ignore_goal_current_writes = True  # the register stays at 910 mA
        code, out = calibrate(handler)
        self.assertEqual(code, 1)
        self.assertNotIn(('torque', True), handler.calls)
        self.assertFalse(handler.torque)

    def test_exception_during_the_closing_still_switches_the_torque_off(self):
        handler = FakeHandler()

        def explode(h):
            if h.torque and h.position > 1300:
                raise RuntimeError('the usb adapter fell out')
            h.on_read = explode
        handler.on_read = explode
        with self.assertRaises(RuntimeError):
            calibrate(handler)
        self.assertFalse(handler.torque)
        self.assertEqual(handler.calls[-1], ('torque', False))
        kinds = handler.kinds()
        # the goal was set to the present position after the torque went on
        self.assertGreater(len(kinds) - 1 - kinds[::-1].index('hold'), kinds.index('torque'))

    def test_ctrl_c_during_the_closing_holds_and_switches_the_torque_off(self):
        handler = FakeHandler()

        def interrupt(h):
            if h.torque and h.position > 1300:
                raise KeyboardInterrupt
            h.on_read = interrupt
        handler.on_read = interrupt
        code, out = calibrate(handler)
        self.assertEqual(code, 1)
        self.assertFalse(handler.torque)
        self.assertEqual(handler.calls[-1], ('torque', False))
        self.assertGreater(handler.goal, 1086)  # it was not opened, it holds where it was
        self.assertLess(handler.goal, 1500)
        self.assertLessEqual(abs(handler.goal - handler.position), 1)

    def test_hardware_error_during_the_closing_stops_it(self):
        handler = FakeHandler()

        def fail(h):
            if h.torque and h.position > 1300:
                h.errors = ['overload']
            else:
                h.on_read = fail
        handler.on_read = fail
        code, out = calibrate(handler)
        self.assertEqual(code, 1)
        self.assertIn('overload', out)
        self.assertFalse(handler.torque)
        self.assertLess(max(handler.goals()), 1700)  # it did not keep pushing to the end

    def test_warm_servo_during_the_closing_stops_it(self):
        handler = FakeHandler()

        def heat(h):
            if h.torque and h.position > 1300:
                h.temperature = gripper_test.MAX_TEMPERATURE + 1
            else:
                h.on_read = heat
        handler.on_read = heat
        code, out = calibrate(handler)
        self.assertEqual(code, 1)
        self.assertIn('warmer than', out)
        self.assertFalse(handler.torque)
        self.assertEqual(handler.calls[-1], ('torque', False))

    def test_one_failed_read_is_tried_again_but_three_in_a_row_stop_it(self):
        handler = FakeHandler()

        def blip(h):
            if h.torque and h.position > 1300:
                h.failing_reads = 2
            else:
                h.on_read = blip
        handler.on_read = blip
        self.assertEqual(calibrate(handler)[0], 0)

        handler = FakeHandler()

        def outage(h):
            if h.torque and h.position > 1300:
                h.failing_reads = 3
            else:
                h.on_read = outage
        handler.on_read = outage
        code, out = calibrate(handler)
        self.assertEqual(code, 1)
        self.assertFalse(handler.torque)
        self.assertEqual(handler.calls[-1], ('torque', False))

    def test_torque_that_cannot_be_confirmed_off_returns_1(self):
        handler = FakeHandler()
        # it reads as off before the torque was enabled, and stays on after
        handler.read_torque_enabled = lambda ids=None: {GRIPPER: ('torque', True) in handler.calls}
        code, out = calibrate(handler)
        self.assertEqual(code, 1)
        self.assertIn('COULD NOT CONFIRM', out)

    def test_question_before_moving_is_asked_without_yes_and_nothing_happens_on_no(self):
        handler = FakeHandler()
        questions = []
        code, out = quietly(gripper_test.calibrate, handler, yes=False, ask=lambda q: questions.append(q) or 'no',
                            sleep=handler.sleep, clock=handler.clock)
        self.assertEqual(code, 0)
        self.assertEqual(len(questions), 1)
        self.assertIn('type yes to continue', questions[0])
        self.assertEqual(handler.calls, [])

    def test_yes_option_skips_the_question_before_moving(self):
        handler = FakeHandler()
        self.assertEqual(calibrate(handler, yes=True)[0], 0)  # no_questions raises when it is asked

    def test_typed_yes_starts_it(self):
        handler = FakeHandler()
        code, out = quietly(gripper_test.calibrate, handler, yes=False, ask=lambda q: 'yes', sleep=handler.sleep,
                            clock=handler.clock)
        self.assertEqual(code, 0)
        self.assertIn(('torque', True), handler.calls)


class WriteTests(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = os.path.join(self.directory.name, 'servo_config.yaml')
        with open(self.path, 'wb') as config_file:
            config_file.write('\r\n'.join(YAML_LINES + ['']).encode())
        self.original = self.read()

    def read(self):
        with open(self.path, 'rb') as config_file:
            return config_file.read()

    def test_replace_positions_changes_only_the_positions_of_the_servo(self):
        text = self.original.decode()
        new = gripper_test.replace_positions(text, 7, 1086, 1900)
        old_lines, new_lines = text.splitlines(keepends=True), new.splitlines(keepends=True)
        self.assertEqual(len(old_lines), len(new_lines))
        changed = [i for i, (a, b) in enumerate(zip(old_lines, new_lines)) if a != b]
        self.assertEqual(changed, [old_lines.index('    positions: [1024, 2048]\r\n')])
        self.assertEqual(new_lines[changed[0]], '    positions: [1086, 1900]\r\n')

    def test_replace_positions_keeps_a_comment_and_adds_the_note_to_a_line_without_one(self):
        text = self.original.decode()
        self.assertIn('    positions: [5, 10]  # a full turn\r\n', gripper_test.replace_positions(text, 6, 5, 10))
        new = gripper_test.replace_positions(text, 7, 1086, 1900, 'measured')
        self.assertIn('    positions: [1086, 1900]  # measured\r\n', new)

    def test_replace_positions_keeps_unix_line_endings(self):
        text = self.original.decode().replace('\r\n', '\n')
        new = gripper_test.replace_positions(text, 7, 1086, 1900)
        self.assertNotIn('\r', new)
        self.assertIn('    positions: [1086, 1900]\n', new)

    def test_replace_positions_of_a_missing_servo_is_an_error(self):
        with self.assertRaises(ValueError):
            gripper_test.replace_positions(self.original.decode(), 9, 1, 2)

    def test_write_replaces_only_the_positions_line_of_servo_7(self):
        handler = FakeHandler(closed_end=1900)
        code, out = calibrate(handler, write=True, config_path=self.path)
        self.assertEqual(code, 0)
        self.assertIn('written', out)
        new = self.read()
        old_lines, new_lines = self.original.split(b'\r\n'), new.split(b'\r\n')
        self.assertEqual(len(old_lines), len(new_lines))
        changed = [(a, b) for a, b in zip(old_lines, new_lines) if a != b]
        self.assertEqual(len(changed), 1)
        self.assertEqual(changed[0][0], b'    positions: [1024, 2048]')
        self.assertTrue(changed[0][1].startswith(b'    positions: [1086, 19'))
        self.assertEqual(new.count(b'\r\n'), self.original.count(b'\r\n'))  # every line ending is still CRLF
        self.assertEqual(new.count(b'\n'), new.count(b'\r\n'))
        config = load_servo_config(self.path)
        gripper = config.servos[GRIPPER]
        self.assertEqual(gripper.min_position, 1086)
        self.assertLessEqual(abs(gripper.max_position - 1900), 3)
        self.assertEqual((config.servos[6].min_position, config.servos[6].max_position), (0, 4096))

    def test_nothing_is_written_without_the_write_option(self):
        code, out = calibrate(FakeHandler(), config_path=self.path)
        self.assertEqual(code, 0)
        self.assertEqual(self.read(), self.original)
        self.assertIn('nothing was written', out)
        self.assertIn('positions: [1086, 19', out)

    def test_write_asks_first_without_yes(self):
        handler = FakeHandler()
        answers = ['yes', 'no']
        questions = []
        code, out = quietly(gripper_test.calibrate, handler, write=True, config_path=self.path,
                            ask=lambda q: questions.append(q) or answers.pop(0), sleep=handler.sleep,
                            clock=handler.clock)
        self.assertEqual(code, 0)
        self.assertEqual(len(questions), 2)
        self.assertIn('type yes', questions[1])
        self.assertEqual(self.read(), self.original)
        self.assertIn('not writing', out)

        handler = FakeHandler()
        answers = ['yes', 'yes']
        quietly(gripper_test.calibrate, handler, write=True, config_path=self.path, ask=lambda q: answers.pop(0),
                sleep=handler.sleep, clock=handler.clock)
        self.assertNotEqual(self.read(), self.original)

    def test_failed_calibration_writes_nothing(self):
        code, out = calibrate(FakeHandler(closed_end=1100), write=True, config_path=self.path)
        self.assertEqual(code, 1)
        self.assertEqual(self.read(), self.original)
        code, out = calibrate(FakeHandler(closed_end=4000), write=True, config_path=self.path, max_travel=200)
        self.assertEqual(code, 1)
        self.assertEqual(self.read(), self.original)

    def test_closing_down_writes_the_open_position_first(self):
        handler = FakeHandler(open_position=1086, closed_end=900, open_end=1100)
        code, out = calibrate(handler, closing='down', write=True, config_path=self.path)
        self.assertEqual(code, 0)
        gripper = load_servo_config(self.path).servos[GRIPPER]
        self.assertEqual(gripper.min_position, 1086)  # state 0 is the open position
        self.assertLess(gripper.max_position, 1086)

    def test_states_in_the_other_order_fail_before_the_file_is_changed(self):
        with open(self.path, 'wb') as config_file:
            config_file.write(self.original.replace(b'states: [0, 100]', b'states: [100, 0]'))
        original = self.read()
        with self.assertRaises(ValueError):
            gripper_test.write_positions(self.path, GRIPPER, 1086, 1900)
        self.assertEqual(self.read(), original)

    def test_real_servo_config_gets_the_positions_and_nothing_else_changes(self):
        path = os.path.join(self.directory.name, 'real.yaml')
        shutil.copyfile(DEFAULT_SERVO_CONFIG_PATH, path)
        with open(path, 'rb') as config_file:
            before = config_file.read()
        gripper_test.write_positions(path, GRIPPER, 1086, 1900)
        with open(path, 'rb') as config_file:
            after = config_file.read()
        old_lines, new_lines = before.splitlines(keepends=True), after.splitlines(keepends=True)
        self.assertEqual(len(old_lines), len(new_lines))
        self.assertEqual(sum(a != b for a, b in zip(old_lines, new_lines)), 1)
        gripper = load_servo_config(path).servos[GRIPPER]
        self.assertEqual((gripper.min_position, gripper.max_position), (1086, 1900))
        self.assertEqual(load_servo_config(path).servos[6].max_position,
                         load_servo_config(DEFAULT_SERVO_CONFIG_PATH).servos[6].max_position)

    def test_the_real_servo_config_is_not_touched_by_the_tests(self):
        # a guard: the default path of calibrate is the real yaml, the tests always pass their own
        self.assertNotEqual(os.path.abspath(self.path), os.path.abspath(DEFAULT_SERVO_CONFIG_PATH))


class InteractiveTests(unittest.TestCase):

    def test_squeeze_currents_are_capped_at_the_limit_of_the_model(self):
        self.assertEqual(gripper_test.squeeze_currents(910), [200, 400, 600, 800, 910])
        self.assertEqual(gripper_test.squeeze_currents(1750), gripper_test.GOAL_CURRENTS)
        self.assertEqual(gripper_test.squeeze_currents(100), [100])

    def test_current_limit_of_the_models(self):
        self.assertEqual(gripper_test.model_current_limit('XC330-T288-T'), 910)
        self.assertEqual(gripper_test.model_current_limit('XL330-M288'), 1750)
        self.assertEqual(gripper_test.model_current_limit('XL330-M288-T'), 1750)
        self.assertEqual(gripper_test.model_current_limit('XC330-M181-T'), 910)
        self.assertEqual(gripper_test.model_current_limit('Z999'), gripper_test.DEFAULT_CURRENT_LIMIT)
        self.assertEqual(gripper_test.model_current_limit(None), gripper_test.DEFAULT_CURRENT_LIMIT)

    def test_the_real_gripper_has_a_limit_of_910(self):
        config = load_servo_config()
        self.assertEqual(gripper_test.model_current_limit(config.models[GRIPPER]), 910)

    def test_squeeze_tests_only_use_goal_currents_the_model_allows(self):
        handler = FakeHandler(torque=True)
        with mock.patch.object(gripper_test, 'sleep', handler.sleep), \
                mock.patch.object(gripper_test, 'perf_counter', handler.clock), \
                mock.patch('builtins.input', return_value=''):
            result, out = quietly(gripper_test.squeeze_tests, handler, gripper_test.squeeze_currents(910))
        self.assertTrue(result)
        written = [call[1] for call in handler.calls if call[0] == 'goal_current']
        self.assertEqual(written, [200, 400, 600, 800, 910])
        self.assertIn('configured_goal_current', handler.kinds())  # the goal current of the yaml is back at the end

    def test_distance_outside_range(self):
        servo = make_servo((1024, 2048))
        self.assertEqual(gripper_test.distance_outside_range(servo, 1086), 0)
        self.assertEqual(gripper_test.distance_outside_range(servo, 1000), 24)
        self.assertEqual(gripper_test.distance_outside_range(servo, 2100), 52)
        servo.constant_offset = 10
        self.assertEqual(gripper_test.distance_outside_range(servo, 1030), 4)
        self.assertEqual(gripper_test.distance_outside_range(make_servo((2048, 1024)), 1086), 0)

    def test_range_check_accepts_a_calibrated_or_nearly_calibrated_gripper(self):
        for position in (1086, 1024, 2048, 974, 2098):  # the last two are exactly 50 outside
            handler = FakeHandler(open_position=position, open_end=position - 100, closed_end=position + 100)
            result, out = quietly(gripper_test.gripper_is_calibrated, handler)
            self.assertTrue(result, position)

    def test_range_check_refuses_an_uncalibrated_gripper_and_says_what_to_run(self):
        for position in (973, 2099, 3000, 100):
            handler = FakeHandler(open_position=position, open_end=position - 50, closed_end=position + 50)
            result, out = quietly(gripper_test.gripper_is_calibrated, handler)
            self.assertFalse(result, position)
            self.assertIn('--calibrate', out)

    def test_range_check_refuses_when_the_position_cannot_be_read(self):
        handler = FakeHandler()
        handler.failing_reads = 1
        self.assertFalse(quietly(gripper_test.gripper_is_calibrated, handler)[0])


def measure(handler, **kwargs):
    kwargs.setdefault('yes', True)
    kwargs.setdefault('ask', no_questions)
    return quietly(gripper_test.measure_lead, handler, sleep=handler.sleep, clock=handler.clock, **kwargs)


def measure_with_rows(handler, **kwargs):
    """returns the code, the output and the rows that measure_leads gave (None when it was not reached)"""
    captured = []
    original = gripper_test.measure_leads

    def spy(*args, **kw):
        rows = original(*args, **kw)
        captured.append(rows)
        return rows
    with mock.patch.object(gripper_test, 'measure_leads', spy):
        code, out = measure(handler, **kwargs)
    return code, out, captured[0] if captured else None


def goals_after_yaml_current(handler):
    """the goals written after the goal current of the yaml (150 mA)"""
    start = handler.calls.index(('goal_current', 150))
    return [call[1] for call in handler.calls[start:] if call[0] == 'goal']


def make_row(lead, distance, mean_current, goal_current=150):
    return gripper_test.LeadRow(lead, 1500.0, distance, mean_current, int(mean_current),
                                100.0 * mean_current / goal_current)


LEADS = [2, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100]


class MeasureLeadTests(unittest.TestCase):
    """the gripper holds an object at position 1500 that pushes back with 3 mA per position unless told otherwise"""

    @staticmethod
    def object_in_the_gripper(**kwargs):
        kwargs.setdefault('closed_end', 1500)
        kwargs.setdefault('stiffness', 3.0)
        return FakeHandler(**kwargs)

    def test_contact_is_found_at_the_object(self):
        handler = self.object_in_the_gripper()
        code, out, rows = measure_with_rows(handler)
        self.assertEqual(code, 0)
        self.assertIn('the object stops the gripper at position 1500', out)
        self.assertEqual([row.position for row in rows], [1500.0] * len(rows))

    def test_goal_current_is_the_gentle_one_while_closing_and_the_one_of_the_yaml_while_the_leads_are_measured(self):
        handler = self.object_in_the_gripper()
        measure(handler)
        first_lead = [goal for goal, register in handler.goal_log].index(1502)
        self.assertGreater(first_lead, 10)  # the closing wrote many goals
        self.assertTrue(all(register == 60 for goal, register in handler.goal_log[:first_lead]))
        self.assertTrue(all(register == 150 for goal, register in handler.goal_log[first_lead:]))
        self.assertEqual(handler.goal_current_at_torque_on, 60)

    def test_each_goal_current_is_read_back_before_it_is_used(self):
        handler = self.object_in_the_gripper()
        measure(handler)
        for value in (60, 150):
            written = handler.calls.index(('goal_current', value))
            self.assertIn((written + 1, value), handler.register_reads, value)  # the read straight after the write
        self.assertLess(handler.calls.index(('goal_current', 60)), handler.calls.index(('torque', True)))
        # the current is raised after the goal is back at the object, before the first lead
        raised = handler.calls.index(('goal_current', 150))
        self.assertEqual(handler.calls[raised - 1], ('goal', 1500))
        self.assertEqual(handler.calls[raised + 1], ('goal', 1502))

    def test_current_option_is_the_goal_current_while_closing(self):
        handler = self.object_in_the_gripper()
        measure(handler, current=40)
        self.assertEqual(handler.goal_current_at_torque_on, 40)
        self.assertNotIn(('goal_current', 60), handler.calls)

    def test_rows_have_the_commanded_leads_in_ascending_order(self):
        handler = self.object_in_the_gripper()
        code, out, rows = measure_with_rows(handler, leads=[50, 10, 30, 10])
        self.assertEqual([row.lead for row in rows], [10, 30, 50])
        self.assertEqual(goals_after_yaml_current(handler)[:3], [1510, 1530, 1550])

    def test_default_leads(self):
        handler = self.object_in_the_gripper()
        code, out, rows = measure_with_rows(handler)
        self.assertEqual([row.lead for row in rows], LEADS)
        self.assertEqual(gripper_test.DEFAULT_LEADS, LEADS)

    def test_distances_and_currents_of_the_rows(self):
        handler = self.object_in_the_gripper()
        code, out, rows = measure_with_rows(handler)
        for row in rows:
            self.assertEqual(row.distance, row.lead)  # the object does not give way
            self.assertAlmostEqual(row.mean_current, min(150, 3 * row.lead))
            self.assertEqual(row.peak_current, min(150, 3 * row.lead))
            self.assertAlmostEqual(row.percent, 100 * min(150, 3 * row.lead) / 150)

    def test_object_that_gives_way_shows_in_the_distance(self):
        handler = self.object_in_the_gripper(give=10.0)  # 10 positions per 100 mA
        code, out, rows = measure_with_rows(handler, leads=[10, 20, 30, 40])
        self.assertEqual([row.lead for row in rows], [10, 20, 30, 40])
        contact = int(re.search(r'stops the gripper at position (\d+)', out).group(1))
        self.assertLess(contact, 1510)  # the closing with 60 mA already squeezed the object a little
        for row in rows:
            self.assertGreater(row.position, 1500)
            self.assertEqual(row.distance, contact + row.lead - row.position)
            self.assertLess(row.distance, row.lead)  # the gripper squeezes the object, it is not at the contact
            # the object pushes back 3 mA per position of what the servo sees, the positions are whole
            self.assertAlmostEqual(row.mean_current, 3 * row.distance, delta=2)

    def test_peak_current_is_the_largest_read_and_the_mean_the_average(self):
        handler = self.object_in_the_gripper()
        handler.torque = True
        handler.goal_current_register = 150
        handler.goal = 1500
        handler.position = 1500.0
        levels = iter([10, -40, 20] * 20)  # the sign is dropped

        def noisy(servo_ids=None):  # a current that changes with every read
            return {GRIPPER: next(levels)}
        handler.read_present_current = noisy
        rows, out = quietly(gripper_test.measure_leads, handler, 1500, 1, [10], 150, handler.sleep, handler.clock)
        self.assertEqual(rows[0].peak_current, 40)
        self.assertGreater(rows[0].mean_current, 10)
        self.assertLess(rows[0].mean_current, 40)

    def test_next_goal_comes_one_and_a_half_seconds_after_the_previous_one(self):
        handler = self.object_in_the_gripper()
        times = {}
        original = handler.move_servo_to_pos

        def timed(servo_id, position):
            times.setdefault(position, handler.now)
            return original(servo_id, position)
        handler.move_servo_to_pos = timed
        measure(handler, leads=[10, 20, 30])
        for earlier, later in ((1510, 1520), (1520, 1530)):
            self.assertGreaterEqual(times[later] - times[earlier], 1.5)  # 0.5 s settle, 1.0 s of reads
            self.assertLess(times[later] - times[earlier], 1.75)

    def test_current_and_position_are_read_about_every_100_ms_for_one_second(self):
        handler = self.object_in_the_gripper()
        handler.torque = True
        handler.goal_current_register = 150
        handler.goal = 1500
        handler.position = 1500.0
        reads = []

        def count(h):
            reads.append(h.now)
            h.on_read = count
        handler.on_read = count
        quietly(gripper_test.measure_leads, handler, 1500, 1, [10, 20], 150, handler.sleep, handler.clock)
        self.assertGreaterEqual(len(reads), 18)
        self.assertLessEqual(len(reads), 22)
        self.assertGreaterEqual(reads[0], 0.5 + 0.09)  # not before the settle time
        self.assertAlmostEqual(reads[1] - reads[0], 0.1)

    def test_whole_squeeze_from_50_positions_and_the_recommendation(self):
        handler = self.object_in_the_gripper()
        code, out, rows = measure_with_rows(handler)
        self.assertEqual(code, 0)
        self.assertEqual(gripper_test.saturation_distance(rows), 50)
        self.assertAlmostEqual(gripper_test.current_per_position(rows), 3.0)
        self.assertEqual(gripper_test.recommended_lead(50), 65)
        self.assertIn('the squeeze is whole (at least 95% of the goal current) from 50.0 positions', out)
        self.assertIn('grows by 3.00 mA per position', out)
        self.assertIn('lead for the gamepad: 65 positions (6.3 states', out)  # 65 * 100 / 1024 of the fake yaml
        self.assertIn('the goal current is 150 mA', out)

    def test_table_has_a_line_per_lead(self):
        handler = self.object_in_the_gripper()
        code, out, rows = measure_with_rows(handler, leads=[10, 50])
        lines = out.splitlines()
        header = [i for i, line in enumerate(lines) if line.split()[:3] == ['lead', 'goal', '-']]
        self.assertEqual(len(header), 1)
        table = lines[header[0] + 1:header[0] + 3]
        self.assertEqual(table[0].split(), ['10', '10.0', '30.0', '20', '30', '1500.0'])
        self.assertEqual(table[1].split(), ['50', '50.0', '150.0', '100', '150', '1500.0'])

    def test_squeeze_that_is_not_whole_within_the_leads_says_so(self):
        handler = self.object_in_the_gripper()
        code, out, rows = measure_with_rows(handler, leads=[2, 5, 10, 20])
        self.assertEqual(code, 0)  # the table was measured
        self.assertIsNone(gripper_test.saturation_distance(rows))
        self.assertIn('did not reach 95% of the goal current within the leads tried', out)
        self.assertNotIn('lead for the gamepad', out)
        self.assertIn('grows by 3.00 mA per position', out)

    def test_stiff_object_has_a_small_whole_squeeze_distance(self):
        handler = self.object_in_the_gripper(stiffness=30.0)  # 150 mA from 5 positions
        code, out, rows = measure_with_rows(handler)
        self.assertEqual(gripper_test.saturation_distance(rows), 5)
        self.assertIn('lead for the gamepad: 10 positions', out)  # 5 * 1.25 = 6.25

    def test_leads_beyond_the_range_of_the_servo_end_the_measuring_but_keep_the_smaller_ones(self):
        handler = self.object_in_the_gripper(open_position=3900, open_end=3800, closed_end=4040)
        code, out, rows = measure_with_rows(handler, leads=[10, 50, 100, 200])
        self.assertEqual(code, 0)
        self.assertEqual([row.lead for row in rows], [10, 50])
        self.assertIn('outside the range of the servo', out)
        self.assertLessEqual(max(handler.goals()), cfg.MAX_POSITION)

    def test_closing_down(self):
        handler = self.object_in_the_gripper(open_position=1086, closed_end=900, open_end=1100)
        code, out, rows = measure_with_rows(handler, closing='down')
        self.assertEqual(code, 0)
        self.assertIn('the object stops the gripper at position 900', out)
        self.assertEqual([row.distance for row in rows], LEADS)
        self.assertEqual(goals_after_yaml_current(handler)[:2], [898, 895])
        self.assertEqual(gripper_test.saturation_distance(rows), 50)
        self.assertLessEqual(abs(handler.position - 1086), gripper_test.ARRIVED)

    def test_gripper_is_opened_again_and_the_torque_is_off(self):
        handler = self.object_in_the_gripper()
        measure(handler)
        self.assertFalse(handler.torque)
        self.assertEqual(handler.calls[-1], ('torque', False))
        self.assertLessEqual(abs(handler.position - 1086), gripper_test.ARRIVED)
        self.assertEqual(handler.goal, 1086)
        goals = goals_after_yaml_current(handler)
        self.assertEqual(goals[:len(LEADS)], [1500 + lead for lead in LEADS])
        self.assertEqual(goals[len(LEADS)], 1500)  # it stops pushing first
        back = goals[len(LEADS):]
        self.assertTrue(all(abs(b - a) <= 10 for a, b in zip(back, back[1:])))  # and opens in small steps

    def test_output_says_what_goal_current_the_servo_is_left_with(self):
        handler = self.object_in_the_gripper()
        code, out = measure(handler)
        self.assertIn('left with a goal current of 150 mA', out)
        self.assertIn('writes the goal current of the yaml when it starts', out)
        self.assertEqual(handler.goal_current_register, 150)

    def test_exception_during_the_leads_holds_where_it_is_and_switches_the_torque_off(self):
        handler = self.object_in_the_gripper()

        def explode(h):
            if h.goal_current_register == 150 and h.torque and h.goal >= 1530:  # while the leads are measured
                raise RuntimeError('the usb adapter fell out')
            h.on_read = explode
        handler.on_read = explode
        with self.assertRaises(RuntimeError):
            measure(handler)
        self.assertFalse(handler.torque)
        self.assertEqual(handler.calls[-1], ('torque', False))
        self.assertLessEqual(abs(handler.goal - handler.position), 1)  # not pushing any more
        self.assertGreater(handler.goal, 1086 + 300)  # and not opened either, the gripper holds

    def test_ctrl_c_during_the_leads_holds_where_it_is_and_switches_the_torque_off(self):
        handler = self.object_in_the_gripper()

        def interrupt(h):
            if h.goal_current_register == 150 and h.torque and h.goal >= 1530:  # while the leads are measured
                raise KeyboardInterrupt
            h.on_read = interrupt
        handler.on_read = interrupt
        code, out = measure(handler)
        self.assertEqual(code, 1)
        self.assertFalse(handler.torque)
        self.assertEqual(handler.calls[-1], ('torque', False))
        self.assertLessEqual(abs(handler.goal - handler.position), 1)
        self.assertNotIn('the squeeze is whole', out)
        self.assertIn('left with a goal current of 150 mA', out)

    def test_hardware_error_between_two_leads_ends_the_measuring(self):
        handler = self.object_in_the_gripper()

        def fail(h):
            if h.goal_current_register == 150 and h.torque and h.goal >= 1520:  # the fifth lead
                h.errors = ['overload']
            else:
                h.on_read = fail
        handler.on_read = fail
        code, out = measure(handler)
        self.assertEqual(code, 1)
        self.assertIn('overload', out)
        self.assertEqual(goals_after_yaml_current(handler), [1502, 1505, 1510, 1515, 1520])
        self.assertFalse(handler.torque)
        self.assertEqual(handler.calls[-1], ('torque', False))
        self.assertNotIn('the squeeze is whole', out)

    def test_warm_servo_between_two_leads_ends_the_measuring(self):
        handler = self.object_in_the_gripper()

        def heat(h):
            if h.goal_current_register == 150 and h.torque and h.goal >= 1520:
                h.temperature = gripper_test.MAX_TEMPERATURE + 1
            else:
                h.on_read = heat
        handler.on_read = heat
        code, out = measure(handler)
        self.assertEqual(code, 1)
        self.assertIn('warmer than', out)
        self.assertEqual(goals_after_yaml_current(handler), [1502, 1505, 1510, 1515, 1520])
        self.assertFalse(handler.torque)

    def test_three_failed_reads_in_a_row_end_the_measuring(self):
        handler = self.object_in_the_gripper()

        def outage(h):
            if h.goal_current_register == 150 and h.torque and h.goal >= 1520:
                h.failing_reads = 3
            else:
                h.on_read = outage
        handler.on_read = outage
        code, out = measure(handler)
        self.assertEqual(code, 1)
        self.assertFalse(handler.torque)
        self.assertEqual(handler.calls[-1], ('torque', False))

    def test_stop_within_the_first_30_positions_returns_1_without_squeezing(self):
        handler = self.object_in_the_gripper(closed_end=1100)
        code, out = measure(handler)
        self.assertEqual(code, 1)
        self.assertIn('did not move in that direction', out)
        self.assertNotIn(('goal_current', 150), handler.calls)  # the squeeze current is never written
        self.assertFalse(handler.torque)
        self.assertEqual(handler.calls[-1], ('torque', False))
        self.assertEqual(handler.goal, 1086)
        self.assertIn('left with a goal current of 60 mA', out)

    def test_no_contact_within_the_travel_returns_1_without_squeezing(self):
        handler = self.object_in_the_gripper(closed_end=4000)
        code, out = measure(handler, max_travel=300)
        self.assertEqual(code, 1)
        self.assertIn('did not stop within 300 positions', out)
        self.assertNotIn(('goal_current', 150), handler.calls)
        self.assertFalse(handler.torque)
        self.assertEqual(handler.calls[-1], ('torque', False))
        self.assertLessEqual(abs(handler.position - 1086), gripper_test.ARRIVED)

    def test_failed_write_of_the_goal_current_of_the_yaml_opens_again_and_returns_1(self):
        for acknowledged in (True, False):
            handler = self.object_in_the_gripper()
            original = handler.set_goal_current

            def set_goal_current(servo_id, value, original=original, handler=handler, acknowledged=acknowledged):
                if value == 150:  # the register is not changed, and the write is or is not acknowledged
                    handler.calls.append(('goal_current', value))
                    return acknowledged
                return original(servo_id, value)
            handler.set_goal_current = set_goal_current
            code, out = measure(handler)
            self.assertEqual(code, 1, acknowledged)
            self.assertIn('not squeezing', out)
            self.assertNotIn(1502, handler.goals())  # no lead was tried
            self.assertFalse(handler.torque)
            self.assertEqual(handler.calls[-1], ('torque', False))
            self.assertLessEqual(abs(handler.position - 1086), gripper_test.ARRIVED)  # opened again
            self.assertIn('left with a goal current of 60 mA', out)

    def test_failed_goal_current_write_before_the_start_does_not_enable_the_torque(self):
        handler = self.object_in_the_gripper()
        handler.ignore_goal_current_writes = True
        code, out = measure(handler)
        self.assertEqual(code, 1)
        self.assertNotIn(('torque', True), handler.calls)

    def test_hardware_error_torque_on_or_a_failed_read_before_the_start_refuse_without_writing(self):
        handler = self.object_in_the_gripper()
        handler.errors = ['overload']
        self.assertEqual(measure(handler)[0], 1)
        self.assertEqual(handler.calls, [])
        handler = self.object_in_the_gripper(torque=True)
        self.assertEqual(measure(handler)[0], 1)
        self.assertEqual(handler.calls, [])
        handler = self.object_in_the_gripper()
        handler.failing_reads = 3
        self.assertEqual(measure(handler)[0], 1)
        self.assertEqual(handler.calls, [])

    def test_bad_arguments_are_refused_before_anything_is_written(self):
        for kwargs in ({'leads': [0, 5]}, {'leads': [-3]}, {'leads': []}, {'current': 0}, {'current': 151},
                       {'step': 0}, {'max_travel': 30}):
            handler = self.object_in_the_gripper()
            self.assertEqual(measure(handler, **kwargs)[0], 1, kwargs)
            self.assertEqual(handler.calls, [], kwargs)

    def test_a_gripper_without_a_goal_current_in_the_yaml_is_refused(self):
        handler = self.object_in_the_gripper()
        handler.servo.goal_current = None
        code, out = measure(handler)
        self.assertEqual(code, 1)
        self.assertIn('no goal_current', out)
        self.assertEqual(handler.calls, [])

    def test_torque_that_cannot_be_confirmed_off_returns_1(self):
        handler = self.object_in_the_gripper()
        handler.read_torque_enabled = lambda ids=None: {GRIPPER: ('torque', True) in handler.calls}
        code, out = measure(handler)
        self.assertEqual(code, 1)
        self.assertIn('COULD NOT CONFIRM', out)

    def test_question_before_moving_is_asked_without_yes_and_nothing_happens_on_no(self):
        handler = self.object_in_the_gripper()
        questions = []
        code, out = quietly(gripper_test.measure_lead, handler, yes=False, ask=lambda q: questions.append(q) or 'no',
                            sleep=handler.sleep, clock=handler.clock)
        self.assertEqual(code, 0)
        self.assertEqual(len(questions), 1)
        self.assertIn('type yes to continue', questions[0])
        self.assertEqual(handler.calls, [])

    def test_typed_yes_starts_it(self):
        handler = self.object_in_the_gripper()
        code, out = quietly(gripper_test.measure_lead, handler, yes=False, ask=lambda q: 'yes', sleep=handler.sleep,
                            clock=handler.clock)
        self.assertEqual(code, 0)
        self.assertIn(('torque', True), handler.calls)


class LeadArithmeticTests(unittest.TestCase):

    def test_saturation_distance_is_the_smallest_distance_at_95_percent(self):
        rows = [make_row(10, 10, 30), make_row(40, 38, 120), make_row(50, 49, 143), make_row(60, 59, 150)]
        self.assertEqual(gripper_test.saturation_distance(rows), 49)

    def test_saturation_distance_with_one_row(self):
        self.assertEqual(gripper_test.saturation_distance([make_row(50, 48, 150)]), 48)
        self.assertIsNone(gripper_test.saturation_distance([make_row(10, 10, 30)]))

    def test_saturation_distance_with_all_rows_saturated_is_the_smallest_distance(self):
        rows = [make_row(5, 5, 150), make_row(10, 10, 150), make_row(20, 20, 150)]
        self.assertEqual(gripper_test.saturation_distance(rows), 5)

    def test_saturation_distance_with_no_row_saturated_is_none(self):
        rows = [make_row(5, 5, 15), make_row(10, 10, 30), make_row(20, 20, 60)]
        self.assertIsNone(gripper_test.saturation_distance(rows))
        self.assertIsNone(gripper_test.saturation_distance([]))

    def test_95_percent_counts_and_just_below_does_not(self):
        self.assertEqual(gripper_test.saturation_distance([make_row(10, 10, 142.5)]), 10)
        self.assertIsNone(gripper_test.saturation_distance([make_row(10, 10, 142.4)]))

    def test_distance_not_the_lead_is_what_counts(self):
        rows = [make_row(60, 41, 150), make_row(80, 62, 150)]  # the object gave way
        self.assertEqual(gripper_test.saturation_distance(rows), 41)

    def test_current_per_position_of_the_rows_below_saturation(self):
        rows = [make_row(10, 10, 30), make_row(20, 20, 60), make_row(40, 40, 120), make_row(50, 50, 150),
                make_row(80, 80, 150)]
        self.assertAlmostEqual(gripper_test.current_per_position(rows), 3.0)

    def test_current_per_position_is_a_least_squares_line_through_the_origin(self):
        rows = [make_row(10, 10, 35), make_row(20, 20, 55)]
        self.assertAlmostEqual(gripper_test.current_per_position(rows), (10 * 35 + 20 * 55) / (100 + 400))

    def test_current_per_position_needs_two_rows_below_saturation(self):
        self.assertIsNone(gripper_test.current_per_position([make_row(10, 10, 30)]))  # one row
        self.assertIsNone(gripper_test.current_per_position([make_row(10, 10, 30), make_row(50, 50, 150)]))
        self.assertIsNone(gripper_test.current_per_position([make_row(50, 50, 150), make_row(80, 80, 150)]))  # none
        self.assertIsNone(gripper_test.current_per_position([]))

    def test_90_percent_is_already_saturation(self):
        rows = [make_row(10, 10, 30), make_row(20, 20, 60), make_row(45, 45, 135)]  # the last one is at 90%
        self.assertAlmostEqual(gripper_test.current_per_position(rows), 3.0)
        rows = [make_row(10, 10, 30), make_row(45, 45, 135)]
        self.assertIsNone(gripper_test.current_per_position(rows))

    def test_rows_without_a_distance_are_not_on_the_slope(self):
        rows = [make_row(2, 0, 5), make_row(5, -1, 4), make_row(10, 10, 30), make_row(20, 20, 60)]
        self.assertAlmostEqual(gripper_test.current_per_position(rows), 3.0)

    def test_recommended_lead_is_the_distance_times_1_25_rounded_up_to_5_positions(self):
        self.assertEqual(gripper_test.recommended_lead(50), 65)  # 62.5
        self.assertEqual(gripper_test.recommended_lead(40), 50)  # 50.0 exactly
        self.assertEqual(gripper_test.recommended_lead(48), 60)
        self.assertEqual(gripper_test.recommended_lead(49.3), 65)  # 61.6
        self.assertEqual(gripper_test.recommended_lead(100), 125)
        self.assertEqual(gripper_test.recommended_lead(2), 5)  # 2.5
        self.assertEqual(gripper_test.recommended_lead(0), 5)  # never zero

    def test_states_per_position_follows_the_positions_of_the_yaml(self):
        self.assertAlmostEqual(gripper_test.states_per_position(make_servo((1024, 2048))), 100 / 1024)
        self.assertAlmostEqual(gripper_test.states_per_position(make_servo((1086, 2037))), 100 / 951)
        self.assertAlmostEqual(gripper_test.states_per_position(make_servo((2048, 1024))), 100 / 1024)
        self.assertAlmostEqual(65 * gripper_test.states_per_position(make_servo((1086, 2037))), 6.83, places=2)

    def test_report_with_a_whole_squeeze(self):
        rows = [make_row(10, 10, 30), make_row(40, 40, 120), make_row(50, 50, 150)]
        report = gripper_test.lead_report(rows, 150, 100 / 951)
        self.assertIn('from 50.0 positions', report)
        self.assertIn('grows by 3.00 mA per position', report)
        self.assertIn('lead for the gamepad: 65 positions (6.8 states', report)
        self.assertEqual(len(gripper_test.lead_table(rows).splitlines()), 4)

    def test_report_with_all_rows_saturated_has_no_slope(self):
        rows = [make_row(10, 10, 150), make_row(20, 20, 150)]
        report = gripper_test.lead_report(rows, 150, 100 / 951)
        self.assertIn('from 10.0 positions', report)
        self.assertIn('no mA per position', report)
        self.assertIn('lead for the gamepad: 15 positions', report)  # 12.5

    def test_report_with_no_row_saturated_has_no_recommendation(self):
        rows = [make_row(10, 10, 30), make_row(20, 20, 60)]
        report = gripper_test.lead_report(rows, 150, 100 / 951)
        self.assertIn('did not reach 95%', report)
        self.assertIn('(largest distance 20.0 positions)', report)
        self.assertNotIn('lead for the gamepad', report)
        self.assertIn('grows by 3.00 mA per position', report)

    def test_report_with_one_row(self):
        report = gripper_test.lead_report([make_row(10, 10, 30)], 150, 100 / 951)
        self.assertIn('no mA per position', report)
        self.assertNotIn('lead for the gamepad', report)

    def test_check_lead_arguments(self):
        servo = make_servo()
        self.assertIsNone(gripper_test.check_lead_arguments(servo, [1, 50]))
        self.assertIsNotNone(gripper_test.check_lead_arguments(servo, []))
        self.assertIsNotNone(gripper_test.check_lead_arguments(servo, [5, 0]))
        self.assertIn('goal_current', gripper_test.check_lead_arguments(make_servo(goal_current=None), [5]))


class MainTests(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = os.path.join(self.directory.name, 'servo_config.yaml')
        with open(self.path, 'wb') as config_file:
            config_file.write('\r\n'.join(YAML_LINES + ['']).encode())
        self.port_handler = mock.Mock()

    def run_main(self, handler, argv):
        """main with the port and the handler faked, and no real waiting"""
        with mock.patch.object(gripper_test, 'setup_dynamixel_handlers',
                               return_value=(self.port_handler, mock.Mock())) as setup, \
                mock.patch.object(gripper_test, 'ServoHandler', return_value=handler), \
                mock.patch.object(gripper_test, 'sleep', handler.sleep), \
                mock.patch.object(gripper_test, 'perf_counter', handler.clock), \
                mock.patch('builtins.input', side_effect=AssertionError('asked')):
            result, out = quietly(gripper_test.main, argv + ['--config', self.path])
        return result, out, setup

    def test_calibrate_runs_and_closes_the_port(self):
        handler = FakeHandler()
        code, out, setup = self.run_main(handler, ['--calibrate', '--yes', '--current', '40', '--step', '20'])
        self.assertEqual(code, 0)
        self.assertEqual(handler.goal_current_at_torque_on, 40)
        self.assertEqual(handler.goals()[0], 1106)
        self.assertIn('positions: [1086, 19', out)
        self.port_handler.closePort.assert_called_once_with()
        self.assertFalse(handler.torque)

    def test_calibrate_closing_and_write_options(self):
        handler = FakeHandler(open_position=1086, closed_end=900, open_end=1100)
        code, out, setup = self.run_main(handler, ['--calibrate', '--yes', '--closing', 'down', '--write'])
        self.assertEqual(code, 0)
        self.assertEqual(load_servo_config(self.path).servos[GRIPPER].min_position, 1086)

    def test_current_above_the_yaml_is_refused_before_the_port_is_opened(self):
        handler = FakeHandler()
        code, out, setup = self.run_main(handler, ['--calibrate', '--yes', '--current', '200'])
        self.assertEqual(code, 1)
        setup.assert_not_called()
        self.assertEqual(handler.calls, [])

    def test_calibration_options_need_calibrate(self):
        for option in (['--write'], ['--current', '40'], ['--step', '5'], ['--max-travel', '100'],
                       ['--closing', 'up']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                gripper_test.main(option + ['--config', self.path])

    def test_measure_lead_runs_and_closes_the_port(self):
        handler = FakeHandler(closed_end=1500, stiffness=3.0)
        code, out, setup = self.run_main(handler, ['--measure-lead', '--yes', '--current', '40', '--step', '20',
                                                   '--leads', '10', '50', '30'])
        self.assertEqual(code, 0)
        self.assertEqual(handler.goal_current_at_torque_on, 40)
        self.assertEqual(handler.goals()[0], 1106)  # the step option
        self.assertEqual(goals_after_yaml_current(handler)[:3], [1510, 1530, 1550])
        self.assertIn('from 50.0 positions', out)
        self.assertIn('lead for the gamepad: 65 positions', out)
        self.port_handler.closePort.assert_called_once_with()
        self.assertFalse(handler.torque)

    def test_measure_lead_without_leads_uses_the_default_ones(self):
        handler = FakeHandler(closed_end=1500, stiffness=3.0)
        code, out, setup = self.run_main(handler, ['--measure-lead', '--yes'])
        self.assertEqual(code, 0)
        self.assertEqual(goals_after_yaml_current(handler)[:11], [1500 + lead for lead in gripper_test.DEFAULT_LEADS])

    def test_measure_lead_closing_option(self):
        handler = FakeHandler(open_position=1086, closed_end=900, open_end=1100, stiffness=3.0)
        code, out, setup = self.run_main(handler, ['--measure-lead', '--yes', '--closing', 'down'])
        self.assertEqual(code, 0)
        self.assertIn('the object stops the gripper at position 900', out)

    def test_measure_lead_returns_1_when_nothing_was_measured(self):
        handler = FakeHandler(closed_end=1100, stiffness=3.0)
        code, out, setup = self.run_main(handler, ['--measure-lead', '--yes'])
        self.assertEqual(code, 1)
        self.port_handler.closePort.assert_called_once_with()

    def test_bad_measure_lead_arguments_are_refused_before_the_port_is_opened(self):
        for options in (['--leads', '0'], ['--leads', '10', '-5'], ['--current', '200'], ['--step', '0']):
            handler = FakeHandler(closed_end=1500, stiffness=3.0)
            code, out, setup = self.run_main(handler, ['--measure-lead', '--yes'] + options)
            self.assertEqual(code, 1, options)
            setup.assert_not_called()
            self.assertEqual(handler.calls, [], options)

    def test_leads_need_measure_lead(self):
        for options in (['--leads', '10'], ['--calibrate', '--leads', '10']):
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
                gripper_test.main(options + ['--config', self.path])
            self.assertIn('--leads only works with --measure-lead', stderr.getvalue())

    def test_measure_lead_and_calibrate_exclude_each_other(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            gripper_test.main(['--measure-lead', '--calibrate', '--config', self.path])
        self.assertIn('can not be used together', stderr.getvalue())

    def test_write_does_not_apply_to_measure_lead(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            gripper_test.main(['--measure-lead', '--write', '--config', self.path])
        self.assertIn('--write only works with --calibrate', stderr.getvalue())

    def test_leads_must_be_whole_numbers(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            gripper_test.main(['--measure-lead', '--leads', '2.5', '--config', self.path])

    def test_shared_options_need_a_mode(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            gripper_test.main(['--step', '5', '--config', self.path])
        self.assertIn('--calibrate or --measure-lead', stderr.getvalue())

    def test_interactive_tests_refuse_an_uncalibrated_gripper(self):
        handler = FakeHandler(open_position=3000, open_end=2900, closed_end=3100)
        code, out, setup = self.run_main(handler, [])
        self.assertEqual(code, 1)
        self.assertIn('--calibrate', out)
        self.assertEqual(handler.calls, [])  # nothing was enabled, and no question was asked
        self.port_handler.closePort.assert_called_once_with()

    def test_interactive_tests_start_with_a_gripper_inside_the_range(self):
        handler = FakeHandler(open_position=1086)
        answers = iter(['yes', '', '', 'n'])  # start, remove the object, put an object, no squeeze test
        with mock.patch.object(gripper_test, 'setup_dynamixel_handlers',
                               return_value=(self.port_handler, mock.Mock())), \
                mock.patch.object(gripper_test, 'ServoHandler', return_value=handler), \
                mock.patch.object(gripper_test, 'sleep', handler.sleep), \
                mock.patch.object(gripper_test, 'perf_counter', handler.clock), \
                mock.patch('builtins.input', side_effect=lambda prompt='': next(answers, '')):
            code, out = quietly(gripper_test.main, ['--config', self.path])
        self.assertEqual(code, 0)
        self.assertIn(('torque', True), handler.calls)
        self.assertIn(('configured_goal_current',), handler.calls)  # the interactive tests use the yaml current
        self.assertFalse(handler.torque)
        self.port_handler.closePort.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
