import contextlib
import io
import os
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
    """
    SPEED = 1500.0  # positions per second
    WORKING_CURRENT = 25
    IDLE_CURRENT = 5

    def __init__(self, open_position=1086, closed_end=1900, open_end=1040, positions=(1024, 2048), torque=False):
        self.servo = make_servo(positions)
        self.now = 0.0
        self.position = float(open_position)
        self.goal = open_position
        self.lower_end = min(open_end, closed_end)
        self.upper_end = max(open_end, closed_end)
        self.torque = torque
        self.goal_current_register = 910  # what the servo holds until it is written, its maximum
        self.goal_current_at_torque_on = None
        self.ignore_goal_current_writes = False
        self.current = self.IDLE_CURRENT
        self.errors: list[str] = []
        self.temperature = 30
        self.failing_reads = 0  # the next reads of the position that fail
        self.on_read = None  # called with the handler at every read of the position, once
        self.calls: list[tuple] = []
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
        if blocked:
            self.current = self.goal_current_register
        elif abs(move) > 0.5:
            self.current = self.WORKING_CURRENT
        else:
            self.current = self.IDLE_CURRENT

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
