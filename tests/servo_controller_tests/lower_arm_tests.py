import contextlib
import io
import unittest
from unittest import mock

import numpy as np

from src.global_constants import dynamixel_robot_config
from src.kinematics.kinematics import inverse_kinematics
from src.robot_controllers.dynamixel_robot.calibration import lower_arm, record_static_load
from src.robot_controllers.dynamixel_robot.calibration.tune_servo_gains import parse_pose
from src.robot_controllers.dynamixel_robot.dynamixel_utils import DynamixelError
from tests.servo_controller_tests.static_load_scripts_tests import FakeHandler, RestRobot


class TorqueHandler(FakeHandler):
    """the servo handler of the fake robot with the torque of the 7 servos, which the robot switches off"""

    def __init__(self, servo_config, torque):
        super().__init__(servo_config)
        self.torque = dict(torque)
        self.torque_failures = 0  # the next reads of the torque fail (return None), counting down
        self.torque_reads = 0

    def read_torque_enabled(self, servo_ids=None):
        self.torque_reads += 1
        if self.torque_failures:
            self.torque_failures -= 1
            return None
        return {servo_id: on for servo_id, on in self.torque.items() if servo_ids is None or servo_id in servo_ids}


class ArmRobot(RestRobot):
    """the rest robot with a torque state and the reads of the status"""

    def __init__(self, torque=True, angles=None, **kwargs):
        super().__init__(parse_pose('elbow_out') if angles is None else angles, **kwargs)
        on = {servo_id: torque for servo_id in range(1, 8)} if isinstance(torque, bool) else torque
        self.servo_handler = TorqueHandler(self.servo_config, on)

    def disable_servos(self):
        super().disable_servos()
        self.servo_handler.torque = {servo_id: False for servo_id in self.servo_handler.torque}

    def get_servo_temperatures(self):
        return dict(self.temperatures)


class LowerArmTestCase(unittest.TestCase):

    def setUp(self):
        patcher = mock.patch('src.utils.movement.sleep')  # the planned moves run at their real speed otherwise
        patcher.start()
        self.addCleanup(patcher.stop)
        self.pauses = []

    def sleep(self, seconds):
        self.pauses.append(seconds)

    def run_lower_arm(self, robot, attempts=3, answer=''):
        """lower_arm.lower_arm with the input answered by answer (or raising it); returns (result, output, input)"""
        prompt = mock.Mock(side_effect=answer) if isinstance(answer, BaseException) else mock.Mock(return_value=answer)
        output = io.StringIO()
        with mock.patch('builtins.input', prompt), contextlib.redirect_stdout(output):
            result = lower_arm.lower_arm(robot, attempts, self.sleep)
        return result, output.getvalue(), prompt

    def run_main(self, robot, *argv, starts=None):
        """
        lower_arm.main with get_robot giving the robot (or the sequence in starts, exceptions are raised), the fake
        counting as a CombinedRobot; returns (result, output, input mock, get_robot mock)
        """
        prompt = mock.Mock(return_value='')
        output = io.StringIO()
        start = mock.Mock(side_effect=starts) if starts is not None else mock.Mock(return_value=robot)
        with mock.patch('builtins.input', prompt), contextlib.redirect_stdout(output), \
                mock.patch.object(lower_arm, 'get_robot', start), \
                mock.patch.object(lower_arm, 'CombinedRobot', RestRobot):
            result = lower_arm.main(list(argv), self.sleep)
        return result, output.getvalue(), prompt, start


class LowerArmTests(LowerArmTestCase):

    def test_the_arm_is_lowered_and_the_torque_goes_off(self):
        robot = ArmRobot()
        result, output, prompt = self.run_lower_arm(robot)
        self.assertEqual(0, result)
        prompt.assert_not_called()
        self.assertTrue(robot.disabled)
        np.testing.assert_allclose(inverse_kinematics(record_static_load.REST_POSE, dynamixel_robot_config),
                                   robot.moves[-1], atol=1e-6)
        self.assertGreater(len(robot.moves), 1)
        self.assertEqual(['disable'], robot.events[-1:])
        self.assertIn('the arm is lowered and the torque is off', output)
        self.assertIn('the torque is on for servos [1, 2, 3, 4, 5, 6]', output)

    def test_a_torque_that_is_off_already_moves_and_writes_nothing(self):
        robot = ArmRobot(torque=False)
        result, output, prompt = self.run_lower_arm(robot)
        self.assertEqual(0, result)
        self.assertEqual([], robot.events, 'no move, no torque write')
        prompt.assert_not_called()
        self.assertIn('the torque of the arm servos is off already, nothing is moved', output)
        self.assertEqual([], self.pauses)

    def test_only_the_arm_servos_decide_whether_the_torque_is_off(self):
        torque = {servo_id: False for servo_id in range(1, 8)}
        robot = ArmRobot(torque={**torque, 7: True})  # the gripper holds, the arm lies on the table
        result, output, _ = self.run_lower_arm(robot)
        self.assertEqual(0, result)
        self.assertEqual([], robot.events)
        self.assertIn('off already', output)
        # one arm servo that is on is enough to lower the arm
        robot = ArmRobot(torque={**torque, 4: True})
        result, output, _ = self.run_lower_arm(robot)
        self.assertEqual(0, result)
        self.assertTrue(robot.disabled)
        self.assertIn('the torque is on for servos [4]', output)

    def test_the_torque_is_read_for_the_arm_servos(self):
        robot = ArmRobot(torque=False)
        with mock.patch.object(robot.servo_handler, 'read_torque_enabled',
                               wraps=robot.servo_handler.read_torque_enabled) as read:
            lower_arm.torque_of_arm(robot)
        read.assert_called_once_with((1, 2, 3, 4, 5, 6))

    def test_a_torque_that_can_not_be_read_is_said_and_the_arm_is_lowered_anyway(self):
        robot = ArmRobot()
        robot.servo_handler.torque_failures = 10 ** 6
        result, output, prompt = self.run_lower_arm(robot)
        self.assertEqual(0, result)
        self.assertIn('could not be read, lowering the arm anyway', output)
        self.assertTrue(robot.disabled)
        prompt.assert_not_called()

    def test_a_torque_read_that_fails_now_and_then_is_tried_again(self):
        robot = ArmRobot(torque=False)
        robot.servo_handler.torque_failures = lower_arm.STATUS_READS - 1
        result, output, _ = self.run_lower_arm(robot)
        self.assertEqual(0, result)
        self.assertIn('off already', output)
        self.assertEqual([], robot.events)

    def test_a_failed_lowering_is_tried_again_without_asking(self):
        robot = ArmRobot()
        failing = {'count': 2}  # the first two attempts find a hardware error, then it is gone
        original = robot.get_hardware_errors

        def flaky():
            if failing['count']:
                failing['count'] -= 1
                return {3: ['overload']}
            return original()
        robot.get_hardware_errors = flaky
        result, output, prompt = self.run_lower_arm(robot, attempts=5)
        self.assertEqual(0, result)
        prompt.assert_not_called()
        self.assertTrue(robot.disabled)
        self.assertEqual(2, output.count('did not go through'))
        self.assertIn('lowering attempt 1 of 5 did not go through: servos [3] have a hardware error', output)
        self.assertIn('lowering attempt 2 of 5 did not go through', output)
        self.assertNotIn('lowering attempt 3', output)
        self.assertEqual([lower_arm.PAUSE, lower_arm.PAUSE], [p for p in self.pauses if p == lower_arm.PAUSE],
                        'a pause after every failed attempt')

    def test_every_attempt_failing_gives_up_with_one_question(self):
        robot = ArmRobot()
        robot.hardware_errors = {3: ['overload']}
        result, output, prompt = self.run_lower_arm(robot, attempts=4)
        self.assertEqual(1, result)
        self.assertEqual(1, prompt.call_count, 'the prompt of disable_torque_on_request, once')
        self.assertEqual(4, output.count('did not go through'))
        self.assertIn('lowering attempt 4 of 4 did not go through', output)
        self.assertIn('the arm is NOT lowered after 4 attempts, the servos still hold', output)
        self.assertEqual([lower_arm.PAUSE] * 3, self.pauses, 'no pause after the last attempt')
        self.assertEqual([], robot.moves)
        # the question was answered with enter: the torque goes off, as the user supports the arm
        self.assertTrue(robot.disabled)

    def test_giving_up_without_input_leaves_the_servos_holding(self):
        robot = ArmRobot()
        robot.hardware_errors = {3: ['overload']}
        result, output, prompt = self.run_lower_arm(robot, attempts=2, answer=EOFError())
        self.assertEqual(1, result)
        self.assertEqual(1, prompt.call_count)
        self.assertFalse(robot.disabled)
        self.assertIn('the servos still hold', output)
        self.assertIn('no input to answer the question', output)
        self.assertIn('torque: {1: True', output, 'the status shows the torque is still on')

    def test_ctrl_c_at_the_question_leaves_the_torque_on(self):
        robot = ArmRobot()
        robot.hardware_errors = {3: ['overload']}
        result, output, _ = self.run_lower_arm(robot, attempts=1, answer=KeyboardInterrupt())
        self.assertEqual(1, result)
        self.assertFalse(robot.disabled)

    def test_the_status_is_printed_after_a_lowering_and_after_giving_up_and_when_the_torque_was_off(self):
        for name, robot in (('lowered', ArmRobot()), ('off', ArmRobot(torque=False))):
            with self.subTest(case=name):
                robot.temperatures[2] = 41
                _, output, _ = self.run_lower_arm(robot)
                self.assertIn('torque: {1: False', output)
                self.assertIn('hardware errors: {}', output)
                self.assertIn('temperatures: {1: 30, 2: 41', output)
        robot = ArmRobot()
        robot.hardware_errors = {3: ['overload']}
        _, output, _ = self.run_lower_arm(robot, attempts=1)
        self.assertIn("hardware errors: {3: ['overload']}", output)
        self.assertEqual(1, output.count('temperatures:'))

    def test_the_status_reads_are_tried_again_and_a_failing_one_prints_none(self):
        robot = ArmRobot(torque=False)
        calls = []

        def flaky():
            calls.append(1)
            return None if len(calls) < 4 else {1: 30}
        robot.get_servo_temperatures = flaky
        robot.get_hardware_errors = lambda: None
        _, output, _ = self.run_lower_arm(robot)
        self.assertIn('temperatures: {1: 30}', output)
        self.assertEqual(4, len(calls))
        self.assertIn('hardware errors: None', output)

    def test_read_with_retries(self):
        results = [None, None, 5]
        self.assertEqual(5, lower_arm.read_with_retries(lambda: results.pop(0), 3))
        self.assertIsNone(lower_arm.read_with_retries(lambda: None, 4))
        calls = []
        lower_arm.read_with_retries(lambda: calls.append(1), 4)  # None every time
        self.assertEqual(4, len(calls))
        self.assertEqual({}, lower_arm.read_with_retries(lambda: {}, 4), 'an empty result is a result')


class MainTests(LowerArmTestCase):

    def test_the_robot_is_started_and_lowered(self):
        robot = ArmRobot()
        result, output, prompt, start = self.run_main(robot)
        self.assertEqual(0, result)
        start.assert_called_once_with()
        self.assertTrue(robot.disabled)
        prompt.assert_not_called()
        self.assertEqual([], [p for p in self.pauses if p == lower_arm.PAUSE])

    def test_a_failed_start_is_tried_again(self):
        robot = ArmRobot()
        result, output, _, start = self.run_main(
            robot, starts=[DynamixelError('failed to read the initial position'), DynamixelError('no answer'), robot])
        self.assertEqual(0, result)
        self.assertEqual(3, start.call_count)
        self.assertIn('start attempt 1 of 15: failed to read the initial position', output)
        self.assertIn('start attempt 2 of 15: no answer', output)
        self.assertEqual([lower_arm.PAUSE] * 2, [p for p in self.pauses if p == lower_arm.PAUSE])
        self.assertTrue(robot.disabled)

    def test_a_robot_that_never_starts_gives_return_code_1_and_the_servos_still_hold(self):
        result, output, prompt, start = self.run_main(None, '--attempts', '4', starts=DynamixelError('no answer'))
        self.assertEqual(1, result)
        self.assertEqual(4, start.call_count)
        self.assertIn('could not start the robot after 4 attempts, the servos still hold', output)
        self.assertEqual([lower_arm.PAUSE] * 3, self.pauses, 'no pause after the last attempt')
        prompt.assert_not_called()
        self.assertNotIn('torque:', output, 'there is no robot to read')

    def test_the_simulated_robot_is_refused(self):
        simulated = mock.Mock(spec=['move_servos'])  # not a CombinedRobot
        result, output, prompt, _ = self.run_main(simulated)
        self.assertEqual(1, result)
        self.assertIn('only be lowered on the real robot', output)
        simulated.move_servos.assert_not_called()
        prompt.assert_not_called()

    def test_something_that_is_not_a_combined_robot_is_refused_without_the_patch_of_the_tests(self):
        output = io.StringIO()
        with mock.patch.object(lower_arm, 'get_robot', return_value=mock.Mock()), contextlib.redirect_stdout(output):
            self.assertEqual(1, lower_arm.main([], self.sleep))
        self.assertIn('only be lowered on the real robot', output.getvalue())

    def test_the_attempts_option_limits_the_lowering(self):
        robot = ArmRobot()
        robot.hardware_errors = {3: ['overload']}
        result, output, prompt, _ = self.run_main(robot, '--attempts', '2')
        self.assertEqual(1, result)
        self.assertEqual(2, output.count('did not go through'))
        self.assertEqual(1, prompt.call_count)

    def test_attempts_is_a_whole_number_of_at_least_one(self):
        for value in ('0', '-1', 'x', '1.5'):
            with self.subTest(value=value), self.assertRaises(SystemExit), mock.patch('sys.stderr'):
                lower_arm.main(['--attempts', value], self.sleep)

    def test_the_default_number_of_attempts_is_15(self):
        self.assertEqual(15, lower_arm.DEFAULT_ATTEMPTS)
        robot = ArmRobot()
        robot.hardware_errors = {3: ['overload']}
        _, output, _, _ = self.run_main(robot)
        self.assertEqual(15, output.count('did not go through'))


if __name__ == '__main__':
    unittest.main()
