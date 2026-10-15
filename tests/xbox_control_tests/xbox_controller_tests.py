import threading
import unittest
from unittest import mock

from inputs import UnpluggedError

from src.xbox_control.xbox360controller import XboxController as xbox_controller_module
from src.xbox_control.xbox360controller.XboxController import XboxController


class Event:

    def __init__(self, code, state, ev_type='Absolute'):
        self.code = code
        self.state = state
        self.ev_type = ev_type


class FakeGamepad:
    """returns the queued results of get_gamepad one at a time, an exception is raised instead of returned"""

    def __init__(self, results):
        self.results = list(results)
        self.done = threading.Event()

    def get_gamepad(self):
        if not self.results:
            self.done.set()
            threading.Event().wait()  # block the monitor thread forever, like waiting for the next event
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class XboxControllerTests(unittest.TestCase):

    def run_controller(self, results):
        gamepad = FakeGamepad(results)
        with mock.patch.object(xbox_controller_module, 'get_gamepad', gamepad.get_gamepad), \
                mock.patch.object(xbox_controller_module, 'sleep'), \
                mock.patch.object(xbox_controller_module, '_find_gamepads') as find_gamepads:
            controller = XboxController(dead_zone=0, scale=1)
            self.assertTrue(gamepad.done.wait(timeout=5))
        return controller, find_gamepads

    def test_missing_gamepad_does_not_stop_the_monitor_thread(self):
        controller, find_gamepads = self.run_controller([UnpluggedError(), UnpluggedError(),
                                                         [Event('ABS_X', 2 ** 14)]])
        self.assertTrue(controller.is_connected())
        self.assertEqual(2, find_gamepads.call_count)
        self.assertAlmostEqual(0.5, controller.get_left_thumb()[0])

    def test_unplugging_the_gamepad_stops_the_robot(self):
        controller, _ = self.run_controller([[Event('ABS_X', 2 ** 14), Event('ABS_RZ', 100), Event('BTN_TR', 1)],
                                             UnpluggedError()])
        self.assertFalse(controller.is_connected())
        self.assertEqual((0, 0), controller.get_left_thumb())
        self.assertEqual(0, controller.get_lr_trigger())
        self.assertFalse(controller.get_buttons().rb)

    def test_wait_until_connected(self):
        controller, _ = self.run_controller([UnpluggedError()])
        with mock.patch.object(xbox_controller_module, 'sleep'):
            self.assertFalse(controller.wait_until_connected(timeout=1.0))


if __name__ == '__main__':
    unittest.main()
