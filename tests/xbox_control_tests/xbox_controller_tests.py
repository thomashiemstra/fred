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

    def run_controller(self, results, dead_zone=0, scale=1, invert_yaxis=False):
        gamepad = FakeGamepad(results)
        with mock.patch.object(xbox_controller_module, 'get_gamepad', gamepad.get_gamepad), \
                mock.patch.object(xbox_controller_module, 'sleep'), \
                mock.patch.object(xbox_controller_module, '_find_gamepads') as find_gamepads:
            controller = XboxController(dead_zone=dead_zone, scale=scale, invert_yaxis=invert_yaxis)
            self.assertTrue(gamepad.done.wait(timeout=5))
        return controller, find_gamepads

    def test_sticks(self):
        controller, _ = self.run_controller([[Event('ABS_X', 2 ** 14), Event('ABS_Y', -2 ** 14),
                                              Event('ABS_RX', -2 ** 15), Event('ABS_RY', 2 ** 13)]])
        self.assertEqual((0.5, -0.5), controller.get_left_thumb())
        self.assertEqual((-1, 0.25), controller.get_right_thumb())

    def test_dead_zone_scale_and_inverted_y_axis(self):
        controller, _ = self.run_controller([[Event('ABS_X', 0.2 * 2 ** 15), Event('ABS_Y', 0.5 * 2 ** 15),
                                              Event('ABS_RX', -0.29 * 2 ** 15), Event('ABS_RY', -0.75 * 2 ** 15)]],
                                            dead_zone=30, scale=100, invert_yaxis=True)
        self.assertEqual((0, -50), controller.get_left_thumb())
        self.assertEqual((0, 75), controller.get_right_thumb())

    def test_triggers_move_up_and_down(self):
        controller, _ = self.run_controller([[Event('ABS_Z', 128)]])
        self.assertEqual(-0.5, controller.get_lr_trigger())

        controller, _ = self.run_controller([[Event('ABS_RZ', 64)]])
        self.assertEqual(0.25, controller.get_lr_trigger())

    def test_buttons_are_reset_after_reading_them_but_bumpers_are_held(self):
        controller, _ = self.run_controller([[Event('BTN_SOUTH', 1, 'Key'), Event('BTN_NORTH', 1, 'Key'),
                                              Event('BTN_WEST', 1, 'Key'), Event('BTN_EAST', 1, 'Key'),
                                              Event('BTN_START', 1, 'Key'), Event('BTN_TL', 1, 'Key'),
                                              Event('BTN_TR', 1, 'Key'), Event('ABS_HAT0Y', -1),
                                              Event('ABS_HAT0X', 1)]])
        buttons = controller.get_buttons()
        for name in ('a', 'y', 'x', 'b', 'start', 'lb', 'rb', 'pad_up', 'pad_right'):
            self.assertTrue(getattr(buttons, name), name)
        self.assertFalse(buttons.pad_down)
        self.assertFalse(buttons.pad_left)

        # a press is handled once, while the bumpers keep moving the gripper as long as they are held
        buttons = controller.get_buttons()
        for name in ('a', 'y', 'x', 'b', 'start', 'pad_up', 'pad_right'):
            self.assertFalse(getattr(buttons, name), name)
        self.assertTrue(buttons.lb)
        self.assertTrue(buttons.rb)

    def test_released_buttons_and_d_pad(self):
        controller, _ = self.run_controller([[Event('BTN_SOUTH', 1, 'Key'), Event('BTN_TL', 1, 'Key'),
                                              Event('ABS_HAT0Y', 1), Event('ABS_HAT0X', -1)],
                                             [Event('BTN_SOUTH', 0, 'Key'), Event('BTN_TL', 0, 'Key'),
                                              Event('ABS_HAT0Y', 0), Event('ABS_HAT0X', 0)]])
        buttons = controller.get_buttons()
        for name in ('a', 'lb', 'pad_up', 'pad_down', 'pad_left', 'pad_right'):
            self.assertFalse(getattr(buttons, name), name)

    def test_d_pad_down_and_left(self):
        controller, _ = self.run_controller([[Event('ABS_HAT0Y', 1), Event('ABS_HAT0X', -1)]])
        buttons = controller.get_buttons()
        self.assertTrue(buttons.pad_down)
        self.assertTrue(buttons.pad_left)
        self.assertFalse(buttons.pad_up)
        self.assertFalse(buttons.pad_right)

    def test_sync_and_unknown_events_are_ignored(self):
        controller, _ = self.run_controller([[Event('ABS_X', 2 ** 14, ev_type='Sync'), Event('BTN_SELECT', 1, 'Key'),
                                              Event('BTN_THUMBL', 1, 'Key')]])
        self.assertTrue(controller.is_connected())
        self.assertEqual((0, 0), controller.get_left_thumb())
        buttons = controller.get_buttons()
        self.assertFalse(any(vars(buttons).values()))

    def test_wait_until_connected_with_a_gamepad(self):
        controller, _ = self.run_controller([[]])
        self.assertTrue(controller.wait_until_connected(timeout=0))

    def test_buttons_are_reset_when_the_gamepad_is_unplugged(self):
        controller, _ = self.run_controller([[Event('BTN_SOUTH', 1, 'Key'), Event('BTN_START', 1, 'Key'),
                                              Event('ABS_HAT0Y', -1), Event('ABS_RX', 2 ** 14)],
                                             UnpluggedError()])
        buttons = controller.get_buttons()
        self.assertFalse(any(vars(buttons).values()))
        self.assertEqual((0, 0), controller.get_right_thumb())

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
