import threading
import unittest
from unittest import mock

from src.xbox_control.xbox360controller import Xbox360Controller as xbox360_controller_module
from src.xbox_control.xboxonecontroller import XboxOneController as xbox_one_controller_module
from src.xbox_control.xboxonecontroller.XboxOneController import XboxOneController, EV_KEY, EV_ABS
from tests.xbox_control_tests.xbox360_controller_tests import FakeGamepad, Event as GamepadEvent

EV_SYN = 0
EV_MSC = 4


class Event:

    def __init__(self, code, value, type=EV_ABS):
        self.code = code
        self.value = value
        self.type = type


class FakeInputDevices:
    """
    _find_controller returns the queued results one at a time: None when the controller isn't connected, otherwise a
    device that yields the events of the list. An exception in the list is raised instead, like when the controller
    is switched off.
    """

    def __init__(self, results):
        self.results = list(results)
        self.done = threading.Event()
        self.searches = 0
        self.closed = 0

    def find_controller(self):
        self.searches += 1
        if not self.results:
            self.wait_forever()
        events = self.results.pop(0)
        return None if events is None else FakeDevice(self, events)

    def wait_forever(self):
        self.done.set()
        threading.Event().wait()  # block the monitor thread forever, like waiting for the next event


class FakeDevice:

    def __init__(self, devices, events):
        self.devices = devices
        self.events = events

    def read_loop(self):
        for event in self.events:
            if isinstance(event, Exception):
                raise event
            yield event
        self.devices.wait_forever()

    def close(self):
        self.devices.closed += 1


SWITCHED_OFF = OSError(19, 'No such device')


class XboxOneControllerTests(unittest.TestCase):

    def run_controller(self, results):
        devices = FakeInputDevices(results)
        with mock.patch.object(xbox360_controller_module, 'is_linux', return_value=True), \
                mock.patch.object(xbox_one_controller_module, 'is_linux', return_value=True), \
                mock.patch.object(xbox_one_controller_module, 'sleep'), \
                mock.patch.object(xbox_one_controller_module, '_find_controller', devices.find_controller):
            controller = XboxOneController(dead_zone=0, scale=1)
            self.assertTrue(devices.done.wait(timeout=5))
        return controller, devices

    def test_sticks_deliver_what_the_wired_controller_does(self):
        # full left is 0, the centre 2 ** 15
        controller, _ = self.run_controller([[Event(0, 0), Event(1, 2 ** 14),
                                              Event(2, 2 ** 15 + 2 ** 14), Event(5, 2 ** 15 + 2 ** 13)]])
        self.assertEqual((-1, -0.5), controller.get_left_thumb())
        # the right stick is ABS_Z and ABS_RZ, with the signs of Xbox360Controller on Linux: up is positive
        self.assertEqual((0.5, -0.25), controller.get_right_thumb())

    def test_triggers_report_up_to_1023(self):
        # ABS_GAS is the right trigger, ABS_BRAKE the left one
        controller, _ = self.run_controller([[Event(9, 256), Event(10, 512)]])
        self.assertEqual(0.25 - 0.5, controller.get_lr_trigger())

    def test_buttons(self):
        controller, _ = self.run_controller([[Event(304, 1, EV_KEY), Event(305, 1, EV_KEY), Event(307, 1, EV_KEY),
                                              Event(308, 1, EV_KEY), Event(310, 1, EV_KEY), Event(311, 1, EV_KEY),
                                              Event(314, 1, EV_KEY), Event(17, -1), Event(16, 1)]])
        buttons = controller.get_buttons()
        for name in ('a', 'b', 'x', 'y', 'lb', 'rb', 'start', 'pad_up', 'pad_right'):
            self.assertTrue(getattr(buttons, name), name)
        self.assertFalse(buttons.pad_down)
        self.assertFalse(buttons.pad_left)

    def test_x_and_y_are_not_mixed_up(self):
        controller, _ = self.run_controller([[Event(307, 1, EV_KEY)]])
        buttons = controller.get_buttons()
        self.assertTrue(buttons.x)
        self.assertFalse(buttons.y)

    def test_other_events_are_ignored(self):
        # a sync event has code 0 and value 0, like the left stick all the way left; the menu and xbox buttons do
        # nothing
        controller, _ = self.run_controller([[Event(0, 0, EV_SYN), Event(4, 589825, EV_MSC), Event(315, 1, EV_KEY),
                                              Event(316, 1, EV_KEY)]])
        self.assertTrue(controller.is_connected())
        self.assertEqual((0, 0), controller.get_left_thumb())
        self.assertFalse(any(vars(controller.get_buttons()).values()))

    def test_switching_the_controller_off_stops_the_robot(self):
        controller, devices = self.run_controller([[Event(0, 2 ** 15 + 2 ** 14), Event(9, 400),
                                                    Event(311, 1, EV_KEY), SWITCHED_OFF]])
        self.assertFalse(controller.is_connected())
        self.assertEqual((0, 0), controller.get_left_thumb())
        self.assertEqual(0, controller.get_lr_trigger())
        self.assertFalse(controller.get_buttons().rb)
        self.assertEqual(1, devices.closed)

    def test_the_controller_is_looked_for_until_it_is_connected(self):
        controller, devices = self.run_controller([None, None, [Event(0, 2 ** 15 + 2 ** 14)]])
        self.assertTrue(controller.is_connected())
        self.assertEqual(3, devices.searches)
        self.assertEqual(0.5, controller.get_left_thumb()[0])

    def test_it_is_found_again_after_it_was_switched_off(self):
        controller, _ = self.run_controller([[SWITCHED_OFF], None, [Event(1, 2 ** 14)]])
        self.assertTrue(controller.is_connected())
        self.assertEqual((0, -0.5), controller.get_left_thumb())

    def test_on_windows_the_inputs_library_reads_it_like_the_wired_controller(self):
        gamepad = FakeGamepad([[GamepadEvent('ABS_X', 2 ** 14)]])
        with mock.patch.object(xbox360_controller_module, 'is_linux', return_value=False), \
                mock.patch.object(xbox_one_controller_module, 'is_linux', return_value=False), \
                mock.patch.object(xbox360_controller_module, 'get_gamepad', gamepad.get_gamepad), \
                mock.patch.object(xbox_one_controller_module, '_find_controller') as find_controller:
            controller = XboxOneController(dead_zone=0, scale=1)
            self.assertTrue(gamepad.done.wait(timeout=5))
        self.assertEqual((0.5, 0), controller.get_left_thumb())
        find_controller.assert_not_called()


if __name__ == '__main__':
    unittest.main()
