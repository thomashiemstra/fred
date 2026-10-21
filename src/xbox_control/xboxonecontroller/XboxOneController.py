"""
The wireless xbox one controller. It delivers the same inputs as the wired Xbox360Controller and is used the same way.

On Windows the wireless controller is an xinput device like the wired one, so the inputs library reads it and
nothing differs. On Linux the inputs library doesn't find it over Bluetooth (it only looks in /dev/input/by-id and
/dev/input/by-path, where Bluetooth devices aren't listed), so it is read with evdev, which only exists on Linux and
isn't in requirements.txt. The kernel's hid driver gives it other codes and ranges than the xpad driver gives the
wired controller: the right stick is ABS_Z and ABS_RZ, the triggers are ABS_BRAKE and ABS_GAS with 0..1023 and the
sticks report 0..65535. Every input is turned into what the wired controller reports, so the scaling, the dead zone
and the signs are those of Xbox360Controller. The xpadneo driver uses yet other codes, it isn't supported. Without
read access to /dev/input/event* (the input group) the controller isn't found.
"""
from __future__ import annotations

from time import sleep
from typing import TYPE_CHECKING

from src.utils.os_utils import is_linux
from src.xbox_control.xbox360controller.Xbox360Controller import Xbox360Controller, RECONNECT_INTERVAL

if TYPE_CHECKING:
    from evdev import InputDevice

# the name the controller has over Bluetooth
DEVICE_NAME = 'Xbox Wireless Controller'

# the evdev event types of the buttons and of the sticks, the triggers and the d-pad
EV_KEY = 1
EV_ABS = 3

# the evdev codes of the wireless controller, with the name the inputs library gives the same input of the wired one
LINUX_CODES = {
    0: 'ABS_X',  # ABS_X, left stick
    1: 'ABS_Y',  # ABS_Y
    2: 'ABS_RX',  # ABS_Z, right stick
    5: 'ABS_RY',  # ABS_RZ
    10: 'ABS_Z',  # ABS_BRAKE, left trigger
    9: 'ABS_RZ',  # ABS_GAS, right trigger
    16: 'ABS_HAT0X',  # ABS_HAT0X, d-pad
    17: 'ABS_HAT0Y',  # ABS_HAT0Y
    304: 'BTN_SOUTH',  # BTN_A
    305: 'BTN_EAST',  # BTN_B
    307: 'BTN_WEST',  # BTN_X, on Windows the inputs library reports the x button as BTN_WEST
    308: 'BTN_NORTH',  # BTN_Y
    310: 'BTN_TL',  # BTN_TL, left bumper
    311: 'BTN_TR',  # BTN_TR
    # BTN_SELECT, the view button: on Windows the inputs library reports the left one of the two buttons in the
    # middle as BTN_START, so start is the same button on both
    314: 'BTN_START',
}
STICKS = {'ABS_X', 'ABS_Y', 'ABS_RX', 'ABS_RY'}
TRIGGERS = {'ABS_Z', 'ABS_RZ'}
# the sticks report 0..65535 around this, the wired controller -32768..32767 around 0
STICK_CENTER = 2 ** 15
# the triggers report 0..1023, the wired controller 0..255
TRIGGER_RANGE = 2 ** 10


class XboxOneController(Xbox360Controller):

    def _monitor_controller(self) -> None:
        if not is_linux():
            super()._monitor_controller()
            return

        while True:
            device = _find_controller()
            if device is None:
                self._set_connected(False)
                sleep(RECONNECT_INTERVAL)
                continue
            self._set_connected(True)

            try:
                for event in device.read_loop():
                    if event.type in (EV_KEY, EV_ABS):
                        self._handle_linux_event(event.code, event.value)
            except OSError:
                # switched off, out of range or out of battery
                self._set_connected(False)
            device.close()

    def _handle_linux_event(self, code: int, value: int) -> None:
        name = LINUX_CODES.get(code)
        if name is None:
            return
        if name in STICKS:
            value -= STICK_CENTER
        elif name in TRIGGERS:
            value = value * 2 ** 8 // TRIGGER_RANGE
        self._handle_event(name, value)


def _find_controller() -> InputDevice | None:
    """the wireless controller among the input devices, None when it isn't connected"""
    from evdev import InputDevice, list_devices
    for path in list_devices():
        try:
            device = InputDevice(path)
        except OSError:
            # gone since it was listed
            continue
        if device.name == DEVICE_NAME:
            return device
        device.close()
    return None


if __name__ == '__main__':
    joy = XboxOneController(scale=100)
    while True:
        print(joy.get_left_thumb(), joy.get_right_thumb(), joy.get_lr_trigger(), joy.get_buttons())
        sleep(0.1)
