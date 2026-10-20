import logging

import inputs
from inputs import get_gamepad, UnpluggedError
import math
import threading
from time import sleep

from src.utils.decorators import synchronized_with_lock
from src.utils.os_utils import is_linux

logger = logging.getLogger(__name__)

# how often to look for a gamepad when none is connected
RECONNECT_INTERVAL = 1.0  # seconds


class Buttons:
    def __init__(self, start: bool = False, x: bool = False, y: bool = False, a: bool = False, b: bool = False,
                 rb: bool = False, lb: bool = False, pad_up: bool = False, pad_down: bool = False,
                 pad_left: bool = False, pad_right: bool = False) -> None:
        self.rb = rb
        self.lb = lb
        self.b = b
        self.a = a
        self.y = y
        self.x = x
        self.start = start
        self.pad_up = pad_up
        self.pad_down = pad_down
        self.pad_left = pad_left
        self.pad_right = pad_right

    def __str__(self) -> str:
        return 'Buttons: start={} x={} y={} a={}, b={}, rb={}, lb={}, ' \
               'pad_up={}. pad_down={}. pad_left={}, pad_right={}'\
            .format(self.start, self.x, self.y, self.a, self.b, self.rb, self.lb,
                    self.pad_up, self.pad_down, self.pad_left, self.pad_right)


class XboxController(object):
    MAX_TRIG_VAL = math.pow(2, 8)
    MAX_JOY_VAL = math.pow(2, 15)

    def __init__(self,
                 dead_zone: float = 30,
                 scale: float = 1,
                 invert_yaxis: bool = False) -> None:
        self.lower_dead_zone = dead_zone * -1
        self.upper_dead_zone = dead_zone

        self.scale = scale
        self.invert_yaxis_val = -1 if invert_yaxis else 1
        # the inputs library reports the x axis of the right stick with opposite signs on Linux and Windows; the
        # sign makes Windows behave like Linux, where the controls were tuned. The y axis of the right stick is
        # turned around so that up is positive, like the gamma it turns
        self._right_x_sign = 1 if is_linux() else -1
        self.lock = threading.RLock()

        self.l_thumb_x: float = 0
        self.l_thumb_y: float = 0
        self.r_thumb_x: float = 0
        self.r_thumb_y: float = 0
        self.lr_trigger: float = 0

        self.start = False
        self.x = False
        self.y = False
        self.a = False
        self.b = False
        self.lb = False
        self.rb = False
        self.pad_up = False
        self.pad_down = False
        self.pad_left = False
        self.pad_right = False
        self.connected = False

        self._monitor_thread = threading.Thread(target=self._monitor_controller, args=())
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

    @synchronized_with_lock("lock")
    def stop(self) -> None:
        pass

    @synchronized_with_lock("lock")
    def is_connected(self) -> bool:
        return self.connected

    def wait_until_connected(self, timeout: float = 2.0) -> bool:
        """returns True if a gamepad is connected within the timeout"""
        waited = 0.0
        while not self.is_connected() and waited < timeout:
            sleep(0.1)
            waited += 0.1
        return self.is_connected()

    @synchronized_with_lock("lock")
    def _set_connected(self, connected: bool) -> None:
        if connected == self.connected:
            return
        self.connected = connected
        if connected:
            logger.info('gamepad connected')
        else:
            # the last values of the sticks would keep the robot moving
            logger.warning('no gamepad connected')
            self.l_thumb_x = self.l_thumb_y = self.r_thumb_x = self.r_thumb_y = self.lr_trigger = 0
            self.lb = self.rb = False
            self.reset_button()

    @synchronized_with_lock("lock")
    def get_left_thumb(self) -> tuple[float, float]:
        return self.l_thumb_x, self.l_thumb_y

    @synchronized_with_lock("lock")
    def get_right_thumb(self) -> tuple[float, float]:
        return self.r_thumb_x, self.r_thumb_y

    @synchronized_with_lock("lock")
    def get_lr_trigger(self) -> float:
        return self.lr_trigger

    @synchronized_with_lock("lock")
    def get_buttons(self) -> Buttons:
        buttons = Buttons(self.start, self.x, self.y, self.a, self.b, self.rb, self.lb,
                          self.pad_up, self.pad_down, self.pad_left, self.pad_right)
        self.reset_button()
        return buttons

    def reset_button(self) -> None:
        self.start = False
        self.x = False
        self.y = False
        self.a = False
        self.b = False
        self.pad_up = False
        self.pad_down = False
        self.pad_left = False
        self.pad_right = False

    def _monitor_controller(self) -> None:
        while True:
            try:
                events = get_gamepad()
            except UnpluggedError:
                self._set_connected(False)
                sleep(RECONNECT_INTERVAL)
                _find_gamepads()
                continue
            self._set_connected(True)

            for event in events:
                if event.ev_type == 'Sync':
                    continue
                if event.code == 'ABS_Y':
                    self.__left_thumb_y(event.state)
                elif event.code == 'ABS_X':
                    self.__left_thumb_x(event.state)
                elif event.code == 'ABS_RY':
                    self.__right_thumb_y(event.state)
                elif event.code == 'ABS_RX':
                    self.__right_thumb_x(event.state)
                elif event.code == 'ABS_Z':
                    self._left_trigger(event.state)
                elif event.code == 'ABS_RZ':
                    self._right_trigger(event.state)
                elif event.code == 'BTN_TL':
                    self.___left_bumper(event.state)
                elif event.code == 'BTN_TR':
                    self.__right_bumper(event.state)
                elif event.code == 'BTN_SOUTH':
                    self.__a(event.state)
                elif event.code == 'BTN_NORTH':
                    self.__y(event.state)
                elif event.code == 'BTN_WEST':
                    self.__x(event.state)
                elif event.code == 'BTN_EAST':
                    self.__b(event.state)
                # elif event.code == 'BTN_THUMBL':
                #     self.LeftThumb = event.state
                # elif event.code == 'BTN_THUMBR':
                #     self.RightThumb = event.state
                # elif event.code == 'BTN_SELECT':
                #     self.Back = event.state
                elif event.code == 'BTN_START':
                    self._start(event.state)
                elif event.code == 'ABS_HAT0X':
                    self._pad_left_right(event.state)
                elif event.code == 'ABS_HAT0Y':
                    self._pad_up_down(event.state)

    @synchronized_with_lock("lock")
    def __left_thumb_x(self, x_value: float) -> None:
        self.l_thumb_x = self._handle_x_axis_value(x_value)

    @synchronized_with_lock("lock")
    def __left_thumb_y(self, y_value: float) -> None:
        self.l_thumb_y = self._handle_y_axis_value(y_value)

    @synchronized_with_lock("lock")
    def __right_thumb_x(self, x_value: float) -> None:
        self.r_thumb_x = self._handle_x_axis_value(x_value) * self._right_x_sign

    @synchronized_with_lock("lock")
    def __right_thumb_y(self, y_value: float) -> None:
        self.r_thumb_y = -self._handle_y_axis_value(y_value)

    @synchronized_with_lock("lock")
    def __x(self, value: int) -> None:
        self.x = True if value == 1 else False

    @synchronized_with_lock("lock")
    def __y(self, value: int) -> None:
        self.y = True if value == 1 else False

    @synchronized_with_lock("lock")
    def __a(self, value: int) -> None:
        self.a = True if value == 1 else False

    @synchronized_with_lock("lock")
    def __b(self, value: int) -> None:
        self.b = True if value == 1 else False

    @synchronized_with_lock("lock")
    def __right_bumper(self, value: int) -> None:
        self.rb = True if value == 1 else False

    @synchronized_with_lock("lock")
    def ___left_bumper(self, value: int) -> None:
        self.lb = True if value == 1 else False

    @synchronized_with_lock("lock")
    def _left_trigger(self, left_trigger_value: float) -> None:
        self.lr_trigger = -(left_trigger_value / XboxController.MAX_TRIG_VAL) * self.scale

    @synchronized_with_lock("lock")
    def _right_trigger(self, right_trigger_value: float) -> None:
        self.lr_trigger = (right_trigger_value / XboxController.MAX_TRIG_VAL) * self.scale

    @synchronized_with_lock("lock")
    def _pad_left_right(self, value: int) -> None:
        if value == -1:
            self.pad_left = True
        elif value == 1:
            self.pad_right = True
        else:
            self.pad_right, self.pad_left = False, False

    @synchronized_with_lock("lock")
    def _pad_up_down(self, value: int) -> None:
        if value == -1:
            self.pad_up = True
        elif value == 1:
            self.pad_down = True
        else:
            self.pad_down, self.pad_up = False, False

    @synchronized_with_lock("lock")
    def _start(self, value: int) -> None:
        self.start = value == 1

    def _handle_x_axis_value(self, value: float) -> float:
        value = (value / XboxController.MAX_JOY_VAL) * self.scale

        if self.upper_dead_zone > value > self.lower_dead_zone:
            value = 0
        return value

    def _handle_y_axis_value(self, value: float) -> float:
        value = (value / XboxController.MAX_JOY_VAL) * self.scale * self.invert_yaxis_val

        if self.upper_dead_zone > value > self.lower_dead_zone:
            value = 0
        return value


def _find_gamepads() -> None:
    """the inputs library only looks for gamepads when it is imported, look again for gamepads connected later"""
    devices = inputs.devices
    if devices.gamepads or not hasattr(devices, '_detect_gamepads') or devices.xinput is None:
        # a gamepad that was found before is reused when it is plugged back in, on linux the devices can't be
        # searched again without restarting
        return
    try:
        devices._detect_gamepads()
        devices._update_all_devices()
    except RuntimeError as e:
        logger.warning('failed to look for gamepads: %s', e)


if __name__ == '__main__':
    joy = XboxController(scale=100)
    while True:
        print(joy.get_buttons())
