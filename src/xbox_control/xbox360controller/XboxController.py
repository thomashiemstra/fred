from inputs import get_gamepad
import math
import threading
from time import sleep

from src.utils.decorators import synchronized_with_lock


class Buttons:
    def __init__(self, start=False, x=False, y=False, a=False, b=False, rb=False, lb=False,
                 pad_up=False, pad_down=False, pad_left=False, pad_right=False):
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

    def __str__(self):
        return 'Buttons: start={} x={} y={} a={}, b={}, rb={}, lb={}, ' \
               'pad_up={}. pad_down={}. pad_left={}, pad_right={}'\
            .format(self.start, self.x, self.y, self.a, self.b, self.rb, self.lb,
                    self.pad_up, self.pad_down, self.pad_left, self.pad_right)


class XboxController(object):
    MAX_TRIG_VAL = math.pow(2, 8)
    MAX_JOY_VAL = math.pow(2, 15)

    def __init__(self,
                 dead_zone=30,
                 scale=1,
                 invert_yaxis=False):
        self.lower_dead_zone = dead_zone * -1
        self.upper_dead_zone = dead_zone

        self.scale = scale
        self.invert_yaxis_val = -1 if invert_yaxis else 1
        self.lock = threading.RLock()

        self.l_thumb_x = 0
        self.l_thumb_y = 0
        self.r_thumb_x = 0
        self.r_thumb_y = 0
        self.lr_trigger = 0

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

        self._monitor_thread = threading.Thread(target=self._monitor_controller, args=())
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

    @synchronized_with_lock("lock")
    def stop(self):
        pass

    @synchronized_with_lock("lock")
    def get_left_thumb(self):
        return self.l_thumb_x, self.l_thumb_y

    @synchronized_with_lock("lock")
    def get_right_thumb(self):
        return self.r_thumb_x, self.r_thumb_y

    @synchronized_with_lock("lock")
    def get_lr_trigger(self):
        return self.lr_trigger

    @synchronized_with_lock("lock")
    def get_buttons(self):
        buttons = Buttons(self.start, self.x, self.y, self.a, self.b, self.rb, self.lb,
                          self.pad_up, self.pad_down, self.pad_left, self.pad_right)
        self.reset_button()
        return buttons

    def reset_button(self):
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

    def _monitor_controller(self):
        while True:
            events = get_gamepad()
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
    def __left_thumb_x(self, x_value):
        self.l_thumb_x = self._handle_x_axis_value(x_value)

    @synchronized_with_lock("lock")
    def __left_thumb_y(self, y_value):
        self.l_thumb_y = self._handle_y_axis_value(y_value)

    @synchronized_with_lock("lock")
    def __right_thumb_x(self, x_value):
        self.r_thumb_x = self._handle_x_axis_value(x_value)

    @synchronized_with_lock("lock")
    def __right_thumb_y(self, y_value):
        self.r_thumb_y = self._handle_y_axis_value(y_value)

    @synchronized_with_lock("lock")
    def __x(self, value):
        self.x = True if value == 1 else False

    @synchronized_with_lock("lock")
    def __y(self, value):
        self.y = True if value == 1 else False

    @synchronized_with_lock("lock")
    def __a(self, value):
        self.a = True if value == 1 else False

    @synchronized_with_lock("lock")
    def __b(self, value):
        self.b = True if value == 1 else False

    @synchronized_with_lock("lock")
    def __right_bumper(self, value):
        self.rb = True if value == 1 else False

    @synchronized_with_lock("lock")
    def ___left_bumper(self, value):
        self.lb = True if value == 1 else False

    @synchronized_with_lock("lock")
    def _left_trigger(self, left_trigger_value):
        self.lr_trigger = -(left_trigger_value / XboxController.MAX_TRIG_VAL) * self.scale

    @synchronized_with_lock("lock")
    def _right_trigger(self, right_trigger_value):
        self.lr_trigger = (right_trigger_value / XboxController.MAX_TRIG_VAL) * self.scale

    @synchronized_with_lock("lock")
    def _pad_left_right(self, value):
        if value == -1:
            self.pad_left = True
        elif value == 1:
            self.pad_right = True
        else:
            self.pad_right, self.pad_left = False, False

    @synchronized_with_lock("lock")
    def _pad_up_down(self, value):
        if value == -1:
            self.pad_up = True
        elif value == 1:
            self.pad_down = True
        else:
            self.pad_down, self.pad_up = False, False

    @synchronized_with_lock("lock")
    def _start(self, value):
        self.start = True if value == 1 else 0

    def _handle_x_axis_value(self, value):
        value = (value / XboxController.MAX_JOY_VAL) * self.scale

        if self.upper_dead_zone > value > self.lower_dead_zone:
            value = 0
        return value

    def _handle_y_axis_value(self, value):
        value = (value / XboxController.MAX_JOY_VAL) * self.scale * self.invert_yaxis_val

        if self.upper_dead_zone > value > self.lower_dead_zone:
            value = 0
        return value


if __name__ == '__main__':
    joy = XboxController(scale=100)
    while True:
        print(joy.get_buttons())
