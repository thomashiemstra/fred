from xbox_controller.XboxController import XboxController
import threading
import time
import numpy as np
from utils.decorators import synchronized_with_lock


class Buttons:
    def __init__(self, start=False, x=False, y=False, a=False, b=False, rb=False, lb=False):
        self.rb = rb
        self.lb = lb
        self.b = b
        self.a = a
        self.y = y
        self.x = x
        self.start = start

    def __str__(self):
        return 'Buttons: start ={} x={} y={} a={}, b={}, rb={}, lb={}'\
            .format(self.start, self.x, self.y, self.a, self.b, self.rb, self.lb)


class XboxPoller:

    def __init__(self) -> None:
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

        self.xbox_controller = XboxController(None, deadzone=30, scale=100, invertYAxis=True)
        self.setup_callbacks()
        self.xbox_controller.start()

    def setup_callbacks(self):
        self.xbox_controller.setupControlCallback(self.xbox_controller.XboxControls.LTHUMBX, self.__left_thumb_x)
        self.xbox_controller.setupControlCallback(self.xbox_controller.XboxControls.LTHUMBY, self.__left_thumb_y)
        self.xbox_controller.setupControlCallback(self.xbox_controller.XboxControls.RTHUMBX, self.__right_thumb_x)
        self.xbox_controller.setupControlCallback(self.xbox_controller.XboxControls.RTHUMBY, self.__right_thumb_y)
        self.xbox_controller.setupControlCallback(self.xbox_controller.XboxControls.LRTRIGGER,
                                                  self.__left_right_trigger)
        self.xbox_controller.setupControlCallback(self.xbox_controller.XboxControls.START, self.__start)

        self.xbox_controller.setupControlCallback(self.xbox_controller.XboxControls.A, self.__a)
        self.xbox_controller.setupControlCallback(self.xbox_controller.XboxControls.B, self.__b)
        self.xbox_controller.setupControlCallback(self.xbox_controller.XboxControls.X, self.__x)
        self.xbox_controller.setupControlCallback(self.xbox_controller.XboxControls.Y, self.__y)
        self.xbox_controller.setupControlCallback(self.xbox_controller.XboxControls.RB, self.__right_bumper)
        self.xbox_controller.setupControlCallback(self.xbox_controller.XboxControls.LB, self.___left_bumper)

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
        buttons = Buttons(self.start, self.x, self.y, self.a, self.b, self.rb, self.lb)
        self.reset_buttons()
        return buttons

    @synchronized_with_lock("lock")
    def stop(self):
        self.xbox_controller.stop()

    @synchronized_with_lock("lock")
    def __start(self, value):
        if value == 1:
            self.start = True

    @synchronized_with_lock("lock")
    def __x(self, value):
        if value == 1:
            self.x = True

    @synchronized_with_lock("lock")
    def __y(self, value):
        if value == 1:
            self.y = True

    @synchronized_with_lock("lock")
    def __a(self, value):
        if value == 1:
            self.a = True

    @synchronized_with_lock("lock")
    def __b(self, value):
        if value == 1:
            self.b = True

    @synchronized_with_lock("lock")
    def __right_bumper(self, value):
        if value == 1:
            self.rb = True

    @synchronized_with_lock("lock")
    def ___left_bumper(self, value):
        if value == 1:
            self.lb = True

    @synchronized_with_lock("lock")
    def __left_thumb_x(self, x_value):
        with self.lock:
            self.l_thumb_x = x_value

    @synchronized_with_lock("lock")
    def __left_thumb_y(self, y_value):
        self.l_thumb_y = y_value

    @synchronized_with_lock("lock")
    def __right_thumb_x(self, x_value):
        self.r_thumb_x = x_value

    @synchronized_with_lock("lock")
    def __right_thumb_y(self, y_value):
        self.r_thumb_y = y_value

    @synchronized_with_lock("lock")
    def __left_right_trigger(self, lr_trigger_value):
        if np.abs(lr_trigger_value) < 0.1:
            lr_trigger_value = 0
        self.lr_trigger = lr_trigger_value

    def reset_buttons(self):
        self.start = False
        self.x = False
        self.y = False
        self.a = False
        self.b = False
        self.rb = False
        self.lb = False


if __name__ == '__main__':

    poller = XboxPoller()

    try:
        while True:
            buttons = poller.get_buttons()
            print(buttons)
            time.sleep(1 / 5.0)

    except KeyboardInterrupt:
        print("stopped")

    finally:
        poller.stop()
