from xbox_controller.XboxController import XboxController
import threading
import time


class XboxPoller:

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.l_thumb_x = 0
        self.l_thumb_y = 0

        self.r_thumb_x = 0
        self.r_thumb_y = 0

        self.lr_trigger = 0

        self.xbox_controller = XboxController(None, deadzone=30, scale=100, invertYAxis=True)
        self.setup_callbacks()
        self.xbox_controller.start()

    def setup_callbacks(self):
        self.xbox_controller.setupControlCallback(self.xbox_controller.XboxControls.LTHUMBX, self.__left_thumb_x)
        self.xbox_controller.setupControlCallback(self.xbox_controller.XboxControls.LTHUMBY, self.__left_thumb_y)
        self.xbox_controller.setupControlCallback(self.xbox_controller.XboxControls.RTHUMBX, self.__right_thumb_x)
        self.xbox_controller.setupControlCallback(self.xbox_controller.XboxControls.RTHUMBY, self.__right_thumb_y)
        self.xbox_controller.setupControlCallback(self.xbox_controller.XboxControls.LRTRIGGER, self.__left_right_trigger)

    def get_left_thumb(self):
        with self.lock:
            return self.l_thumb_x, self.l_thumb_y

    def get_right_thumb(self):
        with self.lock:
            return self.r_thumb_x, self.r_thumb_y

    def get_lr_trigger(self):
        with self.lock:
            return self.lr_trigger

    def stop(self):
        self.xbox_controller.stop()

    def __left_thumb_x(self, x_value):
        with self.lock:
            self.l_thumb_x = x_value

    def __left_thumb_y(self, y_value):
        with self.lock:
            self.l_thumb_y = y_value

    def __right_thumb_x(self, x_value):
        with self.lock:
            self.r_thumb_x = x_value

    def __right_thumb_y(self, y_value):
        with self.lock:
            self.r_thumb_y = y_value

    def __left_right_trigger(self, lr_trigger_value):
        with self.lock:
            self.lr_trigger = lr_trigger_value


if __name__ == '__main__':

    poller = XboxPoller()

    try:
        while True:
            # x, y = poller.get_left_thumb()
            # x1, y1 = poller.get_right_thumb()
            # print(x,y,x1,y1)
            trigger = poller.get_lr_trigger()
            print(trigger)
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("stopped")

    finally:
        poller.stop()
