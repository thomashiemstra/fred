from xbox_controller.XboxController import XboxController
import threading

import time
lock = threading.Lock()

l_tumb_x = 0
l_tumb_y = 0


def controlCallBack(xboxControlId, value):
    pass # print("Control Id = {}, Value = {}".format(xboxControlId, value))


def leftThumbX(xValue):
    global l_tumb_x
    with lock:
        l_tumb_x = xValue

    # print("LX {}".format(xValue))

def leftThumbY(yValue):
    global l_tumb_y
    with lock:
        l_tumb_y = yValue
    # print("LY {}".format(yValue))


if __name__ == '__main__':

    xboxCont = XboxController(controlCallBack, deadzone=30, scale=100, invertYAxis=True)

    # setup the left thumb (X & Y) callbacks
    xboxCont.setupControlCallback(xboxCont.XboxControls.LTHUMBX, leftThumbX)
    xboxCont.setupControlCallback(xboxCont.XboxControls.LTHUMBY, leftThumbY)

    try:
        # start the controller
        xboxCont.start()
        while True:
            print("LX = ", l_tumb_x, " LY = ", l_tumb_y)
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("stopped")

    finally:
        xboxCont.stop()
