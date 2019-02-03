from xbox_controller.xbox_poller import XboxPoller
import numpy as np
import time


if __name__ == '__main__':
    v_x = 0
    v_y = 0
    v_z = 0

    x_max = 5

    steps_per_second = 5
    dt = 1.0 / steps_per_second
    maximum_speed = 5  # cm/sec
    ramp_up_time = 1.0  # 1 second to reach max speed
    dv = (maximum_speed * ramp_up_time) / steps_per_second  # v/step

    pos_x = 0
    pos_y = 15
    pos_z = 5

    poller = XboxPoller()

    try:
        while True:
            x, y = poller.get_left_thumb()

            v_x_max = maximum_speed*(x / 100)

            # TODO reduce, if you dare...
            if x > 0:
                v_x = v_x + dv if v_x < v_x_max else v_x_max
            elif x < 0:
                v_x = v_x - dv if v_x > v_x_max else v_x_max
            else:
                if v_x > 0:
                    v_x = v_x - dv if v_x - dv > 0 else 0
                elif v_x < 0:
                    v_x = v_x + dv if v_x + dv < 0 else 0

            pos_x += dt*v_x_max
            if pos_x > x_max:
                pos_x = x_max
            if pos_x < -x_max:
                pos_x = -x_max

            print(v_x)

            time.sleep(dt)

    except KeyboardInterrupt:
        print("stopped")

    finally:
        poller.stop()
