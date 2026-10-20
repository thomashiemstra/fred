# Lists the connected usb adapters for dynamixel servos and the servos that respond on them
# usage: python -m src.robot_controllers.dynamixel_robot.scan_servos [max servo id]
import sys

import serial

from src.robot_controllers.dynamixel_robot import dynamixel_x_config as cfg
from src.robot_controllers.dynamixel_robot.dynamixel_utils import list_dynamixel_adapters, ping_servos

# model numbers of the servos used in the robot
MODEL_NAMES = {
    1020: 'XM430-W350',
    1120: 'XM540-W270',
    1070: 'XC430-W150',
    1080: 'XC430-W240',
    1200: 'XL330-M288',
}


def main(max_servo_id: int = 20) -> None:
    adapters = list_dynamixel_adapters()
    if not adapters:
        print("no dynamixel usb adapter found")
        return

    for port, name in adapters:
        print("%s: %s" % (port, name))
        try:
            models = ping_servos(port, cfg, range(1, max_servo_id + 1))
        except serial.SerialException as e:
            print("  could not open the port, is it in use by another program? %s" % e)
            continue

        if not models:
            print("  no servos respond with ids 1-%d at %d bps, is the power on?" % (max_servo_id, cfg.BAUDRATE))
        for servo_id, model_number in models.items():
            print("  id %d: %s (model number %d)" % (servo_id, MODEL_NAMES.get(model_number, 'unknown'), model_number))


if __name__ == '__main__':
    main(*[int(arg) for arg in sys.argv[1:]])
