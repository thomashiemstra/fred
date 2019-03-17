from src.servo_handling import dynamixel_x_config as cfg
from src.servo_handling.dynamixel_utils import setup_dynamixel_handlers
from src.servo_handling.servo import Servo
from src.servo_handling.servo_handler import ServoHandler
import time
from numpy import pi
import logging


logging.basicConfig(level=logging.DEBUG)

PORT = 'COM5'


if __name__ == "__main__":
    port_handler, packet_handler, group_bulk_write, group_bulk_read = setup_dynamixel_handlers(PORT, cfg)

    servo1 = Servo(1024, 3072, 0, pi)
    servo2 = Servo(1024, 3072, 0, pi)
    servo3 = Servo(1024, 3072, 0, pi)
    base_servos = {1: servo1, 2: servo2, 3: servo3}

    base_servo_handler = ServoHandler(base_servos, cfg, port_handler, packet_handler, group_bulk_write, group_bulk_read)
    base_servo_handler.set_torque(enable=True)

    servo4 = Servo(0, 4095, -pi, pi)
    servo5 = Servo(0, 4095, -pi, pi)
    servo6 = Servo(0, 4095, -pi, pi)
    wrist_servos = {4: servo4, 5: servo5, 6: servo6}

    wrist_servo_handler = ServoHandler(wrist_servos, cfg, port_handler, packet_handler, group_bulk_write, group_bulk_read)
    wrist_servo_handler.set_torque(enable=True)

    angle = 0

    for i in range(40):
        if i % 2 == 0:
            angle = pi
        else:
            angle = 0

        base_servo_handler.set_angle(1, angle)
        base_servo_handler.set_angle(2, angle)
        base_servo_handler.set_angle(3, angle)
        base_servo_handler.move_to_angles()

        wrist_servo_handler.set_angle(4, angle)
        wrist_servo_handler.set_angle(5, angle)
        wrist_servo_handler.set_angle(6, angle)
        wrist_servo_handler.move_to_angles()

        time.sleep(0.2)

        # base_servo_handler.read_current_pos()
        # wrist_servo_handler.read_current_pos()
        # time.sleep(0.1)
        # print("-----------base servos------------")
        # print("servo1", servo1.current_position)
        # print("servo2", servo2.current_position)
        # print("servo3", servo3.current_position)
        # print("-----------wrist servos------------")
        # print("servo4", servo4.current_position)
        # print("servo5", servo5.current_position)
        # print("servo6", servo6.current_position)
        #
        # time.sleep(0.5)

    base_servo_handler.set_torque(enable=False)
    wrist_servo_handler.set_torque(enable=False)

    port_handler.closePort()
