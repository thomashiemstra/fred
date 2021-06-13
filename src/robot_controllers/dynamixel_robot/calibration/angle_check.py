from src import global_constants
from src.robot_controllers.dynamixel_robot.dynamixel_robot_controller import DynamixelRobotController
import numpy as np
from time import sleep
import json


robot = DynamixelRobotController("COM3", global_constants.dynamixel_robot_config)
servo_id = 5
servo = robot.servo5

robot.enable_servos()
robot.move_servo(servo_id, 0)
sleep(2)


def get_offset():
    current_positions = robot.get_current_positions()
    servo2_target = servo.target_position
    return int(current_positions[servo_id] - servo2_target)


result = {}


for i in range(0, 21, 1):
    if servo_id == 5:
        angle = ((i * (np.pi + 1)) / 20) - (np.pi/2 + 0.5)
    else:
        angle = (2 * i * np.pi / 20) - np.pi
    robot.move_servo(servo_id, angle)
    sleep(1)
    offset = get_offset()
    print(offset)
    result[str(angle)] = offset

print(json.dumps(result))
robot.disable_servos()
