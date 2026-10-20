from src.global_objects import get_robot
from src.robot_controllers.combined_robot.combined_robot import CombinedRobot
import numpy as np
from time import sleep
import json


real_robot = get_robot()
assert isinstance(real_robot, CombinedRobot), 'this script needs the real robot'
robot = real_robot
servo_id = 5
servo = robot.servo5

robot.enable_servos()
robot.move_servo(servo_id, 0)
sleep(2)


def get_offset() -> int:
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
