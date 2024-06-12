from src.global_constants import EXTENDED_POSITION_CONTROL, CURRENT_BASED_POSITION_CONTROL_MODE, POSITION_CONTROL_MODE
from src.robot_controllers.dynamixel_robot.servo import Servo, ServoWithOffsetFunction
from numpy import pi, cos


def servo2_offset_function_going_up(all_angles):
    angle2_input = all_angles[2]
    angle3_input = all_angles[3] - 0.5 * pi
    # function fitting output:
    # [12.00445  5.08356]
    return int(12.0 * cos(angle2_input) + 5.1 * cos(angle2_input + angle3_input))


def servo2_offset_function_going_down(all_angles):
    angle2_input = all_angles[2]
    angle3_input = all_angles[3] - 0.5 * pi
    # function fitting output:
    # [ 3.24279 -0.89222]
    return int(2.6 * cos(angle2_input) - 0.9 * cos(angle2_input + angle3_input))


def servo3_offset_function_going_up(all_angles):
    angle2_input = all_angles[2]
    angle3_input = all_angles[3] - 0.5 * pi
    # function fitting output:
    # 29.6
    return int(29.6 * cos(angle2_input + angle3_input))


def servo3_offset_function_going_down(all_angles):
    angle2_input = all_angles[2]
    angle3_input = all_angles[3] - 0.5 * pi
    # function fitting output:
    # 9.75
    return int(9.75 * cos(angle2_input + angle3_input))


servo1 = Servo(1024, 3072, 0, pi, POSITION_CONTROL_MODE, 250, 100, p=1500, i=0, d=500, offset=-25)
servo2 = Servo(1024, 3072, 0, pi, POSITION_CONTROL_MODE, 150, 50, p=1000, i=0, d=0, offset=20)
servo3 = Servo(3072, 1024, -pi/2, pi/2, POSITION_CONTROL_MODE,150, 50, p=1500, i=0, d=0, offset=0)
servo4 = Servo(0, 4096, -pi, pi, POSITION_CONTROL_MODE, 300, 100, p=1000, i=0, d=0, offset=0)
servo5 = Servo(0, 4096, -pi, pi, POSITION_CONTROL_MODE, 300, 100, p=2000, i=0, d=0, offset=-27)
servo6 = Servo(0, 4096, -pi, pi, POSITION_CONTROL_MODE, 300, 100, p=600, i=0, d=0, offset=0)
servo7 = Servo(1024, 2048, 0, 100, CURRENT_BASED_POSITION_CONTROL_MODE, 50, 50, p=500, i=0, d=0, goal_current=50)

servo_configs = [servo1, servo2, servo3, servo4, servo5, servo6, servo7]
