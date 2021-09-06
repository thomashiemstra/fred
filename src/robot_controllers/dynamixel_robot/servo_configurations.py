from src.global_constants import EXTENDED_POSITION_CONTROL, CURRENT_BASED_POSITION_CONTROL_MODE, POSITION_CONTROL_MODE
from src.robot_controllers.dynamixel_robot.servo import Servo, ServoWithOffsetFunction
from numpy import pi, cos


def servo2_offset_function_going_up(all_angles):
    angle2_input = all_angles[2]
    angle3_input = all_angles[3] - 0.5 * pi
    # function fitting output:
    # [12.89047 -0.81075  2.5941  -0.7554 ]
    return int(12.9 * cos(-0.8 * angle2_input) + 2.6 * cos(-0.75 * (angle2_input + angle3_input)))


def servo2_offset_function_going_down(all_angles):
    angle2_input = all_angles[2]
    angle3_input = all_angles[3] - 0.5 * pi
    # function fitting output:
    # [ 2.64251  1.21949  0.23948 -2.62854]
    return int(2.6 * cos(1.2 * angle2_input) + 0.2 * cos(-2.6 * (angle2_input + angle3_input)))


def servo3_offset_function_going_up(all_angles):
    angle2_input = all_angles[2]
    angle3_input = all_angles[3] - 0.5 * pi
    # function fitting output:
    # [29.91475285  1.01794437]
    return int(30 * cos(1.0 * (angle2_input + angle3_input)))


def servo3_offset_function_going_down(all_angles):
    angle2_input = all_angles[2]
    angle3_input = all_angles[3] - 0.5 * pi
    # function fitting output:
    # [9.84893724 1.02064491]
    return int(9.8 * cos(1 * (angle2_input + angle3_input)))


servo1 = Servo(0, 6144, 0, pi, EXTENDED_POSITION_CONTROL, 250, 100, p=1000, i=0, d=500, offset=50)
servo2 = ServoWithOffsetFunction(0, 6144, 0, pi, EXTENDED_POSITION_CONTROL,
                                 servo2_offset_function_going_up, servo2_offset_function_going_down,
                                 150, 5, p=1000, i=0, d=0, offset=50)
servo3 = ServoWithOffsetFunction(3072, 1024, -pi/2, pi/2, POSITION_CONTROL_MODE,
                                 servo3_offset_function_going_up, servo3_offset_function_going_down,
                                 150, 5, p=1500, i=0, d=0, offset=0)
servo4 = Servo(0, 4096, -pi, pi, POSITION_CONTROL_MODE, 300, 100, p=1000, i=0, d=0, offset=40)
servo5 = Servo(0, 4096, -pi, pi, POSITION_CONTROL_MODE, 300, 100, p=2000, i=0, d=0, offset=15)
servo6 = Servo(0, 4096, -pi, pi, POSITION_CONTROL_MODE, 300, 100, p=600, i=0, d=0, offset=0)
servo7 = Servo(2048, 3072, 0, 100, CURRENT_BASED_POSITION_CONTROL_MODE, 50, 50, p=500, i=0, d=0, goal_current=100)

servo_configs = [servo1, servo2, servo3, servo4, servo5, servo6, servo7]
