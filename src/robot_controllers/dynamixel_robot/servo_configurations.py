from src.robot_controllers.dynamixel_robot.servo import Servo, ServoWithOffsetFunction
from numpy import pi, cos


def servo2_offset_function_going_up(all_angles):
    angle2_input = all_angles[2]
    angle3_input = all_angles[3] - 0.5 * pi
    # function fitting output:
    # [-5.78395335  1.85416765 -7.71692822  0.21412491]
    return int(-5.8 * cos(1.8 * angle2_input) - 7.7 * cos(0.2 * (angle2_input + angle3_input)))


def servo2_offset_function_going_down(all_angles):
    angle2_input = all_angles[2]
    angle3_input = all_angles[3] - 0.5 * pi
    # function fitting output:
    # [-1.26041  1.03593 -1.97792  0.00002]
    return int(-1.3 * cos(1.0 * angle2_input) - 2.0)


def servo3_offset_function_going_up(all_angles):
    angle2_input = all_angles[2]
    angle3_input = all_angles[3] - 0.5 * pi
    # function fitting output:
    # [11.58150402  1.0308894 ]
    return int(11.5 * cos(1.0 * (angle2_input + angle3_input)))


def servo3_offset_function_going_down(all_angles):
    angle2_input = all_angles[2]
    angle3_input = all_angles[3] - 0.5 * pi
    # function fitting output:
    # [5.35683678 1.00077923]
    return int(5.3 * cos(1.0 * (angle2_input + angle3_input)))


servo1 = Servo(0, 6144, 0, pi, 150, 50, p=1500, i=0, d=500, offset=0)
servo2 = ServoWithOffsetFunction(0, 6144, 0, pi, servo2_offset_function_going_up, servo2_offset_function_going_down, 150, 50, p=1000, i=0, d=0, offset=0)
servo3 = ServoWithOffsetFunction(3072, 1024, -pi/2, pi/2,
                                 servo3_offset_function_going_up, servo3_offset_function_going_down,
                                 150, 15, p=2000, i=0, d=0, offset=10)
servo4 = Servo(0, 4096, -pi, pi, 250, 50, p=3000, i=0, d=0, offset=0)
servo5 = Servo(0, 4096, -pi, pi, 250, 50, p=5500, i=0, d=2000, offset=0)
servo6 = Servo(0, 4096, -pi, pi, 250, 50, p=2000, i=0, d=0, offset=0)
servo7 = Servo(2048, 3072, 0, 100, 50, 50, p=500, i=0, d=0, goal_current=100)

servo_configs = [servo1, servo2, servo3, servo4, servo5, servo6, servo7]
