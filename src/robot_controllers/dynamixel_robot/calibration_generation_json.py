import json

import jsonpickle

from src.robot_controllers.dynamixel_robot.servo import Servo, ServoEncoder, ServoDecoder
from numpy import pi

# TODO fix values for new fancy robot
servo1 = Servo(1024, 3072, 0, pi, 80, 30, p=800, i=0, d=2500)
servo2 = Servo(1024, 3072, 0, pi, 80, 30, p=1500, i=0, d=500)
servo3 = Servo(1024, 3072, -pi/2, pi/2, 80, 30, p=1500, i=100, d=500)
servo4 = Servo(0, 4095, -pi, pi, 150, 50, p=2500, i=0, d=3500)
servo5 = Servo(0, 4095, -pi, pi, 150, 50, p=2500, i=0, d=3500)
servo6 = Servo(0, 4095, -pi, pi, 150, 50, p=2500, i=0, d=3500)
servo7 = Servo(0, 4095, -pi, pi, 150, 50, p=2500, i=0, d=3500)

servos = {
    "servo1": servo1,
    "servo2": servo2,
    "servo3": servo3,
    "servo4": servo4,
    "servo5": servo5,
    "servo6": servo6,
    "servo7": servo7,
}

json_string = json.dumps(servos, cls=ServoEncoder,  indent=4)

print(json_string)

decoded_servo = ServoDecoder().decode(json_string)

print(decoded_servo)

