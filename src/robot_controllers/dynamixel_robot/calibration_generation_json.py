import json
from src.robot_controllers.dynamixel_robot.servo import Servo, ServoEncoder, ServoDecoder
from numpy import pi

servo1 = Servo(0, 6144, 0, pi, 150, 40, p=1500, i=0, d=1500, offset=-130)
servo2 = Servo(0, 6144, 0, pi, 150, 40, p=1000, i=0, d=8500, offset=20)
servo3 = Servo(3072, 1024, -pi/2, pi/2, 150, 40, p=1500, i=0, d=4500, offset=10)
servo4 = Servo(0, 4096, -pi, pi, 250, 100, p=2500, i=0, d=4500, offset=-10)
servo5 = Servo(0, 4096, -pi, pi, 250, 100, p=2500, i=0, d=500, offset=-10)
servo6 = Servo(0, 4096, -pi, pi, 250, 100, p=1500, i=0, d=500, offset=0)
servo7 = Servo(2048, 3072, 0, 100, 50, 50, p=500, i=0, d=500, goal_current=300)

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

# print(json_string)

with open("resources/servo_config.json", "w") as outfile:
    outfile.write(json_string)

decoded_servo = ServoDecoder().decode(json_string)

print(decoded_servo)

