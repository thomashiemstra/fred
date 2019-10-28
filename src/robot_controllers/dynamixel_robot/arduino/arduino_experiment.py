from Arduino import Arduino  # arduino-python3 0.6
import time

gripper_min_pwm = 60
gripper_max_pwm = 160
gripper_servo_pin = 9

def convert_gripper_state_to_pwm(state):
    return gripper_min_pwm + ((gripper_max_pwm - gripper_min_pwm) * (state / 100))


board = Arduino()  # plugged in via USB, serial com at rate 115200

board.Servos.attach(9, min=720, max=1240)

for i in range(100):
    pwm = convert_gripper_state_to_pwm(i)
    board.Servos.write(9, pwm)
    time.sleep(0.05)


for i in reversed(range(100)):
    pwm = convert_gripper_state_to_pwm(i)
    board.Servos.write(9, pwm)
    time.sleep(0.05)

# while True:
#     board.Servos.write(9, 80)
#
#     time.sleep(1)
#
#     board.Servos.write(9, 180)
#     time.sleep(1)
