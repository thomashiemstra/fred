import time

from src.robot_controllers.dynamixel_robot.arduino.arduino import Arduino

board = Arduino('9600', port='COM9')

board.Servos.attach(9)

while True:
    print('writing 0')
    board.Servos.write(9, 0)

    time.sleep(1)

    print('writing 180')
    board.Servos.write(9, 180)

    time.sleep(1)
