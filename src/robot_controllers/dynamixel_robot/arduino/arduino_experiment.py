from Arduino import Arduino  # arduino-python3 0.6
import time

board = Arduino()  # plugged in via USB, serial com at rate 115200

board.Servos.attach(9, min=720, max=1240)

while True:
    board.Servos.write(9, 80)

    time.sleep(1)

    board.Servos.write(9, 180)
    time.sleep(1)
