from time import sleep

from src.robot_controllers.dynamixel_robot.calibration.gravity_compensation import Servo2Manager


manager = Servo2Manager(debug=True)
manager.start_no_keyboard()

sleep(1)

for _ in range(int(0.25 / manager.step_size)):
    manager.move_current_servo_up()

sleep(1)
manager.switch_servo()

for _ in range(int(0.75 / manager.step_size)):
    manager.move_current_servo_down()


sleep(3)
manager.record_offset()

for _ in range(int(0.75 / manager.step_size)):
    manager.move_current_servo_up()
    sleep(2)
    manager.record_offset()

manager.print_offset_json()

