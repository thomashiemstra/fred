from time import sleep, time
import matplotlib.pyplot as plt
from src.global_objects import get_robot
from src.robot_controllers.combined_robot.combined_robot import CombinedRobot

servo_id = 1
target = 2048


robot = get_robot()
assert isinstance(robot, CombinedRobot), 'this script needs the real robot'

robot.enable_servos()
# p  i  d
robot.set_pid_single_servo(servo_id, 1500, 0, 500)

x = []
y = []
y_target = []

prev_pos = robot.get_servo_position(servo_id)
assert prev_pos is not None, 'could not read servo {}'.format(servo_id)
if prev_pos > 1500:
    target = 1024
else:
    target = 3072

robot.set_servo_position(servo_id, target)
elapsed_time: float = 0
for i in range(300):
    start = time()
    pos = robot.get_servo_position(servo_id)
    stop = time()

    elapsed_time += stop - start
    x.append(elapsed_time)

    y.append(pos)
    y_target.append(target)

# robot.disable_servos()

plt.plot(x, y, 'g')
plt.plot(x, y_target, 'r')
plt.show()

print("hoi")