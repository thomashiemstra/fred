from src.robot_controllers.dynamixel_robot.servo import Servo


class ServoPositionRecorder:

    def __init__(self):
        self._positions = []
        super().__init__()

    def record(self, servos: list[Servo]):
        data =  [[1, servos[1].current_position, servos[1].target_position],
                 [2, servos[2].current_position, servos[2].target_position],
                 [3, servos[3].current_position, servos[3].target_position],
                 [4, servos[4].current_position, servos[4].target_position],
                 [5, servos[5].current_position, servos[5].target_position],
                 [6, servos[6].current_position, servos[6].target_position]]

        self._positions.append(data)

    def print(self):
        print(self._positions)

    def write(self):
        with open("output.txt", "w") as txt_file:
            for line in self._positions:
                txt_file.write(" ".join(line) + "\n")  # works with any number of elements in a line

