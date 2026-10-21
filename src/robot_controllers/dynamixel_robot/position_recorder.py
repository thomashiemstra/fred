from src.robot_controllers.dynamixel_robot.servo import Servo


class ServoPositionRecorder:

    def __init__(self) -> None:
        self._positions: list[list[list[int]]] = []
        super().__init__()

    def record(self, servos: list[Servo]) -> None:
        """records the arm servos, servos[0] is servo 1 like in DynamixelRobotController._servos"""
        data = [[servo_id, servo.current_position, servo.target_position]
                for servo_id, servo in enumerate(servos[:6], start=1)]

        self._positions.append(data)

    def print(self) -> None:
        print(self._positions)

    def write(self) -> None:
        """one line per recording: id, current position and target position of each servo"""
        with open("output.txt", "w") as txt_file:
            for recording in self._positions:
                txt_file.write(" ".join(str(value) for servo in recording for value in servo) + "\n")

