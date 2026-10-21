import os
import tempfile
import types
import unittest

from src.robot_controllers.dynamixel_robot.position_recorder import ServoPositionRecorder


class ServoPositionRecorderTests(unittest.TestCase):

    def test_each_recording_is_written_as_one_line(self):
        # servo 1 first, like DynamixelRobotController._servos; the gripper (7) is not recorded
        servos = [types.SimpleNamespace(current_position=100 + servo_id, target_position=200 + servo_id)
                  for servo_id in range(1, 8)]
        recorder = ServoPositionRecorder()
        recorder.record(servos)
        recorder.record(servos)

        with tempfile.TemporaryDirectory() as directory:
            cwd = os.getcwd()
            os.chdir(directory)
            try:
                recorder.write()
                with open('output.txt') as output:
                    lines = output.read().splitlines()
            finally:
                os.chdir(cwd)

        expected = '1 101 201 2 102 202 3 103 203 4 104 204 5 105 205 6 106 206'
        self.assertEqual([expected, expected], lines)


if __name__ == '__main__':
    unittest.main()
