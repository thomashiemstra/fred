from __future__ import print_function

import inspect
import json
import os
from time import sleep

import keyboard
from numpy import pi

# root window
from src.robot_controllers.dynamixel_robot.arduino.ArduinoAngleReader import ArduinoAngleReader
from src.global_objects import get_robot
from src.robot_controllers.combined_robot.combined_robot import CombinedRobot


class Servo2Manager:

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug
        robot = get_robot()
        assert isinstance(robot, CombinedRobot), 'servo 2 can only be managed on the real robot'
        self.robot = robot

        self.servo_id = 2
        self.servo = self.robot.servo2

        self.angle2 = 0.0
        self.angle3 = 0.5
        self.step_size = 0.05

        self.servo_2offsets_dict: dict[float, dict[float, int]] = {}
        self.servo_3offsets_dict: dict[float, dict[float, int]] = {}
        self.servo_3_direct_offsets_dict: dict[float, list[int]] = {}
        self.angle2_reader = ArduinoAngleReader()
        self.angle2_reader.start()

    def _add_keyboard_shortcuts(self) -> None:
        keyboard.add_hotkey('s', lambda: self.switch_servo())
        keyboard.add_hotkey('up', lambda: self.move_current_servo_up())
        keyboard.add_hotkey('down', lambda: self.move_current_servo_down())
        keyboard.add_hotkey('space', lambda: self.record_offset())
        keyboard.add_hotkey('p', lambda: self.print_offset_json())

    def start(self) -> None:
        self._add_keyboard_shortcuts()
        self._start_internal()
        keyboard.wait()

    def start_no_keyboard(self) -> None:
        self._start_internal()

    def _start_internal(self) -> None:
        self.robot.enable_servos()
        self.robot.move_servo(3, self.angle3 * pi)
        self.robot.move_servo(2, self.angle2 * pi)

    def print_offset_json(self) -> None:
        print(json.dumps(self.servo_2offsets_dict))
        print(json.dumps(self.servo_3_direct_offsets_dict))

    def record_offset(self) -> None:
        current_positions = self.robot.get_current_positions()
        servo2_target = self.robot.servo2.target_position
        servo2_offset = current_positions[2] - servo2_target

        servo3_target = self.robot.servo3.target_position
        servo3_offset = current_positions[3] - servo3_target

        # go from 0.300000001 to 0.3, floats....
        angle2rounded = round(self.angle2, 2)
        angle3rounded = round(self.angle3, 2)

        if self.debug:
            print(angle2rounded, angle3rounded, servo2_offset, servo3_offset)

        angle3_dict = {} if angle2rounded not in self.servo_2offsets_dict else self.servo_2offsets_dict[angle2rounded]
        angle3_dict[angle3rounded] = int(servo2_offset)
        self.servo_2offsets_dict[angle2rounded] = angle3_dict

        angle2_dict = {} if angle3rounded not in self.servo_3offsets_dict else self.servo_3offsets_dict[angle3rounded]
        angle2_dict[angle2rounded] = int(servo3_offset)
        self.servo_3offsets_dict[angle3rounded] = angle2_dict

        combined_angle = round(angle2rounded + angle3rounded, 2)
        angle2_list = [] if combined_angle not in self.servo_3_direct_offsets_dict else self.servo_3_direct_offsets_dict[combined_angle]
        angle2_list.append(int(servo3_offset))
        self.servo_3_direct_offsets_dict[combined_angle] = angle2_list

    def switch_servo(self) -> None:
        if self.servo_id == 2:
            self.servo_id = 3
            self.servo = self.robot.servo3
            if self.debug:
                print("switched to servo3")
        elif self.servo_id == 3:
            self.servo_id = 2
            self.servo = self.robot.servo2
            if self.debug:
                print("switched to servo2")

    def move_current_servo_up(self) -> None:
        if self.servo_id == 2:
            self.angle2 = self.angle2 + self.step_size if self.angle2 < 1 else self.angle2
            self.robot.move_servo(self.servo_id, self.angle2 * pi)
        if self.servo_id == 3:
            self.angle3 = self.angle3 + self.step_size if self.angle3 < 0.5 else self.angle3
            self.robot.move_servo(self.servo_id, self.angle3 * pi)

    def move_current_servo_down(self) -> None:
        if self.servo_id == 2:
            self.angle2 = self.angle2 - self.step_size if self.angle2 > 0 else self.angle2
            self.robot.move_servo(self.servo_id, self.angle2 * pi)
        if self.servo_id == 3:
            self.angle3 = self.angle3 - self.step_size if self.angle3 > -0.5 else self.angle3
            self.robot.move_servo(self.servo_id, self.angle3 * pi)


if __name__ == '__main__':
    manager = Servo2Manager(debug=True)
    manager.start()
