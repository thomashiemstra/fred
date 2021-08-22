from enum import Enum
from typing import Any, Optional, Callable, Dict, List, Tuple

import numpy as np
import logging
import json
from numpy import pi


class ServoEncoder(json.JSONEncoder):

    def default(self, o: Any) -> Any:
        if isinstance(o, Servo):
            return {
                "min_position": o.min_position,
                "max_position": o.max_position,
                "max_angle": o.max_angle,
                "min_angle": o.min_angle,
                "profile_velocity": o.profile_velocity,
                "profile_acceleration": o.profile_acceleration,
                "p": o.p,
                "i": o.i,
                "d": o.d,
                "offset": o.constant_offset,
                "goal_current": o.goal_current,
                "operating_mode" : o.operating_mode,
                "class": "servo"
            }
        else:
            return super().default(o)


class ServoDecoder(json.JSONDecoder):

    def __init__(self) -> None:
        super().__init__(object_hook=self.dict_to_object)

    @staticmethod
    def dict_to_object(dictionary):
        if "class" in dictionary.keys() and dictionary["class"] == "servo":
            return Servo(
                dictionary["min_position"],
                dictionary["max_position"],
                dictionary["min_angle"],
                dictionary["max_angle"],
                dictionary["profile_velocity"],
                dictionary["profile_acceleration"],
                dictionary["p"],
                dictionary["i"],
                dictionary["d"],
                dictionary["offset"],
                dictionary["goal_current"],
                dictionary["operating_mode"]
            )
        else:
            return dictionary


class Servo:
    min_position = 0
    max_position = 0
    target_position = 0
    current_position = -1  # set by the servo handler

    def __init__(self, min_position, max_position, min_angle, max_angle, operating_mode, profile_velocity=100,
                 profile_acceleration=50, p=1000, i=600, d=500, offset=0, goal_current=None):
        self.p = p
        self.i = i
        self.d = d
        self.max_angle = max_angle
        self.min_angle = min_angle
        self.min_position = min_position
        self.max_position = max_position
        self.operating_mode = operating_mode
        self.profile_velocity = profile_velocity
        self.profile_acceleration = profile_acceleration
        self.constant_offset = offset
        self.goal_current = goal_current
        self.unmodified_target_position = 0

    def __str__(self):
        return "p={} i={} d={} max_angle={} min_angle={} min_position={} max_position={} operating_mode={} " \
               "profile_velocity={} profile_acceleration={} offset={} goal_current={}" \
            .format(self.p, self.i, self.d, self.max_angle, self.min_angle, self.min_position, self.max_position,
                    self.operating_mode, self.profile_velocity, self.profile_acceleration, self.constant_offset,
                    self.goal_current)

    # updates the target position of this servo
    def set_target_position_from_angle(self, angle, all_angles=None):
        if angle > self.max_angle:
            logging.debug("input angle is bigger than max input angle")
        if angle < self.min_angle:
            logging.debug("input angle is lower than min input angle")

        self.target_position = int(np.rint(np.interp(angle, [self.min_angle, self.max_angle],
                                                     [self.min_position, self.max_position])))
        self.target_position += self.constant_offset
        self.unmodified_target_position = self.target_position

    def get_angle_from_position(self, position):
        if position is None:
            return 0
        if position > self.max_position:
            logging.debug("input position is larger than max position")
        if position < self.min_position:
            logging.debug("input position is smaller than min position")

        if self.min_position > self.max_position:
            return np.interp(position, [self.max_position, self.min_position],
                             [self.max_angle, self.min_angle])
        else:
            return np.interp(position, [self.min_position, self.max_position],
                             [self.min_angle, self.max_angle])


class AdjustmentDirections(Enum):
    UP = 1
    DOWN = 2


class ServoWithOffsetFunction(Servo):

    def __init__(self, min_position, max_position, min_angle, max_angle, operating_mode, offset_function_up,
                 offset_function_down, profile_velocity=100, profile_acceleration=50, p=1000, i=600, d=500,
                 offset=0, goal_current=None):
        super().__init__(min_position, max_position, min_angle, max_angle, operating_mode, profile_velocity,
                         profile_acceleration, p, i, d, offset, goal_current)
        if offset_function_up is None or offset_function_down is None:
            raise ValueError("Should have gotten offset functions")
        self.offset_function_up = offset_function_up
        self.offset_function_down = offset_function_down
        self._last_commanded_angle = 0.0
        self._last_adjustment_direction = AdjustmentDirections.UP
        self.unmodified_target_position = 0

    def set_target_position_from_angle(self, angle, all_angles=None):
        super().set_target_position_from_angle(angle)
        if all_angles is None:
            return

        direction = self._get_adjustment_direction(angle)
        if direction == AdjustmentDirections.UP:
            self.target_position -= self.offset_function_up(all_angles)
        else:
            self.target_position -= self.offset_function_down(all_angles)
        self._last_commanded_angle = angle

    def _get_adjustment_direction(self, angle):
        if angle == self._last_commanded_angle:
            return self._last_adjustment_direction
        elif angle > self._last_commanded_angle:
            self._last_adjustment_direction = AdjustmentDirections.UP
            return AdjustmentDirections.UP
        else:
            self._last_adjustment_direction = AdjustmentDirections.DOWN
            return AdjustmentDirections.DOWN


class Servo2(Servo):

    def __init__(self, min_position, max_position, min_angle, max_angle, operating_mode, profile_velocity=100,
                 profile_acceleration=50, p=1000, i=600, d=500, offset=0, goal_current=None, dynamic_offsets=None):
        super().__init__(min_position, max_position, min_angle, max_angle, operating_mode, profile_velocity,
                         profile_acceleration, p, i, d, offset, goal_current)
        self.dynamic_offsets = dynamic_offsets
        self.dynamic_offsets_info = self._get_dynamic_offsets_info(dynamic_offsets)
        self.angle2_lookup_min = self.dynamic_offsets_info['min']
        self.angle2_lookup_max = self.dynamic_offsets_info['max']

    def set_target_position_from_angle(self, angle, all_angles=None):
        super().set_target_position_from_angle(angle)
        self.target_position -= self._get_dynamic_offsets(all_angles)

    @staticmethod
    def _get_dynamic_offsets_info(dynamic_offsets):
        info = {}
        angle2_values = []

        for angle2 in dynamic_offsets:
            angle2_values.append(round(float(angle2), 1))
            servo3_dict = dynamic_offsets[angle2]
            angle3_values = []
            for angle3 in servo3_dict:
                angle3_values.append(round(float(angle3), 1))
            sorted_angle3_values = sorted(angle3_values)
            info[angle2] = {}
            info[angle2]["min"] = sorted_angle3_values[0]
            info[angle2]["max"] = sorted_angle3_values[-1]

        sorted_angle2_values = sorted(angle2_values)
        info["min"] = sorted_angle2_values[0]
        info["max"] = sorted_angle2_values[-1]

        return info

    def _get_dynamic_offsets(self, all_angles):
        if self.dynamic_offsets is None or all_angles is None:
            return 0
        angle2rounded = round(all_angles[2] / pi, 1)
        angle2lookup = str(angle2rounded)
        if angle2rounded < self.angle2_lookup_min:
            angle2lookup = str(self.angle2_lookup_min)
        elif angle2rounded > self.angle2_lookup_max:
            angle2lookup = str(self.angle2_lookup_max)

        angle3rounded = round(all_angles[3] / pi, 1)
        angle3_lookup = str(angle3rounded)
        if angle3rounded < self.dynamic_offsets_info[angle2lookup]['min']:
            angle3_lookup = str(self.dynamic_offsets_info[angle2lookup]['min'])
        if angle3rounded > self.dynamic_offsets_info[angle2lookup]['max']:
            angle3_lookup = str(self.dynamic_offsets_info[angle2lookup]['max'])

        return self.dynamic_offsets[angle2lookup][angle3_lookup]


class Servo3(Servo):

    def __init__(self, min_position, max_position, min_angle, max_angle, operating_mode, profile_velocity=100,
                 profile_acceleration=50, p=1000, i=600, d=500, offset=0, goal_current=None, dynamic_offsets=None):
        super().__init__(min_position, max_position, min_angle, max_angle, operating_mode,
                         profile_velocity, profile_acceleration, p, i, d, offset, goal_current)
        self.dynamic_offsets = dynamic_offsets
        self.angle_lookup_min, self.angle_lookup_max = self._get_dynamic_offsets_info(dynamic_offsets)

    def set_target_position_from_angle(self, angle, all_angles=None):
        super().set_target_position_from_angle(angle)
        self.target_position -= self._get_dynamic_offsets(all_angles)

    @staticmethod
    def _get_dynamic_offsets_info(dynamic_offset):
        angles = []
        for angle in dynamic_offset:
            angles.append(round(float(angle), 1))
        sorted_angles = sorted(angles)
        min_angle = sorted_angles[0]
        max_angle = sorted_angles[-1]
        return min_angle, max_angle

    def _get_dynamic_offsets(self, all_angles):
        if self.dynamic_offsets is None or all_angles is None:
            return 0
        angle_rounded = round((all_angles[2] + all_angles[3])/pi, 1)
        angle_lookup = str(angle_rounded)
        if angle_rounded < self.angle_lookup_min:
            angle_lookup = str(self.angle_lookup_min)
        if angle_rounded > self.angle_lookup_max:
            angle_lookup = str(self.angle_lookup_max)
        return self.dynamic_offsets[angle_lookup]
