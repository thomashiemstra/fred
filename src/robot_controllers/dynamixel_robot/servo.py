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
                "max_angle": o.max_angle/pi,
                "min_angle": o.min_angle/pi,
                "profile_velocity": o.profile_velocity,
                "profile_acceleration": o.profile_acceleration,
                "p": o.p,
                "i": o.i,
                "d": o.d,
                "offset": o.offset,
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
                dictionary["max_angle"],
                dictionary["min_angle"],
                dictionary["profile_velocity"],
                dictionary["profile_acceleration"],
                dictionary["p"],
                dictionary["i"],
                dictionary["d"],
                dictionary["offset"]
            )
        else:
            return dictionary


class Servo:
    min_position = 0
    max_position = 0
    target_position = 0
    current_position = -1  # set by the servo handler

    def __init__(self, min_position, max_position, min_angle, max_angle, profile_velocity=100, profile_acceleration=50,
                 p=1000, i=600, d=500, offset=0, goal_current=300):
        self.p = p
        self.i = i
        self.d = d
        self.max_angle = max_angle
        self.min_angle = min_angle
        self.min_position = min_position
        self.max_position = max_position
        self.profile_velocity = profile_velocity
        self.profile_acceleration = profile_acceleration
        self.offset = offset
        self.goal_current = goal_current

    # updates the target position of this servo
    def set_target_position_from_angle(self, angle):
        if angle > self.max_angle:
            logging.debug("input angle is bigger than max input angle")
        if angle < self.min_angle:
            logging.debug("input angle is lower than min input angle")

        self.target_position = int(np.rint(np.interp(angle, [self.min_angle, self.max_angle],
                                                     [self.min_position, self.max_position])))
        self.target_position += self.offset

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
