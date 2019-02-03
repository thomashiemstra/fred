import numpy as np
import logging


class Servo:
    min_position = 0
    max_position = 0
    target_position = 0
    current_position = -1  # set by the servo handler

    def __init__(self, min_position, max_position, min_angle, max_angle, profile_velocity=0, profile_acceleration=0, p=1200, i=800, d=100):
        self.p = p
        self.i = i
        self.d = d
        self.max_angle = max_angle
        self.min_angle = min_angle
        self.min_position = min_position
        self.max_position = max_position
        self.profile_velocity = profile_velocity
        self.profile_acceleration = profile_acceleration

    # convert an angle to servo position
    def set_target_position_from_angle(self, angle):
        if angle > self.max_angle:
            logging.debug("input angle is bigger than max input angle")
        if angle < self.min_angle:
            logging.debug("input angle is lower than min input angle")

        self.target_position = int(np.rint(np.interp(angle, [self.min_angle, self.max_angle],
                                                    [self.min_position, self.max_position])))

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

