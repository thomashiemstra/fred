from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.global_constants import recommended_max_servo_speed
import numpy as np

if TYPE_CHECKING:
    from src.robot_controllers.dynamixel_robot.servo import Servo

logger = logging.getLogger(__name__)

def get_recommended_wait_time(current_angles: np.ndarray, new_angles: np.ndarray) -> float:
    recommended_time = 0
    for current_angle, new_angle in zip(current_angles, new_angles):
        delta_angle = current_angle - new_angle
        if delta_angle == 0:
            continue
        delta_angle = np.abs(delta_angle)
        time = delta_angle / recommended_max_servo_speed
        recommended_time = np.maximum(time, recommended_time)

    return recommended_time


def position_to_degrees(servo: Servo, position: int) -> float:
    """
    the angle of the servo at the raw position in degrees, with the calibration of the servo. Unlike
    Servo.get_angle_from_position this does not clamp to the calibrated range, a position outside of it gives
    the angle the servo is really at.
    """
    angle_per_position = (servo.max_angle - servo.min_angle) / (servo.max_position - servo.min_position)
    angle = servo.min_angle + (position - servo.constant_offset - servo.min_position) * angle_per_position
    return float(np.degrees(angle))
