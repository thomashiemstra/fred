from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.global_constants import recommended_max_servo_speed
import numpy as np

if TYPE_CHECKING:
    from src.robot_controllers.dynamixel_robot.servo import Servo

logger = logging.getLogger(__name__)

# the range of servo 1 and 2 in which the arm is not below the 0 degree line, in degrees. These are the angles of the
# raw positions 0 and 2500 (servo 1: -1370 and 1130 positions from its 0 degree position, servo 2: -457 and 2043)
# with the offsets of servo_config.yaml at the time: a new calibration moves the raw positions, not these angles
SERVO_1_SAFE_ANGLES = (-120.41015625, 99.31640625)
SERVO_2_SAFE_ANGLES = (-40.166015625, 179.560546875)
# float noise of the conversion, the positions are whole numbers 0.088 degrees apart
ANGLE_TOLERANCE = 1e-6


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


def _check_in_safe_range(servo_id: int, servo: Servo, position: int, safe_angles: tuple[float, float]) -> bool:
    angle = position_to_degrees(servo, position)
    lowest, highest = safe_angles
    if lowest - ANGLE_TOLERANCE <= angle <= highest + ANGLE_TOLERANCE:
        return True
    logger.error('DANGER servo %d is at raw position %d (%.1f degrees), outside of %.1f to %.1f degrees, it might be '
                 'below the 0 degree line. Move the arm by hand into that range and try again',
                 servo_id, position, angle, lowest, highest)
    return False


def servo_1_check(positions: np.ndarray, servo: Servo) -> bool:
    """False, with a logged error, when servo 1 is outside of its safe range. Never asks for input."""
    return _check_in_safe_range(1, servo, int(positions[1]), SERVO_1_SAFE_ANGLES)


def servo_2_check(positions: np.ndarray, servo: Servo) -> bool:
    """False, with a logged error, when servo 2 is outside of its safe range. Never asks for input."""
    return _check_in_safe_range(2, servo, int(positions[2]), SERVO_2_SAFE_ANGLES)
