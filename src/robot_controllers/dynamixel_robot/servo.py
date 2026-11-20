from __future__ import annotations

from collections.abc import Callable, Iterable
from math import ceil
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import logging
from numpy import pi

from src.robot_controllers.dynamixel_robot.dynamixel_x_config import MAX_PROFILE_TIME, \
    POSITIONS_PER_SECOND_PER_VELOCITY_UNIT

if TYPE_CHECKING:
    from src.robot_controllers.dynamixel_robot.static_compensation import StaticCompensation


class Servo:
    min_position = 0
    max_position = 0
    target_position = 0
    current_position = -1  # set by the servo handler

    def __init__(self, min_position: int, max_position: int, min_angle: float, max_angle: float, operating_mode: int,
                 profile_velocity: int = 100, profile_acceleration: int = 50, p: int = 1000, i: int = 600,
                 d: int = 500, offset: int = 0, goal_current: int | None = None,
                 compensation: StaticCompensation | None = None, move_time: int | None = None, ff1: int = 0) -> None:
        """
        :param compensation: a StaticCompensation, the positions it gives for the commanded angles of all joints
            and the P gain of this servo are added to every target position (see static_compensation.py)
        :param move_time: ms the servo gets for each small streamed goal step, None for the velocity-based profile;
            profile_velocity and profile_acceleration then limit a goal that is far away
        :param ff1: velocity feedforward gain of the position controller
        """
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
        self.compensation = compensation
        self.move_time = move_time
        self.ff1 = ff1
        self.compensation_positions = 0  # the positions the compensation added to the last target

    def __str__(self) -> str:
        return "p={} i={} d={} max_angle={} min_angle={} min_position={} max_position={} operating_mode={} " \
               "profile_velocity={} profile_acceleration={} offset={} goal_current={} compensation={} move_time={} " \
               "ff1={}" \
            .format(self.p, self.i, self.d, self.max_angle, self.min_angle, self.min_position, self.max_position,
                    self.operating_mode, self.profile_velocity, self.profile_acceleration, self.constant_offset,
                    self.goal_current, self.compensation, self.move_time, self.ff1)

    @property
    def time_based(self) -> bool:
        """whether the servo uses the time-based profile"""
        return self.move_time is not None

    def profile_registers(self) -> tuple[int, int]:
        """the values of the profile velocity and the profile acceleration register the servo holds between moves"""
        if self.move_time is not None:
            return self.move_time, 0
        return self.profile_velocity, self.profile_acceleration

    def goal_timing(self, distance: int, speed_factor: float = 1.0) -> tuple[int, int]:
        """
        the time of the move and the acceleration time in ms to send with a goal of a time-based servo
        :param distance: the positions from the previous goal, the sign doesn't matter
        :param speed_factor: the part of the top speed (profile_velocity) the move may use, above 0 up to 1
        :raises ValueError: if the servo is not time-based or the speed factor is not in (0, 1]
        """
        if self.move_time is None:
            raise ValueError('goal_timing is only for a servo with a move_time')
        if not 0 < speed_factor <= 1:
            raise ValueError('speed_factor should be above 0 and at most 1, not {}'.format(speed_factor))
        if self.profile_velocity == 0:  # 0 means no speed limit
            return self.move_time, 0
        velocity = self.profile_velocity * speed_factor  # velocity units
        # ms the move takes at the top speed
        cruise = ceil(1000 * abs(distance) / (velocity * POSITIONS_PER_SECOND_PER_VELOCITY_UNIT))
        if cruise <= self.move_time:  # a streamed step: the servo interpolates over the move time
            return self.move_time, 0
        # a goal that is far away: a trapezoid with the top speed and the acceleration of the yaml, as the
        # velocity-based profile gives; the servo manual: acceleration time in ms = 64 * profile velocity / profile
        # acceleration
        ramp = round(64 * velocity / self.profile_acceleration) if self.profile_acceleration > 0 else 0
        ramp = min(ramp, cruise)
        total = min(cruise + ramp, MAX_PROFILE_TIME)
        return total, min(ramp, total // 2)

    # updates the target position of this servo
    def set_target_position_from_angle(self, angle: float, all_angles: np.ndarray | None = None) -> None:
        """
        :param angle: the angle to move to
        :param all_angles: the commanded angles of all joints, indexed by servo id; with a compensation the
            positions for the static load of that pose at the P gain of this servo are added to the target
        """
        if angle > self.max_angle:
            logging.debug("input angle is bigger than max input angle")
        if angle < self.min_angle:
            logging.debug("input angle is lower than min input angle")

        self.target_position = int(np.rint(np.interp(angle, [self.min_angle, self.max_angle],
                                                     [self.min_position, self.max_position])))
        self.target_position += self.constant_offset

        self.compensation_positions = 0
        if self.compensation is not None and all_angles is not None:
            self.compensation_positions = self.compensation.positions(all_angles, self.p)
            self.target_position += self.compensation_positions

    def calibrated_range(self) -> tuple[int, int]:
        """the lowest and the highest raw position of the calibrated range of this servo, offset included"""
        return (min(self.min_position, self.max_position) + self.constant_offset,
                max(self.min_position, self.max_position) + self.constant_offset)

    def clamp_position(self, position: float) -> int:
        """clamp the position to the calibrated range of this servo, including the constant offset"""
        lower, upper = self.calibrated_range()
        return int(np.clip(position, lower, upper))

    def clamp_target_position(self) -> int:
        """
        the target position with the uncompensated part clamped to the calibrated range, so the compensation is
        kept at the ends of the range (the goal position may exceed the calibrated range by the compensation, the
        servo handler clamps it to the physical range of the servo)
        """
        return self.clamp_position(self.target_position - self.compensation_positions) + self.compensation_positions

    def get_angle_from_position(self, position: int | None) -> float:
        if position is None:
            return 0
        if position > self.max_position:
            logging.debug("input position is larger than max position")
        if position < self.min_position:
            logging.debug("input position is smaller than min position")

        if self.min_position > self.max_position:
            return np.interp(position - self.constant_offset, [self.max_position, self.min_position],  # type: ignore[return-value]  # scalar in, scalar out
                             [self.max_angle, self.min_angle])
        else:
            return np.interp(position - self.constant_offset, [self.min_position, self.max_position],  # type: ignore[return-value]  # scalar in, scalar out
                             [self.min_angle, self.max_angle])

    def degrees_at_position(self, position: int) -> float:
        """
        the angle of the servo at the raw position in degrees, with the calibration of the servo. Unlike
        get_angle_from_position this does not clamp to the calibrated range, a position outside of it gives
        the angle the servo is really at.
        """
        angle_per_position = (self.max_angle - self.min_angle) / (self.max_position - self.min_position)
        angle = self.min_angle + (position - self.constant_offset - self.min_position) * angle_per_position
        return float(np.degrees(angle))


class AdjustmentDirections(Enum):
    UP = 1
    DOWN = 2


class ServoWithOffsetFunction(Servo):

    def __init__(self, min_position: int, max_position: int, min_angle: float, max_angle: float, operating_mode: int,
                 offset_function_up: Callable[[np.ndarray], int], offset_function_down: Callable[[np.ndarray], int],
                 profile_velocity: int = 100, profile_acceleration: int = 50, p: int = 1000, i: int = 600,
                 d: int = 500, offset: int = 0, goal_current: int | None = None, move_time: int | None = None,
                 ff1: int = 0) -> None:
        super().__init__(min_position, max_position, min_angle, max_angle, operating_mode, profile_velocity,
                         profile_acceleration, p, i, d, offset, goal_current, move_time=move_time, ff1=ff1)
        if offset_function_up is None or offset_function_down is None:
            raise ValueError("Should have gotten offset functions")
        self.offset_function_up = offset_function_up
        self.offset_function_down = offset_function_down
        self._last_commanded_angle = 0.0
        self._last_adjustment_direction = AdjustmentDirections.UP

    def set_target_position_from_angle(self, angle: float, all_angles: np.ndarray | None = None) -> None:
        super().set_target_position_from_angle(angle)
        if all_angles is None:
            return

        direction = self._get_adjustment_direction(angle)
        if direction == AdjustmentDirections.UP:
            self.target_position -= self.offset_function_up(all_angles)
        else:
            self.target_position -= self.offset_function_down(all_angles)
        self._last_commanded_angle = angle

    def _get_adjustment_direction(self, angle: float) -> AdjustmentDirections:
        if angle == self._last_commanded_angle:
            return self._last_adjustment_direction
        elif angle > self._last_commanded_angle:
            self._last_adjustment_direction = AdjustmentDirections.UP
            return AdjustmentDirections.UP
        else:
            self._last_adjustment_direction = AdjustmentDirections.DOWN
            return AdjustmentDirections.DOWN


def _nearest_key(angle: float, table: Iterable[str]) -> str:
    """
    the key of the table (an angle in units of pi as text) closest to the angle, the first one for a tie. An angle
    outside the table gets the key at the end of the table, one between two keys the nearer of them.
    """
    return min(table, key=lambda key: (abs(float(key) - angle), float(key)))


class Servo2(Servo):

    def __init__(self, min_position: int, max_position: int, min_angle: float, max_angle: float, operating_mode: int,
                 profile_velocity: int = 100, profile_acceleration: int = 50, p: int = 1000, i: int = 600,
                 d: int = 500, offset: int = 0, goal_current: int | None = None,
                 dynamic_offsets: dict[str, dict[str, int]] | None = None, move_time: int | None = None,
                 ff1: int = 0) -> None:
        super().__init__(min_position, max_position, min_angle, max_angle, operating_mode, profile_velocity,
                         profile_acceleration, p, i, d, offset, goal_current, move_time=move_time, ff1=ff1)
        if dynamic_offsets is None:
            raise ValueError('Servo2 needs the dynamic offsets')
        self.dynamic_offsets = dynamic_offsets

    def set_target_position_from_angle(self, angle: float, all_angles: np.ndarray | None = None) -> None:
        super().set_target_position_from_angle(angle)
        self.target_position -= self._get_dynamic_offsets(all_angles)

    def _get_dynamic_offsets(self, all_angles: np.ndarray | None) -> int:
        if self.dynamic_offsets is None or all_angles is None:
            return 0
        angle2_lookup = _nearest_key(round(all_angles[2] / pi, 1), self.dynamic_offsets)
        angle3_lookup = _nearest_key(round(all_angles[3] / pi, 1), self.dynamic_offsets[angle2_lookup])
        return self.dynamic_offsets[angle2_lookup][angle3_lookup]


class Servo3(Servo):

    def __init__(self, min_position: int, max_position: int, min_angle: float, max_angle: float, operating_mode: int,
                 profile_velocity: int = 100, profile_acceleration: int = 50, p: int = 1000, i: int = 600,
                 d: int = 500, offset: int = 0, goal_current: int | None = None,
                 dynamic_offsets: dict[str, int] | None = None, move_time: int | None = None, ff1: int = 0) -> None:
        super().__init__(min_position, max_position, min_angle, max_angle, operating_mode,
                         profile_velocity, profile_acceleration, p, i, d, offset, goal_current,
                         move_time=move_time, ff1=ff1)
        if dynamic_offsets is None:
            raise ValueError('Servo3 needs the dynamic offsets')
        self.dynamic_offsets = dynamic_offsets

    def set_target_position_from_angle(self, angle: float, all_angles: np.ndarray | None = None) -> None:
        super().set_target_position_from_angle(angle)
        self.target_position -= self._get_dynamic_offsets(all_angles)

    def _get_dynamic_offsets(self, all_angles: np.ndarray | None) -> int:
        if self.dynamic_offsets is None or all_angles is None:
            return 0
        angle_lookup = _nearest_key(round((all_angles[2] + all_angles[3]) / pi, 1), self.dynamic_offsets)
        return self.dynamic_offsets[angle_lookup]
