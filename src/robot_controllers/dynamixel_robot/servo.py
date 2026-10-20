from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import logging
from numpy import pi

from src.robot_controllers.dynamixel_robot.static_compensation import UP, DOWN

if TYPE_CHECKING:
    from src.robot_controllers.dynamixel_robot.static_compensation import StaticCompensation


class Servo:
    min_position = 0
    max_position = 0
    target_position = 0
    current_position = -1  # set by the servo handler
    DIRECTION_DEADBAND_POSITIONS = 3

    def __init__(self, min_position: int, max_position: int, min_angle: float, max_angle: float, operating_mode: int,
                 profile_velocity: int = 100, profile_acceleration: int = 50, p: int = 1000, i: int = 600,
                 d: int = 500, offset: int = 0, goal_current: int | None = None,
                 compensation: StaticCompensation | None = None) -> None:
        """
        :param compensation: a StaticCompensation, the positions it gives for the commanded angles of all joints
            are added to every target position (see static_compensation.py)
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
        self.compensation_positions = 0  # the positions the compensation added to the last target
        self._direction: str | None = None
        self._extreme_angle: float | None = None  # the furthest commanded angle in the current direction
        # the direction only turns around when the commanded angle has come back more than this from the furthest
        # point, so a target that dithers around a value doesn't flip the up/down compensation every step
        positions_per_radian = abs(max_position - min_position) / (max_angle - min_angle) if max_angle != min_angle else 0
        self.direction_deadband = self.DIRECTION_DEADBAND_POSITIONS / positions_per_radian if positions_per_radian else 0.0

    def __str__(self) -> str:
        return "p={} i={} d={} max_angle={} min_angle={} min_position={} max_position={} operating_mode={} " \
               "profile_velocity={} profile_acceleration={} offset={} goal_current={} compensation={}" \
            .format(self.p, self.i, self.d, self.max_angle, self.min_angle, self.min_position, self.max_position,
                    self.operating_mode, self.profile_velocity, self.profile_acceleration, self.constant_offset,
                    self.goal_current, self.compensation)

    # updates the target position of this servo
    def set_target_position_from_angle(self, angle: float, all_angles: np.ndarray | None = None) -> None:
        """
        :param angle: the angle to move to
        :param all_angles: the commanded angles of all joints, indexed by servo id; with a compensation the
            positions for the static load of that pose are added to the target
        """
        if angle > self.max_angle:
            logging.debug("input angle is bigger than max input angle")
        if angle < self.min_angle:
            logging.debug("input angle is lower than min input angle")

        self.target_position = int(np.rint(np.interp(angle, [self.min_angle, self.max_angle],
                                                     [self.min_position, self.max_position])))
        self.target_position += self.constant_offset

        direction = self._update_direction(angle)
        self.compensation_positions = 0
        if self.compensation is not None and all_angles is not None:
            self.compensation_positions = self.compensation.positions(all_angles, direction)
            self.target_position += self.compensation_positions

    def _update_direction(self, angle: float) -> str | None:
        """
        the direction the commanded angle moves in: UP, DOWN or None before it first moved. The direction only turns
        around when the angle has come back more than the deadband from the furthest point it reached.
        """
        if self._extreme_angle is None:
            self._extreme_angle = angle
            return None
        if self._direction == UP and angle >= self._extreme_angle or \
                self._direction == DOWN and angle <= self._extreme_angle:
            self._extreme_angle = angle
        elif angle > self._extreme_angle + self.direction_deadband:
            self._direction, self._extreme_angle = UP, angle
        elif angle < self._extreme_angle - self.direction_deadband:
            self._direction, self._extreme_angle = DOWN, angle
        return self._direction

    def clamp_position(self, position: float) -> int:
        """clamp the position to the calibrated range of this servo, including the constant offset"""
        lower = min(self.min_position, self.max_position) + self.constant_offset
        upper = max(self.min_position, self.max_position) + self.constant_offset
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


class AdjustmentDirections(Enum):
    UP = 1
    DOWN = 2


class ServoWithOffsetFunction(Servo):

    def __init__(self, min_position: int, max_position: int, min_angle: float, max_angle: float, operating_mode: int,
                 offset_function_up: Callable[[np.ndarray], int], offset_function_down: Callable[[np.ndarray], int],
                 profile_velocity: int = 100, profile_acceleration: int = 50, p: int = 1000, i: int = 600,
                 d: int = 500, offset: int = 0, goal_current: int | None = None) -> None:
        super().__init__(min_position, max_position, min_angle, max_angle, operating_mode, profile_velocity,
                         profile_acceleration, p, i, d, offset, goal_current)
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


class Servo2(Servo):

    def __init__(self, min_position: int, max_position: int, min_angle: float, max_angle: float, operating_mode: int,
                 profile_velocity: int = 100, profile_acceleration: int = 50, p: int = 1000, i: int = 600,
                 d: int = 500, offset: int = 0, goal_current: int | None = None,
                 dynamic_offsets: dict[str, dict[str, int]] | None = None) -> None:
        super().__init__(min_position, max_position, min_angle, max_angle, operating_mode, profile_velocity,
                         profile_acceleration, p, i, d, offset, goal_current)
        if dynamic_offsets is None:
            raise ValueError('Servo2 needs the dynamic offsets')
        self.dynamic_offsets = dynamic_offsets
        self.dynamic_offsets_info = self._get_dynamic_offsets_info(dynamic_offsets)
        self.angle2_lookup_min = self.dynamic_offsets_info['min']
        self.angle2_lookup_max = self.dynamic_offsets_info['max']

    def set_target_position_from_angle(self, angle: float, all_angles: np.ndarray | None = None) -> None:
        super().set_target_position_from_angle(angle)
        self.target_position -= self._get_dynamic_offsets(all_angles)

    @staticmethod
    def _get_dynamic_offsets_info(dynamic_offsets: dict[str, dict[str, int]]) -> dict[str, Any]:
        info: dict[str, Any] = {}
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

    def _get_dynamic_offsets(self, all_angles: np.ndarray | None) -> int:
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

    def __init__(self, min_position: int, max_position: int, min_angle: float, max_angle: float, operating_mode: int,
                 profile_velocity: int = 100, profile_acceleration: int = 50, p: int = 1000, i: int = 600,
                 d: int = 500, offset: int = 0, goal_current: int | None = None,
                 dynamic_offsets: dict[str, int] | None = None) -> None:
        super().__init__(min_position, max_position, min_angle, max_angle, operating_mode,
                         profile_velocity, profile_acceleration, p, i, d, offset, goal_current)
        if dynamic_offsets is None:
            raise ValueError('Servo3 needs the dynamic offsets')
        self.dynamic_offsets = dynamic_offsets
        self.angle_lookup_min, self.angle_lookup_max = self._get_dynamic_offsets_info(dynamic_offsets)

    def set_target_position_from_angle(self, angle: float, all_angles: np.ndarray | None = None) -> None:
        super().set_target_position_from_angle(angle)
        self.target_position -= self._get_dynamic_offsets(all_angles)

    @staticmethod
    def _get_dynamic_offsets_info(dynamic_offset: dict[str, int]) -> tuple[float, float]:
        angles = []
        for angle in dynamic_offset:
            angles.append(round(float(angle), 1))
        sorted_angles = sorted(angles)
        min_angle = sorted_angles[0]
        max_angle = sorted_angles[-1]
        return min_angle, max_angle

    def _get_dynamic_offsets(self, all_angles: np.ndarray | None) -> int:
        if self.dynamic_offsets is None or all_angles is None:
            return 0
        angle_rounded = round((all_angles[2] + all_angles[3])/pi, 1)
        angle_lookup = str(angle_rounded)
        if angle_rounded < self.angle_lookup_min:
            angle_lookup = str(self.angle_lookup_min)
        if angle_rounded > self.angle_lookup_max:
            angle_lookup = str(self.angle_lookup_max)
        return self.dynamic_offsets[angle_lookup]
