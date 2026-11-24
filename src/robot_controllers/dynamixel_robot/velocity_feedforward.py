"""
Velocity lead: the positions a servo has to be commanded ahead of the target of an angle to give its motor the drive
it needs to turn at the commanded speed.

The servos run a P-only position loop, pwm = P / 128 * (goal - present). A turning motor needs a drive in proportion
to its speed (its back voltage), and the only source of drive is P times the error, so a moving joint trails its
trajectory by an error that grows with the speed. The lead shifts every goal by exactly the error that produces the
drive the motor needs at the commanded speed, lead in positions = drive * 128 / P. It is pure feedforward from the
commanded motion and zero when the joint stands still.

The drive has two terms, both on the commanded velocity of the joint:
    drive = coefficient * velocity + pwm_for_friction * sign(velocity)
The first grows with the speed (the back voltage of the motor), the coefficient is in pwm units per degree/s of the
joint. The second does not depend on the speed: friction needs a fixed extra drive to keep the joint sliding, in the
direction of the motion, pwm_for_friction in pwm units (the sign is smoothed at the lowest speeds, see
FRICTION_FULL_SPEED). Both are in pwm units, like the coefficients of the static compensation, so they stay valid
when P is retuned: only the conversion to positions follows the P the servo runs with. The friction term is zero when
the joint stands still, so it can not push a joint that rests off its goal: a joint that is not commanded to move
gets no lead at all. A common delay of the servo's own interpolation between goals is deliberately not cancelled: it
is the same on every joint, so it makes the arm a little late without bending the path.
"""
import numpy as np

from src.robot_controllers.dynamixel_robot.dynamixel_x_config import PWM_PER_POSITION_PER_P

# degrees per second: below this speed the friction drive is in proportion to the speed, at and above it it is whole.
# A hard sign would make the goal jump by twice the friction every time the joint reverses or the commanded speed
# wobbles around zero
FRICTION_FULL_SPEED = 1.0


class VelocityFeedforward:
    """
    The drive a joint needs for its commanded velocity, and the positions to add to the target of the servo to
    produce it.

    :param pwm_per_degree_per_second: the coefficient in pwm units per degree/s of the joint, 0 or more
    :param max_positions: the lead is clipped to this magnitude in positions (not pwm units), so a wrong coefficient
        or a fast move can't command a big jump
    :param pwm_for_friction: the drive in pwm units the motor needs to keep the joint moving against friction, in the
        direction of the motion whatever the speed, 0 or more; it is whole from FRICTION_FULL_SPEED degrees/s and in
        proportion to the speed below it
    :raises ValueError: if the coefficient or the friction drive is negative (or not a number) or max_positions is not
        a positive whole number
    """

    def __init__(self, pwm_per_degree_per_second: float, max_positions: int, pwm_for_friction: float = 0.0) -> None:
        if pwm_per_degree_per_second < 0:
            raise ValueError('pwm_per_degree_per_second should be 0 or more, got {}'.format(
                pwm_per_degree_per_second))
        if isinstance(max_positions, bool) or int(max_positions) != max_positions or max_positions <= 0:
            raise ValueError('max_positions should be a positive whole number, got {}'.format(max_positions))
        if not pwm_for_friction >= 0:  # also refuses nan
            raise ValueError('pwm_for_friction should be 0 or more, got {}'.format(pwm_for_friction))
        self.pwm_per_degree_per_second = float(pwm_per_degree_per_second)
        self.max_positions = int(max_positions)
        self.pwm_for_friction = float(pwm_for_friction)

    def drive(self, velocity: float) -> float:
        """
        the pwm the motor needs to turn at the velocity: the coefficient times the speed in degrees/s, plus the
        friction drive, which is whole from FRICTION_FULL_SPEED degrees/s and in proportion to the speed below it
        (0 at rest)
        :param velocity: the commanded velocity of the joint in radians per second
        :return: the pwm in the direction of the velocity, not rounded
        """
        degrees_per_second = float(np.degrees(velocity))
        friction = self.pwm_for_friction * float(np.clip(degrees_per_second / FRICTION_FULL_SPEED, -1.0, 1.0))
        return self.pwm_per_degree_per_second * degrees_per_second + friction

    def positions(self, velocity: float, p: float) -> float:
        """
        the positions to add to the target for the commanded velocity: the drive divided by the pwm the servo gives
        per position of error, P / 128
        :param velocity: the commanded velocity of the joint in radians per second
        :param p: the P gain the servo runs with
        :return: positions with the sign of the velocity, not rounded, clipped to +- max_positions; 0.0 when p is not
            positive
        """
        if p <= 0:
            return 0.0
        positions = self.drive(velocity) / (p * PWM_PER_POSITION_PER_P)
        return float(np.clip(positions, -self.max_positions, self.max_positions))

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, VelocityFeedforward)
                and self.pwm_per_degree_per_second == other.pwm_per_degree_per_second
                and self.max_positions == other.max_positions
                and self.pwm_for_friction == other.pwm_for_friction)

    def __repr__(self) -> str:
        return 'VelocityFeedforward(pwm_per_degree_per_second={}, max_positions={}, pwm_for_friction={})'.format(
            self.pwm_per_degree_per_second, self.max_positions, self.pwm_for_friction)
