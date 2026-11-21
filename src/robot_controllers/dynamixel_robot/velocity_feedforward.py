"""
Velocity lead: the positions a servo has to be commanded ahead of the target of an angle to give its motor the drive
it needs to turn at the commanded speed.

The servos run a P-only position loop, pwm = P / 128 * (goal - present). A turning motor needs a drive in proportion
to its speed (its back voltage), and the only source of drive is P times the error, so a moving joint trails its
trajectory by an error that grows with the speed. The lead shifts every goal by exactly the error that produces the
drive the motor needs at the commanded speed, lead in positions = coefficient * velocity * 128 / P, with the velocity
in degrees per second of the joint. It is pure feedforward from the commanded motion and zero when the joint stands
still.

The coefficient is in pwm units per degree/s, like the coefficients of the static compensation, so it stays valid when
P is retuned: only the conversion to positions follows the P the servo runs with. Friction is not in it yet, it will
be a later term on the same commanded velocity. A common delay of the servo's own interpolation between goals is
deliberately not cancelled: it is the same on every joint, so it makes the arm a little late without bending the path.
"""
import numpy as np

from src.robot_controllers.dynamixel_robot.dynamixel_x_config import PWM_PER_POSITION_PER_P


class VelocityFeedforward:
    """
    The drive a joint needs for its commanded velocity, and the positions to add to the target of the servo to
    produce it.

    :param pwm_per_degree_per_second: the coefficient in pwm units per degree/s of the joint, 0 or more
    :param max_positions: the lead is clipped to this magnitude in positions (not pwm units), so a wrong coefficient
        or a fast move can't command a big jump
    :raises ValueError: if the coefficient is negative or max_positions is not a positive whole number
    """

    def __init__(self, pwm_per_degree_per_second: float, max_positions: int) -> None:
        if pwm_per_degree_per_second < 0:
            raise ValueError('pwm_per_degree_per_second should be 0 or more, got {}'.format(
                pwm_per_degree_per_second))
        if isinstance(max_positions, bool) or int(max_positions) != max_positions or max_positions <= 0:
            raise ValueError('max_positions should be a positive whole number, got {}'.format(max_positions))
        self.pwm_per_degree_per_second = float(pwm_per_degree_per_second)
        self.max_positions = int(max_positions)

    def drive(self, velocity: float) -> float:
        """
        the pwm the motor needs to turn at the velocity
        :param velocity: the commanded velocity of the joint in radians per second
        :return: the pwm in the direction of the velocity, not rounded
        """
        return self.pwm_per_degree_per_second * float(np.degrees(velocity))

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
                and self.max_positions == other.max_positions)

    def __repr__(self) -> str:
        return 'VelocityFeedforward(pwm_per_degree_per_second={}, max_positions={})'.format(
            self.pwm_per_degree_per_second, self.max_positions)
