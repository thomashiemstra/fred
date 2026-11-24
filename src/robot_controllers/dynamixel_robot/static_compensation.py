"""
Static load compensation: the positions a servo has to be commanded beyond the target of an angle to hold the arm
there against gravity. See Handover Gravity and Spring Compensation.md. (A spring between the housings of servos 2 and
3 was removed from the robot on 23 October 2026 and its term from the code on 19 November 2026.)

This compensates gravity only, and gravity does not depend on the direction a joint moves in. A joint also rests off
its goal by its friction, with the sign of the direction it arrived from; that is not compensated here. The recorders
measure every pose from both directions so the fit can take their mean, which is the gravity load.

The holding load of every joint is a linear combination of a few sin/cos terms of the joint angles, written out as
one function per joint below (joint_2_drive to joint_5_drive), in the conventions of src/kinematics/kinematics.py:
angles are indexed by servo id in radians, angle 2 = 0 is the lower arm horizontal and phi = angle 2 + angle 3 is
the angle of the upper arm: pi/2 is the upper arm horizontal, 0 is it hanging straight down. The functions are the
only definition of the model: the names of the terms of a joint are the names of its coefficient arguments, and
term_values and regressor_row (the columns of the fit) are computed by calling the function with one coefficient at
1.0 and the others at 0.0.

The coefficients are the holding drive per joint in pwm units: the pwm the servo needs to hold the pose, positive
drives towards higher positions (885 is full drive). They are fitted by calibration/fit_static_load.py from the data
of calibration/record_static_load.py and live under `compensation` of the servo in servo_config.yaml. The servo
holds a load with a steady position error because its position loop has only a P gain: pwm = P / 128 * error. So
the positions to add to the goal follow from the drive with the P gain the servo runs with, positions = drive * 128
/ P, and a fit stays valid when the gains are retuned (a fit in positions would only be valid for the P it was
recorded at). A Servo with a compensation adds those positions to every target it gets together with all the angles
(DynamixelRobotController.move_servos passes them).

Terms per joint (h = mass times centre of mass of a link, a along the link, p perpendicular to it in the arm plane):
    constant: the same drive at every pose: friction that is not the same in both directions, or a term the model
              lacks (not an offset of the zero of the servo: the error is present minus goal in the frame of the
              servo itself, which a zero offset does not change)
    joint 2:  cos_q2 (lower arm h2a + everything beyond joint 3 times a2), sin_q2 (-h2p), and the terms of joint 3
    joint 3:  sin_phi (h3a), cos_phi (h3p), wrist_reach (the gripper, m6 c6)
    joint 5:  wrist_pitch (the gripper)
    joint 4:  wrist_roll (the gripper when the wrist is bent and rolled, small), off_axis (the weight that rolls with
              joint 4 off its axis whatever the bend of the wrist: the body of servo 5, mounted sideways)
"""
import inspect
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from src.robot_controllers.dynamixel_robot.dynamixel_x_config import PWM_PER_POSITION_PER_P


def joint_2_drive(q2: float, q3: float, q4: float, q5: float, constant: float = 0.0, cos_q2: float = 0.0,
                  sin_q2: float = 0.0, sin_phi: float = 0.0, cos_phi: float = 0.0, wrist_reach: float = 0.0) -> float:
    """the pwm that holds joint 2: the lower arm and everything beyond it, hanging on the shoulder"""
    phi = q2 + q3  # pi/2 is the upper arm horizontal
    gripper_reach = np.sin(phi) * np.cos(q5) + np.cos(phi) * np.cos(q4) * np.sin(q5)
    return float(constant
                 + cos_q2 * np.cos(q2)  # the lower arm's weight along the link, and all that is beyond joint 3 on a2
                 + sin_q2 * np.sin(q2)  # the lower arm's weight off the axis of the link
                 + sin_phi * np.sin(phi)  # the upper arm's weight along the link, and the wrist at its end
                 + cos_phi * np.cos(phi)  # its weight off the axis of the link
                 + wrist_reach * gripper_reach)  # the gripper beyond the wrist


def joint_3_drive(q2: float, q3: float, q4: float, q5: float, constant: float = 0.0, sin_phi: float = 0.0,
                  cos_phi: float = 0.0, wrist_reach: float = 0.0) -> float:
    """the pwm that holds joint 3: the upper arm and everything beyond it, hanging on the elbow"""
    phi = q2 + q3  # pi/2 is the upper arm horizontal
    gripper_reach = np.sin(phi) * np.cos(q5) + np.cos(phi) * np.cos(q4) * np.sin(q5)
    return float(constant
                 + sin_phi * np.sin(phi)  # the upper arm's weight along the link
                 + cos_phi * np.cos(phi)  # its weight off the axis of the link
                 + wrist_reach * gripper_reach)  # the gripper beyond the wrist


def joint_4_drive(q2: float, q3: float, q4: float, q5: float, constant: float = 0.0, wrist_roll: float = 0.0,
                  off_axis: float = 0.0) -> float:
    """
    the pwm that holds joint 4, the first wrist joint: the gripper, when the wrist is bent and rolled, and what rolls
    with the joint off its axis whatever the bend of the wrist
    """
    phi = q2 + q3  # pi/2 is the upper arm horizontal
    return float(constant
                 + wrist_roll * (np.sin(q4) * np.sin(q5) * np.sin(phi))  # the gripper's weight off the roll axis
                 # the weight that rolls with joint 4 and sits off the roll axis (the body of servo 5, mounted
                 # sideways), whatever the bend of the wrist. The part of it that sits to the side of the axis when the
                 # roll is at 0 follows cos(q4), which the constant takes over the range the roll is used in
                 + off_axis * (np.sin(q4) * np.sin(phi)))


def joint_5_drive(q2: float, q3: float, q4: float, q5: float, constant: float = 0.0,
                  wrist_pitch: float = 0.0) -> float:
    """the pwm that holds joint 5, the wrist pitch: the gripper hanging on it"""
    phi = q2 + q3  # pi/2 is the upper arm horizontal
    return float(constant
                 + wrist_pitch * (np.cos(phi) * np.sin(q5)
                                  + np.cos(q4) * np.cos(q5) * np.sin(phi)))  # the gripper's reach beyond the wrist


# the formula of every joint that has a static load, called with the angles q2 to q5 and the coefficients by name
JOINT_DRIVES: dict[int, Callable[..., float]] = {2: joint_2_drive, 3: joint_3_drive, 4: joint_4_drive,
                                                 5: joint_5_drive}
_ANGLE_ARGUMENTS = ('q2', 'q3', 'q4', 'q5')
# the terms of every joint in the order of regressor_row: the coefficient arguments of its formula
TERMS_OF_JOINT: dict[int, tuple[str, ...]] = {
    joint: tuple(name for name in inspect.signature(formula).parameters if name not in _ANGLE_ARGUMENTS)
    for joint, formula in JOINT_DRIVES.items()}


def term_names(joint: int) -> tuple[str, ...]:
    """the names of the terms of the holding load of the joint, in the order of regressor_row"""
    if joint not in TERMS_OF_JOINT:
        raise ValueError('joint {} has no static load terms, only joints {} have'.format(joint, sorted(TERMS_OF_JOINT)))
    return TERMS_OF_JOINT[joint]


def _joint_angles(angles: np.ndarray) -> tuple[float, float, float, float]:
    """q2 to q5 from the angles of all joints, indexed by servo id"""
    return float(angles[2]), float(angles[3]), float(angles[4]), float(angles[5])


def term_values(joint: int, angles: np.ndarray) -> dict[str, float]:
    """
    the value of every term of the joint at the angles: the formula of the joint with only that coefficient at 1.0
    :param joint: 2 to 5
    :param angles: the angles of all joints, indexed by servo id, in radians
    :return: mapping of term name to value
    """
    names = term_names(joint)
    q2, q3, q4, q5 = _joint_angles(angles)
    return {name: JOINT_DRIVES[joint](q2, q3, q4, q5, **{term: 1.0 if term == name else 0.0 for term in names})
            for name in names}


def regressor_row(joint: int, angles: np.ndarray, names: Sequence[str] | None = None) -> np.ndarray:
    """
    the values of the terms of the joint as a row of the regressor
    :param names: the terms to include, by default all terms of the joint
    """
    values = term_values(joint, angles)
    if names is None:
        names = tuple(values)
    missing = [name for name in names if name not in values]
    if missing:
        raise ValueError('joint {} has no terms {}, it has {}'.format(joint, missing, list(values)))
    return np.array([values[name] for name in names], dtype=np.float64)


class StaticCompensation:
    """
    The holding drive of a joint for the gravity load of the pose, and the positions to add to the target of the
    servo to produce it. Friction is not compensated, see the top of this module.

    :param joint: the joint (servo id) the compensation is for, 2 to 5
    :param terms: mapping of term name to coefficient in pwm units, the holding drive fitted on the mean of both
        directions: the pwm the servo needs to hold the pose, positive drives towards higher positions
    :param max_positions: the compensation is clipped to this magnitude in positions (not pwm units), so a wrong
        model near the end of a range can't command a big jump
    """

    def __init__(self, joint: int, terms: Mapping[str, float], max_positions: int) -> None:
        self.joint = joint
        self.max_positions = int(max_positions)
        if self.max_positions <= 0:
            raise ValueError('max_positions should be positive, got {}'.format(max_positions))
        self.terms = self._checked(terms)

    def _checked(self, terms: Mapping[str, float]) -> dict[str, float]:
        names = term_names(self.joint)
        unknown = [name for name in terms if name not in names]
        if unknown:
            raise ValueError('joint {} has no terms {}, it has {}'.format(self.joint, unknown, list(names)))
        return {name: float(coefficient) for name, coefficient in terms.items()}

    def drive(self, angles: np.ndarray) -> float:
        """
        the holding drive for the gravity load at the angles: the formula of the joint, joint_2_drive to joint_5_drive
        :param angles: the commanded angles of all joints, indexed by servo id, in radians
        :return: the pwm the servo needs to hold the pose, not rounded, 0.0 without coefficients
        """
        q2, q3, q4, q5 = _joint_angles(angles)
        return JOINT_DRIVES[self.joint](q2, q3, q4, q5, **self.terms)

    def positions(self, angles: np.ndarray, p: float) -> int:
        """
        the positions to add to the target for the gravity load at the angles: the holding drive divided by the
        pwm the servo gives per position of error, P / 128
        :param angles: the commanded angles of all joints, indexed by servo id, in radians
        :param p: the P gain the servo runs with
        :return: whole positions, clipped to +- max_positions; 0 when p is not positive
        """
        if p <= 0:
            return 0
        positions = self.drive(angles) / (p * PWM_PER_POSITION_PER_P)
        return int(np.clip(np.rint(positions), -self.max_positions, self.max_positions))

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, StaticCompensation) and self.joint == other.joint and self.terms == other.terms
                and self.max_positions == other.max_positions)

    def __repr__(self) -> str:
        return 'StaticCompensation(joint={}, terms={}, max_positions={})'.format(
            self.joint, self.terms, self.max_positions)
