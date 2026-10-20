"""
Static load compensation: the positions a servo has to be commanded beyond the target of an angle to hold the arm
there against gravity and the spring. See Handover Gravity and Spring Compensation.md.

The holding load of every joint is a linear combination of a few sin/cos terms of the joint angles, the basis
below, in the conventions of src/kinematics/kinematics.py: angles are indexed by servo id in radians, angle 2 = 0 is
the lower arm horizontal and phi = angle 2 + angle 3 = pi/2 is the upper arm in line with the lower arm. The
coefficients are fitted in positions per joint by calibration/fit_static_load.py from the data of
calibration/record_static_load.py, so they absorb the stiffness of the servo, and live under `compensation` of the
servo in servo_config.yaml. A Servo with a compensation adds its positions to every target it gets together with
all the angles (DynamixelRobotController.move_servos passes them).

Terms per joint (h = mass times centre of mass of a link, a along the link, p perpendicular to it in the arm plane):
    constant: the same number of positions at every pose; in the up/down fits that is the friction deadband, in
              the mean fit an offset of the zero of the servo
    joint 2:  cos_q2 (lower arm h2a + everything beyond joint 3 times a2), sin_q2 (-h2p), and the terms of joint 3
    joint 3:  sin_phi (h3a), cos_phi (h3p), wrist_reach (the gripper, m6 c6)
    joint 5:  wrist_pitch (the gripper)
    joint 4:  wrist_roll (the gripper when the wrist is bent and rolled, small)
    spring_dl on joints 2 and 3: dL/dq of the spring length, coefficient -k L0 (in positions)

A linear spring of stiffness k and free length L0 loads a joint with k (L - L0) dL/dq. The k L dL/dq part is a spring
of zero free length, and such a spring pulls exactly like a mass does: its torque is a combination of the same
sin/cos terms as gravity (the regressor of L dL/dq is collinear with the gravity terms, condition numbers of 1e11).
So that part is absorbed by the gravity coefficients and only dL/dq is a term of its own.
"""
import numpy as np

UP = 'up'
DOWN = 'down'
DIRECTIONS = (UP, DOWN)

SPRING_TERMS = ('spring_dl',)
TERMS_OF_JOINT = {
    2: ('constant', 'cos_q2', 'sin_q2', 'sin_phi', 'cos_phi', 'wrist_reach') + SPRING_TERMS,
    3: ('constant', 'sin_phi', 'cos_phi', 'wrist_reach') + SPRING_TERMS,
    4: ('constant', 'wrist_roll'),
    5: ('constant', 'wrist_pitch'),
}


class SpringGeometry:
    """
    The spring between the housing of servo 2, on the base plate, and the housing of servo 3, which turns with the
    upper arm, so its length depends on angle 2 and angle 3 and it loads both joints.

    All lengths in cm, in the vertical plane of the arm.
    :param base_anchor: (radial, up) of the anchor on the servo 2 housing relative to the axis of joint 2, radial
        is away from the base axis
    :param arm_anchor: (along, perpendicular) of the anchor on the servo 3 housing relative to the axis of joint 3,
        along the upper arm towards the wrist, perpendicular is the direction the upper arm turns towards when
        angle 3 increases
    :param a2: the length of the lower arm (joint 2 to joint 3)
    """

    def __init__(self, base_anchor, arm_anchor, a2):
        self.base_anchor = np.array(base_anchor, dtype=np.float64)
        self.arm_anchor = np.array(arm_anchor, dtype=np.float64)
        self.a2 = float(a2)
        if self.base_anchor.shape != (2,) or self.arm_anchor.shape != (2,):
            raise ValueError('the anchors of the spring are (radial, up) and (along, perpendicular) pairs')

    def length(self, q2, q3):
        """the length of the spring at the angles, in cm"""
        phi = q2 + q3
        along, perpendicular = self.arm_anchor
        # the elbow, then along the upper arm (sin phi, -cos phi) and perpendicular to it (cos phi, sin phi)
        anchor = np.array([self.a2 * np.cos(q2) + along * np.sin(phi) + perpendicular * np.cos(phi),
                           self.a2 * np.sin(q2) - along * np.cos(phi) + perpendicular * np.sin(phi)])
        return float(np.linalg.norm(anchor - self.base_anchor))

    def length_and_gradient(self, q2, q3, h=1e-6):
        """returns the length and its derivatives to angle 2 and angle 3, in cm and cm per radian"""
        length = self.length(q2, q3)
        d_q2 = (self.length(q2 + h, q3) - self.length(q2 - h, q3)) / (2 * h)
        d_q3 = (self.length(q2, q3 + h) - self.length(q2, q3 - h)) / (2 * h)
        return length, d_q2, d_q3

    def __eq__(self, other):
        return (isinstance(other, SpringGeometry) and np.array_equal(self.base_anchor, other.base_anchor)
                and np.array_equal(self.arm_anchor, other.arm_anchor) and self.a2 == other.a2)


def term_names(joint):
    """the names of the terms of the holding load of the joint, in the order of regressor_row"""
    if joint not in TERMS_OF_JOINT:
        raise ValueError('joint {} has no static load terms, only joints {} have'.format(joint, sorted(TERMS_OF_JOINT)))
    return TERMS_OF_JOINT[joint]


def term_values(joint, angles, spring=None):
    """
    the value of every term of the joint at the angles
    :param joint: 2 to 5
    :param angles: the angles of all joints, indexed by servo id, in radians
    :param spring: a SpringGeometry, without it the spring terms are left out
    :return: mapping of term name to value
    """
    q2, q3, q4, q5 = (float(angles[i]) for i in (2, 3, 4, 5))
    phi = q2 + q3
    values = {
        'constant': 1.0,
        'cos_q2': np.cos(q2),
        'sin_q2': np.sin(q2),
        'sin_phi': np.sin(phi),
        'cos_phi': np.cos(phi),
        # the radial reach of the gripper beyond the wrist centre, per cm of gripper
        'wrist_reach': np.sin(phi) * np.cos(q5) + np.cos(phi) * np.cos(q4) * np.sin(q5),
        'wrist_pitch': np.cos(phi) * np.sin(q5) + np.cos(q4) * np.cos(q5) * np.sin(phi),
        'wrist_roll': np.sin(q4) * np.sin(q5) * np.sin(phi),
    }
    if spring is not None and joint in (2, 3):
        _, d_q2, d_q3 = spring.length_and_gradient(q2, q3)
        values['spring_dl'] = d_q2 if joint == 2 else d_q3
    return {name: values[name] for name in term_names(joint) if name in values}


def regressor_row(joint, angles, spring=None, names=None):
    """
    the values of the terms of the joint as a row of the regressor
    :param names: the terms to include, by default all terms of the joint (the spring ones when a spring is given)
    """
    values = term_values(joint, angles, spring)
    if names is None:
        names = tuple(values)
    missing = [name for name in names if name not in values]
    if missing:
        raise ValueError('joint {} has no terms {} (a spring is needed for the spring terms)'.format(joint, missing))
    return np.array([values[name] for name in names], dtype=np.float64)


class StaticCompensation:
    """
    The positions to add to the target of a servo for the static load of the pose.

    :param joint: the joint (servo id) the compensation is for, 2 to 5
    :param terms: mapping of term name to coefficient in positions, the static load fitted on the mean of both
        directions
    :param max_positions: the compensation is clipped to this magnitude, so a wrong model near the end of a range
        can't command a big jump
    :param up: optional mapping of term name to coefficient used when the joint moves up (the angle increases)
    :param down: the same for moving down; with up and down the friction hysteresis is compensated too. Both are
        needed, without a direction (the first command) the mean terms are used
    :param spring: a SpringGeometry, needed for the spring terms
    """

    def __init__(self, joint, terms, max_positions, up=None, down=None, spring=None):
        self.joint = joint
        self.max_positions = int(max_positions)
        self.spring = spring
        if self.max_positions <= 0:
            raise ValueError('max_positions should be positive, got {}'.format(max_positions))
        if (up is None) != (down is None):
            raise ValueError('up and down have to be given together')
        self.terms = self._checked(terms)
        self.directional = {UP: self._checked(up), DOWN: self._checked(down)} if up is not None else {}

    def _checked(self, terms):
        names = term_names(self.joint)
        unknown = [name for name in terms if name not in names]
        if unknown:
            raise ValueError('joint {} has no terms {}, it has {}'.format(self.joint, unknown, list(names)))
        if self.spring is None and any(name in SPRING_TERMS for name in terms):
            raise ValueError('the spring terms of joint {} need the spring geometry'.format(self.joint))
        return {name: float(coefficient) for name, coefficient in terms.items()}

    def coefficients(self, direction=None):
        """the coefficients used for a direction, the mean terms without one or without a directional fit"""
        return self.directional.get(direction, self.terms)

    def positions(self, angles, direction=None):
        """
        the positions to add to the target for the static load at the angles
        :param angles: the commanded angles of all joints, indexed by servo id, in radians
        :param direction: UP or DOWN, the direction the joint moves in, or None
        :return: whole positions, clipped to +- max_positions
        """
        coefficients = self.coefficients(direction)
        if not coefficients:
            return 0
        values = term_values(self.joint, angles, self.spring)
        total = sum(coefficient * values[name] for name, coefficient in coefficients.items())
        return int(np.clip(np.rint(total), -self.max_positions, self.max_positions))

    def __eq__(self, other):
        return (isinstance(other, StaticCompensation) and self.joint == other.joint and self.terms == other.terms
                and self.max_positions == other.max_positions and self.directional == other.directional
                and self.spring == other.spring)

    def __repr__(self):
        return 'StaticCompensation(joint={}, terms={}, max_positions={}, directional={})'.format(
            self.joint, self.terms, self.max_positions, self.directional)
