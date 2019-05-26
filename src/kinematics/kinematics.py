# from __future__ import division
import numpy as np
from numpy import sqrt
from numpy import arctan2, sin, cos, pi, power


def inverse_kinematics(pose, robot_config):
    """
    Given a target pose and link lengths of a 6 DOF robot arm, calculate the corresponding angles to reach it.
    For a more detailed explanation, see ik_documentation.pdf.
    :param pose: target pose, encodes both position and orientation
    :param robot_config: link lengths
    :return: array of angles to reach this pose, this array starts at 1.
    """
    # Link lengths
    d1 = robot_config.d1
    d4 = robot_config.d4
    d6 = robot_config.d6
    a2 = robot_config.a2

    # Target values
    x, y, z = pose.x, pose.y, pose.z
    t = pose.get_euler_matrix()
    flip = pose.flip

    # First find the position of the wrist
    xc = x - d6 * t[0, 2]
    yc = y - d6 * t[1, 2]
    zc = z - d6 * t[2, 2]
    angles = np.zeros(7, dtype=np.float64)

    # The first 3 angles only depend on the position of the wrist
    angles[1] = arctan2(yc, xc)

    d = (power(xc, 2) + power(yc, 2) + power((zc - d1), 2) - power(a2, 2) - power(d4, 2)) / (2.0 * a2 * d4)
    if d >= 1 or d <= -1:
        d = 1
    angles[3] = arctan2(-sqrt(1 - d ** 2), d)

    k1 = a2 + d4 * cos(angles[3])
    k2 = d4 * sin(angles[3])
    # The positive square root is picked meaning elbow up.
    angles[2] = arctan2((zc - d1), sqrt(power(xc, 2) + power(yc, 2))) - arctan2(k2, k1)

    # Because of the DH-parameters used in the forward kinematics angle3 behaves a bit weird.
    # When the arm is stretched out angle3 should be 0 like the above calculation assumes,
    # but instead it's shifted by pi/2. So we add that here to have a correct angle for the forward kinematics.
    # angle3 = pi/2 means the arm is stretched out, angle3=0 means a 90 degree turn towards the base.
    angles[3] += pi / 2

    q1 = angles[1]
    q2 = angles[2]
    q3 = angles[3]
    q23 = q2 + q3

    t11 = t[0, 0]
    t12 = t[0, 1]
    t13 = t[0, 2]
    t21 = t[1, 0]
    t22 = t[1, 1]
    t23 = t[1, 2]
    t31 = t[2, 0]
    t32 = t[2, 1]
    t33 = t[2, 2]

    # For rotation: R_13.R_46 = T where R_13 is the rotation of frame 3 relative to the base, R_46 the rotation of
    # frame 4 to 6 and T is the target rotation. We know R_13 because we have the first 3 angles, and we know T because
    # that was the input. So we have R_46 = (R_13**T).T (inverse of rotation matrix is it's transpose).
    # The left side is a matrix of the last 3 angles.
    # Then it's just solving an euler matrix for which we need a few elements of (R_13**T).T:
    r13 = t13 * cos(q1) * cos(q23) + t23 * cos(q23) * sin(q1) + t33 * sin(q23)
    r23 = -t23 * cos(q1) + t13 * sin(q1)
    r33 = -t33 * cos(q23) + t13 * cos(q1) * sin(q23) + t23 * sin(q1) * sin(q23)
    r32 = -t32 * cos(q23) + t12 * cos(q1) * sin(q23) + t22 * sin(q1) * sin(q23)
    r31 = -t31 * cos(q23) + t11 * cos(q1) * sin(q23) + t21 * sin(q1) * sin(q23)

    # If you flip the last 3 angles you reach the same pose. Picking the right flip will prevent singularities.
    if flip:
        angles[4] = arctan2(-r23, -r13)
        angles[5] = arctan2(-sqrt(r13 * r13 + r23 * r23), r33)
        angles[6] = arctan2(-r32, r31)
    else:
        angles[4] = arctan2(r23, r13)
        angles[5] = arctan2(sqrt(r13 * r13 + r23 * r23), r33)
        angles[6] = arctan2(r32, -r31)

    return angles


def forward_position_kinematics(angles, robot_config):
    d1 = robot_config.d1
    d4 = robot_config.d4
    d6 = robot_config.d6
    a2 = robot_config.a2

    q1 = angles[1]
    q2 = angles[2]
    q3 = angles[3]
    q4 = angles[4]
    q5 = angles[5]

    p1 = np.zeros(3, dtype=np.float64)
    p2 = np.zeros(3, dtype=np.float64)
    p3 = np.zeros(3, dtype=np.float64)
    p4 = np.zeros(3, dtype=np.float64)
    p6 = np.zeros(3, dtype=np.float64)

    p1[0] = 0
    p1[1] = 0
    p1[2] = d1

    p2[0] = a2 * cos(q1) * cos(q2)
    p2[1] = a2 * cos(q2) * sin(q1)
    p2[2] = d1 + a2 * sin(q2)

    p3[0] = cos(q1) * (a2 * cos(q2) + (d4 / 2.0) * sin(q2 + q3))
    p3[1] = sin(q1) * (a2 * cos(q2) + (d4 / 2.0) * sin(q2 + q3))
    p3[2] = d1 - (d4 / 2.0) * cos(q2 + q3) + a2 * sin(q2)

    p4[0] = cos(q1) * (a2 * cos(q2) + d4 * sin(q2 + q3))
    p4[1] = sin(q1) * (a2 * cos(q2) + d4 * sin(q2 + q3))
    p4[2] = d1 - d4 * cos(q2 + q3) + a2 * sin(q2)

    p6[0] = d6 * sin(q1) * sin(q4) * sin(q5) + cos(q1) * (
                a2 * cos(q2) + (d4 + d6 * cos(q5)) * sin(q2 + q3) + d6 * cos(q2 + q3) * cos(q4) * sin(q5))
    p6[1] = cos(q3) * (d4 + d6 * cos(q5)) * sin(q1) * sin(q2) - d6 * (
                cos(q4) * sin(q1) * sin(q2) * sin(q3) + cos(q1) * sin(q4)) * sin(q5) + cos(q2) * sin(q1) * (
                        a2 + (d4 + d6 * cos(q5)) * sin(q3) + d6 * cos(q3) * cos(q4) * sin(q5))
    p6[2] = d1 - cos(q2 + q3) * (d4 + d6 * cos(q5)) + a2 * sin(q2) + d6 * cos(q4) * sin(q2 + q3) * sin(q5)

    return p1, p2, p3, p4, p6


def forward_orientation_kinematics(angles):
    q1 = angles[1]
    q2 = angles[2]
    q3 = angles[3]
    q4 = angles[4]
    q5 = angles[5]
    q6 = angles[6]

    sx = cos(q6) * (cos(q4) * sin(q1) - cos(q1) * cos(q2 + q3) * sin(q4)) - (cos(q5) * sin(q1) * sin(q4) + cos(q1) * (
                cos(q2 + q3) * cos(q4) * cos(q5) - sin(q2 + q3) * sin(q5))) * sin(q6)
    sy = cos(q1) * (-cos(q4) * cos(q6) + cos(q5) * sin(q4) * sin(q6)) - sin(q1) * (
                -sin(q2 + q3) * sin(q5) * sin(q6) + cos(q2 + q3) * (cos(q6) * sin(q4) + cos(q4) * cos(q5) * sin(q6)))
    sz = -cos(q6) * sin(q2 + q3) * sin(q4) - (cos(q4) * cos(q5) * sin(q2 + q3) + cos(q2 + q3) * sin(q5)) * sin(q6)

    ax = sin(q1) * sin(q4) * sin(q5) + cos(q1) * (cos(q5) * sin(q2 + q3) + cos(q2 + q3) * cos(q4) * sin(q5))
    ay = cos(q5) * sin(q1) * sin(q2 + q3) + (cos(q2 + q3) * cos(q4) * sin(q1) - cos(q1) * sin(q4)) * sin(q5)
    az = -cos(q2 + q3) * cos(q5) + cos(q4) * sin(q2 + q3) * sin(q5)

    s = np.array([sx, sy, sz])
    a = np.array([ax, ay, az])
    n = np.cross(s, a)
    res = np.column_stack((n, s, a))

    return res


def jacobian_transpose_on_f(workspace_force, angles, robot_config, c1_location):
    """
    compute the jacobian transpose on 3 control points on the robot, this translates world forces on each joint
    into joint forces. The join forces add up in joint space.
    :param workspace_force: 3x3 numpy array of the workspace forces on each of the control points
    :param angles: current angles of the robot
    :param robot_config: robot configuration of link lengths
    :param c1_location: distance of control point 1 from frame 3 in the direction of frame 4, should be less than d4
    :return:
    """
    x_comp, y_comp, z_comp = 0, 1, 2
    a2, d4, d6 = robot_config.a2, robot_config.d4, robot_config.d6
    c1, c2, c3, c4, c5 = cos(angles[1]), cos(angles[2]), cos(angles[3]), cos(angles[4]), cos(angles[5])
    c23 = cos(angles[2] + angles[3])
    s1, s2, s3, s4, s5 = sin(angles[1]), sin(angles[2]), sin(angles[3]), sin(angles[4]), sin(angles[5])
    s23 = sin(angles[2] + angles[3])
    
    joint_forces = np.zeros(7)

    # first control point, somewhere between frame 3 and frame 4
    fx, fy, fz = workspace_force[0][x_comp], workspace_force[0][y_comp], workspace_force[0][z_comp]
    joint_forces[1] += (fy*c1 - fx*s1)*(a2*c2 + c1_location*s23)
    joint_forces[2] += a2*fz*c2 + (fx*c2 + fy*s1)*(c1_location*c23 - a2*s2) + c1_location*fz*s23
    joint_forces[3] += c1_location*c23*(fx*c1 + fy*s1) + c1_location*fz*s23

    # second control point, origin of frame 4
    fx, fy, fz = workspace_force[1][x_comp], workspace_force[1][y_comp], workspace_force[1][z_comp]
    joint_forces[1] += (fy*c1 - fx*s1)*(a2*c2 + d4*s23)
    joint_forces[2] += a2*fz*c2 + (fx*c2 + fy*s1)*(d4*c23 - a2*s2) + d4*fz*s23
    joint_forces[3] += d4*c23*(fx*c1 + fy*s1) + d4*fz*s23

    # third control point, origin of frame 6
    fx, fy, fz = workspace_force[2][x_comp], workspace_force[2][y_comp], workspace_force[2][z_comp]
    joint_forces[1] += (fy*c1 - fx*s1)*(a2*c2 + (d4 + d6*c5)*s23) + d6*(c23*c4*(fy*c1 - fx*s1) + (fx*c1 + fy*s1)*s4)*s5
    joint_forces[2] += a2*fz*c2 + c23*(d4 + d6*c5)*(fx*c1 + fy*s1) + fz*(d4 + d6*c5)*s23 + d6*fz*c23*c4*s5 - (fx*c1 + fy*s1)*(a2*s2 + d6*c4*s23*s5)
    joint_forces[3] += (d4 + d6*c5)*(c23*(fx*c1 + fy*s1) + fz*s23) + d6*c4*(fz*c23 - (fx*c1 + fy*s1)*s23)*s5
    joint_forces[4] += -d6*(-fx*c4*s1 + (fy*c23*s1 + fz*s23)*s4 + c1*(fy*c4 + fx*c23*s4))*s5
    joint_forces[5] += d6*c5*(c4*(c23*(fx*c1 + fy*s1) + fz*s23) + (-fy*c1 + fx*s1)*s4) + d6*(fz*c23 - (fx*c1 + fy*s1)*s23)*s5

    return joint_forces
