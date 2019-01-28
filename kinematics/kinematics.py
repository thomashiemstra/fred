import numpy as np
from numpy import sqrt
from numpy import arctan2, sin, cos, pi


def inverse_kinematics(pose, robot_config):
    d1 = robot_config.d1
    d4 = robot_config.d4
    d6 = robot_config.d6
    a2 = robot_config.a2

    x, y, z = pose.x, pose.y, pose.z
    t = pose.orientation
    flip = pose.flip

    xc = x - d6 * t[0, 2]
    yc = y - d6 * t[1, 2]
    zc = z - d6 * t[2, 2]
    angles = np.zeros(7, dtype=np.float64)

    angles[1] = arctan2(yc, xc)

    d = (xc ** 2 + yc ** 2 + (zc - d1) ** 2 - a2 ** 2 - d4 ** 2) / (2 * a2 * d4)
    if np.isclose(d, 1):
        d = d - 0.00001
    angles[3] = arctan2(-sqrt(1 - d ** 2), d)

    k1 = a2 + d4 * cos(angles[3])
    k2 = d4 * sin(angles[3])
    angles[2] = arctan2((zc - d1), sqrt(xc ** 2 + yc ** 2)) - arctan2(k2, k1)
    1 - d ** 2
    angles[3] += pi / 2

    q1 = angles[1]
    q2 = angles[2]
    q3 = angles[3]
    q23 = q2 + q3

    r11 = t[0, 0]
    r12 = t[0, 1]
    r13 = t[0, 2]
    r21 = t[1, 0]
    r22 = t[1, 1]
    r23 = t[1, 2]
    r31 = t[2, 0]
    r32 = t[2, 1]
    r33 = t[2, 2]

    ax = r13 * cos(q1) * cos(q23) + r23 * cos(q23) * sin(q1) + r33 * sin(q23)
    ay = -r23 * cos(q1) + r13 * sin(q1)
    az = -r33 * cos(q23) + r13 * cos(q1) * sin(q23) + r23 * sin(q1) * sin(q23)
    sz = -r32 * cos(q23) + r12 * cos(q1) * sin(q23) + r22 * sin(q1) * sin(q23)
    nz = -r31 * cos(q23) + r11 * cos(q1) * sin(q23) + r21 * sin(q1) * sin(q23)

    if flip:
        angles[4] = arctan2(-ay, -ax)
        angles[5] = arctan2(-sqrt(ax * ax + ay * ay), az)
        angles[6] = arctan2(-sz, nz)
    else:
        angles[4] = arctan2(ay, ax)
        angles[5] = arctan2(sqrt(ax * ax + ay * ay), az)
        angles[6] = arctan2(sz, -nz)

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
