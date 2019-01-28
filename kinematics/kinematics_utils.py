import numpy as np
from numpy import sin, cos


class Pose:
    orientation = np.eye(3, dtype=np.float64)

    def __init__(self, x, y, z, flip=False, alpha=0, beta=0, gamma=0):
        self.x = x
        self.y = y
        self.z = z
        self.flip = flip
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.euler_matrix(alpha, beta, gamma)

    def euler_matrix(self, alpha, beta, gamma):
        """alpha is a turn around the world z-axis"""
        """beta is a turn around the world y-axis"""
        """gamma is a turn around the world x-axis"""
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        ca = cos(alpha)
        cb = cos(beta)
        cy = cos(gamma)
        sa = sin(alpha)
        sb = sin(beta)
        sy = sin(gamma)
        self.orientation[0, 0] = ca * sb * cy + sa * sy
        self.orientation[1, 0] = sa * sb * cy - ca * sy
        self.orientation[2, 0] = cb * cy

        self.orientation[0, 1] = ca * cb
        self.orientation[1, 1] = sa * cb
        self.orientation[2, 1] = -sb

        self.orientation[0, 2] = ca * sb * sy - sa * cy
        self.orientation[1, 2] = sa * sb * sy + ca * cy
        self.orientation[2, 2] = cb * sy

    def __copy__(self):
        res = Pose(self.x, self.y, self.z, self.flip, self.alpha, self.beta, self.gamma)
        res.orientation = self.orientation.copy()
        return res


class RobotConfig:

    def __init__(self, d1=9.1, a2=15.8, d4=21.9, d6=2):
        self.initial_d1 = self.d1 = d1  # ground to joint 2
        self.initial_a2 = self.a2 = a2  # joint 2 to join 3
        self.initial_d4 = self.d4 = d4  # joint 3 to wrist centre
        self.initial_d6 = self.d6 = d6  # wrist centre to tip of the end effector

    def set_end_effector_length(self, length):
        self.d6 = length

    def restore_initial_values(self):
        self.d1 = self.initial_d1
        self.a2 = self.initial_a2
        self.d4 = self.initial_d4
        self.d6 = self.initial_d6
