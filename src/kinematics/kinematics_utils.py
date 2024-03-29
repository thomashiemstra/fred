import numpy as np
from numba import jit
from numpy import sin, cos


@jit(nopython=True)
def calculate_euler_matrix_from_angles(alpha, beta, gamma):
    orientation = np.eye(3, dtype=np.float64)

    ca = cos(alpha)
    cb = cos(beta)
    cy = cos(gamma)
    sa = sin(alpha)
    sb = sin(beta)
    sy = sin(gamma)
    orientation[0, 0] = ca * sb * cy + sa * sy
    orientation[1, 0] = sa * sb * cy - ca * sy
    orientation[2, 0] = cb * cy

    orientation[0, 1] = ca * cb
    orientation[1, 1] = sa * cb
    orientation[2, 1] = -sb

    orientation[0, 2] = ca * sb * sy - sa * cy
    orientation[1, 2] = sa * sb * sy + ca * cy
    orientation[2, 2] = cb * sy
    return orientation


class Pose:

    def __init__(self, x, y, z, flip=False, alpha=0.0, beta=0.0, gamma=0.0, time=2.0, euler_matrix=None):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.time = time
        self.flip = flip
        self.euler_matrix = euler_matrix
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)

    def get_euler_matrix(self):
        if self.euler_matrix is not None:
            return self.euler_matrix
        else:
            return self._get_euler_matrix_from_angles()

    def _get_euler_matrix_from_angles(self):
        """alpha is a turn around the world z-axis"""
        """beta is a turn around the world y-axis"""
        """gamma is a turn around the world x-axis"""
        return calculate_euler_matrix_from_angles(self.alpha, self.beta, self.gamma)

    def reset_orientation(self):
        self.alpha = 0
        self.gamma = 0
        self.beta = 0

    def __copy__(self):
        res = Pose(self.x, self.y, self.z, self.flip, self.alpha, self.beta, self.gamma, self.time)
        return res

    def __str__(self):
        return 'POSE: x={} y={} z={}, a={} b={} g={} ,time={} filp={}'\
            .format(self.x, self.y, self.z, self.alpha, self.beta, self.gamma, self.time, self.flip)

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, self.__class__)
                and self.__dict__ == other.__dict__)


class RobotConfig:

    def __init__(self, d1=9.1, a2=15.8, d4=22.0, d6=2.0):
        self.initial_d1 = self.d1 = d1  # ground to joint 2
        self.initial_a2 = self.a2 = a2  # joint 2 to join 3
        self.initial_d4 = self.d4 = d4  # joint 3 to wrist centre
        self.initial_d6 = self.d6 = d6  # wrist centre to tip of the end effector

    @property
    def d6(self):
        return self._d6

    @d6.setter
    def d6(self, length):
        if length < self.initial_d6:
            self._d6 = self.initial_d6
        else:
            self._d6 = length

    def restore_initial_values(self):
        self.d1 = self.initial_d1
        self.a2 = self.initial_a2
        self.d4 = self.initial_d4
        self.d6 = self.initial_d6
