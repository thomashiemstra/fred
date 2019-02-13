import numpy as np
from numpy import sin, cos
import yaml
from copy import copy


class Pose(yaml.YAMLObject):
    yaml_tag = '!Pose'

    @classmethod
    def to_yaml(cls, dumper, data):

        to_save_pose = copy(data)
        for attr, value in data.__dict__.items():
            if attr != 'orientation' and attr != 'flip':
                setattr(to_save_pose, attr, float(value))

        return dumper.represent_yaml_object(cls.yaml_tag, to_save_pose, cls,
                                            flow_style=cls.yaml_flow_style)

    def __init__(self, x, y, z, flip=False, alpha=0, beta=0, gamma=0, time=2):
        self.x = x
        self.y = y
        self.z = z
        self.time = time
        self.flip = flip
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def get_euler_matrix(self):
        """alpha is a turn around the world z-axis"""
        """beta is a turn around the world y-axis"""
        """gamma is a turn around the world x-axis"""
        orientation = np.eye(3, dtype=np.float64)

        ca = cos(self.alpha)
        cb = cos(self.beta)
        cy = cos(self.gamma)
        sa = sin(self.alpha)
        sb = sin(self.beta)
        sy = sin(self.gamma)
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


if __name__ == '__main__':
    pose = Pose(1.0, 2.0, 3.0)

    print(yaml.dump(pose))

    with open('test.yml', 'w') as outfile:
        yaml.dump(pose, outfile)

    with open('test.yml', 'r') as infile:
        print(yaml.load(infile))
