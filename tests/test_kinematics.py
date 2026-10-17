from unittest import TestCase

from src.kinematics.kinematics import inverse_kinematics, forward_position_kinematics
from src.kinematics.kinematics_utils import RobotConfig, Pose
from numpy import pi, cos, sin


class Test(TestCase):

    # expected robot pose:
    #     __________
    #     |
    #     |
    #     |
    #     |
    # ----------- y-->
    def test_upper_arm_up_and_forearm_forward(self):
        config = RobotConfig(d1=0, a2=10, d4=10, d6=0)
        x = 0
        y = config.d4 + config.d6
        z = config.d1 + config.a2

        pose = Pose(x, y, z)

        angles = inverse_kinematics(pose, config)

        self.assertAlmostEqual(pi / 2, angles[1], places=6, msg='got a wrong value for angle 1')
        self.assertAlmostEqual(pi / 2, angles[2], places=6, msg='got a wrong value for angle 2')
        self.assertAlmostEqual(0, angles[3], places=6, msg='got a wrong value for angle 3')
        self.assertAlmostEqual(0, angles[5], places=6, msg='got a wrong value for angle 5')
        # with a straight wrist angle 4 and 6 turn around the same axis, only their sum is defined
        self.assertAlmostEqual(0, angles[4] + angles[6], places=6, msg='got a wrong value for angle 4 + 6')
        for expected, actual in zip([x, y, z], forward_position_kinematics(angles, config)[-1]):
            self.assertAlmostEqual(expected, actual, places=6)

    # expected robot pose:
    #     |\    /
    #     | \  /
    #     |  \/
    #     |
    #     |
    # ----------- y-->
    def test_forearm_down_and_end_effector_up(self):
        config = RobotConfig(d1=5, a2=20, d4=10, d6=10.0)
        x = 0
        y = config.d4 * cos(- pi/4) + config.d6 * cos(pi / 4)
        z = config.d1 + config.a2

        pose = Pose(x, y, z, gamma=pi/4)

        angles = inverse_kinematics(pose, config)

        self.assertAlmostEqual(pi / 2, angles[1], places=2, msg='got a wrong value for angle 1')
        self.assertAlmostEqual(pi / 2, angles[2], places=2, msg='got a wrong value for angle 2')
        self.assertAlmostEqual(- pi / 4, angles[3], places=2, msg='got a wrong value for angle 3')
        self.assertAlmostEqual(0, angles[4], places=2, msg='got a wrong value for angle 4')
        self.assertAlmostEqual(pi / 2, angles[5], places=2, msg='got a wrong value for angle 5')
        self.assertAlmostEqual(0, angles[6], places=2, msg='got a wrong value for angle 6')

    # expected robot pose:
    #     /----\
    #    /      \
    #   /
    #   |
    # ------------- y-->
    def test_upper_arm_at_45_degrees_and_end_effector_down(self):
        config = RobotConfig(d1=5, a2=20, d4=10, d6=10.0)
        x = 0.01
        y = config.a2 * cos(pi / 4) + config.d4 + config.d6 * cos(- pi / 4)
        z = config.d1 + config.a2 * sin(pi / 4) + config.d6 * sin(- pi / 4)

        pose = Pose(x, y, z, gamma=-pi / 4)

        angles = inverse_kinematics(pose, config)

        # Angle 4 and 6 hit a singularity here
        self.assertAlmostEqual(pi / 2, angles[1], places=2, msg='got a wrong value for angle 1')
        self.assertAlmostEqual(pi / 4, angles[2], places=2, msg='got a wrong value for angle 2')
        self.assertAlmostEqual(pi / 4, angles[3], places=2, msg='got a wrong value for angle 3')
        self.assertAlmostEqual(pi / 4, angles[5], places=2, msg='got a wrong value for angle 5')