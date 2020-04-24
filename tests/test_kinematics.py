from unittest import TestCase

from src.kinematics.kinematics import inverse_kinematics
from src.kinematics.kinematics_utils import RobotConfig, Pose
from numpy import pi, cos, sin

from src.simulation.simulation_utils import start_simulated_robot


def show_robot(angles, config):
    robot = start_simulated_robot(True, config)
    robot.move_servos(angles)
    input("Press Enter to continue...")


class Test(TestCase):

    # expected robot pose:
    #     |\    /
    #     | \  /
    #     |  \/
    #     |
    #     |
    # ----------- y-->
    def test_inverse_kinematics_1(self):
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
    def test_inverse_kinematics_2(self):
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