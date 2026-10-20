import unittest

from src.kinematics.kinematics import inverse_kinematics, forward_kinematics, forward_position_kinematics, \
    jacobian_transpose_on_f
from src.kinematics.kinematics_utils import RobotConfig, Pose
from numpy import pi, sin, cos
import numpy as np

test_config = RobotConfig(d1=10.0, a2=15.0, d4=25.0, d6=5.0)
# the link lengths of the real robot
robot_config = RobotConfig(d1=13.92, a2=20, d4=22, d6=12)


class InverseKinematicsTests(unittest.TestCase):

    def test_upper_arm_up_and_forearm_forward_along_y(self):
        # the default orientation points the end effector along +y, so the wrist is straight
        x = 0
        y = test_config.d4 + test_config.d6
        z = test_config.d1 + test_config.a2
        pose = Pose(x, y, z)

        angles = inverse_kinematics(pose, test_config)

        # angle 3 = 0 means the forearm makes a 90 degree turn relative to the upper arm
        expected = [0, pi / 2, pi / 2, 0.0, 0.0, 0.0, 0.0]
        self.assert_angles(angles[:4], expected[:4])
        self.assertAlmostEqual(0.0, angles[5])
        # with a straight wrist angle 4 and 6 turn around the same axis, so only their sum is defined
        self.assertAlmostEqual(0.0, angles[4] + angles[6])

    def test_forearm_pointing_down_at_45_degrees_along_y(self):
        x = 0
        y = test_config.d4 * sin(pi / 4) + test_config.d6
        z = (test_config.d1 + test_config.a2) - test_config.d4 * cos(pi / 4)
        pose = Pose(x, y, z)

        angles = inverse_kinematics(pose, test_config)

        # the wrist bends up by pi/4 to keep the end effector horizontal
        expected = [0, pi / 2, pi / 2, -pi / 4, 0.0, pi / 4, 0.0]
        self.assert_angles(angles, expected)

    def test_forearm_pointing_down_at_45_degrees_along_x(self):
        # the wrist is on the x-axis, but the end effector still points along +y so the wrist turns sideways
        x = test_config.d4 * sin(pi / 4)
        y = test_config.d6
        z = (test_config.d1 + test_config.a2) - test_config.d4 * cos(pi / 4)
        pose = Pose(x, y, z)

        angles = inverse_kinematics(pose, test_config)

        expected = [0, 0, pi / 2, -pi / 4, -pi / 2, pi / 2, pi / 4]
        self.assert_angles(angles, expected)

    def test_unreachable_pose_stretches_the_arm_towards_it(self):
        # the inverse kinematics never fails, callers have to check the reached position (see movement._angles_of_poses)
        far_away = Pose(0, 80, robot_config.d1)
        angles = inverse_kinematics(far_away, robot_config)

        self.assertTrue(np.all(np.isfinite(angles)))
        self.assertAlmostEqual(pi / 2, angles[3], msg='the arm should be stretched out')
        reached = forward_position_kinematics(angles, robot_config)[-1]
        np.testing.assert_allclose([0, robot_config.a2 + robot_config.d4 + robot_config.d6, robot_config.d1],
                                   reached, atol=1e-9)

    def assert_angles(self, actual_angles, expected):
        if len(actual_angles) != len(expected):
            self.fail("different sizes of angle arrays")
        for i in range(len(actual_angles)):
            self.assertAlmostEqual(actual_angles[i], expected[i], places=6,
                                   msg="angle {} does not match, actual:{}, expected:{}"
                                   .format(i, actual_angles[i], expected[i]))


def random_angles(rng, flip):
    """
    random angles within the range the inverse kinematics returns: the wrist in front of the base (angle 1 is the
    direction of the wrist), the elbow bent one way (angle 3 in [-pi/2, pi/2]) and angle 5 positive when not flipped
    """
    while True:
        angles = np.zeros(7)
        angles[1] = rng.uniform(0.05, pi - 0.05)
        angles[2] = rng.uniform(0.05, pi - 0.05)
        angles[3] = rng.uniform(-pi / 2 + 0.05, pi / 2 - 0.05)
        angles[4] = rng.uniform(-pi + 0.05, pi - 0.05)
        angles[5] = rng.uniform(0.05, pi - 0.05) * (-1 if flip else 1)
        angles[6] = rng.uniform(-pi + 0.05, pi - 0.05)
        wrist_distance_from_z_axis = robot_config.a2 * cos(angles[2]) + robot_config.d4 * sin(angles[2] + angles[3])
        if wrist_distance_from_z_axis > 1:
            return angles


class InverseOfForwardKinematicsTests(unittest.TestCase):

    def assert_same_pose(self, expected, actual, atol=1e-9):
        np.testing.assert_allclose([expected.x, expected.y, expected.z], [actual.x, actual.y, actual.z], atol=atol)
        np.testing.assert_allclose(expected.get_euler_matrix(), actual.get_euler_matrix(), atol=atol)

    def test_inverse_kinematics_finds_the_angles_back(self):
        rng = np.random.default_rng(2)
        for flip in (False, True):
            for _ in range(500):
                angles = random_angles(rng, flip)
                pose = forward_kinematics(angles, robot_config)
                self.assertEqual(flip, pose.flip)
                np.testing.assert_allclose(angles, inverse_kinematics(pose, robot_config), atol=1e-7)

    def test_both_flips_reach_the_same_pose(self):
        rng = np.random.default_rng(3)
        for _ in range(200):
            pose = forward_kinematics(random_angles(rng, False), robot_config)
            pose.flip = True
            flipped = inverse_kinematics(pose, robot_config)
            self.assert_same_pose(pose, forward_kinematics(flipped, robot_config))

    def test_flip_turns_angles_4_and_6_by_half_a_turn_and_mirrors_angle_5(self):
        pose = Pose(25, 25, 10, alpha=0.3, gamma=-0.2)
        angles = inverse_kinematics(pose, robot_config)
        pose.flip = True
        flipped = inverse_kinematics(pose, robot_config)

        np.testing.assert_allclose(angles[:4], flipped[:4], atol=1e-12)
        self.assertAlmostEqual(-angles[5], flipped[5])
        for joint in (4, 6):
            turn = np.mod(flipped[joint] - angles[joint], 2 * pi)
            self.assertAlmostEqual(pi, turn, msg='joint {}'.format(joint))

    def test_nearly_straight_wrist(self):
        # close to angle 5 = 0 angle 4 and 6 turn around almost the same axis, their sum is still reached
        rng = np.random.default_rng(4)
        for angle_5 in (1e-6, -1e-6):
            for _ in range(100):
                angles = random_angles(rng, angle_5 < 0)
                angles[5] = angle_5
                pose = forward_kinematics(angles, robot_config)
                self.assert_same_pose(pose, forward_kinematics(inverse_kinematics(pose, robot_config), robot_config),
                                      atol=1e-8)

    def test_arm_stretched_out(self):
        # angle 3 = pi/2 is the elbow singularity at the edge of the workspace, where rounding errors can push the
        # wrist just out of reach
        rng = np.random.default_rng(5)
        for _ in range(100):
            angles = random_angles(rng, False)
            angles[3] = pi / 2
            pose = forward_kinematics(angles, robot_config)
            self.assert_same_pose(pose, forward_kinematics(inverse_kinematics(pose, robot_config), robot_config),
                                  atol=1e-6)

    def test_upper_arm_leaning_back_over_the_base(self):
        # angle 2 > pi/2 with the wrist still in front of the base
        angles = np.array([0, pi / 2, 2.0, 0.3, 0.0, 0.5, 0.0])
        pose = forward_kinematics(angles, robot_config)
        np.testing.assert_allclose(angles, inverse_kinematics(pose, robot_config), atol=1e-9)


class ForwardPositionKinematicsTests(unittest.TestCase):

    def test_joint_positions_are_the_link_lengths_apart(self):
        rng = np.random.default_rng(6)
        for _ in range(50):
            p1, p2, p3, p4, p6 = forward_position_kinematics(random_angles(rng, False), robot_config)
            np.testing.assert_allclose([0, 0, robot_config.d1], p1)
            self.assertAlmostEqual(robot_config.a2, np.linalg.norm(p2 - p1))
            self.assertAlmostEqual(robot_config.d4 / 2, np.linalg.norm(p3 - p2))
            self.assertAlmostEqual(robot_config.d4, np.linalg.norm(p4 - p2))
            self.assertAlmostEqual(robot_config.d6, np.linalg.norm(p6 - p4))

    def test_end_effector_is_d6_along_the_approach_vector_from_the_wrist(self):
        rng = np.random.default_rng(7)
        for _ in range(50):
            angles = random_angles(rng, False)
            _, _, _, wrist, end_effector = forward_position_kinematics(angles, robot_config)
            approach = forward_kinematics(angles, robot_config).get_euler_matrix()[:, 2]
            np.testing.assert_allclose(wrist + robot_config.d6 * approach, end_effector, atol=1e-9)


class JacobianTransposeTests(unittest.TestCase):

    def test_end_effector_force_matches_the_numerical_jacobian(self):
        # the joint forces of a force on the end effector are J^T.f, with J the derivative of its position
        rng = np.random.default_rng(8)
        h = 1e-6
        for _ in range(20):
            angles = random_angles(rng, False)
            force = np.zeros((3, 3))
            force[2] = rng.normal(size=3)
            expected = np.zeros(7)
            for joint in range(1, 7):
                delta = np.zeros(7)
                delta[joint] = h
                derivative = (forward_position_kinematics(angles + delta, robot_config)[-1] -
                              forward_position_kinematics(angles - delta, robot_config)[-1]) / (2 * h)
                expected[joint] = force[2] @ derivative
            np.testing.assert_allclose(expected, jacobian_transpose_on_f(force, angles, robot_config, 10.0),
                                       atol=1e-5)


class PoseTest(unittest.TestCase):

    def test_default_euler_matrix(self):
        # the end effector points along +y (third column) with its side along +x (second column)
        pose = Pose(0, 0, 0, alpha=0, beta=0, gamma=0)
        matrix = pose.get_euler_matrix()

        expected_grid = np.array(
            [[0,  1,  0],
             [0,  0,  1],
             [1, 0, 0]]
        )

        self.assertTrue(np.array_equal(matrix, expected_grid),
                        "did not get the expected matrix. Expected: {}, actual: {}".format(expected_grid, matrix))

    def test_alpha_turns_around_the_z_axis(self):
        a = np.pi/4
        pose = Pose(0, 0, 0, alpha=a, beta=0, gamma=0)
        matrix = pose.get_euler_matrix()

        ca = np.cos(a)

        expected_grid = np.array(
            [[0,  ca,  -ca],
             [0,  ca,  ca],
             [1, 0, 0]]
        )

        self.assertTrue(np.array_equal(matrix, expected_grid),
                        "did not get the expected matrix.Expected: {}, actual: {}".format(expected_grid, matrix))

    def test_beta_turns_around_the_y_axis(self):
        b = np.pi/4
        pose = Pose(0, 0, 0, alpha=0, beta=b, gamma=0)
        matrix = pose.get_euler_matrix()

        cb = np.cos(b)

        expected_grid = np.array(
            [[cb,  cb, 0],
             [0,  0,  1],
             [cb, -cb, 0]]
        )

        self.assertTrue(np.array_equal(matrix, expected_grid),
                        "did not get the expected matrix.Expected: {}, actual: {}".format(expected_grid, matrix))

    def test_gamma_turns_around_the_x_axis(self):
        g = np.pi/4
        pose = Pose(0, 0, 0, alpha=0, beta=0, gamma=g)
        matrix = pose.get_euler_matrix()

        cg = np.cos(g)

        expected_grid = np.array(
            [[0,  1, 0],
             [-cg,  0,  cg],
             [cg, 0, cg]]
        )

        self.assertTrue(np.array_equal(matrix, expected_grid),
                        "did not get the expected matrix.Expected: {}, actual: {}".format(expected_grid, matrix))

    def test_euler_matrix_is_a_rotation(self):
        rng = np.random.default_rng(9)
        for _ in range(100):
            matrix = Pose(0, 0, 0, alpha=rng.uniform(-pi, pi), beta=rng.uniform(-pi, pi),
                          gamma=rng.uniform(-pi, pi)).get_euler_matrix()
            np.testing.assert_allclose(np.eye(3), matrix @ matrix.T, atol=1e-12)
            self.assertAlmostEqual(1.0, np.linalg.det(matrix))

    def test_given_euler_matrix_overrides_the_angles(self):
        matrix = Pose(0, 0, 0, alpha=0.5).get_euler_matrix()
        pose = Pose(25, 25, 10, alpha=0.1, gamma=0.2, euler_matrix=matrix)

        np.testing.assert_array_equal(matrix, pose.get_euler_matrix())
        np.testing.assert_allclose(inverse_kinematics(Pose(25, 25, 10, alpha=0.5), robot_config),
                                   inverse_kinematics(pose, robot_config))


class RobotConfigTests(unittest.TestCase):

    def test_d6_can_not_be_shorter_than_the_initial_length(self):
        config = RobotConfig(d1=13.92, a2=20, d4=22, d6=12)
        config.d6 = 5
        self.assertEqual(12, config.d6)
        config.d6 = 20
        self.assertEqual(20, config.d6)
        config.restore_initial_values()
        self.assertEqual(12, config.d6)


if __name__ == '__main__':
    unittest.main()
