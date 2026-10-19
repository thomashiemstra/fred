"""The urdf of the simulated robot has to agree with the link lengths of src/robot_config.yaml"""
import os
import unittest

import numpy as np
import pybullet as p

from src.kinematics.kinematics_utils import RobotConfig
from src.robot_config import load_robot_configs
from src.simulation import build_urdf

# the joint whose frame sits on the axis of each servo, see fred.xacro
JOINT_OF_SERVO = {1: 'base_to_base_plate', 2: 'base_plate_to_arm_lower', 3: 'arm_lower_to_arm_upper',
                  4: 'arm_upper_to_wrist', 5: 'wrist_to_wrist_servo', 6: 'wrist_servo_to_gripper'}


def joint_frames(urdf_path):
    """returns {joint name: (world position of its frame, world direction of its axis)} with every joint at zero,
    in metres"""
    client = p.connect(p.DIRECT)
    try:
        body = p.loadURDF(urdf_path, physicsClientId=client)
        frames = {}
        for index in range(p.getNumJoints(body, physicsClientId=client)):
            info = p.getJointInfo(body, index, physicsClientId=client)
            p.resetJointState(body, index, 0.0, physicsClientId=client)
            position, orientation = p.getLinkState(body, index, physicsClientId=client)[4:6]
            axis = p.rotateVector(orientation, info[13])
            frames[info[1].decode()] = (np.array(position), np.array(axis))
        return frames
    finally:
        p.disconnect(physicsClientId=client)


def joint_positions(urdf_path):
    """returns {joint name: world position of its frame} with every joint at zero, in metres"""
    return {name: position for name, (position, _) in joint_frames(urdf_path).items()}


def distance_between_axes(frame_a, frame_b):
    """the distance between two parallel joint axes, in metres: an offset along the axes doesn't count"""
    (position_a, axis_a), (position_b, _) = frame_a, frame_b
    delta = position_b - position_a
    return np.linalg.norm(delta - np.dot(delta, axis_a) * axis_a)


class UrdfChainTests(unittest.TestCase):

    def assert_chain_matches(self, urdf_path, config, with_spheres):
        """the distances between the joint axes of the urdf are the link lengths of the config (cm)"""
        joints = joint_frames(urdf_path)
        shoulder, elbow, wrist_centre = (joints[JOINT_OF_SERVO[servo_id]] for servo_id in (2, 3, 5))
        self.assertAlmostEqual(shoulder[0][2] * 100, config.d1, places=4, msg='d1: height of the shoulder axis')
        # the shoulder servo sits beside the arm plane along its own axis and the elbow is back in the plane, so the
        # length of the lower arm is the distance between the two (parallel) axes, not between the frame origins
        self.assertTrue(np.allclose(shoulder[1], elbow[1]), 'the shoulder and elbow axes are parallel')
        self.assertAlmostEqual(distance_between_axes(shoulder, elbow) * 100, config.a2, places=4, msg='a2')
        self.assertAlmostEqual(np.linalg.norm(wrist_centre[0] - elbow[0]) * 100, config.d4, places=4, msg='d4')
        self.assertAlmostEqual(np.dot(elbow[0], shoulder[1]), 0.0, places=6,
                               msg='the elbow is in the vertical plane through the base axis')
        if with_spheres:
            tip = joints['gripper_to_sphere3'][0]
            self.assertAlmostEqual(np.linalg.norm(tip - wrist_centre[0]) * 100, config.d6, places=4, msg='d6')
            self.assertTrue(np.allclose(joints['wrist_servo_to_sphere2'][0], wrist_centre[0]),
                            'control point 2 is the wrist centre')

    def test_the_urdf_the_simulation_loads_matches_the_yaml(self):
        config = load_robot_configs()['simulated_robot']
        self.assert_chain_matches(os.path.join(build_urdf.URDF_DIR, 'fred_with_spheres_new.urdf'), config, True)
        self.assert_chain_matches(os.path.join(build_urdf.URDF_DIR, 'fred_new.urdf'), config, False)

    def test_the_urdf_files_are_built_from_the_xacro_and_the_yaml(self):
        outdated = build_urdf.outdated_files()
        self.assertEqual(outdated, [], 'run python -m src.simulation.build_urdf after changing fred.xacro or '
                                       'robot_config.yaml')

    def test_other_lengths_move_the_joints(self):
        config = RobotConfig(d1=10.0, a2=30.0, d4=25.0, d6=8.0)
        path = os.path.join(build_urdf.URDF_DIR, 'fred_test_build.urdf')
        with open(path, 'w') as urdf_file:
            urdf_file.write(build_urdf.build(config, spheres=True))
        try:
            self.assert_chain_matches(path, config, True)
        finally:
            os.remove(path)

    def test_the_servo_joints_are_in_servo_order(self):
        client = p.connect(p.DIRECT)
        try:
            body = p.loadURDF(os.path.join(build_urdf.URDF_DIR, 'fred_with_spheres_new.urdf'), physicsClientId=client)
            names = [p.getJointInfo(body, index, physicsClientId=client)[1].decode()
                     for index in range(p.getNumJoints(body, physicsClientId=client))]
        finally:
            p.disconnect(physicsClientId=client)
        self.assertEqual(names[:6], [JOINT_OF_SERVO[servo_id] for servo_id in range(1, 7)])


if __name__ == '__main__':
    unittest.main()
