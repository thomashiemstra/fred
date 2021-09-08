from src.kinematics.kinematics import inverse_kinematics, forward_position_kinematics
import pybullet as p
import numpy as np

from src.robot_controllers.abstract_robot_controller import AbstractRobotController
from src.simulation.simulation_utils import generate_control_points


class SimulatedRobotController(AbstractRobotController):

    def __init__(self, robot_config, physics_client, body_id):
        self.robot_config = robot_config
        self.body_id = body_id
        self.physics_client = physics_client

        self._motors = [i for i in range(6)]
        self._current_angles = self.get_current_angles()
        self._status = False

        # Used for gradient descent control and reinforcement learning
        self.control_points = generate_control_points(self.body_id, self.physics_client)

    def enable_servos(self):
        pass

    def disable_servos(self):
        pass

    def set_profile_velocity_percentage(self, val):
        pass

    def reset_to_pose(self, pose):
        angles = inverse_kinematics(pose, self.robot_config)
        self.reset_servos(angles)

    def reset_to_angels(self, angles):
        self.reset_servos(angles)

    def reset_servos(self, angles):
        """
        Move the robot arm to the target angles ignoring physics, just teleport it to the target angles.

        :param angles: angles to reset the servos to
        """
        self._current_angles = angles
        p.setJointMotorControlArray(self.body_id, self._motors, controlMode=p.POSITION_CONTROL,
                                    targetPositions=angles[1:7], physicsClientId=self.physics_client)
        for i in range(1, 7):
            p.resetJointState(self.body_id, i-1, angles[i], physicsClientId=self.physics_client, targetVelocity=0)

    def move_to_pose(self, pose):
        angles = inverse_kinematics(pose, self.robot_config)
        self.move_servos(angles)
        recommended_time = 0
        return recommended_time, 0

    def move_to_pose_and_give_new_angles(self, pose):
        angles = inverse_kinematics(pose, self.robot_config)
        self.move_servos(angles)
        return angles

    def move_servos(self, angles):
        self._current_angles = angles
        # First set the target_position variable of all servos
        p.setJointMotorControlArray(self.body_id, self._motors, controlMode=p.POSITION_CONTROL,
                                    targetPositions=angles[1:7], physicsClientId=self.physics_client)

    def forward_position_kinematics(self, angles):
        return forward_position_kinematics(angles, self.robot_config)

    def set_gripper(self, new_gripper_state):
        pass

    def set_velocity_profile(self):
        pass

    def set_pid(self):
        pass

    def set_servo_torque(self, servo_id, enable):
        pass

    def get_current_angles(self):
        angles = np.zeros(7, dtype=np.float64)
        for i in range(1, 7):
            angles[i] = p.getJointState(self.body_id, i - 1, physicsClientId=self.physics_client)[0]
        return angles

    def pose_to_angles(self, pose):
        return inverse_kinematics(pose, self.robot_config)

    def get_status(self):
        return self._status

    def change_status(self, new_state):
        self._status = new_state

    def set_pid_single_servo(self, servo_id, p, i, d):
        pass

    def set_servo_position(self, servo_id, pos):
        pass

    def get_servo_position(self, servo_id):
        pass


