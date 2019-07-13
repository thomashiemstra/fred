from src.kinematics.kinematics import inverse_kinematics
import pybullet as p
import numpy as np

from src.robot_controllers.abstract_robot_controller import AbstractRobotController


class SimulatedRobotController(AbstractRobotController):

    def __init__(self, robot_config, physics_client):
        self.robot_config = robot_config
        start_pos = [0, 0, 0]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.body_id = p.loadURDF("urdf/fred_with_spheres.urdf", start_pos, start_orientation, physicsClientId=physics_client)
        self.motors = [i for i in range(6)]
        self.physics_client = physics_client

    def enable_servos(self):
        pass

    def disable_servos(self):
        pass

    def reset_to_pose(self, pose):
        angles = inverse_kinematics(pose, self.robot_config)
        self.reset_servo(angles)

    def reset_servo(self, angles):
        for i in range(1, 7):
            p.resetJointState(self.body_id, i-1, angles[i], physicsClientId=self.physics_client)

    def move_to_pose(self, pose):
        angles = inverse_kinematics(pose, self.robot_config)
        self.move_servos(angles)

    def move_servos(self, angles):
        # First set the target_position variable of all servos
        p.setJointMotorControlArray(self.body_id, self.motors, controlMode=p.POSITION_CONTROL,
                                    targetPositions=angles[1:7], physicsClientId=self.physics_client)

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

    @staticmethod
    def get_status():
        return True

    def change_status(self, new_state):
        pass

    def set_pid_single_servo(self, servo_id, p, i, d):
        pass

    def set_servo_position(self, servo_id, pos):
        pass

    def get_servo_position(self, servo_id):
        pass

