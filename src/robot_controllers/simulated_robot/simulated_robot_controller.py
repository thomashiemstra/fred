from src.kinematics.kinematics import inverse_kinematics
import pybullet as p
import numpy as np

from src.robot_controllers.abstract_robot_controller import AbstractRobotController
from src.utils.robot_controller_utils import get_recommended_wait_time


class ControlPoint:

    def __init__(self, point_id, radius, body_id, physics_client, weight=1):
        self.point_id = point_id
        self.body_id = body_id
        self.radius = radius
        self._physics_client = physics_client
        self._position = None
        self.weight = weight

    @property
    def position(self):
        self.update_position()
        return self._position

    def update_position(self):
        """
        Function used to sync the position of the control point with the robot,
        needs to be called before getting the position of the control point
        """
        _, _, _, _, pos, _ = p.getLinkState(self.body_id, self.point_id, physicsClientId=self._physics_client)
        self._position = np.array(pos) * 100  # convert from meters to centimeters


def generate_control_points(body_id, physics_client):
    c1 = ControlPoint(8, 4, body_id, physics_client)  # in between frame 3 and the wrist
    c2 = ControlPoint(7, 6, body_id, physics_client)  # wrist (frame 4)
    c3 = ControlPoint(6, 6, body_id, physics_client)  # tip of the gripper
    return c1, c2, c3


class SimulatedRobotController(AbstractRobotController):

    def __init__(self, robot_config, physics_client, body_id):
        self.robot_config = robot_config
        self.body_id = body_id
        self.motors = [i for i in range(6)]
        self.physics_client = physics_client
        self.control_points = generate_control_points(self.body_id, self.physics_client)
        self._current_angles = self.get_current_angles()

    def enable_servos(self):
        pass

    def disable_servos(self):
        pass

    def reset_to_pose(self, pose):
        angles = inverse_kinematics(pose, self.robot_config)
        self.reset_servos(angles)

    def reset_servos(self, angles):
        self._current_angles = angles
        p.setJointMotorControlArray(self.body_id, self.motors, controlMode=p.POSITION_CONTROL,
                                    targetPositions=angles[1:7], physicsClientId=self.physics_client)
        for i in range(1, 7):
            p.resetJointState(self.body_id, i-1, angles[i], physicsClientId=self.physics_client, targetVelocity=0)

    def move_to_pose(self, pose):
        angles = inverse_kinematics(pose, self.robot_config)
        recommended_time = get_recommended_wait_time(self._current_angles, angles)
        self.move_servos(angles)
        return recommended_time

    def move_servos(self, angles):
        self._current_angles = angles
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

