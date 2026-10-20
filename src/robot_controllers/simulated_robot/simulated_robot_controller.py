from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from src.kinematics.kinematics import inverse_kinematics, forward_position_kinematics
import pybullet as p
import numpy as np

from src.robot_controllers.abstract_robot_controller import AbstractRobotController

if TYPE_CHECKING:
    from src.kinematics.kinematics_utils import Pose, RobotConfig


class ControlPoint:

    def __init__(self, point_id: int, radius: float, body_id: int, physics_client: int, weight: float = 1) -> None:
        self.point_id = point_id
        self.body_id = body_id
        self.radius = radius
        self._physics_client = physics_client
        self._position: np.ndarray | None = None
        self.weight = weight

    @property
    def position(self) -> np.ndarray:
        self._update_position()
        assert self._position is not None
        return self._position

    def _update_position(self) -> None:
        """
        Function used to sync the position of the control point with the robot,
        needs to be called before getting the position of the control point
        """
        _, _, _, _, pos, _ = p.getLinkState(self.body_id, self.point_id, physicsClientId=self._physics_client)
        self._position = np.array(pos) * 100  # convert from meters to centimeters


def generate_control_points(body_id: int, physics_client: int) -> tuple[ControlPoint, ControlPoint, ControlPoint]:
    c1 = ControlPoint(8, 6, body_id, physics_client)   # in between frame 3 and the wrist
    c2 = ControlPoint(7, 6, body_id, physics_client, weight=2)  # wrist (frame 4)
    c3 = ControlPoint(6, 4, body_id, physics_client)  # tip of the gripper
    return c1, c2, c3


class SimulatedRobotController(AbstractRobotController):

    def __init__(self, robot_config: RobotConfig, physics_client: int, body_id: int) -> None:
        self.robot_config = robot_config
        self.body_id = body_id
        self.physics_client = physics_client

        self._motors = [i for i in range(6)]
        self._current_angles: np.ndarray | Sequence[float] = self.get_current_angles()
        self._status = False

        # Used for gradient descent control and reinforcement learning
        self.control_points = generate_control_points(self.body_id, self.physics_client)

    def enable_servos(self) -> None:
        pass

    def disable_servos(self) -> None:
        pass

    def set_profile_velocity_percentage(self, val: float) -> None:
        pass

    def reset_to_pose(self, pose: Pose) -> None:
        angles = inverse_kinematics(pose, self.robot_config)
        self.reset_servos(angles)

    def reset_to_angels(self, angles: np.ndarray | Sequence[float]) -> None:
        self.reset_servos(angles)

    def reset_servos(self, angles: np.ndarray | Sequence[float]) -> None:
        """
        Move the robot arm to the target angles ignoring physics, just teleport it to the target angles.

        :param angles: angles to reset the servos to
        """
        self._current_angles = angles
        p.setJointMotorControlArray(self.body_id, self._motors, controlMode=p.POSITION_CONTROL,
                                    targetPositions=angles[1:7], physicsClientId=self.physics_client)
        for i in range(1, 7):
            p.resetJointState(self.body_id, i-1, angles[i], physicsClientId=self.physics_client, targetVelocity=0)

    def move_to_pose(self, pose: Pose) -> tuple[float, float]:
        angles = inverse_kinematics(pose, self.robot_config)
        self.move_servos(angles)
        recommended_time = 0
        return recommended_time, 0

    def move_to_pose_and_give_new_angles(self, pose: Pose) -> np.ndarray:
        angles = inverse_kinematics(pose, self.robot_config)
        self.move_servos(angles)
        return angles

    def move_servos(self, angles: np.ndarray | Sequence[float]) -> None:
        self._current_angles = angles
        # First set the target_position variable of all servos
        p.setJointMotorControlArray(self.body_id, self._motors, controlMode=p.POSITION_CONTROL,
                                    targetPositions=angles[1:7], physicsClientId=self.physics_client)

    def forward_position_kinematics(
            self, angles: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return forward_position_kinematics(angles, self.robot_config)

    def set_gripper(self, new_gripper_state: float) -> None:
        pass

    def set_velocity_profile(self) -> None:
        pass

    def set_pid(self) -> None:
        pass

    def set_servo_torque(self, servo_id: int, enable: bool) -> None:
        pass

    def get_current_angles(self) -> np.ndarray:
        angles = np.zeros(7, dtype=np.float64)
        for i in range(1, 7):
            angles[i] = p.getJointState(self.body_id, i - 1, physicsClientId=self.physics_client)[0]
        return angles

    def pose_to_angles(self, pose: Pose) -> np.ndarray:
        return inverse_kinematics(pose, self.robot_config)

    def get_status(self) -> bool:
        return self._status

    def change_status(self, new_state: bool) -> None:
        self._status = new_state

    def set_pid_single_servo(self, servo_id: int, p: int, i: int, d: int) -> None:
        pass

    def set_servo_position(self, servo_id: int, pos: int) -> None:
        pass

    def get_servo_position(self, servo_id: int) -> None:
        pass


