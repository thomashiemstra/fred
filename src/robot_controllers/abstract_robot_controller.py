from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from src.global_constants import WorkSpaceLimits
from src.utils import workspace

if TYPE_CHECKING:
    import numpy as np

    from src.kinematics.kinematics_utils import Pose, RobotConfig


class AbstractRobotController(ABC):
    # where the arm may go, see global_constants.WorkSpaceLimits; a robot can be given other limits by assigning
    # this attribute. Whoever moves the robot checks against them with check_workspace before executing a move,
    # move_servos doesn't check anything
    workspace_limits = WorkSpaceLimits
    robot_config: RobotConfig

    def workspace_violation(self, angles: np.ndarray) -> str | None:
        """returns why the pose of the angles is outside the workspace limits of this robot, or None"""
        return workspace.workspace_violation(angles, self.robot_config, self.workspace_limits)

    def check_workspace(self, angles: np.ndarray) -> None:
        """
        :raises MovementException: when the pose of the angles is outside the workspace limits of this robot
        """
        workspace.check_workspace(angles, self.robot_config, self.workspace_limits)

    @abstractmethod
    def enable_servos(self) -> bool | None:
        pass

    @abstractmethod
    def disable_servos(self) -> bool | None:
        pass

    @abstractmethod
    def move_to_pose(self, pose: Pose) -> tuple[float, float]:
        pass

    @abstractmethod
    def move_to_pose_and_give_new_angles(self, pose: Pose) -> np.ndarray:
        pass

    @abstractmethod
    def move_servos(self, angles: np.ndarray) -> float | None:
        pass

    @abstractmethod
    def get_current_angles(self) -> np.ndarray | None:
        pass

    @abstractmethod
    def pose_to_angles(self, pose: Pose) -> np.ndarray:
        pass

    @abstractmethod
    def set_gripper(self, new_gripper_state: float) -> bool | None:
        pass

    @abstractmethod
    def reset_to_pose(self, pose: Pose) -> None:
        pass

    def get_gripper_state(self) -> float | None:
        """returns the actual state of the gripper (0 to 100), or None if it is unknown"""
        return None

    def get_angle_limits(self) -> dict[int, tuple[float, float]] | None:
        """returns a mapping of joint number to the (min angle, max angle) the joint can reach, or None if unknown"""
        return None

    def get_overheating_servos(self) -> dict[int, int] | None:
        """returns a mapping of servo id to temperature of the servos that are too warm, or None if the temperatures
        could not be read"""
        return {}

    def get_hardware_errors(self) -> dict[int, list[str]] | None:
        """returns a mapping of servo id to the names of its hardware errors for the servos that shut down, or None
        if the servos could not be read"""
        return {}
