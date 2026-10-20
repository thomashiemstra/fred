from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.kinematics.kinematics import inverse_kinematics
from src.robot_controllers.abstract_robot_controller import AbstractRobotController

if TYPE_CHECKING:
    import numpy as np

    from src.kinematics.kinematics_utils import Pose
    from src.robot_controllers.dynamixel_robot.dynamixel_robot_controller import DynamixelRobotController
    from src.robot_controllers.simulated_robot.simulated_robot_controller import SimulatedRobotController


class CombinedRobot(AbstractRobotController):
    """
    the real robot mirrored in the simulation: moves go to both, readings come from the physical robot. Everything
    this class doesn't define itself (the servo level of the DynamixelRobotController: positions, currents,
    temperatures, the servo handler, used by the calibration scripts) is forwarded to the physical robot.
    """

    def __init__(self, dynamixel_robot: DynamixelRobotController, simulated_robot: SimulatedRobotController) -> None:
        self.dynamixel_robot = dynamixel_robot
        self.simulated_robot = simulated_robot

        self.robot_config = dynamixel_robot.robot_config
        self.physics_client = simulated_robot.physics_client
        self.body_id = simulated_robot.body_id
        self.control_points = simulated_robot.control_points

    def __getattr__(self, name: str) -> Any:
        # only called for attributes this class doesn't have; before __init__ ran there is no physical robot yet
        if name == 'dynamixel_robot':
            raise AttributeError(name)
        return getattr(self.dynamixel_robot, name)

    def enable_servos(self) -> bool:
        """returns True if the servos of the physical robot were enabled"""
        if not self.dynamixel_robot.enable_servos():
            return False
        self.simulated_robot.enable_servos()
        return True

    def disable_servos(self) -> None:
        self.dynamixel_robot.disable_servos()
        self.simulated_robot.disable_servos()

    def move_to_pose(self, pose: Pose) -> tuple[float, float]:
        self.simulated_robot.move_to_pose(pose)
        return self.dynamixel_robot.move_to_pose(pose)

    def move_to_pose_and_give_new_angles(self, pose: Pose) -> np.ndarray:
        self.simulated_robot.move_to_pose_and_give_new_angles(pose)
        return self.dynamixel_robot.move_to_pose_and_give_new_angles(pose)

    def move_servos(self, angles: np.ndarray) -> float | None:
        self.simulated_robot.move_servos(angles)
        return self.dynamixel_robot.move_servos(angles)

    def get_current_angles(self) -> np.ndarray | None:
        return self.dynamixel_robot.get_current_angles()

    def get_hardware_errors(self) -> dict[int, list[str]] | None:
        return self.dynamixel_robot.get_hardware_errors()

    def get_overheating_servos(self) -> dict[int, int] | None:
        return self.dynamixel_robot.get_overheating_servos()

    def pose_to_angles(self, pose: Pose) -> np.ndarray:
        return inverse_kinematics(pose, self.dynamixel_robot.robot_config)

    def set_gripper(self, new_gripper_state: float) -> bool:
        return self.dynamixel_robot.set_gripper(new_gripper_state)

    def get_gripper_state(self) -> float | None:
        return self.dynamixel_robot.get_gripper_state()

    def get_angle_limits(self) -> dict[int, tuple[float, float]]:
        return self.dynamixel_robot.get_angle_limits()

    def reset_to_pose(self, pose: Pose) -> None:
        self.simulated_robot.reset_to_pose(pose)
        self.dynamixel_robot.reset_to_pose(pose)

    def set_profile_velocity_percentage(self, percentage: float) -> None:
        self.simulated_robot.set_profile_velocity_percentage(percentage)
        self.dynamixel_robot.set_profile_velocity_percentage(percentage)
