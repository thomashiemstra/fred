from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from timeit import default_timer as timer
from typing import TYPE_CHECKING, Any

import numpy as np

from src.kinematics.kinematics import inverse_kinematics, forward_position_kinematics, forward_kinematics
from src.robot_controllers.abstract_robot_controller import AbstractRobotController
from src.robot_controllers.dynamixel_robot import dynamixel_x_config as cfg
from src.robot_controllers.dynamixel_robot.dynamixel_utils import DynamixelError
from src.robot_controllers.dynamixel_robot.servo_config import load_servo_config
from src.robot_controllers.dynamixel_robot.servo_handler import ServoHandler
from src.utils.decorators import synchronized_with_lock
from src.utils.movement import move_from_current_angles
from src.utils.robot_controller_utils import get_recommended_wait_time, servo_2_check, servo_1_check

if TYPE_CHECKING:
    from src.kinematics.kinematics_utils import Pose, RobotConfig
    from src.robot_controllers.dynamixel_robot.position_recorder import ServoPositionRecorder
    from src.robot_controllers.dynamixel_robot.servo_config import ServoConfig

logger = logging.getLogger(__name__)

ARM_IDS = (1, 2, 3, 4, 5, 6)
GRIPPER_ID = 7


# Facade for the robot as a whole, abstracting away the servo handling
class DynamixelRobotController(AbstractRobotController):

    def __init__(self, robot_config: RobotConfig, port_handler: Any, packet_handler: Any, servo_config: ServoConfig,
                 perform_safety_checks: bool = True, print_position_error: bool = False) -> None:
        """
        :param robot_config: a RobotConfig object
        :param port_handler: the port handler of the usb adapter the servos are connected to
        :param packet_handler: the packet handler for the protocol of the servos; both come from
        dynamixel_utils.setup_dynamixel_handlers, global_objects.get_robot sets them up
        :param servo_config: a ServoConfig, by default it is loaded from servo_config.yaml
        :param perform_safety_checks: raise when the servos can't be set up and check the arm position before
        enabling the torque
        :param print_position_error: periodically print the difference between the target and actual position
        """
        self.robot_config = robot_config
        self.perform_safety_checks = perform_safety_checks
        self.print_position_error = print_position_error

        missing = set(ARM_IDS + (GRIPPER_ID,)) - set(servo_config.servos)
        if missing:
            raise DynamixelError('the servo configuration has no servos with ids {}'.format(sorted(missing)))
        self.servo_config = servo_config
        servos = [servo_config.servos[servo_id] for servo_id in ARM_IDS + (GRIPPER_ID,)]

        # the individual servos are used directly by the calibration scripts
        self.servo1 = servos[0]
        self.servo2 = servos[1]
        self.servo3 = servos[2]
        self.servo4 = servos[3]
        self.servo5 = servos[4]
        self.servo6 = servos[5]
        self.servo7 = servos[6]
        self._servos = servos

        # all servos share the same usb adapter, the handler reads them per bus
        self.servo_handler = ServoHandler(dict(servo_config.servos), cfg, port_handler, packet_handler,
                                          servo_config.buses)
        self.bus_lock = self.servo_handler.bus_lock
        # the servos that are warmer than the warning temperature, so the warning is only logged once
        self._warm_servos: set[int] = set()

        self.initialize_servos()
        self._current_angles = self._read_initial_angles()
        self.status = False
        self.lock = threading.RLock()

        self.gripper_state = 0  # 0 is completely open 100 is completely closed
        self.counter = 0
        self._recorder: ServoPositionRecorder | None = None

    def reboot_all_servos(self) -> None:
        """reboot the servos and set them up again, the torque is disabled after a reboot"""
        if not self.servo_handler.reboot():
            raise DynamixelError("not all servos came back after the reboot")
        self.initialize_servos()

    def set_recorder(self, recorder: ServoPositionRecorder) -> None:
        self._recorder = recorder

    def initialize_servos(self) -> None:
        success = self.set_control_mode()
        success = success & self.set_velocity_profile()
        success = success & self.set_pid()
        success = success & self.set_goal_current()
        # TODO set min and max pos in servo directly
        if self.perform_safety_checks and not success:
            raise DynamixelError("failed to setup the dynamixel robot")

    def _read_initial_angles(self) -> np.ndarray | None:
        angles = self.get_current_angles()
        if angles is None and self.perform_safety_checks:
            raise DynamixelError("failed to read the initial position of the dynamixel robot")
        return angles

    @synchronized_with_lock("bus_lock")
    def enable_servos(self) -> bool:
        """enable the torque of all servos, returns True on success"""
        if self.perform_safety_checks and not self.safety_check():
            logger.error('failed the safety check, disabling servos!')
            self.disable_servos()
            return False

        if self.get_hardware_errors() != {}:
            logger.error('not all servos are working, not enabling the servos!')
            return False

        if self.get_overheating_servos() != {}:
            logger.error('not all servos are cool enough, not enabling the servos!')
            return False

        if not self.servo_handler.hold_current_position():
            logger.error('failed to set the goal position of the servos, not enabling the servos!')
            return False

        success = self.servo_handler.set_torque(enable=True)
        if not success:
            # don't leave the robot with only part of the servos holding their position
            logger.error('failed to enable all servos, disabling servos!')
            self.disable_servos()
        return success

    def safety_check(self) -> bool:
        positions = self.get_current_positions()
        if positions is None:
            return False
        return servo_1_check(positions) and servo_2_check(positions)

    def disable_servos(self) -> bool:
        """disable the torque of all servos, returns True on success"""
        success = self.servo_handler.set_torque(enable=False)
        if not success:
            logger.error('failed to disable all servos!')
        return success

    def move_to_pose(self, pose: Pose) -> tuple[float, float]:
        angles = inverse_kinematics(pose, self.robot_config)
        # unknown only before the first move when the initial read failed without safety checks
        previous_angles = self._current_angles if self._current_angles is not None else self.get_current_angles()
        if previous_angles is None:
            raise DynamixelError('the current angles of the robot are unknown, not moving')
        recommended_time = get_recommended_wait_time(previous_angles, angles)
        time_taken = self.move_servos(angles)
        return recommended_time, time_taken

    def move_to_pose_and_give_new_angles(self, pose: Pose) -> np.ndarray:
        angles = inverse_kinematics(pose, self.robot_config)
        self.move_servos(angles)
        return angles

    def get_current_pose(self) -> Pose | None:
        """returns the current pose of the end effector, or None if the servos could not be read"""
        angles = self.get_current_angles()
        if angles is None:
            return None
        return forward_kinematics(angles, self.robot_config)

    def get_angle_limits(self) -> dict[int, tuple[float, float]]:
        """returns a mapping of joint number to the (min angle, max angle) the servo can reach"""
        limits = {}
        for servo_id in ARM_IDS:
            servo = self.servo_handler.get_servo(servo_id)
            limits[servo_id] = (min(servo.min_angle, servo.max_angle), max(servo.min_angle, servo.max_angle))
        return limits

    def get_hardware_errors(self) -> dict[int, list[str]] | None:
        """
        returns a mapping of servo id to the names of its hardware errors, for the servos that shut down because of
        one, or None if the servos could not be read. A servo with a hardware error disables its torque and ignores
        goal positions until it is rebooted. The temperature and input voltage of those servos are logged.
        """
        errors = self.servo_handler.read_hardware_errors()
        if errors:
            status = self.servo_handler.read_temperature_and_voltage(list(errors)) or {}
            for servo_id, names in errors.items():
                temperature, voltage = status.get(servo_id, ('?', '?'))
                logger.error('servo %d shut down because of: %s (temperature %sC, input voltage %sV), reboot the '
                             'servos to clear the error', servo_id, ', '.join(names), temperature, voltage)
        return errors

    def get_overheating_servos(self) -> dict[int, int] | None:
        """
        returns a mapping of servo id to temperature of the servos that are warmer than the temperature limit of the
        servo configuration, or None if the temperatures could not be read. A warning is logged once when a servo gets
        warmer than the warning temperature, and again after it cooled down.
        """
        warning, limit = self.servo_config.temperature_warning, self.servo_config.temperature_limit
        temperatures = self.servo_handler.read_temperatures()
        if temperatures is None:
            return None

        for servo_id, temperature in temperatures.items():
            if temperature > warning and servo_id not in self._warm_servos:
                self._warm_servos.add(servo_id)
                logger.warning('servo %d is getting warm: %dC, the robot stops at %dC', servo_id, temperature, limit)
            elif temperature <= warning and servo_id in self._warm_servos:
                self._warm_servos.remove(servo_id)
                logger.info('servo %d cooled down: %dC', servo_id, temperature)

        overheating = {servo_id: temperature for servo_id, temperature in temperatures.items() if temperature > limit}
        for servo_id, temperature in overheating.items():
            logger.error('servo %d is too warm: %dC, the limit is %dC', servo_id, temperature, limit)
        return overheating

    def get_servos_outside_calibrated_range(self) -> dict[int, tuple[int, tuple[int, int]]] | None:
        """returns a mapping of servo id to (present position, calibrated range) of the arm servos outside their
        calibrated range, their angle is clamped to the range so it differs from the actual angle"""
        outside = {}
        if not self.servo_handler.read_current_pos(ARM_IDS):
            return None
        for servo_id in ARM_IDS:
            servo = self.servo_handler.get_servo(servo_id)
            if servo.clamp_position(servo.current_position) != servo.current_position:
                calibrated_range = (servo.clamp_position(-2 ** 31), servo.clamp_position(2 ** 31))
                outside[servo_id] = (servo.current_position, calibrated_range)
        return outside

    def get_current_gripper_position(self) -> np.ndarray | None:
        """the position of the gripper or None if the servos could not be read"""
        angles = self.get_current_angles()
        if angles is None:
            return None
        _, _, _, _, p6 = forward_position_kinematics(angles, self.robot_config)
        return p6

    # returns the time in seconds it took to move the servos
    @synchronized_with_lock("bus_lock")
    def move_servos(self, angles: np.ndarray) -> float:
        start = timer()

        if self.print_position_error:
            if self.counter > 10:
                self._print_position_error()
                self.counter = 0
            self.counter += 1

        if self._recorder is not None and self.servo_handler.read_current_pos(ARM_IDS):
            self._recorder.record(self._servos)

        self._current_angles = angles
        # First set the target_position variable of all servos, some servos compensate based on the other angles
        for servo_id in ARM_IDS:
            self.servo_handler.set_angle(servo_id, angles[servo_id], angles)

        # Next physically move the servos to their target_position
        self.servo_handler.move_to_angles(ARM_IDS)
        end = timer()
        return end - start

    def _print_position_error(self) -> None:
        """print the difference between the current position and the previous target position of each servo, the
        target without its static load compensation, so the number is the remaining sag"""
        current_positions = self.get_current_positions()
        if current_positions is None:
            logger.warning("failed to read the current positions")
            return

        diff = np.zeros(7, dtype=np.intp)
        for servo_id in ARM_IDS:
            servo = self.servo_handler.get_servo(servo_id)
            diff[servo_id] = current_positions[servo_id] - (servo.target_position - servo.compensation_positions)

        print("1: {} 2:{} 3:{}, 4:{}, 5:{}, 6:{}".format(diff[1], diff[2], diff[3], diff[4], diff[5], diff[6]))

    def get_present_currents(self, servo_ids: Iterable[int] = ARM_IDS) -> dict[int, int] | None:
        """returns a mapping of servo id to the signed present current register of the servos, or None if they
        could not be read; the unit depends on the model, see ServoHandler.read_present_current"""
        return self.servo_handler.read_present_current(list(servo_ids))

    def get_temperatures_and_voltages(self, servo_ids: Iterable[int] = ARM_IDS) -> dict[int, tuple[int, float]] | None:
        """returns a mapping of servo id to (temperature in degrees celsius, input voltage), or None on failure"""
        return self.servo_handler.read_temperature_and_voltage(list(servo_ids))

    def set_gripper(self, new_gripper_state: float) -> bool:
        """
        move the gripper to the state, this returns right away. The servo moves smoothly at its profile velocity, and
        when an object blocks the gripper it squeezes the object with at most the goal current of the servo.
        :param new_gripper_state: value between 0 (open) and 100 (closed)
        :return: True if the gripper state was sent to the servo
        """
        self.servo_handler.set_angle(GRIPPER_ID, new_gripper_state)
        return self.servo_handler.move_to_angles([GRIPPER_ID])

    def get_gripper_state(self) -> float | None:
        """returns the actual state of the gripper (0 to 100), or None if it could not be read"""
        if not self.servo_handler.read_current_pos([GRIPPER_ID]):
            return None
        return self.servo_handler.get_angle(GRIPPER_ID, self.servo7.current_position)

    def set_velocity_profile(self) -> bool:
        return self.servo_handler.set_profile_velocity_and_acceleration()

    def set_goal_current(self) -> bool:
        return self.servo_handler.set_configured_goal_current()

    def set_control_mode(self) -> bool:
        return self.servo_handler.set_configured_operating_mode()

    def set_pid(self) -> bool:
        return self.servo_handler.set_pid()

    def set_profile_velocity_percentage(self, percentage: float) -> bool:
        return self.servo_handler.set_profile_velocity_percentage(percentage)

    # debug function to control single servo
    def move_servo(self, servo_id: int, angle: float, all_angles: np.ndarray | None = None) -> None:
        self._check_servo_id(servo_id)
        self.servo_handler.set_angle(servo_id, angle, all_angles)
        self.servo_handler.move_to_angles([servo_id])

    @synchronized_with_lock("bus_lock")
    def get_current_angles(self) -> np.ndarray | None:
        if not self.servo_handler.read_current_pos(ARM_IDS):
            return None

        angles = np.zeros(7, dtype=np.float64)
        for servo_id in ARM_IDS:
            servo = self.servo_handler.get_servo(servo_id)
            angles[servo_id] = servo.get_angle_from_position(servo.current_position)
        return angles

    @synchronized_with_lock("bus_lock")
    def get_current_positions(self) -> np.ndarray | None:
        """returns the current positions of the servos or None if they could not be read"""
        if not self.servo_handler.read_current_pos(ARM_IDS):
            return None

        positions = np.zeros(7, dtype=np.int_)
        for servo_id in ARM_IDS:
            positions[servo_id] = self.servo_handler.get_servo(servo_id).current_position
        return positions

    def pose_to_angles(self, pose: Pose) -> np.ndarray:
        return inverse_kinematics(pose, self.robot_config)

    @synchronized_with_lock("lock")
    def get_status(self) -> bool:
        return self.status

    @synchronized_with_lock("lock")
    def change_status(self, new_state: bool) -> None:
        self.status = new_state

    def set_pid_single_servo(self, servo_id: int, p: int, i: int, d: int) -> bool:
        self._check_servo_id(servo_id)
        return self.servo_handler.set_pid_single_servo(servo_id, p, i, d)

    def set_servo_position(self, servo_id: int, pos: int) -> bool:
        self._check_servo_id(servo_id)
        return self.servo_handler.move_servo_to_pos(servo_id, pos)

    def get_servo_position(self, servo_id: int) -> int | None:
        self._check_servo_id(servo_id)
        return self.servo_handler.read_current_pos_single_servo(servo_id)

    def reset_to_pose(self, pose: Pose) -> None:
        move_from_current_angles(self, pose, 4)

    def _check_servo_id(self, servo_id: int) -> None:
        if servo_id not in self.servo_handler.servo_map:
            raise ValueError("unknown servo id: {}".format(servo_id))
