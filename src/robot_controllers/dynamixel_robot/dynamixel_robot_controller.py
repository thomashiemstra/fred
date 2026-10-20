import logging
import threading
from timeit import default_timer as timer

import numpy as np

from src.kinematics.kinematics import inverse_kinematics, forward_position_kinematics, forward_kinematics
from src.robot_controllers.abstract_robot_controller import AbstractRobotController
from src.robot_controllers.dynamixel_robot import dynamixel_x_config as cfg
from src.robot_controllers.dynamixel_robot.dynamixel_utils import setup_dynamixel_handlers, DynamixelError
from src.robot_controllers.dynamixel_robot.servo_config import load_servo_config
from src.robot_controllers.dynamixel_robot.servo_handler import ServoHandler
from src.utils.decorators import synchronized_with_lock
from src.utils.movement import move_from_current_angles
from src.utils.robot_controller_utils import get_recommended_wait_time, servo_2_check, servo_1_check

logger = logging.getLogger(__name__)

ARM_IDS = (1, 2, 3, 4, 5, 6)
GRIPPER_ID = 7


# Facade for the robot as a whole, abstracting away the servo handling
class DynamixelRobotController(AbstractRobotController):

    def __init__(self, port, robot_config, servo_config=None, perform_safety_checks=True,
                 print_position_error=False):
        """
        :param port: a string representing the usb port the robot is connected to, None to find it automatically
        :param robot_config: a RobotConfig object
        :param servo_config: a ServoConfig, by default it is loaded from servo_config.yaml
        :param perform_safety_checks: raise when the servos can't be set up and check the arm position before
        enabling the torque
        :param print_position_error: periodically print the difference between the target and actual position
        """
        self.robot_config = robot_config
        self.perform_safety_checks = perform_safety_checks
        self.print_position_error = print_position_error

        if servo_config is None:
            servo_config = load_servo_config()
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

        port_handler, packet_handler = setup_dynamixel_handlers(port, cfg, ARM_IDS)
        # all servos share the same usb adapter, the handler reads them per bus
        self.servo_handler = ServoHandler(dict(servo_config.servos), cfg, port_handler, packet_handler,
                                          servo_config.buses)
        self.bus_lock = self.servo_handler.bus_lock
        # the servos that are warmer than the warning temperature, so the warning is only logged once
        self._warm_servos = set()

        self.initialize_servos()
        self._current_angles = self._read_initial_angles()
        self.status = False
        self.lock = threading.RLock()

        self.gripper_state = 0  # 0 is completely open 100 is completely closed
        self.counter = 0
        self._recorder = None

    def reboot_all_servos(self):
        """reboot the servos and set them up again, the torque is disabled after a reboot"""
        if not self.servo_handler.reboot():
            raise DynamixelError("not all servos came back after the reboot")
        self.initialize_servos()

    def set_recorder(self, recorder):
        self._recorder = recorder

    def initialize_servos(self):
        success = self.set_control_mode()
        success = success & self.set_velocity_profile()
        success = success & self.set_pid()
        success = success & self.set_goal_current()
        # TODO set min and max pos in servo directly
        if self.perform_safety_checks and not success:
            raise DynamixelError("failed to setup the dynamixel robot")

    def _read_initial_angles(self):
        angles = self.get_current_angles()
        if angles is None and self.perform_safety_checks:
            raise DynamixelError("failed to read the initial position of the dynamixel robot")
        return angles

    @synchronized_with_lock("bus_lock")
    def enable_servos(self):
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

    def safety_check(self):
        positions = self.get_current_positions()
        if positions is None:
            return False
        return servo_1_check(positions) and servo_2_check(positions)

    def disable_servos(self):
        """disable the torque of all servos, returns True on success"""
        success = self.servo_handler.set_torque(enable=False)
        if not success:
            logger.error('failed to disable all servos!')
        return success

    def move_to_pose(self, pose):
        angles = inverse_kinematics(pose, self.robot_config)
        recommended_time = get_recommended_wait_time(self._current_angles, angles)
        time_taken = self.move_servos(angles)
        return recommended_time, time_taken

    def move_to_pose_and_give_new_angles(self, pose):
        angles = inverse_kinematics(pose, self.robot_config)
        self.move_servos(angles)
        return angles

    def get_current_pose(self):
        """returns the current pose of the end effector, or None if the servos could not be read"""
        angles = self.get_current_angles()
        if angles is None:
            return None
        return forward_kinematics(angles, self.robot_config)

    def get_angle_limits(self):
        """returns a mapping of joint number to the (min angle, max angle) the servo can reach"""
        limits = {}
        for servo_id in ARM_IDS:
            servo = self.servo_handler.get_servo(servo_id)
            limits[servo_id] = (min(servo.min_angle, servo.max_angle), max(servo.min_angle, servo.max_angle))
        return limits

    def get_hardware_errors(self):
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

    def get_overheating_servos(self):
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

    def get_servos_outside_calibrated_range(self):
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

    def get_current_gripper_position(self):
        angles = self.get_current_angles()
        _, _, _, _, p6 = forward_position_kinematics(angles, self.robot_config)
        return p6

    # returns the time in seconds it took to move the servos
    @synchronized_with_lock("bus_lock")
    def move_servos(self, angles):
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

    def _print_position_error(self):
        """print the difference between the current position and the previous target position of each servo"""
        current_positions = self.get_current_positions()
        if current_positions is None:
            logger.warning("failed to read the current positions")
            return

        diff = np.zeros(7, dtype=np.intp)
        for servo_id in ARM_IDS:
            diff[servo_id] = current_positions[servo_id] - self.servo_handler.get_servo(servo_id).target_position

        print("1: {} 2:{} 3:{}, 4:{}, 5:{}, 6:{}".format(diff[1], diff[2], diff[3], diff[4], diff[5], diff[6]))

    def set_gripper(self, new_gripper_state):
        """
        move the gripper to the state, this returns right away. The servo moves smoothly at its profile velocity, and
        when an object blocks the gripper it squeezes the object with at most the goal current of the servo.
        :param new_gripper_state: value between 0 (open) and 100 (closed)
        :return: True if the gripper state was sent to the servo
        """
        self.servo_handler.set_angle(GRIPPER_ID, new_gripper_state)
        return self.servo_handler.move_to_angles([GRIPPER_ID])

    def get_gripper_state(self):
        """returns the actual state of the gripper (0 to 100), or None if it could not be read"""
        if not self.servo_handler.read_current_pos([GRIPPER_ID]):
            return None
        return self.servo_handler.get_angle(GRIPPER_ID, self.servo7.current_position)

    def set_velocity_profile(self):
        return self.servo_handler.set_profile_velocity_and_acceleration()

    def set_goal_current(self):
        return self.servo_handler.set_configured_goal_current()

    def set_control_mode(self):
        return self.servo_handler.set_configured_operating_mode()

    def set_pid(self):
        return self.servo_handler.set_pid()

    def set_profile_velocity_percentage(self, percentage):
        return self.servo_handler.set_profile_velocity_percentage(percentage)

    # debug function to control single servo
    def move_servo(self, servo_id, angle, all_angles=None):
        self._check_servo_id(servo_id)
        self.servo_handler.set_angle(servo_id, angle, all_angles)
        self.servo_handler.move_to_angles([servo_id])

    @synchronized_with_lock("bus_lock")
    def get_current_angles(self):
        if not self.servo_handler.read_current_pos(ARM_IDS):
            return None

        angles = np.zeros(7, dtype=np.float64)
        for servo_id in ARM_IDS:
            servo = self.servo_handler.get_servo(servo_id)
            angles[servo_id] = servo.get_angle_from_position(servo.current_position)
        return angles

    @synchronized_with_lock("bus_lock")
    def get_current_positions(self):
        """returns the current positions of the servos or None if they could not be read"""
        if not self.servo_handler.read_current_pos(ARM_IDS):
            return None

        positions = np.zeros(7, dtype=np.int_)
        for servo_id in ARM_IDS:
            positions[servo_id] = self.servo_handler.get_servo(servo_id).current_position
        return positions

    def pose_to_angles(self, pose):
        return inverse_kinematics(pose, self.robot_config)

    @synchronized_with_lock("lock")
    def get_status(self):
        return self.status

    @synchronized_with_lock("lock")
    def change_status(self, new_state):
        self.status = new_state

    def set_pid_single_servo(self, servo_id, p, i, d):
        self._check_servo_id(servo_id)
        return self.servo_handler.set_pid_single_servo(servo_id, p, i, d)

    def set_servo_position(self, servo_id, pos):
        self._check_servo_id(servo_id)
        return self.servo_handler.move_servo_to_pos(servo_id, pos)

    def get_servo_position(self, servo_id):
        self._check_servo_id(servo_id)
        return self.servo_handler.read_current_pos_single_servo(servo_id)

    def reset_to_pose(self, pose):
        move_from_current_angles(self, pose, 4)

    def _check_servo_id(self, servo_id):
        if servo_id not in self.servo_handler.servo_map:
            raise ValueError("unknown servo id: {}".format(servo_id))
