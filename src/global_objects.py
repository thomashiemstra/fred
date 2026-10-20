"""
The objects shared by the whole process, built on their first use.

get_robot() is the one place that builds the robot: the server resources, the calibration scripts and the move tests
all go through it, so what the robot is (the simulation, or the real arm mirrored in the simulation, see
global_constants.use_simulation) is decided here and nowhere else. Whether the simulation is shown in a window is
decided once per process by its entry point with configure(): the server shows it, the scripts keep it headless.
Fresh simulated robots for the reinforcement learning environments and the visualiser come from
simulation_utils.start_simulated_robot.
"""
from functools import lru_cache

from src import global_constants
from src.robot_controllers.dynamixel_robot.servo_config import load_servo_config

_show_simulation = False


def configure(show_simulation):
    """
    decides what the robot of this process looks like, to be called by the entry point before the robot is built
    :param show_simulation: open the pybullet window for the simulation (the server does, scripts keep it headless)
    :raises RuntimeError: when the robot is already built
    """
    global _show_simulation
    if get_robot.cache_info().currsize:
        raise RuntimeError('the robot is already built, configure() has to be called before the first get_robot()')
    _show_simulation = show_simulation


@lru_cache(maxsize=1)
def get_robot():
    """
    the robot of this process, built on the first call: the simulated robot, or the real robot mirrored in the
    simulation (headless unless configure(show_simulation=True) was called); the serial port of the real robot is
    found automatically
    :raises DynamixelError: when the servos of the real robot don't answer on any usb adapter or can't be set up
    """
    from src.simulation.simulation_utils import start_simulated_robot
    simulated_robot = start_simulated_robot(use_gui=_show_simulation)
    if global_constants.use_simulation:
        return simulated_robot
    from src.robot_controllers.combined_robot.combined_robot import CombinedRobot
    from src.robot_controllers.dynamixel_robot import dynamixel_x_config
    from src.robot_controllers.dynamixel_robot.dynamixel_robot_controller import DynamixelRobotController, ARM_IDS
    from src.robot_controllers.dynamixel_robot.dynamixel_utils import setup_dynamixel_handlers

    # the real robot first: when its servos don't answer nothing else is started and the next call simply retries
    port_handler, packet_handler = setup_dynamixel_handlers(None, dynamixel_x_config, ARM_IDS)
    servo_config = load_servo_config()
    dynamixel_robot = DynamixelRobotController(global_constants.dynamixel_robot_config, port_handler, packet_handler, servo_config)

    return CombinedRobot(dynamixel_robot, simulated_robot)


@lru_cache(maxsize=1)
def get_xbox_robot_controller():
    from src.xbox_control.xbox360controller.XboxController import XboxController
    from src.xbox_control.xbox360controller.xbox_pose_updater import XboxPoseUpdater
    from src.xbox_control.xbox_robot_controller import XboxRobotController

    robot = get_robot()
    controller = XboxController(dead_zone=30, scale=100)
    pose_updater = XboxPoseUpdater(controller)

    return XboxRobotController(global_constants.dynamixel_robot_config, robot, pose_updater)


@lru_cache(maxsize=1)
def get_camera(camera=0):
    from src.camera.capture import CameraCapture
    return CameraCapture(camera, [])
