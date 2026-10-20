import unittest
from unittest import mock

from src import global_constants, global_objects
from src.robot_controllers.combined_robot.combined_robot import CombinedRobot
from src.robot_controllers.dynamixel_robot import dynamixel_robot_controller, dynamixel_utils
from src.robot_controllers.dynamixel_robot.dynamixel_utils import DynamixelError
from src.simulation import simulation_utils


class FakeSimulatedRobot:
    physics_client = 0
    body_id = 0
    control_points = []


class FakeDynamixelRobot:

    def __init__(self, robot_config, port_handler, packet_handler, servo_config):
        self.robot_config = robot_config
        self.port_handler = port_handler
        self.packet_handler = packet_handler
        self.servo_config = servo_config


class GetRobotTests(unittest.TestCase):
    """get_robot is the one place that builds the robot, nothing else should"""

    def setUp(self):
        global_objects.get_robot.cache_clear()
        global_objects.configure(show_simulation=False)
        # cleanups run in reverse order: the robot has to be gone before the process can be configured again
        self.addCleanup(global_objects.configure, False)
        self.addCleanup(global_objects.get_robot.cache_clear)
        self.simulated = FakeSimulatedRobot()
        self.handlers = (mock.Mock(name='port_handler'), mock.Mock(name='packet_handler'))
        for patcher in (mock.patch.object(simulation_utils, 'start_simulated_robot', return_value=self.simulated),
                        mock.patch.object(dynamixel_utils, 'setup_dynamixel_handlers', return_value=self.handlers),
                        mock.patch.object(dynamixel_robot_controller, 'DynamixelRobotController', FakeDynamixelRobot),
                        mock.patch.object(global_constants, 'use_simulation', False)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_the_real_robot_is_mirrored_in_the_simulation(self):
        robot = global_objects.get_robot()
        self.assertIsInstance(robot, CombinedRobot)
        self.assertIs(self.simulated, robot.simulated_robot)
        self.assertIsInstance(robot.dynamixel_robot, FakeDynamixelRobot)
        self.assertIs(global_constants.dynamixel_robot_config, robot.robot_config)

    def test_the_bus_is_found_once_and_handed_to_the_physical_robot(self):
        robot = global_objects.get_robot()
        dynamixel_utils.setup_dynamixel_handlers.assert_called_once_with(
            None, dynamixel_robot_controller.cfg, dynamixel_robot_controller.ARM_IDS)
        self.assertEqual(self.handlers, (robot.dynamixel_robot.port_handler, robot.dynamixel_robot.packet_handler))

    def test_nothing_is_started_when_the_servos_do_not_answer(self):
        dynamixel_utils.setup_dynamixel_handlers.side_effect = DynamixelError('no servos')
        with self.assertRaises(DynamixelError):
            global_objects.get_robot()
        simulation_utils.start_simulated_robot.assert_not_called()
        # the next call tries again
        dynamixel_utils.setup_dynamixel_handlers.side_effect = None
        self.assertIsInstance(global_objects.get_robot(), CombinedRobot)

    def test_the_simulation_alone(self):
        with mock.patch.object(global_constants, 'use_simulation', True):
            self.assertIs(self.simulated, global_objects.get_robot())
        dynamixel_utils.setup_dynamixel_handlers.assert_not_called()

    def test_there_is_one_robot_per_process(self):
        self.assertIs(global_objects.get_robot(), global_objects.get_robot())
        self.assertEqual(1, simulation_utils.start_simulated_robot.call_count)

    def test_the_simulation_is_headless_by_default(self):
        global_objects.get_robot()
        simulation_utils.start_simulated_robot.assert_called_once_with(use_gui=False)

    def test_the_entry_point_can_show_the_simulation(self):
        global_objects.configure(show_simulation=True)
        global_objects.get_robot()
        simulation_utils.start_simulated_robot.assert_called_once_with(use_gui=True)

    def test_configuring_a_built_robot_is_refused(self):
        global_objects.get_robot()
        with self.assertRaises(RuntimeError):
            global_objects.configure(show_simulation=True)
        # the robot that exists is kept as it is
        self.assertIs(self.simulated, global_objects.get_robot().simulated_robot)


if __name__ == '__main__':
    unittest.main()
