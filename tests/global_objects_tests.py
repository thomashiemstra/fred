import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src import global_constants, global_objects
from src.block_stacking import block_stacker, stacking_config
from src.camera import block_detection
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


class BlockStackingObjectsTests(unittest.TestCase):
    """the block image handler and the block stacker open nothing before they are used"""

    def setUp(self):
        for getter in (global_objects.get_block_image_handler, global_objects.get_block_stacker,
                       global_objects.get_xbox_robot_controller):
            getter.cache_clear()
            self.addCleanup(getter.cache_clear)
        # nothing may open the real robot
        patcher = mock.patch.object(global_objects, 'get_robot', side_effect=AssertionError('real robot'))
        self.get_robot = patcher.start()
        self.addCleanup(patcher.stop)

    def test_importing_the_module_opens_nothing(self):
        # a fresh interpreter, this one has imported everything already
        code = ('import sys\n'
                'from src import global_objects\n'
                'for getter in (global_objects.get_robot, global_objects.get_camera, global_objects.get_block_stacker,\n'
                '               global_objects.get_block_image_handler, global_objects.get_xbox_robot_controller):\n'
                '    assert getter.cache_info().currsize == 0, getter\n'
                'for name in ("src.camera.block_detection", "src.block_stacking.block_stacker", "src.camera.capture"):\n'
                '    assert name not in sys.modules, name\n')
        root = Path(__file__).resolve().parent.parent
        result = subprocess.run([sys.executable, '-c', code], cwd=root, capture_output=True, text=True, timeout=120)
        self.assertEqual(0, result.returncode, result.stderr)

    def test_the_handler_is_made_from_the_stacking_config_once(self):
        config = object()
        handler = object()
        with mock.patch.object(stacking_config, 'load_stacking_config', return_value=config) as load, \
                mock.patch.object(block_detection, 'get_default_block_image_handler',
                                  return_value=handler) as make_handler:
            self.assertIs(handler, global_objects.get_block_image_handler())
            self.assertIs(handler, global_objects.get_block_image_handler())
        load.assert_called_once_with()
        make_handler.assert_called_once_with(config)

    def test_a_missing_calibration_file_is_raised_and_the_next_call_tries_again(self):
        with mock.patch.object(block_detection, 'get_default_block_image_handler',
                               side_effect=[FileNotFoundError('no calibration'), 'handler']):
            with self.assertRaises(FileNotFoundError):
                global_objects.get_block_image_handler()
            self.assertEqual('handler', global_objects.get_block_image_handler())

    def test_the_stacker_gets_the_function_that_gives_the_robot_and_does_not_call_it(self):
        handler = mock.Mock(name='handler')
        config = object()
        with mock.patch.object(global_objects, 'get_block_image_handler', return_value=handler), \
                mock.patch.object(stacking_config, 'load_stacking_config', return_value=config), \
                mock.patch.object(block_stacker, 'BlockStacker') as stacker_class:
            stacker = global_objects.get_block_stacker()
            self.assertIs(stacker, global_objects.get_block_stacker())
        stacker_class.assert_called_once_with(self.get_robot, handler, config)
        self.get_robot.assert_not_called()

    def test_the_clicks_on_the_camera_window_go_to_the_stacker(self):
        handler = mock.Mock(name='handler')
        with mock.patch.object(global_objects, 'get_block_image_handler', return_value=handler), \
                mock.patch.object(stacking_config, 'load_stacking_config', return_value=object()), \
                mock.patch.object(block_stacker, 'BlockStacker') as stacker_class:
            stacker = global_objects.get_block_stacker()
            global_objects.get_block_stacker()
        handler.set_click_listener.assert_called_once_with(stacker_class.return_value.click)
        self.assertIs(stacker_class.return_value, stacker)
        self.get_robot.assert_not_called()

    def test_a_real_stacker_is_the_click_listener_of_a_real_handler(self):
        handler = mock.Mock(name='handler')
        with mock.patch.object(global_objects, 'get_block_image_handler', return_value=handler):
            stacker = global_objects.get_block_stacker()
        handler.set_click_listener.assert_called_once_with(stacker.click)

    def test_a_stacker_that_can_not_be_built_leaves_the_click_listener_alone(self):
        handler = mock.Mock(name='handler')
        with mock.patch.object(global_objects, 'get_block_image_handler', return_value=handler), \
                mock.patch.object(stacking_config, 'load_stacking_config',
                                  side_effect=stacking_config.StackingConfigError('bad')):
            with self.assertRaises(stacking_config.StackingConfigError):
                global_objects.get_block_stacker()
        handler.set_click_listener.assert_not_called()

    def test_a_real_stacker_is_built_without_the_robot(self):
        handler = mock.Mock(name='handler')
        with mock.patch.object(global_objects, 'get_block_image_handler', return_value=handler):
            stacker = global_objects.get_block_stacker()
        self.assertIsInstance(stacker, block_stacker.BlockStacker)
        self.assertFalse(stacker.is_running())
        self.get_robot.assert_not_called()

    def test_the_running_helpers_do_not_build_anything(self):
        self.assertFalse(global_objects.xbox_controller_running())
        self.assertFalse(global_objects.block_stacker_running())
        for getter in (global_objects.get_xbox_robot_controller, global_objects.get_block_stacker):
            self.assertEqual(0, getter.cache_info().currsize)
        self.get_robot.assert_not_called()

    def test_the_running_helpers_do_not_call_the_getters_when_nothing_is_built(self):
        for helper, name in ((global_objects.xbox_controller_running, 'get_xbox_robot_controller'),
                             (global_objects.block_stacker_running, 'get_block_stacker')):
            with self.subTest(helper=name), mock.patch.object(global_objects, name) as getter:
                getter.cache_info.return_value = SimpleNamespace(currsize=0)
                self.assertFalse(helper())
            getter.assert_not_called()

    def test_the_running_helpers_ask_what_is_built(self):
        for helper, name in ((global_objects.xbox_controller_running, 'get_xbox_robot_controller'),
                             (global_objects.block_stacker_running, 'get_block_stacker')):
            for running in (True, False):
                with self.subTest(helper=name, running=running), mock.patch.object(global_objects, name) as getter:
                    getter.cache_info.return_value = SimpleNamespace(currsize=1)
                    getter.return_value.is_running.return_value = running
                    self.assertIs(running, helper())


if __name__ == '__main__':
    unittest.main()
