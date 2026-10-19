import os
import tempfile
import unittest

from src import global_constants
from src.kinematics.kinematics_utils import RobotConfig
from src.robot_config import load_robot_configs, parse_robot_configs, RobotConfigError, DEFAULT_ROBOT_CONFIG_PATH

LENGTHS = {'d1': 15.7, 'a2': 20, 'd4': 22, 'd6': 12}


class ParseRobotConfigsTests(unittest.TestCase):

    def test_both_robots_are_parsed(self):
        configs = parse_robot_configs({'dynamixel_robot': LENGTHS, 'simulated_robot': dict(LENGTHS, d1=10)})
        self.assertIsInstance(configs['dynamixel_robot'], RobotConfig)
        self.assertEqual((configs['dynamixel_robot'].d1, configs['dynamixel_robot'].a2, configs['dynamixel_robot'].d4,
                          configs['dynamixel_robot'].d6), (15.7, 20.0, 22.0, 12.0))
        self.assertEqual(configs['simulated_robot'].d1, 10.0)

    def test_simulated_robot_defaults_to_the_dynamixel_robot(self):
        configs = parse_robot_configs({'dynamixel_robot': LENGTHS})
        self.assertEqual(configs['simulated_robot'].d1, 15.7)
        self.assertIsNot(configs['simulated_robot'], configs['dynamixel_robot'])

    def test_lengths_are_floats(self):
        config = parse_robot_configs({'dynamixel_robot': LENGTHS})['dynamixel_robot']
        for value in (config.d1, config.a2, config.d4, config.d6):
            self.assertIsInstance(value, float)

    def test_initial_values_are_kept_for_restore(self):
        config = parse_robot_configs({'dynamixel_robot': LENGTHS})['dynamixel_robot']
        config.d6 = 30
        config.restore_initial_values()
        self.assertEqual(config.d6, 12.0)

    def test_invalid_configurations(self):
        invalid = [
            None,
            [],
            {},
            {'simulated_robot': LENGTHS},
            {'dynamixel_robot': LENGTHS, 'other_robot': LENGTHS},
            {'dynamixel_robot': {'d1': 15.7, 'a2': 20, 'd4': 22}},
            {'dynamixel_robot': dict(LENGTHS, d7=1)},
            {'dynamixel_robot': dict(LENGTHS, d1='15.7')},
            {'dynamixel_robot': dict(LENGTHS, d1=True)},
            {'dynamixel_robot': dict(LENGTHS, a2=-1)},
            {'dynamixel_robot': 15.7},
        ]
        for data in invalid:
            with self.assertRaises(RobotConfigError, msg=repr(data)):
                parse_robot_configs(data)


class LoadRobotConfigsTests(unittest.TestCase):

    def test_the_real_configuration_loads_and_global_constants_use_it(self):
        configs = load_robot_configs(DEFAULT_ROBOT_CONFIG_PATH)
        for robot, config in configs.items():
            for name in ('d1', 'a2', 'd4', 'd6'):
                self.assertGreater(getattr(config, name), 0, '{} {}'.format(robot, name))
        self.assertEqual(global_constants.dynamixel_robot_config.d1, configs['dynamixel_robot'].d1)
        self.assertEqual(global_constants.simulated_robot_config.d4, configs['simulated_robot'].d4)
        self.assertEqual(global_constants.d6, configs['dynamixel_robot'].d6)
        self.assertEqual(global_constants.WorkSpaceLimits.y_min, configs['dynamixel_robot'].d6 + 0.1)

    def test_every_call_returns_new_objects(self):
        self.assertIsNot(load_robot_configs()['dynamixel_robot'], load_robot_configs()['dynamixel_robot'])

    def test_missing_and_broken_files(self):
        with self.assertRaises(RobotConfigError):
            load_robot_configs(os.path.join(tempfile.gettempdir(), 'does_not_exist_robot_config.yaml'))
        with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as broken:
            broken.write('dynamixel_robot: [1, 2\n')
        try:
            with self.assertRaises(RobotConfigError):
                load_robot_configs(broken.name)
        finally:
            os.remove(broken.name)


if __name__ == '__main__':
    unittest.main()
