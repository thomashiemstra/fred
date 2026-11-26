import copy
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import cv2
import yaml

from src.block_stacking import stacking_config
from src.block_stacking.stacking_config import load_stacking_config, parse_stacking_config, aruco_dictionary, \
    StackingConfigError, DEFAULT_STACKING_CONFIG_PATH, StackingConfig, BoardConfig, BlockConfig, TowerConfig, \
    GripperConfig, MotionConfig, ScanConfig, ClickModeConfig
from src.global_constants import WorkSpaceLimits
from tests.block_stacking_tests.frozen_config import FROZEN_STACKING_CONFIG_PATH, load_frozen_config

REMOVE = object()  # as a value in `with_changes`: take the key out


def default_data() -> dict[str, Any]:
    with open(FROZEN_STACKING_CONFIG_PATH) as config_file:
        data: dict[str, Any] = yaml.safe_load(config_file)
    return data


def with_changes(*changes: tuple[str, str, Any]) -> dict[str, Any]:
    """the default configuration as data, with changes of (section, key, value), the value REMOVE removes the key"""
    data = default_data()
    for section, key, value in changes:
        if value is REMOVE:
            del data[section][key]
        else:
            data[section][key] = value
    return data


class FrozenStackingConfigTests(unittest.TestCase):

    def test_the_frozen_file_loads(self) -> None:
        config = load_frozen_config()
        self.assertIsInstance(config, StackingConfig)
        self.assertEqual(config, load_stacking_config(FROZEN_STACKING_CONFIG_PATH))

    def test_board(self) -> None:
        self.assertEqual(BoardConfig('DICT_4X4_50', 5, 3, 3.0, 2.2, -12.0, 12.0, 0.0), load_frozen_config().board)

    def test_blocks(self) -> None:
        self.assertEqual(BlockConfig('DICT_5X5_50', tuple(range(10)), 2.4, 3.0), load_frozen_config().blocks)

    def test_tower(self) -> None:
        self.assertEqual(TowerConfig(22.0, 18.0, 0.0, 6.0), load_frozen_config().tower)

    def test_gripper(self) -> None:
        self.assertEqual(GripperConfig(finger_gap=6.0, open_width=8.0, finger_depth=3.0, open_state=0, closed_state=100,
                                       empty_state=95, max_lead_positions=85, settle_time=0.3, timeout=4.0),
                         load_frozen_config().gripper)

    def test_motion(self) -> None:
        self.assertEqual(MotionConfig(grip_height=1.0, hover_height=10.0, tower_clearance=5.0, release_height=0.3,
                                      park=(0.0, 16.0, 30.0), linear_speed=5.0, joint_speed=0.3, joint_margin=5.0,
                                      block_clearance=0.5, rest_clear_width=6.0), load_frozen_config().motion)

    def test_scan(self) -> None:
        self.assertEqual(ScanConfig(frames=10, timeout=5.0, max_spread=0.8), load_frozen_config().scan)

    def test_click_mode(self) -> None:
        self.assertEqual(ClickModeConfig(idle_timeout=120.0), load_frozen_config().click_mode)

    def test_types_of_the_values(self) -> None:
        config = load_frozen_config()
        self.assertIsInstance(config.click_mode.idle_timeout, float)
        # ints in the yaml where a float is expected come out as floats, the whole numbers stay whole numbers
        self.assertIsInstance(config.gripper.open_state, float)
        self.assertIsInstance(config.gripper.max_lead_positions, float)
        self.assertIsInstance(config.board.squares_x, int)
        self.assertIsInstance(config.scan.frames, int)
        self.assertIsInstance(config.blocks.marker_ids, tuple)
        self.assertIsInstance(config.motion.park, tuple)

    def test_the_configuration_is_frozen(self) -> None:
        config = load_frozen_config()
        with self.assertRaises(AttributeError):
            config.tower.x = 1.0  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            config.motion = config.motion  # type: ignore[misc]

    def test_the_default_block_ids_are_not_the_ids_of_the_arm_and_the_table(self) -> None:
        self.assertFalse({43, 44, 45, 46} & set(load_frozen_config().blocks.marker_ids))

    def test_every_key_of_the_yaml_is_explained_in_its_header(self) -> None:
        with open(FROZEN_STACKING_CONFIG_PATH) as config_file:
            header = ''.join(line for line in config_file if line.startswith('#'))
        for section, keys in default_data().items():
            self.assertIn(section, header)
            for key in keys:
                self.assertIn(key, header, msg='{}: {}'.format(section, key))


class LiveConfigTests(unittest.TestCase):
    """
    The only tests that read the live stacking_config.yaml, which the user edits on the table: they check that it loads
    and where it is, and never what is in it.
    """

    def test_the_live_file_loads_and_is_valid(self) -> None:
        self.assertIsInstance(load_stacking_config(), StackingConfig)

    def test_the_default_path_is_next_to_the_module(self) -> None:
        self.assertTrue(os.path.isabs(DEFAULT_STACKING_CONFIG_PATH))
        self.assertEqual(os.path.dirname(os.path.abspath(stacking_config.__file__)),
                         os.path.dirname(DEFAULT_STACKING_CONFIG_PATH))
        self.assertEqual('stacking_config.yaml', os.path.basename(DEFAULT_STACKING_CONFIG_PATH))
        self.assertTrue(os.path.isfile(DEFAULT_STACKING_CONFIG_PATH))

    def test_loading_does_not_depend_on_the_working_directory(self) -> None:
        expected = load_stacking_config()
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            try:
                self.assertEqual(expected, load_stacking_config())
            finally:
                os.chdir(previous)

    def test_the_default_path_is_the_same_when_imported_from_another_directory(self) -> None:
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            code = 'import sys; sys.path.insert(0, {!r}); from src.block_stacking.stacking_config import ' \
                   'DEFAULT_STACKING_CONFIG_PATH as p; print(p)'.format(str(root))
            result = subprocess.run([sys.executable, '-c', code], cwd=directory, capture_output=True, text=True,
                                    check=True)
        self.assertEqual(os.path.normcase(DEFAULT_STACKING_CONFIG_PATH), os.path.normcase(result.stdout.strip()))

    def test_a_path_object_is_accepted(self) -> None:
        self.assertEqual(load_stacking_config(), load_stacking_config(Path(DEFAULT_STACKING_CONFIG_PATH)))


class ArucoDictionaryTests(unittest.TestCase):

    def test_the_predefined_dictionary_with_that_name(self) -> None:
        dictionary = aruco_dictionary('DICT_4X4_50')
        self.assertEqual(50, len(dictionary.bytesList))
        self.assertEqual(4, dictionary.markerSize)
        expected = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        self.assertEqual(5, aruco_dictionary('DICT_5X5_50').markerSize)
        self.assertEqual(expected.bytesList.tolist(), aruco_dictionary('DICT_5X5_50').bytesList.tolist())
        self.assertEqual(1024, len(aruco_dictionary('DICT_ARUCO_ORIGINAL').bytesList))

    def test_unknown_names_are_refused(self) -> None:
        for name in ('DICT_9X9_50', 'DICT_', 'DICT_4X4_50 ', 'dict_4x4_50', '', 'Dictionary'):
            with self.subTest(name=name):
                with self.assertRaisesRegex(StackingConfigError, 'is not the name of an aruco dictionary'):
                    aruco_dictionary(name)

    def test_names_that_do_not_start_with_dict_are_refused_even_if_cv2_has_them(self) -> None:
        # cv2.aruco has more attributes than dictionaries
        for name in ('getPredefinedDictionary', 'DetectorParameters', 'CORNER_REFINE_SUBPIX', '__name__'):
            self.assertTrue(hasattr(cv2.aruco, name) or name == '')
            with self.subTest(name=name):
                with self.assertRaisesRegex(StackingConfigError, repr(name)):
                    aruco_dictionary(name)

    def test_attributes_that_start_with_dict_but_are_not_a_dictionary_are_refused(self) -> None:
        for name in [n for n in dir(cv2.aruco) if n.startswith('DICT_') and not isinstance(getattr(cv2.aruco, n), int)]:
            with self.subTest(name=name):
                with self.assertRaises(StackingConfigError):
                    aruco_dictionary(name)

    def test_names_that_are_not_text_are_refused(self) -> None:
        for name in (None, 4, ['DICT_4X4_50']):
            with self.subTest(name=name):
                with self.assertRaises(StackingConfigError):
                    aruco_dictionary(name)  # type: ignore[arg-type]


class InvalidConfigTests(unittest.TestCase):
    """every rule of the loader refuses a bad value with a message that names the key, from a file as the robot does"""

    def load(self, data: Any, name: str = 'stacking.yaml') -> StackingConfig:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, name)
            with open(path, 'w') as config_file:
                yaml.safe_dump(data, config_file)
            return load_stacking_config(path)

    def assert_invalid(self, data: Any, message: str) -> None:
        with self.assertRaisesRegex(StackingConfigError, message):
            self.load(data)

    def assert_all_invalid(self, section: str, key: str, values: list[Any], message: str) -> None:
        for value in values:
            with self.subTest(section=section, key=key, value=value):
                self.assert_invalid(with_changes((section, key, value)), '{}: {}'.format(section, message))

    def test_the_error_is_a_value_error(self) -> None:
        self.assertTrue(issubclass(StackingConfigError, ValueError))

    def test_the_message_names_the_file_the_section_and_the_key(self) -> None:
        self.assert_invalid(with_changes(('board', 'square_length', 'big')),
                            'stacking.yaml, board: square_length should be a number, not \'big\'')

    def test_a_missing_file(self) -> None:
        with self.assertRaisesRegex(StackingConfigError, 'can not read the stacking configuration .*nothing.yaml'):
            load_stacking_config('nothing.yaml')

    def test_invalid_yaml_names_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'stacking.yaml')
            with open(path, 'w') as config_file:
                config_file.write('board: [1, 2\n')
            with self.assertRaisesRegex(StackingConfigError, 'stacking.yaml'):
                load_stacking_config(path)

    def test_a_file_that_is_not_text_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'stacking.yaml')
            with open(path, 'wb') as config_file:
                config_file.write(b'board: \xff\xfe\x00\x80\n')
            with self.assertRaisesRegex(StackingConfigError, 'can not read the stacking configuration'):
                load_stacking_config(path)

    def test_an_empty_file_and_one_that_is_not_a_mapping(self) -> None:
        for content in (None, [1, 2], 'board'):
            with self.subTest(content=content):
                self.assert_invalid(content, 'stacking.yaml: should be a mapping with blocks, board, click_mode, gripper')

    def test_a_missing_section(self) -> None:
        for section in ('board', 'blocks', 'tower', 'gripper', 'motion', 'scan', 'click_mode'):
            data = default_data()
            del data[section]
            with self.subTest(section=section):
                self.assert_invalid(data, 'stacking.yaml: missing {}$'.format(section))

    def test_an_unknown_section(self) -> None:
        data = default_data()
        data['camera'] = {'index': 0}
        self.assert_invalid(data, 'stacking.yaml: unknown camera$')

    def test_a_section_that_is_not_a_mapping(self) -> None:
        for section in ('board', 'blocks', 'tower', 'gripper', 'motion', 'scan', 'click_mode'):
            for value in (None, [1, 2], 3):
                data = default_data()
                data[section] = value
                with self.subTest(section=section, value=value):
                    self.assert_invalid(data, 'stacking.yaml, {}: should be a mapping with'.format(section))

    def test_a_missing_key_in_every_section(self) -> None:
        for section, keys in default_data().items():
            for key in keys:
                with self.subTest(section=section, key=key):
                    self.assert_invalid(with_changes((section, key, REMOVE)),
                                        'stacking.yaml, {}: missing {}$'.format(section, key))

    def test_several_missing_keys_are_all_named(self) -> None:
        self.assert_invalid(with_changes(('scan', 'frames', REMOVE), ('scan', 'timeout', REMOVE)),
                            'scan: missing frames, timeout$')

    def test_an_unknown_key_in_every_section(self) -> None:
        for section in default_data():
            data = default_data()
            data[section]['sped'] = 1
            with self.subTest(section=section):
                self.assert_invalid(data, 'stacking.yaml, {}: unknown sped$'.format(section))

    def test_a_misspelled_key_is_unknown_and_the_right_one_missing(self) -> None:
        data = default_data()
        data['motion']['hover_hight'] = data['motion'].pop('hover_height')
        self.assert_invalid(data, 'motion: missing hover_height$')

    def test_numbers_that_are_not_numbers(self) -> None:
        keys = [('board', 'square_length'), ('board', 'marker_length'), ('board', 'rotation'),
                ('blocks', 'marker_length'), ('blocks', 'size'), ('tower', 'rotation'), ('tower', 'clear_radius'),
                ('gripper', 'finger_gap'), ('gripper', 'open_width'), ('gripper', 'finger_depth'),
                ('gripper', 'open_state'), ('gripper', 'closed_state'), ('gripper', 'empty_state'),
                ('gripper', 'max_lead_positions'), ('gripper', 'settle_time'), ('gripper', 'timeout'),
                ('motion', 'grip_height'), ('motion', 'hover_height'), ('motion', 'tower_clearance'),
                ('motion', 'release_height'), ('motion', 'linear_speed'), ('motion', 'joint_speed'),
                ('motion', 'joint_margin'), ('motion', 'block_clearance'), ('motion', 'rest_clear_width'),
                ('scan', 'timeout'), ('scan', 'max_spread'), ('click_mode', 'idle_timeout')]
        for section, key in keys:
            self.assert_all_invalid(section, key, ['a lot', True, False, None, [1.0], {'a': 1}, float('nan'),
                                                   float('inf'), float('-inf')],
                                    '{} should be a number, not'.format(key))

    def test_a_bool_is_not_a_number_and_a_number_is_not_a_bool_in_the_lists(self) -> None:
        self.assert_all_invalid('board', 'position', [[True, 12.0], [-34.0, 'far'], [None, 1.0]],
                                'position should be a number, not')
        self.assert_all_invalid('tower', 'position', [[22.0, False], ['a', 1.0], [1.0, float('nan')]],
                                'position should be a number, not')
        self.assert_all_invalid('motion', 'park', [[0.0, 16.0, True], [0.0, None, 30.0], ['x', 16.0, 30.0]],
                                'park should be a number, not')

    def test_whole_numbers_are_accepted_where_numbers_are_expected(self) -> None:
        config = parse_stacking_config(with_changes(
            ('board', 'position', [-34, 12]), ('board', 'rotation', 90), ('tower', 'position', [22, 18]),
            ('tower', 'clear_radius', 0), ('motion', 'park', [0, 16, 30]), ('motion', 'linear_speed', 5),
            ('scan', 'max_spread', 1)))
        self.assertEqual((-34.0, 12.0, 90.0), (config.board.x, config.board.y, config.board.rotation))
        self.assertEqual((0.0, 16.0, 30.0), config.motion.park)
        self.assertIsInstance(config.motion.park[0], float)
        self.assertEqual(0.0, config.tower.clear_radius)

    def test_lists_of_the_wrong_length_or_that_are_not_lists(self) -> None:
        for section, key, length in (('board', 'squares', 2), ('board', 'position', 2), ('tower', 'position', 2),
                                     ('motion', 'park', 3)):
            wrong = [list(range(1, n + 1)) for n in range(0, 5) if n != length] + [5, 'ab', None, {'x': 1, 'y': 2}]
            self.assert_all_invalid(section, key, wrong, '{} should be a list of {} values'.format(key, length))

    def test_the_message_of_a_bad_list_gives_the_value(self) -> None:
        self.assert_invalid(with_changes(('motion', 'park', [0.0, 16.0])),
                            r'motion: park should be a list of 3 values, not \[0.0, 16.0\]')

    def test_squares(self) -> None:
        self.assert_all_invalid('board', 'squares', [[7.5, 5], [7, '5'], [True, 5], [7, None]],
                                'squares should be a whole number, not')
        self.assert_all_invalid('board', 'squares', [[1, 5], [7, 1], [0, 0], [-3, 5]],
                                r'squares should both be at least 2, not \[')
        config = parse_stacking_config(with_changes(('board', 'squares', [2, 2])))
        self.assertEqual((2, 2), (config.board.squares_x, config.board.squares_y))

    def test_the_marker_of_the_board_is_smaller_than_its_square(self) -> None:
        for marker_length in (3.0, 3.1, 10):
            with self.subTest(marker_length=marker_length):
                self.assert_invalid(with_changes(('board', 'marker_length', marker_length)),
                                    r'board: marker_length \({}\) should be less than square_length \(3.0\)'.format(
                                        float(marker_length)))
        config = parse_stacking_config(with_changes(('board', 'marker_length', 2.99)))
        self.assertEqual(2.99, config.board.marker_length)

    def test_the_board_needs_markers_that_fit_in_its_dictionary(self) -> None:
        # 7 by 5 squares need 17 markers, 10 by 10 need 50 and 11 by 10 need 55
        self.assertEqual(17, 7 * 5 // 2)
        parse_stacking_config(with_changes(('board', 'squares', [10, 10])))
        self.assert_invalid(with_changes(('board', 'squares', [11, 10])),
                            'board: a board of 11 by 10 squares needs 55 markers, but the dictionary DICT_4X4_50 has '
                            'only 50')
        # a dictionary with fewer markers than the board needs
        self.assertEqual(30, len(aruco_dictionary('DICT_APRILTAG_16h5').bytesList))
        self.assert_invalid(with_changes(('board', 'dictionary', 'DICT_APRILTAG_16h5'), ('board', 'squares', [8, 8])),
                            'board: a board of 8 by 8 squares needs 32 markers, but the dictionary '
                            'DICT_APRILTAG_16h5 has only 30')

    def test_dictionary_names(self) -> None:
        for section in ('board', 'blocks'):
            self.assert_all_invalid(section, 'dictionary', ['DICT_9X9_50', 'DICT_', 'dict_4x4_50', 'ARUCO', '', None, 4,
                                                            'getPredefinedDictionary', ['DICT_4X4_50']],
                                    'dictionary: .* is not the name of an aruco dictionary')

    def test_the_dictionary_name_is_in_the_message(self) -> None:
        self.assert_invalid(with_changes(('blocks', 'dictionary', 'DICT_9X9_50')),
                            "blocks: dictionary: 'DICT_9X9_50' is not the name of an aruco dictionary")

    def test_marker_ids_that_are_not_a_list_of_whole_numbers(self) -> None:
        self.assert_all_invalid('blocks', 'marker_ids', [5, 'abc', None, {'a': 1}], 'marker_ids should be a list, not')
        self.assert_all_invalid('blocks', 'marker_ids', [[1.5, 2], [1, '2'], [True, 2], [1, None], [[1], 2]],
                                'marker_ids should be a whole number, not')

    def test_marker_ids_may_not_be_empty(self) -> None:
        self.assert_invalid(with_changes(('blocks', 'marker_ids', [])),
                            'blocks: marker_ids should have at least one id')

    def test_marker_ids_are_unique(self) -> None:
        self.assert_invalid(with_changes(('blocks', 'marker_ids', [1, 2, 3, 2])),
                            'blocks: marker_ids has the id 2 more than once')

    def test_marker_ids_are_inside_the_dictionary(self) -> None:
        for marker_id in (50, 51, 1000, -1):
            with self.subTest(marker_id=marker_id):
                self.assert_invalid(with_changes(('blocks', 'marker_ids', [3, marker_id])),
                                    'blocks: marker_ids has id {}, which is not in the dictionary DICT_5X5_50 '
                                    r'\(ids 0 to 49\)'.format(marker_id))
        config = parse_stacking_config(with_changes(('blocks', 'marker_ids', [0, 49])))
        self.assertEqual((0, 49), config.blocks.marker_ids)

    def test_the_size_of_the_dictionary_decides_which_ids_are_valid(self) -> None:
        data = with_changes(('blocks', 'dictionary', 'DICT_5X5_250'), ('blocks', 'marker_ids', [100, 249]))
        self.assertEqual((100, 249), parse_stacking_config(data).blocks.marker_ids)
        data = with_changes(('blocks', 'dictionary', 'DICT_5X5_250'), ('blocks', 'marker_ids', [250]))
        with self.assertRaisesRegex(StackingConfigError, r'marker_ids has id 250.*\(ids 0 to 249\)'):
            parse_stacking_config(data)

    def test_a_single_marker_id_is_valid(self) -> None:
        self.assertEqual((7,), parse_stacking_config(with_changes(('blocks', 'marker_ids', [7]))).blocks.marker_ids)

    def test_block_ids_may_not_be_ids_of_the_board_in_the_same_dictionary(self) -> None:
        # the board uses the ids 0 to 6
        same = ('board', 'dictionary', 'DICT_5X5_50')
        self.assert_invalid(with_changes(same), 'blocks: marker_ids has 0, 1, 2, 3, 4, 5, 6 which the board '
                                                'uses too, they have the same dictionary DICT_5X5_50 and the board '
                                                'uses the ids 0 to 6')
        self.assert_invalid(with_changes(same, ('blocks', 'marker_ids', [20, 6, 21])),
                            'blocks: marker_ids has 6 which the board uses too')
        config = parse_stacking_config(with_changes(same, ('blocks', 'marker_ids', [7, 8, 43])))
        self.assertEqual((7, 8, 43), config.blocks.marker_ids)

    def test_the_ids_of_the_board_follow_its_squares(self) -> None:
        same = ('board', 'dictionary', 'DICT_5X5_50')
        parse_stacking_config(with_changes(same, ('board', 'squares', [3, 2]), ('blocks', 'marker_ids', [3, 4])))
        self.assert_invalid(with_changes(same, ('board', 'squares', [3, 2]), ('blocks', 'marker_ids', [2, 3])),
                            'blocks: marker_ids has 2 which the board uses too.*the ids 0 to 2')

    def test_two_names_of_the_same_dictionary_are_the_same_dictionary(self) -> None:
        self.assertEqual(cv2.aruco.DICT_APRILTAG_16H5, cv2.aruco.DICT_APRILTAG_16h5)
        data = with_changes(('board', 'dictionary', 'DICT_APRILTAG_16H5'), ('board', 'squares', [4, 4]),
                            ('blocks', 'dictionary', 'DICT_APRILTAG_16h5'), ('blocks', 'marker_ids', [3]))
        self.assert_invalid(data, 'blocks: marker_ids has 3 which the board uses too')

    def test_the_marker_of_a_block_is_smaller_than_the_block(self) -> None:
        for marker_length in (3.0, 3.1, 8):
            with self.subTest(marker_length=marker_length):
                self.assert_invalid(with_changes(('blocks', 'marker_length', marker_length)),
                                    r'blocks: marker_length \({}\) should be less than size \(3.0\)'.format(
                                        float(marker_length)))

    def test_lengths_speeds_times_and_frames_are_positive(self) -> None:
        keys = [('board', 'square_length'), ('board', 'marker_length'), ('blocks', 'marker_length'),
                ('blocks', 'size'), ('gripper', 'finger_gap'), ('gripper', 'open_width'),
                ('gripper', 'finger_depth'), ('gripper', 'max_lead_positions'), ('gripper', 'settle_time'),
                ('gripper', 'timeout'), ('motion', 'grip_height'), ('motion', 'hover_height'),
                ('motion', 'tower_clearance'), ('motion', 'linear_speed'), ('motion', 'joint_speed'),
                ('scan', 'timeout'), ('scan', 'max_spread'), ('click_mode', 'idle_timeout')]
        for section, key in keys:
            self.assert_all_invalid(section, key, [0, -1, -0.5], '{} should be positive, not'.format(key))
        self.assert_all_invalid('scan', 'frames', [0, -1], 'frames should be positive, not')

    def test_distances_and_margins_may_be_zero_but_not_negative(self) -> None:
        keys = [('tower', 'clear_radius'), ('motion', 'joint_margin'), ('motion', 'block_clearance'),
                ('motion', 'rest_clear_width'), ('motion', 'release_height')]
        for section, key in keys:
            self.assert_all_invalid(section, key, [-0.1, -5], '{} should be 0 or more, not'.format(key))
            with self.subTest(section=section, key=key, value=0):
                parse_stacking_config(with_changes((section, key, 0)))

    def test_the_click_mode_section_is_refused_when_missing_incomplete_or_unknown_and_the_message_names_the_key(
            self) -> None:
        data = default_data()
        del data['click_mode']
        self.assert_invalid(data, 'stacking.yaml: missing click_mode$')
        self.assert_invalid(with_changes(('click_mode', 'idle_timeout', REMOVE)),
                            'stacking.yaml, click_mode: missing idle_timeout$')
        data = default_data()
        data['click_mode']['idle_time'] = 5
        self.assert_invalid(data, 'stacking.yaml, click_mode: unknown idle_time$')
        self.assert_invalid(with_changes(('click_mode', 'idle_timeout', 0)),
                            'stacking.yaml, click_mode: idle_timeout should be positive, not 0.0')
        self.assert_invalid(with_changes(('click_mode', 'idle_timeout', -5)),
                            'stacking.yaml, click_mode: idle_timeout should be positive, not -5.0')
        self.assert_invalid(with_changes(('click_mode', 'idle_timeout', 'soon')),
                            'stacking.yaml, click_mode: idle_timeout should be a number, not \'soon\'')

    def test_the_idle_timeout_may_be_a_whole_number_or_a_fraction(self) -> None:
        self.assertEqual(30.0, parse_stacking_config(with_changes(('click_mode', 'idle_timeout', 30))).click_mode
                         .idle_timeout)
        self.assertEqual(0.5, parse_stacking_config(with_changes(('click_mode', 'idle_timeout', 0.5))).click_mode
                         .idle_timeout)

    def test_frames_are_a_whole_number(self) -> None:
        self.assert_all_invalid('scan', 'frames', [2.5, '10', True, None, 10.0], 'frames should be a whole number, not')
        self.assertEqual(1, parse_stacking_config(with_changes(('scan', 'frames', 1))).scan.frames)

    def test_the_fingers_have_to_open_wider_than_a_block(self) -> None:
        for finger_gap in (3.0, 2.0):
            with self.subTest(finger_gap=finger_gap):
                self.assert_invalid(with_changes(('gripper', 'finger_gap', finger_gap), ('gripper', 'open_width', 9)),
                                    r'gripper: finger_gap \({}\) should be more than the size of a block '
                                    r'\(3.0\)'.format(finger_gap))
        # the size of the block decides, not a fixed number
        self.assert_invalid(with_changes(('blocks', 'size', 6.5), ('blocks', 'marker_length', 2.4)),
                            r'gripper: finger_gap \(6.0\) should be more than the size of a block \(6.5\)')
        parse_stacking_config(with_changes(('gripper', 'finger_gap', 3.1)))

    def test_the_outside_of_the_fingers_is_wider_than_the_gap(self) -> None:
        for open_width in (6.0, 5.0):
            with self.subTest(open_width=open_width):
                self.assert_invalid(with_changes(('gripper', 'open_width', open_width)),
                                    r'gripper: open_width \({}\) should be more than finger_gap \(6.0\)'.format(
                                        open_width))

    def test_gripper_states_are_from_0_to_100(self) -> None:
        for key in ('open_state', 'closed_state', 'empty_state'):
            self.assert_all_invalid('gripper', key, [-1, 100.5, 101, 1000],
                                    '{} should be from 0 to 100, not'.format(key))

    def test_gripper_states_are_in_order(self) -> None:
        message = r'gripper: open_state \({}\) should be less than empty_state \({}\), which should not be more than ' \
                  r'closed_state \({}\)'
        for open_state, empty_state, closed_state in ((95, 95, 100), (96, 95, 100), (0, 0, 100), (0, 95, 90),
                                                      (50, 40, 60), (100, 100, 100)):
            with self.subTest(states=(open_state, empty_state, closed_state)):
                data = with_changes(('gripper', 'open_state', open_state), ('gripper', 'empty_state', empty_state),
                                    ('gripper', 'closed_state', closed_state))
                self.assert_invalid(data, message.format(float(open_state), float(empty_state), float(closed_state)))
        # an empty state equal to the closed state is allowed, and so are fractions
        config = parse_stacking_config(with_changes(('gripper', 'empty_state', 100), ('gripper', 'open_state', 0.5)))
        self.assertEqual((0.5, 100.0, 100.0), (config.gripper.open_state, config.gripper.empty_state,
                                               config.gripper.closed_state))

    def test_the_fingertips_stay_above_the_table_limit(self) -> None:
        self.assertEqual(1, WorkSpaceLimits.z_min)
        for grip_height in (0.9, 0.5, 0.1):
            with self.subTest(grip_height=grip_height):
                self.assert_invalid(with_changes(('motion', 'grip_height', grip_height)),
                                    r'motion: grip_height \({}\) should be at least 1'.format(grip_height))
        self.assertEqual(1.0, parse_stacking_config(with_changes(('motion', 'grip_height', 1))).motion.grip_height)

    def test_the_grip_point_is_within_a_block(self) -> None:
        # less than the size of a block + 1 = 4
        for grip_height in (4.0, 4.5, 9.0):
            with self.subTest(grip_height=grip_height):
                data = with_changes(('motion', 'grip_height', grip_height), ('motion', 'hover_height', 20))
                self.assert_invalid(data,
                                    r'motion: grip_height \({}\) should be less than the size of a block \+ 1 \(4.0\)'
                                    .format(grip_height))
        self.assertEqual(3.9, parse_stacking_config(with_changes(('motion', 'grip_height', 3.9))).motion.grip_height)
        # follows the size of the block
        data = with_changes(('blocks', 'size', 4.0), ('blocks', 'marker_length', 3.0), ('gripper', 'finger_gap', 6.0),
                            ('motion', 'grip_height', 4.5), ('motion', 'hover_height', 12))
        self.assertEqual(4.5, parse_stacking_config(data).motion.grip_height)

    def test_hovering_clears_the_blocks(self) -> None:
        # more than grip_height + size = 4
        for hover_height in (4.0, 3.5, 2.5):
            with self.subTest(hover_height=hover_height):
                self.assert_invalid(with_changes(('motion', 'hover_height', hover_height)),
                                    r'motion: hover_height \({}\) should be more than grip_height \+ the size of a '
                                    r'block \(4.0\)'.format(hover_height))
        self.assertEqual(4.1, parse_stacking_config(with_changes(('motion', 'hover_height', 4.1))).motion.hover_height)

    def test_a_bad_value_of_the_first_section_is_found_before_a_later_one(self) -> None:
        data = with_changes(('board', 'square_length', -1), ('motion', 'grip_height', 0))
        self.assert_invalid(data, 'board: square_length should be positive')

    def test_parse_names_a_default_source(self) -> None:
        with self.assertRaisesRegex(StackingConfigError, '^stacking configuration, scan: missing frames$'):
            parse_stacking_config(with_changes(('scan', 'frames', REMOVE)))

    def test_the_default_configuration_is_not_changed_by_parsing(self) -> None:
        data = default_data()
        before = copy.deepcopy(data)
        parse_stacking_config(data)
        self.assertEqual(before, data)


if __name__ == '__main__':
    unittest.main()
