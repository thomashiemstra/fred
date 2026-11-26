"""Loads the configuration of the block stacking from a yaml file, see stacking_config.yaml for the format."""
import math
import os
from dataclasses import dataclass
from typing import Any

import cv2
import yaml

from src.global_constants import WorkSpaceLimits

DEFAULT_STACKING_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stacking_config.yaml')

# the gripper states of servo_config.yaml: 0 is open and 100 is closed
_MIN_STATE = 0
_MAX_STATE = 100

_BOARD_FIELDS = {'dictionary', 'squares', 'square_length', 'marker_length', 'position', 'rotation'}
_BLOCKS_FIELDS = {'dictionary', 'marker_ids', 'marker_length', 'size'}
_TOWER_FIELDS = {'position', 'rotation', 'clear_radius'}
_GRIPPER_FIELDS = {'finger_gap', 'open_width', 'finger_depth', 'open_state', 'closed_state', 'empty_state',
                   'max_lead_positions', 'settle_time', 'timeout'}
_MOTION_FIELDS = {'grip_height', 'hover_height', 'tower_clearance', 'release_height', 'park', 'linear_speed',
                  'joint_speed', 'joint_margin', 'block_clearance', 'rest_clear_width'}
_SCAN_FIELDS = {'frames', 'timeout', 'max_spread'}
_CLICK_MODE_FIELDS = {'idle_timeout'}
_FIELDS = {'board', 'blocks', 'tower', 'gripper', 'motion', 'scan', 'click_mode'}


class StackingConfigError(ValueError):
    """raised when the stacking configuration is not valid"""


@dataclass(frozen=True)
class BoardConfig:
    """the charuco board on the table, lengths in cm"""
    dictionary: str
    squares_x: int
    squares_y: int
    square_length: float
    marker_length: float
    x: float  # of the bottom-right corner of the board, the one marked as origin on the print, in the robot frame
    y: float
    rotation: float  # degrees, of the x arrow printed on the board relative to the robot's x axis, counter clockwise


@dataclass(frozen=True)
class BlockConfig:
    """the blocks and their markers, lengths in cm"""
    dictionary: str
    marker_ids: tuple[int, ...]
    marker_length: float
    size: float  # the edge of a block


@dataclass(frozen=True)
class TowerConfig:
    """where the tower is built, lengths in cm"""
    x: float  # of the centre of the tower, in the robot frame
    y: float
    rotation: float  # degrees, of the edges of the blocks in the tower
    clear_radius: float


@dataclass(frozen=True)
class GripperConfig:
    """the gripper: lengths in cm, states from 0 (open) to 100 (closed), times in seconds"""
    finger_gap: float
    open_width: float
    finger_depth: float
    open_state: float
    closed_state: float
    empty_state: float
    max_lead_positions: float
    settle_time: float
    timeout: float


@dataclass(frozen=True)
class MotionConfig:
    """how the arm moves: lengths in cm, speeds in cm/s (linear) and rad/s (joint), angles in degrees"""
    grip_height: float
    hover_height: float
    tower_clearance: float
    release_height: float
    park: tuple[float, float, float]  # x, y, z of the tip (the fingertips) in the robot frame
    linear_speed: float
    joint_speed: float
    joint_margin: float
    block_clearance: float
    rest_clear_width: float


@dataclass(frozen=True)
class ScanConfig:
    """how long the camera looks at the blocks"""
    frames: int
    timeout: float  # seconds
    max_spread: float  # cm


@dataclass(frozen=True)
class ClickModeConfig:
    """the mode in which a click on a block in the camera window sends that block to the tower"""
    idle_timeout: float  # seconds the arm waits at the park pose for a click before it lays itself down


@dataclass(frozen=True)
class StackingConfig:
    board: BoardConfig
    blocks: BlockConfig
    tower: TowerConfig
    gripper: GripperConfig
    motion: MotionConfig
    scan: ScanConfig
    click_mode: ClickModeConfig


def load_stacking_config(path: str | os.PathLike[str] = DEFAULT_STACKING_CONFIG_PATH) -> StackingConfig:
    """
    read the stacking configuration
    :raises StackingConfigError: if the file can not be read or the configuration is not valid
    """
    try:
        with open(path, encoding='utf-8') as config_file:
            data = yaml.safe_load(config_file)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as e:
        raise StackingConfigError('can not read the stacking configuration {}: {}'.format(os.fspath(path), e)) from e
    return parse_stacking_config(data, os.fspath(path))


def aruco_dictionary(name: str) -> cv2.aruco.Dictionary:
    """
    the predefined aruco dictionary of cv2.aruco with that name, for example DICT_4X4_50
    :raises StackingConfigError: if there is no such dictionary
    """
    return cv2.aruco.getPredefinedDictionary(_dictionary_constant(name))


def _dictionary_constant(name: Any) -> int:
    """the value of cv2.aruco.DICT_... for the name, only the names of dictionaries are accepted"""
    constant = getattr(cv2.aruco, name, None) if isinstance(name, str) and name.startswith('DICT_') else None
    if isinstance(constant, bool) or not isinstance(constant, int):
        raise StackingConfigError('{!r} is not the name of an aruco dictionary, use a DICT_ name of cv2.aruco such as '
                                  'DICT_4X4_50'.format(name))
    return constant


def parse_stacking_config(data: Any, source: str = 'stacking configuration') -> StackingConfig:
    """create the stacking configuration from the parsed yaml data"""
    _check_fields(data, _FIELDS, source)
    board = _parse_board(data['board'], source + ', board')
    blocks = _parse_blocks(data['blocks'], source + ', blocks', board)
    return StackingConfig(board=board, blocks=blocks, tower=_parse_tower(data['tower'], source + ', tower'),
                          gripper=_parse_gripper(data['gripper'], source + ', gripper', blocks),
                          motion=_parse_motion(data['motion'], source + ', motion', blocks),
                          scan=_parse_scan(data['scan'], source + ', scan'),
                          click_mode=_parse_click_mode(data['click_mode'], source + ', click_mode'))


def _parse_board(data: Any, source: str) -> BoardConfig:
    _check_fields(data, _BOARD_FIELDS, source)
    dictionary = _dictionary(data['dictionary'], source)
    squares = _list(data['squares'], 'squares', source, 2)
    squares_x, squares_y = (_integer(count, 'squares', source) for count in squares)
    for count in (squares_x, squares_y):
        if count < 2:
            raise StackingConfigError('{}: squares should both be at least 2, not {}'.format(source, squares))
    square_length = _positive(data['square_length'], 'square_length', source)
    marker_length = _positive(data['marker_length'], 'marker_length', source)
    if marker_length >= square_length:
        raise StackingConfigError('{}: marker_length ({}) should be less than square_length ({})'.format(
            source, marker_length, square_length))
    markers = _board_marker_count(squares_x, squares_y)
    available = len(aruco_dictionary(dictionary).bytesList)
    if markers > available:
        raise StackingConfigError('{}: a board of {} by {} squares needs {} markers, but the dictionary {} has only {}'
                                  .format(source, squares_x, squares_y, markers, dictionary, available))
    x, y = (_number(value, 'position', source) for value in _list(data['position'], 'position', source, 2))
    return BoardConfig(dictionary, squares_x, squares_y, square_length, marker_length, x, y,
                       _number(data['rotation'], 'rotation', source))


def _board_marker_count(squares_x: int, squares_y: int) -> int:
    """a charuco board has a marker in every other square, so its ids are 0 up to this number - 1"""
    return squares_x * squares_y // 2


def _parse_blocks(data: Any, source: str, board: BoardConfig) -> BlockConfig:
    _check_fields(data, _BLOCKS_FIELDS, source)
    dictionary = _dictionary(data['dictionary'], source)
    available = len(aruco_dictionary(dictionary).bytesList)
    ids = _list(data['marker_ids'], 'marker_ids', source)
    if not ids:
        raise StackingConfigError('{}: marker_ids should have at least one id'.format(source))
    marker_ids = tuple(_integer(marker_id, 'marker_ids', source) for marker_id in ids)
    for marker_id in marker_ids:
        if not 0 <= marker_id < available:
            raise StackingConfigError('{}: marker_ids has id {}, which is not in the dictionary {} (ids 0 to {})'
                                      .format(source, marker_id, dictionary, available - 1))
        if marker_ids.count(marker_id) > 1:
            raise StackingConfigError('{}: marker_ids has the id {} more than once'.format(source, marker_id))
    if _dictionary_constant(dictionary) == _dictionary_constant(board.dictionary):
        board_ids = range(_board_marker_count(board.squares_x, board.squares_y))
        used = [marker_id for marker_id in marker_ids if marker_id in board_ids]
        if used:
            raise StackingConfigError('{}: marker_ids has {} which the board uses too, they have the same dictionary '
                                      '{} and the board uses the ids {} to {}'.format(
                                          source, ', '.join(str(marker_id) for marker_id in used), dictionary,
                                          board_ids[0], board_ids[-1]))
    marker_length = _positive(data['marker_length'], 'marker_length', source)
    size = _positive(data['size'], 'size', source)
    if marker_length >= size:
        raise StackingConfigError('{}: marker_length ({}) should be less than size ({})'.format(
            source, marker_length, size))
    return BlockConfig(dictionary, marker_ids, marker_length, size)


def _parse_tower(data: Any, source: str) -> TowerConfig:
    _check_fields(data, _TOWER_FIELDS, source)
    x, y = (_number(value, 'position', source) for value in _list(data['position'], 'position', source, 2))
    return TowerConfig(x, y, _number(data['rotation'], 'rotation', source),
                       _not_negative(data['clear_radius'], 'clear_radius', source))


def _parse_gripper(data: Any, source: str, blocks: BlockConfig) -> GripperConfig:
    _check_fields(data, _GRIPPER_FIELDS, source)
    finger_gap = _positive(data['finger_gap'], 'finger_gap', source)
    if finger_gap <= blocks.size:
        raise StackingConfigError('{}: finger_gap ({}) should be more than the size of a block ({}), or the open '
                                  'gripper does not fit around it'.format(source, finger_gap, blocks.size))
    open_width = _positive(data['open_width'], 'open_width', source)
    if open_width <= finger_gap:
        raise StackingConfigError('{}: open_width ({}) should be more than finger_gap ({})'.format(
            source, open_width, finger_gap))
    finger_depth = _positive(data['finger_depth'], 'finger_depth', source)
    open_state, closed_state, empty_state = (
        _number(data[name], name, source) for name in ('open_state', 'closed_state', 'empty_state'))
    for name, state in (('open_state', open_state), ('closed_state', closed_state), ('empty_state', empty_state)):
        if not _MIN_STATE <= state <= _MAX_STATE:
            raise StackingConfigError('{}: {} should be from {} to {}, not {}'.format(
                source, name, _MIN_STATE, _MAX_STATE, state))
    if not open_state < empty_state <= closed_state:
        raise StackingConfigError('{}: open_state ({}) should be less than empty_state ({}), which should not be more '
                                  'than closed_state ({})'.format(source, open_state, empty_state, closed_state))
    return GripperConfig(finger_gap, open_width, finger_depth, open_state, closed_state, empty_state,
                         _positive(data['max_lead_positions'], 'max_lead_positions', source),
                         _positive(data['settle_time'], 'settle_time', source),
                         _positive(data['timeout'], 'timeout', source))


def _parse_motion(data: Any, source: str, blocks: BlockConfig) -> MotionConfig:
    _check_fields(data, _MOTION_FIELDS, source)
    grip_height = _positive(data['grip_height'], 'grip_height', source)
    if grip_height < WorkSpaceLimits.z_min:
        raise StackingConfigError('{}: grip_height ({}) should be at least {}, the lowest height the workspace limits '
                                  'allow the tip'.format(source, grip_height, WorkSpaceLimits.z_min))
    if grip_height >= blocks.size + 1:
        raise StackingConfigError('{}: grip_height ({}) should be less than the size of a block + 1 ({}), or the '
                                  'fingers miss the block'.format(source, grip_height, blocks.size + 1))
    hover_height = _positive(data['hover_height'], 'hover_height', source)
    if hover_height <= grip_height + blocks.size:
        raise StackingConfigError('{}: hover_height ({}) should be more than grip_height + the size of a block ({}), '
                                  'or the fingers drag over a block that stands next to the one they approach'.format(
                                      source, hover_height, grip_height + blocks.size))
    park_x, park_y, park_z = (_number(value, 'park', source) for value in _list(data['park'], 'park', source, 3))
    return MotionConfig(grip_height=grip_height, hover_height=hover_height,
                        tower_clearance=_positive(data['tower_clearance'], 'tower_clearance', source),
                        release_height=_not_negative(data['release_height'], 'release_height', source),
                        park=(park_x, park_y, park_z),
                        linear_speed=_positive(data['linear_speed'], 'linear_speed', source),
                        joint_speed=_positive(data['joint_speed'], 'joint_speed', source),
                        joint_margin=_not_negative(data['joint_margin'], 'joint_margin', source),
                        block_clearance=_not_negative(data['block_clearance'], 'block_clearance', source),
                        rest_clear_width=_not_negative(data['rest_clear_width'], 'rest_clear_width', source))


def _parse_scan(data: Any, source: str) -> ScanConfig:
    _check_fields(data, _SCAN_FIELDS, source)
    frames = _integer(data['frames'], 'frames', source)
    if frames <= 0:
        raise StackingConfigError('{}: frames should be positive, not {}'.format(source, frames))
    return ScanConfig(frames, _positive(data['timeout'], 'timeout', source),
                      _positive(data['max_spread'], 'max_spread', source))


def _parse_click_mode(data: Any, source: str) -> ClickModeConfig:
    _check_fields(data, _CLICK_MODE_FIELDS, source)
    return ClickModeConfig(_positive(data['idle_timeout'], 'idle_timeout', source))


def _dictionary(value: Any, source: str) -> str:
    """the name of an aruco dictionary that exists"""
    try:
        _dictionary_constant(value)
    except StackingConfigError as e:
        raise StackingConfigError('{}: dictionary: {}'.format(source, e)) from e
    assert isinstance(value, str)
    return value


def _check_fields(data: Any, required: set[str], source: str) -> None:
    """every key has to be there and no other: a misspelled key would silently leave the default of a later version"""
    if not isinstance(data, dict):
        raise StackingConfigError('{}: should be a mapping with {}'.format(source, ', '.join(sorted(required))))
    missing = required - set(data)
    if missing:
        raise StackingConfigError('{}: missing {}'.format(source, ', '.join(sorted(missing))))
    unknown = set(data) - required
    if unknown:
        raise StackingConfigError('{}: unknown {}'.format(source, ', '.join(sorted(str(key) for key in unknown))))


def _list(value: Any, name: str, source: str, length: int | None = None) -> list[Any]:
    if not isinstance(value, list) or (length is not None and len(value) != length):
        what = 'a list of {} values'.format(length) if length is not None else 'a list'
        raise StackingConfigError('{}: {} should be {}, not {!r}'.format(source, name, what, value))
    return value


def _number(value: Any, name: str, source: str) -> float:
    """a number, a whole number is fine too, but not a bool, nan or infinity"""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise StackingConfigError('{}: {} should be a number, not {!r}'.format(source, name, value))
    return float(value)


def _integer(value: Any, name: str, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StackingConfigError('{}: {} should be a whole number, not {!r}'.format(source, name, value))
    return value


def _positive(value: Any, name: str, source: str) -> float:
    number = _number(value, name, source)
    if number <= 0:
        raise StackingConfigError('{}: {} should be positive, not {}'.format(source, name, number))
    return number


def _not_negative(value: Any, name: str, source: str) -> float:
    number = _number(value, name, source)
    if number < 0:
        raise StackingConfigError('{}: {} should be 0 or more, not {}'.format(source, name, number))
    return number
