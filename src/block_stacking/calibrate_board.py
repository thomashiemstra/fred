"""
Points the closed fingertips of the powered arm at known spots and lets you nudge it with the keyboard, for two
measurements. Run it from the project root with the arm powered and the table clear between the base and the spots:
    python -m src.block_stacking.calibrate_board [corners] [--height 1.0] [--write | --no-write] [--config PATH] [--yes]
    python -m src.block_stacking.calibrate_board sink [--gripper down|forward] [--x 0.0] [--distances 18 25 32 39]
        [--output PATH] [--config PATH] [--yes]

Both modes start the same way: they print what they will do and ask to continue (--yes skips the question), switch the
servos on, close the gripper, move to the park pose of stacking_config.yaml and then to the first spot: a joint move to
above it at motion.hover_height, then straight down at motion.linear_speed to the start height. Every spot is jogged
with the keys below; Enter accepts it. At the end, whatever ended the run (done, abort, an error, Ctrl+C), the arm goes
straight up to the hover height when it is lower, to the park pose, and is laid down on the table by
record_static_load.lower_to_rest, which switches the torque off only there. When that does not work the servos still
hold and python -m src.robot_controllers.dynamixel_robot.calibration.lower_arm lowers the arm.

Heights are the height of the FINGERTIPS above the table whenever the gripper points down. The kinematic tip is d6 from
the wrist centre, the closed fingertips end FINGERTIP_REACH (15 cm, measured) from it whatever d6 says, so with the
gripper down the fingertips are at the tip z minus (FINGERTIP_REACH - d6). The script works for any d6 up to that.
Poses of the arm are planned like those of the block stacker (utils/movement): every move is checked against the
workspace limits of the robot and the ranges of the servos, a jog the planner refuses leaves the arm where it is.

The keys, in a real console (msvcrt): arrow left / right move the spot to robot x minus / plus, arrow up / down to robot
y plus (away from the base) / minus, Page Up / Page Down higher / lower; 1, 2, 3 set the step to 0.5, 1 (default) or
5 mm; Enter accepts the spot, s skips it, Esc or q aborts. Without a console (stdin is not a tty, as in the Run window
of PyCharm) the same things are typed with Enter: l, r, f, b, u, d (optionally with a number of steps, r 3), 1, 2, 3,
ok, s, q. Every jog is one short joint move of the arm (0.25 s) and prints the offset from the start spot in mm and the
height of the fingertips (or the tip) in cm.

The floor: while the gripper points down the script gives the robot workspace limits of its own whose z_min lets the
fingertips down to FINGERTIP_FLOOR (-1 cm, the real arm may be higher than commanded and you stop at the first touch);
the standard limits keep them 1 cm above the table (WorkSpaceLimits.z_min). They are put back when the run ends.

corners: the fingertips go to the two top corners of the printed board, where the yaml has them now, starting
--height cm above the table. Jog them exactly over each corner and accept. The result is both corners as the arm
pointed at them next to where the yaml has them, the distance between them next to the width of the board, and the
new board.position (the origin corner, bottom right) and board.rotation: the direction from the top-left to the
top-right corner is the board's x axis; with one corner only the old rotation is kept and the position shifts. It asks
to type yes before it writes the two lines into the board section of the yaml (--write writes without asking,
--no-write never), keeping every comment and line ending, and loads the file again to make sure it is valid.

sink: how much lower the gripper really is than commanded, at several distances from the base on the line x (default
0, straight ahead of the base, which must be clear). With --gripper down the fingertips start 3 cm above the table at
every distance and you lower them with Page Down until they just touch (a sheet of paper under them starts to drag);
with --gripper forward the gripper points level away from the base, the tip starts at z 6 cm and you lower until the
underside of the gripper touches. The commanded height at the touch is recorded together with the height the encoders
give for the same point at that moment: the difference is what the servos did not hold, the rest is what the encoders
can not see (bending, play, zero errors). The table, a fitted line through the touch heights against the distance and a
csv are the result; nothing is written to a yaml.
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import re
import sys
import time
from collections.abc import Callable, Sequence
from copy import copy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple, cast

import numpy as np

import src.global_constants
from src.block_stacking.stacking_config import BoardConfig, StackingConfig, StackingConfigError, \
    DEFAULT_STACKING_CONFIG_PATH, load_stacking_config
from src.block_stacking.stacking_plan import gripper_down_pose, park_pose
from src.camera.block_detection import board_to_robot
from src.global_constants import WorkSpaceLimits
from src.global_objects import get_robot
from src.kinematics.kinematics import forward_kinematics, forward_position_kinematics
from src.kinematics.kinematics_utils import Pose
from src.robot_controllers.abstract_robot_controller import AbstractRobotController
from src.robot_controllers.dynamixel_robot.calibration import record_static_load
from src.robot_controllers.dynamixel_robot.dynamixel_utils import DynamixelError
from src.utils.movement import MAX_POSITION_ERROR, PlannedMove, execute_move, plan_move_to_angles, \
    plan_straight_line
from src.utils.movement_exception import MovementException

# the closed fingertips end this far (cm) from the wrist centre, measured, whatever d6 of the robot says
FINGERTIP_REACH = 15.0
# the lowest height (cm) the fingertips may be commanded to with the gripper down: the real arm may be higher than
# commanded, and the user stops at the first touch
FINGERTIP_FLOOR = -1.0
DEFAULT_CORNER_HEIGHT = 1.0  # cm, of the fingertips above the table when a corner is first shown
SINK_DOWN_START = 3.0  # cm, of the fingertips above the table at every distance
SINK_FORWARD_START = 6.0  # cm, tip z where the gripper points forward
FORWARD_TRAVEL = 8.0  # cm, tip z where the gripper that points forward moves between the spots
# with the gripper level the tip has to be at least wrist_radius_min + d6 (27.5 cm) from the base axis, so the forward
# distances start at 30
DEFAULT_DISTANCES = {'down': (18.0, 25.0, 32.0, 39.0), 'forward': (30.0, 37.0, 44.0, 51.0)}
# the board is this much (cm) narrower or wider than printed before the result warns about it
WIDTH_TOLERANCE = 0.3

# a jog is a short joint move, stretched when it is long so that the end effector does not go faster than this
JOG_DURATION = 0.25  # seconds
JOG_MAX_SPEED = 4.0  # cm/s
STEP_SIZES_MM = {1: 0.5, 2: 1.0, 3: 5.0}
DEFAULT_STEP_MM = 1.0
GRIPPER_WAIT = 1.0  # seconds the gripper gets to close
SETTLE_TIME = 0.5  # seconds the arm gets to settle before the encoders are read
LOWER_ARM_COMMAND = 'python -m src.robot_controllers.dynamixel_robot.calibration.lower_arm'
SINK_FIELDS = ('gripper', 'x', 'y', 'commanded_height', 'encoder_height', 'd6', 'date')
TUNING_DIRECTORY = Path(__file__).resolve().parent.parent / 'robot_controllers' / 'dynamixel_robot' / 'calibration' \
    / 'tuning'

# what a key press asks for
LEFT, RIGHT, FORWARD, BACK, HIGHER, LOWER = 'left', 'right', 'forward', 'back', 'higher', 'lower'
STEP, ACCEPT, SKIP, ABORT = 'step', 'accept', 'skip', 'abort'
# robot x, y, z of one step of a jog action: forward is away from the base
DIRECTIONS = {LEFT: (-1.0, 0.0, 0.0), RIGHT: (1.0, 0.0, 0.0), FORWARD: (0.0, 1.0, 0.0), BACK: (0.0, -1.0, 0.0),
              HIGHER: (0.0, 0.0, 1.0), LOWER: (0.0, 0.0, -1.0)}
# the second character msvcrt.getwch gives after the prefix of an arrow or page key
_CONSOLE_ARROWS = {'H': FORWARD, 'P': BACK, 'K': LEFT, 'M': RIGHT, 'I': HIGHER, 'Q': LOWER}
_TYPED_ACTIONS = {'l': LEFT, 'r': RIGHT, 'f': FORWARD, 'b': BACK, 'u': HIGHER, 'd': LOWER}

DONE, ABORTED, INTERRUPTED, FAILED = 'done', 'aborted', 'interrupted', 'failed'

CONSOLE_HELP = ('keys: arrows move the spot (left/right: x, up: away from the base, down: toward it), Page Up / Page '
                'Down higher / lower, 1 2 3 step of 0.5 / 1 / 5 mm, Enter accepts the spot, s skips it, Esc or q '
                'aborts')
TYPED_HELP = ('type a command and press Enter: l r (x minus, plus), f b (y plus away from the base, minus), u d '
              '(higher, lower), each with an optional number of steps like "r 3"; 1 2 3 set the step to 0.5 / 1 / 5 '
              'mm; ok accepts the spot, s skips it, q aborts')


class Aborted(Exception):
    """raised to end the run when the user aborts"""


class Command(NamedTuple):
    action: str
    steps: int = 1  # the steps of a jog, or which of the step sizes 1, 2 and 3 a STEP action sets


@dataclass
class KeyReader:
    """reads one command of the user, None for something that means nothing; help says which keys there are"""
    read: Callable[[], Command | None]
    help: str

    def __call__(self) -> Command | None:
        return self.read()


def decode_console_key(getwch: Callable[[], str]) -> Command | None:
    """
    the command of the next key of msvcrt.getwch; arrow and page keys come as a prefix character ('\\x00' or '\\xe0')
    followed by a letter
    :raises KeyboardInterrupt: for Ctrl+C, which getwch returns as a character instead of raising it
    """
    char = getwch()
    if char in ('\x00', '\xe0'):
        action = _CONSOLE_ARROWS.get(getwch())
        return None if action is None else Command(action)
    if char == '\x03':
        raise KeyboardInterrupt
    if char in ('\r', '\n'):
        return Command(ACCEPT)
    if char == '\x1b' or char.lower() == 'q':
        return Command(ABORT)
    if char.lower() == 's':
        return Command(SKIP)
    if char in ('1', '2', '3'):
        return Command(STEP, int(char))
    return None


def parse_typed_command(line: str) -> Command | None:
    """the command of a typed line, for a console that can not read single keys; None when it is not understood"""
    text = line.strip().lower()
    if text == 'ok':
        return Command(ACCEPT)
    if text == 's':
        return Command(SKIP)
    if text == 'q':
        return Command(ABORT)
    if text in ('1', '2', '3'):
        return Command(STEP, int(text))
    match = re.fullmatch(r'([lrfbud])\s*(\d+)?', text)
    if match is None:
        return None
    count = 1 if match.group(2) is None else int(match.group(2))
    return Command(_TYPED_ACTIONS[match.group(1)], count) if count >= 1 else None


def make_key_reader(say: Callable[..., None] = print, console: bool | None = None,
                    getwch: Callable[[], str] | None = None, input_fn: Callable[[str], str] = input) -> KeyReader:
    """
    the reader of the user's commands: single keys through msvcrt in a real console, typed lines otherwise
    :param console: whether to read single keys, by default when stdin is a console on Windows
    :param getwch: the function that reads one key, by default msvcrt.getwch
    """
    if console is None:
        console = sys.stdin is not None and sys.stdin.isatty() and (getwch is not None or sys.platform == 'win32')
    if console:
        if getwch is None:
            import msvcrt
            getwch = msvcrt.getwch
        read_key = getwch
        return KeyReader(lambda: decode_console_key(read_key), CONSOLE_HELP)

    def read_line() -> Command | None:
        try:
            line = input_fn('> ')
        except EOFError:
            say('there is no more input, aborting')
            return Command(ABORT)
        command = parse_typed_command(line)
        if command is None:
            say('not understood: {!r}'.format(line))
        return command

    return KeyReader(read_line, TYPED_HELP)


@dataclass
class Context:
    """everything the run needs from outside, so that tests can give it fakes"""
    robot: AbstractRobotController
    config: StackingConfig
    read_key: Callable[[], Command | None]
    execute: Callable[..., bool] = execute_move  # execute(robot, move, should_stop) like execute_move
    sleep: Callable[[float], None] = time.sleep
    say: Callable[..., None] = print
    ask: Callable[[str], str] = input
    now: Callable[[], datetime] = datetime.now
    key_help: str = CONSOLE_HELP


class Jog(NamedTuple):
    action: str  # ACCEPT, SKIP or ABORT
    x: float
    y: float
    tip_z: float


def limits_with_tip_floor(limits: Any, z_min: float) -> type[WorkSpaceLimits]:
    """
    a copy of the workspace limits with the floor of the tip replaced: a small object with every attribute of the
    limits (the class or the object of a robot), so the robot's own limits are not changed
    """
    attributes = {name: getattr(limits, name) for name in dir(limits)
                  if not name.startswith('_') and not callable(getattr(limits, name))}
    attributes['z_min'] = float(z_min)
    return cast(type[WorkSpaceLimits], SimpleNamespace(**attributes))


class Arm:
    """
    The arm as the run moves it: where the last move ended (the commanded angles, not what the servos read back, which
    sag), and the planned, checked moves. Every move is planned from the commanded angles, so a refused jog changes
    nothing.
    :param down: whether the gripper points straight down (the heights are those of the fingertips) or level forward
        (the heights are those of the tip)
    :param safe_height: the height (of the fingertips when down, else of the tip) where the arm moves from spot to spot
    """

    def __init__(self, ctx: Context, down: bool, safe_height: float) -> None:
        self.ctx = ctx
        self.robot = ctx.robot
        self.down = down
        self.safe_tip_z = self.tip_z_of(safe_height)
        self.angles: np.ndarray | None = None
        self.uncertain = False  # a move was cut short, the angles of the arm have to be read
        self.pointing_forward = False  # the gripper has been turned forward for the spots, forward mode only
        self.step_mm = DEFAULT_STEP_MM
        self.help_shown = False

    # heights

    @property
    def drop(self) -> float:
        """how far the fingertips are below the tip when the gripper points down, 0 when it points forward"""
        return FINGERTIP_REACH - self.robot.robot_config.d6 if self.down else 0.0

    @property
    def point_name(self) -> str:
        return 'fingertips' if self.down else 'tip'

    def tip_z_of(self, height: float) -> float:
        return height + self.drop

    def height_of(self, tip_z: float) -> float:
        return tip_z - self.drop

    def pose(self, x: float, y: float, tip_z: float) -> Pose:
        """the pose of the tip: the gripper pointing straight down (the flip of the park pose) or level forward"""
        if self.down:
            return gripper_down_pose(x, y, tip_z, 0.0, True)
        return Pose(x, y, tip_z)

    def encoder_height(self) -> float | None:
        """the height the encoders give for the point now, None when they can not be read"""
        angles = self.robot.get_current_angles()
        if angles is None:
            return None
        return self.height_of(float(forward_position_kinematics(angles, self.robot.robot_config)[-1][2]))

    # planning and executing moves

    def _angles_now(self) -> np.ndarray:
        if self.angles is None:
            raise RuntimeError('the angles of the arm are not known')
        return self.angles

    def _angles_of(self, pose: Pose) -> np.ndarray:
        """
        the angles of the pose from the inverse kinematics
        :raises MovementException: if the arm can not get to the pose
        """
        angles = np.asarray(self.robot.pose_to_angles(pose), dtype=np.float64)
        if not np.all(np.isfinite(angles)):
            raise MovementException('the inverse kinematics gave no angles')
        reached = forward_position_kinematics(angles, self.robot.robot_config)[-1]
        if np.linalg.norm(reached - [pose.x, pose.y, pose.z]) > MAX_POSITION_ERROR:
            raise MovementException('the arm can not get to x={:.1f} y={:.1f} z={:.1f}'.format(pose.x, pose.y, pose.z))
        return angles

    def _execute(self, move: PlannedMove) -> None:
        self.uncertain = True  # stays set when the move is cut short by an exception
        if not self.ctx.execute(self.robot, move, None):
            raise MovementException('the move was stopped')
        self.angles = np.asarray(move.angles[-1], dtype=np.float64).copy()
        self.uncertain = False

    def joint_move_to(self, target: np.ndarray) -> None:
        """a joint move at motion.joint_speed to the angles"""
        self._execute(plan_move_to_angles(self.robot, self._angles_now(), target,
                                          angular_speed=self.ctx.config.motion.joint_speed))

    def line_to(self, x: float, y: float, tip_z: float) -> None:
        """a straight line at motion.linear_speed of the tip to the point, the orientation stays"""
        angles = self._angles_now()
        start = forward_kinematics(angles, self.robot.robot_config)
        stop = copy(start)
        stop.x, stop.y, stop.z = x, y, tip_z
        self._execute(plan_straight_line(self.robot, start, stop, self.ctx.config.motion.linear_speed,
                                         current_angles=angles))

    def raise_to(self, tip_z: float) -> None:
        """straight up to the tip z when the tip is lower"""
        start = forward_kinematics(self._angles_now(), self.robot.robot_config)
        if start.z < tip_z - 1e-6:
            self.line_to(start.x, start.y, tip_z)

    def prepare(self) -> None:
        """close the gripper, read the angles and move to the park pose"""
        ctx = self.ctx
        ctx.say('closing the gripper and moving to the park pose')
        self.robot.set_gripper(ctx.config.gripper.closed_state)
        ctx.sleep(GRIPPER_WAIT)
        angles = self.robot.get_current_angles()
        if angles is None:
            raise RuntimeError('the angles of the arm could not be read')
        self.angles = np.asarray(angles, dtype=np.float64).copy()
        self.joint_move_to(self._angles_of(park_pose(ctx.config, True)))

    def go_to_spot(self, x: float, y: float, height: float) -> None:
        """
        to the spot at that height: up to the safe height when lower, over the spot (a joint move, with the gripper
        down; with the gripper forward the first spot is reached by a joint move and the others in a straight line
        at the safe height) and straight down
        :raises MovementException: if a move is refused, the arm stays where the last move ended
        """
        self.raise_to(self.safe_tip_z)
        over = self.pose(x, y, self.safe_tip_z)
        if self.down or not self.pointing_forward:
            self.joint_move_to(self._angles_of(over))
            self.pointing_forward = True
        else:
            self.line_to(x, y, self.safe_tip_z)
        self.line_to(x, y, self.tip_z_of(height))

    # the keys

    def status(self, x: float, y: float, tip_z: float, x0: float, y0: float) -> str:
        return 'offset x {:+.1f} mm, y {:+.1f} mm, {} {:.2f} cm'.format((x - x0) * 10, (y - y0) * 10, self.point_name,
                                                                         self.height_of(tip_z))

    def jog(self, name: str, x: float, y: float, tip_z: float) -> Jog:
        """
        let the user move the arm from the spot until one of ACCEPT, SKIP and ABORT, returns that and where the arm is
        """
        say = self.ctx.say
        say('{}: jog the {} {}, then accept the spot'.format(name, self.point_name,
                                                                'over it' if self.down else 'to it'))
        if not self.help_shown:
            say(self.ctx.key_help)
            self.help_shown = True
        say('  ' + self.status(x, y, tip_z, x, y) + ', step {:g} mm'.format(self.step_mm))
        x0, y0 = x, y
        while True:
            command = self.ctx.read_key()
            if command is None:
                continue
            if command.action in (ACCEPT, SKIP, ABORT):
                return Jog(command.action, x, y, tip_z)
            if command.action == STEP:
                self.step_mm = STEP_SIZES_MM[command.steps]
                say('  step {:g} mm'.format(self.step_mm))
                continue
            distance = self.step_mm / 10 * command.steps
            dx, dy, dz = (distance * component for component in DIRECTIONS[command.action])
            try:
                self._jog_to(x + dx, y + dy, tip_z + dz, distance)
            except MovementException as e:
                say('  refused, the arm stays: {}'.format(e))
                continue
            x, y, tip_z = x + dx, y + dy, tip_z + dz
            say('  ' + self.status(x, y, tip_z, x0, y0))

    def _jog_to(self, x: float, y: float, tip_z: float, distance: float) -> None:
        target = self._angles_of(self.pose(x, y, tip_z))
        duration = max(JOG_DURATION, distance / JOG_MAX_SPEED)
        self._execute(plan_move_to_angles(self.robot, self._angles_now(), target, duration=duration))

    # the end

    def finish(self) -> None:
        """straight up to the safe height when lower, then to the park pose; never raises"""
        say = self.ctx.say
        try:
            if self.angles is None or self.uncertain:
                angles = self.robot.get_current_angles()
                if angles is None:
                    say('the angles of the arm can not be read, so it is not moved to the park pose')
                    return
                self.angles = np.asarray(angles, dtype=np.float64).copy()
                self.uncertain = False
            self.raise_to(self.safe_tip_z)
            park = self._angles_of(park_pose(self.ctx.config, True))
            if float(np.abs(self._angles_now()[1:] - park[1:]).max()) > 1e-3:
                say('moving to the park pose')
                self.joint_move_to(park)
        except MovementException as e:
            say('the arm could not go back to the park pose: {}'.format(e))
        except KeyboardInterrupt:
            say('interrupted, the arm stays where it is')
        except Exception as e:
            say('the arm could not go back to the park pose: {}: {}'.format(type(e).__name__, e))


# the board corners


@dataclass(frozen=True)
class BoardEstimate:
    x: float
    y: float
    rotation: float  # degrees


def board_size(board: BoardConfig) -> tuple[float, float]:
    """the width and the height of the board in cm"""
    return board.squares_x * board.square_length, board.squares_y * board.square_length


def corner_board_points(board: BoardConfig) -> dict[str, tuple[float, float]]:
    """the top corners in the board frame: the origin is bottom right, x to the right, y up along the right edge"""
    width, height = board_size(board)
    return {'top-right': (0.0, height), 'top-left': (-width, height)}


def corner_robot_points(board: BoardConfig) -> dict[str, tuple[float, float]]:
    """where the yaml puts the top corners in the robot frame"""
    return {name: (float(point[0]), float(point[1])) for name, point in
            ((name, board_to_robot(corner, board)) for name, corner in corner_board_points(board).items())}


def _rotate(point: tuple[float, float], degrees: float) -> tuple[float, float]:
    cosine, sine = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return point[0] * cosine - point[1] * sine, point[0] * sine + point[1] * cosine


def estimate_board(board: BoardConfig, measured: dict[str, tuple[float, float]]) -> BoardEstimate | None:
    """
    the board position and rotation that put the corners where the arm pointed at them: with both corners the
    direction from the top-left to the top-right corner is the board's x axis; with one the rotation stays
    :param measured: corner name -> where the arm pointed (robot x, y), only the corners that were measured
    :return: None when no corner was measured
    """
    points = corner_board_points(board)
    if not measured:
        return None
    rotation = board.rotation
    if len(measured) == 2:
        right, left = measured['top-right'], measured['top-left']
        new = math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))
        rotation = board.rotation + math.remainder(new - board.rotation, 360.0)  # the same turn count as the yaml has
    name = next(iter(measured))
    offset = _rotate(points[name], rotation)
    return BoardEstimate(measured[name][0] - offset[0], measured[name][1] - offset[1], rotation)


# writing the yaml

_BOARD_HEADER = re.compile(r'board:[ \t]*(#.*)?')
_TOP_LEVEL_KEY = re.compile(r'[^\s#].*')
_POSE_LINE = re.compile(r'(?P<indent>[ \t]+)(?P<key>position|rotation):(?P<gap>[ \t]*)(?P<value>[^#\r\n]*?)'
                        r'(?P<space>[ \t]*)(?:#(?P<comment>[^\r\n]*))?(?P<eol>\r\n|\n|\r)?')
_JOGGED_NOTE = re.compile(r';?\s*jogged to the corners on \d{4}-\d{2}-\d{2}')


def format_number(value: float) -> str:
    """a number for the yaml: two decimals at most, never -0.0"""
    return repr(round(value, 2) + 0.0)


def board_pose_lines(estimate: BoardEstimate, date: str, indent: str = '  ') -> list[str]:
    """the two lines to put under board:"""
    return ['{}position: [{}, {}]  # jogged to the corners on {}'.format(
        indent, format_number(estimate.x), format_number(estimate.y), date),
        '{}rotation: {}'.format(indent, format_number(estimate.rotation))]


def replace_board_pose(yaml_text: str, estimate: BoardEstimate, date: str) -> str:
    """
    the yaml text with the position and rotation lines of the board section replaced, everything else is kept as it is:
    the tower section has lines with the same names, comments, spacing and line endings stay, and the position line
    gets the note that it was jogged to the corners (an earlier note of this script is replaced)
    :raises ValueError: if the board section does not have exactly one position and one rotation line
    """
    lines = yaml_text.splitlines(keepends=True)
    in_board = False
    found: dict[str, int] = {}
    for i, line in enumerate(lines):
        text = line.rstrip('\r\n')
        if _BOARD_HEADER.fullmatch(text):
            in_board = True
        elif _TOP_LEVEL_KEY.fullmatch(text):
            in_board = False
        elif in_board:
            match = _POSE_LINE.fullmatch(line)
            if match is not None:
                if match.group('key') in found:
                    raise ValueError('the board section has more than one {} line'.format(match.group('key')))
                found[match.group('key')] = i
    if set(found) != {'position', 'rotation'}:
        raise ValueError('the board section of the yaml needs a position and a rotation line, only {} found'.format(
            sorted(found) or 'none'))

    for key, value in (('position', '[{}, {}]'.format(format_number(estimate.x), format_number(estimate.y))),
                       ('rotation', format_number(estimate.rotation))):
        match = _POSE_LINE.fullmatch(lines[found[key]])
        assert match is not None
        comment = match.group('comment')
        if key == 'position':
            old = '' if comment is None else _JOGGED_NOTE.sub('', comment).strip()
            comment = ' ' + (old + '; ' if old else '') + 'jogged to the corners on {}'.format(date)
        tail = '' if comment is None else (match.group('space') or '  ') + '#' + comment
        lines[found[key]] = '{}{}:{}{}{}{}'.format(match.group('indent'), key, match.group('gap') or ' ', value, tail,
                                                   match.group('eol') or '')
    return ''.join(lines)


def write_board_pose(path: str | os.PathLike[str], estimate: BoardEstimate, date: str) -> None:
    """
    write the board pose into the yaml and load the file again to make sure it is valid; the old content is put back
    when it is not
    :raises ValueError: if the yaml has no board position and rotation to replace, or the written file does not give
        the values that were written
    :raises StackingConfigError: if the written file is not a valid configuration
    """
    path = Path(path)
    old = path.read_bytes()
    new_text = replace_board_pose(old.decode('utf-8'), estimate, date)  # bytes, so line endings are not translated
    path.write_bytes(new_text.encode('utf-8'))
    try:
        board = load_stacking_config(path).board
        if max(abs(board.x - estimate.x), abs(board.y - estimate.y), abs(board.rotation - estimate.rotation)) > 0.006:
            raise ValueError('the written file has the board at {}, {} turned {} instead'.format(
                board.x, board.y, board.rotation))
    except (StackingConfigError, ValueError):
        path.write_bytes(old)
        raise


# the sessions: what is measured at which spots


class Session:
    """what a mode measures: where the arm goes, how it is reported; the arm is already at the park pose in measure"""
    down = True
    safe_height = 0.0  # where the arm moves between the spots: fingertips above the table when down, else the tip z
    lower_floor = False  # whether the run lowers z_min of the workspace limits for the fingertips, see run_session

    def describe(self) -> list[str]:
        raise NotImplementedError

    def measure(self, ctx: Context, arm: Arm) -> None:
        raise NotImplementedError

    def report(self, ctx: Context, outcome: str) -> int:
        raise NotImplementedError


class CornersSession(Session):
    """the fingertips jogged to the two top corners of the board"""
    lower_floor = True

    def __init__(self, config: StackingConfig, config_path: str, height: float, write: bool | None) -> None:
        self.config = config
        self.config_path = config_path
        self.height = height
        self.write = write  # True writes without asking, False never, None asks
        self.safe_height = config.motion.hover_height
        self.corners = corner_robot_points(config.board)
        self.measured: dict[str, tuple[float, float]] = {}

    def describe(self) -> list[str]:
        lines = ['The fingertips of the closed gripper, pointing down, go to the two top corners of the board where '
                 'the yaml has them,']
        for name, (x, y) in self.corners.items():
            lines.append('  {}: x {:.2f}, y {:.2f} cm'.format(name, x, y))
        lines.append('starting {:g} cm above the table. You jog them exactly over each corner with the keys and '
                     'accept it.'.format(self.height))
        return lines

    def measure(self, ctx: Context, arm: Arm) -> None:
        for name, (x, y) in self.corners.items():
            ctx.say('\nto the {} corner'.format(name))
            try:
                arm.go_to_spot(x, y, self.height)
            except MovementException as e:
                ctx.say('the arm can not go to the {} corner, it is skipped: {}'.format(name, e))
                continue
            jog = arm.jog('{} corner'.format(name), x, y, arm.tip_z_of(self.height))
            if jog.action == ABORT:
                raise Aborted()
            if jog.action == SKIP:
                ctx.say('the {} corner is skipped'.format(name))
            else:
                self.measured[name] = (jog.x, jog.y)

    def report(self, ctx: Context, outcome: str) -> int:
        say = ctx.say
        if outcome != DONE:
            say('\n{}, nothing is written'.format(outcome))
            return 1
        board = self.config.board
        say('')
        for name, (x, y) in self.corners.items():
            if name not in self.measured:
                say('{} corner: not measured'.format(name))
                continue
            mx, my = self.measured[name]
            say('{} corner: the arm pointed at x {:.2f}, y {:.2f} cm, the yaml has it at x {:.2f}, y {:.2f}: shift '
                'x {:+.1f} mm, y {:+.1f} mm'.format(name, mx, my, x, y, (mx - x) * 10, (my - y) * 10))
        estimate = estimate_board(board, self.measured)
        if estimate is None:
            say('no corner was measured, nothing to write')
            return 0
        if len(self.measured) == 2:
            width = board_size(board)[0]
            right, left = self.measured['top-right'], self.measured['top-left']
            distance = math.hypot(right[0] - left[0], right[1] - left[1])
            say('the corners are {:.2f} cm apart, the board is {:.2f} cm wide: difference {:+.2f} cm'.format(
                distance, width, distance - width))
            if abs(distance - width) > WIDTH_TOLERANCE:
                say('WARNING: that is more than {:g} cm, check the print scale of the board and the accuracy of the '
                    'arm before trusting the rotation'.format(WIDTH_TOLERANCE))
        else:
            say('only one corner was measured: the rotation stays {}, only the position shifts'.format(
                format_number(board.rotation)))
        date = ctx.now().strftime('%Y-%m-%d')
        say('the new board pose, under board: in {}'.format(self.config_path))
        for line in board_pose_lines(estimate, date):
            say(line)
        if not self._wants_to_write(ctx):
            say('nothing is written')
            return 0
        try:
            write_board_pose(self.config_path, estimate, date)
        except (OSError, ValueError, StackingConfigError) as e:
            say('NOT written, the old file is as it was: {}'.format(e))
            return 1
        say('written to {}'.format(self.config_path))
        return 0

    def _wants_to_write(self, ctx: Context) -> bool:
        if self.write is not None:
            return self.write
        try:
            answer = ctx.ask('write these lines into {}? type yes: '.format(self.config_path))
        except EOFError:
            ctx.say('there is no input to answer the question, use --write to write without asking')
            return False
        return answer.strip().lower() == 'yes'


@dataclass(frozen=True)
class SinkRow:
    gripper: str
    x: float
    y: float
    commanded: float
    encoder: float | None


def fit_line(distances: Sequence[float], heights: Sequence[float]) -> tuple[float, float] | None:
    """the least squares line height = intercept + slope * distance, None when there is no line to fit"""
    if len(distances) < 2 or max(distances) - min(distances) < 1e-9:
        return None
    slope, intercept = np.polyfit(distances, heights, 1)
    return float(intercept), float(slope)


def default_output(gripper: str, now: datetime) -> str:
    return str(TUNING_DIRECTORY / 'sink_{}_{}.csv'.format(gripper, now.strftime('%Y%m%d_%H%M')))


class SinkSession(Session):
    """the commanded height at which the arm touches the table, at several distances"""

    def __init__(self, config: StackingConfig, gripper: str, x: float, distances: Sequence[float],
                 output: str | None) -> None:
        self.config = config
        self.gripper = gripper
        self.down = gripper == 'down'
        self.x = x
        self.distances = tuple(distances)
        self.output = output
        self.start_height = SINK_DOWN_START if self.down else SINK_FORWARD_START
        self.safe_height = config.motion.hover_height if self.down else FORWARD_TRAVEL
        self.lower_floor = self.down
        self.rows: list[SinkRow] = []

    def describe(self) -> list[str]:
        what = ('the closed fingertips, pointing down, start {:g} cm above the table at every distance and are '
                'lowered until they just touch (a sheet of paper under them starts to drag)'.format(self.start_height)
                if self.down else
                'the gripper points level away from the base, its tip starts at z {:g} cm at every distance and is '
                'lowered until the underside of the gripper touches'.format(self.start_height))
        distances = ', '.join('{:g}'.format(distance) for distance in self.distances)
        return ['The arm goes to the line x {:g} cm at the distances y {} cm, which must be clear of everything;'
                .format(self.x, distances), what + '.',
            'The commanded height at the touch and the height of the encoders are recorded; nothing is written to a '
            'yaml.']

    def measure(self, ctx: Context, arm: Arm) -> None:
        for y in self.distances:
            ctx.say('\nto the distance y {:g} cm'.format(y))
            try:
                arm.go_to_spot(self.x, y, self.start_height)
            except MovementException as e:
                ctx.say('the arm can not go to y {:g} cm, it is skipped: {}'.format(y, e))
                continue
            jog = arm.jog('y {:g} cm'.format(y), self.x, y, arm.tip_z_of(self.start_height))
            if jog.action == ABORT:
                raise Aborted()
            if jog.action == SKIP:
                ctx.say('y {:g} cm is skipped'.format(y))
                continue
            ctx.sleep(SETTLE_TIME)
            encoder = arm.encoder_height()
            commanded = arm.height_of(jog.tip_z)
            self.rows.append(SinkRow(self.gripper, jog.x, jog.y, commanded, encoder))
            ctx.say('recorded: commanded {:.2f} cm, encoders {}'.format(
                commanded, 'unreadable' if encoder is None else '{:.2f} cm'.format(encoder)))

    def report(self, ctx: Context, outcome: str) -> int:
        say = ctx.say
        if outcome != DONE:
            say('\n{}, the spots measured so far are reported'.format(outcome))
        if not self.rows:
            say('no spot was measured')
            return 0 if outcome == DONE else 1
        d6 = ctx.robot.robot_config.d6
        say('\nthe {} of the {} at the touch, d6 {:g}:'.format(
            'height of the fingertips' if self.down else 'tip z', 'gripper pointing ' + self.gripper, d6))
        say('  {:>10}  {:>12}  {:>12}'.format('distance cm', 'commanded cm', 'encoders cm'))
        distances = [math.hypot(row.x, row.y) for row in self.rows]
        for distance, row in zip(distances, self.rows):
            say('  {:>10.1f}  {:>12.2f}  {:>12}'.format(
                distance, row.commanded, '' if row.encoder is None else '{:.2f}'.format(row.encoder)))
        fit = fit_line(distances, [row.commanded for row in self.rows])
        if fit is None:
            say('one distance only, no line to fit')
        else:
            intercept, slope = fit
            say('the touch height changes by {:+.2f} cm per 10 cm of distance (the line is {:+.2f} cm at the base '
                'plus {:+.4f} cm per cm)'.format(slope * 10, intercept, slope))
        self._write_csv(ctx, d6)
        return 0 if outcome == DONE else 1

    def _write_csv(self, ctx: Context, d6: float) -> None:
        now = ctx.now()
        path = self.output or default_output(self.gripper, now)
        try:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(path, 'w', newline='') as output:
                writer = csv.writer(output)
                writer.writerow(SINK_FIELDS)
                for row in self.rows:
                    writer.writerow([row.gripper, '{:g}'.format(row.x), '{:g}'.format(row.y),
                                     '{:.4f}'.format(row.commanded),
                                     '' if row.encoder is None else '{:.4f}'.format(row.encoder),
                                     '{:g}'.format(d6), now.strftime('%Y-%m-%d %H:%M')])
        except OSError as e:
            ctx.say('the csv could not be written: {}'.format(e))
            return
        ctx.say('the rows are in {}'.format(path))


# the run


def lay_down(ctx: Context) -> None:
    """lower the arm to the table and switch the torque off there, or say why that did not work"""
    why = record_static_load.lower_to_rest(cast(Any, ctx.robot), sleep=ctx.sleep)
    if why is not None:
        ctx.say('the arm is not laid down: {}'.format(why))
        ctx.say('the servos still hold the arm; {} lowers it'.format(LOWER_ARM_COMMAND))


def run_session(ctx: Context, session: Session) -> int:
    """
    switch the servos on, run the session and, whatever ends it, bring the arm back and lay it down: straight up to the
    safe height when lower, to the park pose, then record_static_load.lower_to_rest, which switches the torque off
    only after the arm is down. The workspace limits of the robot are put back first. An exception that is not a
    refused move or Ctrl+C goes on to the caller after all that.
    :return: the return code of main: 0 when the run was finished
    """
    robot = ctx.robot
    if robot.enable_servos() is False:
        ctx.say('the servos could not be switched on, nothing was moved')
        return 1
    arm = Arm(ctx, session.down, session.safe_height)
    own_limits = 'workspace_limits' in vars(robot)
    original_limits = robot.workspace_limits
    outcome = DONE
    try:
        if session.lower_floor:
            floor = FINGERTIP_FLOOR + (FINGERTIP_REACH - robot.robot_config.d6)
            robot.workspace_limits = limits_with_tip_floor(original_limits, floor)
            ctx.say('the fingertips may be commanded down to {:g} cm (the tip down to {:g} cm) for this run'.format(
                FINGERTIP_FLOOR, floor))
        try:
            arm.prepare()
            session.measure(ctx, arm)
        except Aborted:
            outcome = ABORTED
        except MovementException as e:
            ctx.say('stopped, a move is refused: {}'.format(e))
            outcome = FAILED
        except KeyboardInterrupt:
            ctx.say('\ninterrupted')
            outcome = INTERRUPTED
        finally:
            arm.finish()
    finally:
        if session.lower_floor:
            if own_limits:
                robot.workspace_limits = original_limits
            else:
                vars(robot).pop('workspace_limits', None)
        lay_down(ctx)
    return session.report(ctx, outcome)


# the command line


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python -m src.block_stacking.calibrate_board',
        description='point the fingertips of the arm at known spots and nudge them with the keyboard: corners '
                    'calibrates board.position and board.rotation of the stacking configuration, sink measures how '
                    'much lower the gripper really is than commanded')
    parser.add_argument('mode', nargs='?', choices=('corners', 'sink'), default='corners',
                        help='what to measure (default corners)')
    parser.add_argument('--config', default=DEFAULT_STACKING_CONFIG_PATH, help='the stacking configuration yaml')
    parser.add_argument('--yes', action='store_true', help="don't ask before the arm moves")
    corners = parser.add_argument_group('corners')
    corners.add_argument('--height', type=float, default=None,
                         help='cm above the table the fingertips start at (default {:g})'.format(DEFAULT_CORNER_HEIGHT))
    write = corners.add_mutually_exclusive_group()
    write.add_argument('--write', action='store_true', help='write the new board pose without asking')
    write.add_argument('--no-write', action='store_true', help='never write the yaml')
    sink = parser.add_argument_group('sink')
    sink.add_argument('--gripper', choices=('down', 'forward'), default=None,
                      help='where the gripper points (default down)')
    sink.add_argument('--x', type=float, default=None, help='the line the distances are on, cm (default 0)')
    sink.add_argument('--distances', type=float, nargs='+', default=None,
                      help='y values in cm (default 18 25 32 39 down, 25 33 41 49 forward)')
    sink.add_argument('--output', default=None, help='the csv (default sink_<gripper>_<date>.csv in the tuning folder)')
    return parser


def make_session(args: argparse.Namespace, parser: argparse.ArgumentParser, config: StackingConfig) -> Session:
    """the session the options ask for, parser.error for options that do not fit the mode"""
    if args.mode == 'corners':
        if any(value is not None for value in (args.gripper, args.x, args.distances, args.output)):
            parser.error('--gripper, --x, --distances and --output are options of sink')
        height = DEFAULT_CORNER_HEIGHT if args.height is None else args.height
        if not FINGERTIP_FLOOR < height <= config.motion.hover_height:
            parser.error('--height should be more than {:g} and at most the hover height {:g}'.format(
                FINGERTIP_FLOOR, config.motion.hover_height))
        return CornersSession(config, args.config, height, True if args.write else False if args.no_write else None)
    if args.height is not None or args.write or args.no_write:
        parser.error('--height, --write and --no-write are options of corners')
    gripper = args.gripper or 'down'
    distances = DEFAULT_DISTANCES[gripper] if args.distances is None else tuple(args.distances)
    if any(distance <= 0 for distance in distances):
        parser.error('--distances should be positive')
    return SinkSession(config, gripper, 0.0 if args.x is None else args.x, distances, args.output)


def main(argv: list[str] | None = None, *, make_robot: Callable[[], AbstractRobotController] = get_robot,
         read_key: KeyReader | None = None, execute: Callable[..., bool] = execute_move,
         sleep: Callable[[float], None] = time.sleep, say: Callable[..., None] = print,
         ask: Callable[[str], str] = input, now: Callable[[], datetime] = datetime.now) -> int:
    """
    run the script; everything that touches the outside has a parameter so that the tests give fakes (only the robot
    is built here, after the question to go on was answered)
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s: %(message)s')
    try:
        config = load_stacking_config(args.config)
    except StackingConfigError as e:
        say(e)
        return 1
    session = make_session(args, parser, config)

    for line in session.describe():
        say(line)
    say('The arm moves: it goes to the park pose, to the spots and at the end lies down on the table. Nothing may be '
        'in its way, in particular the table between the base and the spots has to be clear.')
    if not args.yes:
        try:
            answer = ask('type yes to switch the servos on and start: ')
        except EOFError:
            say('there is no input to answer the question, use --yes for an unattended run')
            return 1
        if answer.strip().lower() != 'yes':
            say('nothing was moved')
            return 0
    if src.global_constants.use_simulation:
        say('the board can only be calibrated on the real robot, use_simulation is set')
        return 1
    try:
        robot = make_robot()
    except DynamixelError as e:
        say(e)
        return 1
    reader = read_key if read_key is not None else make_key_reader(say)
    ctx = Context(robot, config, reader, execute, sleep, say, ask, now, reader.help)
    try:
        return run_session(ctx, session)
    except KeyboardInterrupt:
        say('\ninterrupted')
        return 1


if __name__ == '__main__':
    sys.exit(main())
