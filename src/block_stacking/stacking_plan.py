"""
The plan of the block stacking: given where the blocks lie, which block goes onto the tower in which order, how the
gripper takes it, how it puts it down, and which blocks can not be done. It is a calculation only: it moves nothing,
uses no hardware and no threads. The robot is only asked what it can do (link lengths, workspace limits and the ranges
of its servos). The code that executes the plan comes after this.

What was measured on the arm on 26 November 2026, all the geometry here follows from it:
- The robot frame has its origin on the table under the base, x to the right, y forward and z up. The tip of the
  kinematics is the end of the closed fingertips (d6 = 15 cm from the wrist centre, measured on 24 November 2026),
  so with the gripper pointing down the z of the tip is the height of the fingertips.
- Pose(x, y, z, flip=flip, alpha=a, gamma=-pi/2) (beta is 0) points the gripper straight down, and its fingers then
  close along the direction (cos a, sin a, 0): the fingers close along the second column of the orientation matrix and
  the gripper points along the third.
- The fingers are symmetric, so a and a + pi are the same grip with joint 6 half a turn further.
- With flip=False a gripper-down pose in front of the robot puts joint 4 at -180 degrees, the end of its range. With
  flip=True joint 4 is at 0. So the margin to the ends of the ranges decides between the flips, the whole plan uses one.
- The open gripper is 6 cm between the fingers and 8 cm over the outside, so each finger is 1 cm thick and sits 3 to
  4 cm from the centre along the closing direction. finger_depth is the size of a finger across.
- The arm rests on the table along a line from the base to a rest point, a block that stays behind there is in the
  way when the arm lies down (in_rest_area).

Lengths are in cm and angles in radians here, the configuration has degrees for tower.rotation and
motion.joint_margin.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from src.kinematics.kinematics import forward_position_kinematics, inverse_kinematics
from src.kinematics.kinematics_utils import Pose, RobotConfig
from src.utils.movement import ANGLE_LIMIT_TOLERANCE, MAX_POSITION_ERROR, PlannedMove, plan_straight_line
from src.utils.movement_exception import MovementException

if TYPE_CHECKING:
    from src.block_stacking.stacking_config import StackingConfig
    from src.robot_controllers.abstract_robot_controller import AbstractRobotController

# how far a candidate got before it failed, the reason of the one that got furthest is given to the user
_OUT_OF_REACH = 0
_OUTSIDE_WORKSPACE = 1
_JOINT_LIMIT = 2
_FINGERS = 3
_MOVE = 4


class StackingError(Exception):
    """the whole plan is impossible, the message says why"""


class BlockPosition(Protocol):
    """
    what the planner needs of a block, the DetectedBlock of the vision fits. These are read-only properties so that
    frozen dataclasses and plain attributes both fit.
    """

    @property
    def marker_id(self) -> int: ...

    @property
    def x(self) -> float: ...

    @property
    def y(self) -> float: ...

    @property
    def yaw(self) -> float:
        """radians, the direction of one edge of the block"""
        ...


@dataclass(frozen=True)
class Grip:
    """where the gripper grips (a block or the tower) and how"""
    x: float
    y: float
    closing_angle: float  # radians, the direction the fingers close along
    flip: bool


@dataclass(frozen=True)
class BlockStep:
    marker_id: int
    level: int  # 0 stands on the table
    pick: Grip
    place: Grip


@dataclass(frozen=True)
class StackingPlan:
    steps: tuple[BlockStep, ...]
    skipped: dict[int, str]  # marker id -> why the block is not stacked, a sentence for the user
    flip: bool


class _Failure(Exception):
    """why a candidate grip is not good"""

    def __init__(self, progress: int, reason: str, detail: int = 0) -> None:
        super().__init__(reason)
        self.progress = progress
        self.reason = reason
        self.detail = detail  # of two failures that got equally far, the larger one is the more hopeful


# the poses


def gripper_down_pose(x: float, y: float, z: float, closing_angle: float, flip: bool) -> Pose:
    """the gripper points straight down at (x, y, z) and its fingers close along (cos a, sin a, 0)"""
    return Pose(x, y, z, flip=flip, alpha=closing_angle, gamma=-math.pi / 2)


def park_pose(config: StackingConfig, flip: bool) -> Pose:
    """where the arm waits while the camera looks"""
    x, y, z = config.motion.park
    return gripper_down_pose(x, y, z, 0.0, flip)


def hover_pose(grip: Grip, config: StackingConfig) -> Pose:
    """above a block on the table, high enough for the fingers to pass over the blocks next to it"""
    return gripper_down_pose(grip.x, grip.y, config.motion.hover_height, grip.closing_angle, grip.flip)


def pick_pose(grip: Grip, config: StackingConfig) -> Pose:
    return gripper_down_pose(grip.x, grip.y, config.motion.grip_height, grip.closing_angle, grip.flip)


def _place_height(config: StackingConfig, level: int) -> float:
    """the height of the tip (the fingertips) when a block is let go at that level"""
    return level * config.blocks.size + config.motion.grip_height + config.motion.release_height


def travel_height(config: StackingConfig, level: int) -> float:
    """the height a block is carried at to be put at that level: clear of the tower and never below the hover height"""
    return max(config.motion.hover_height, _place_height(config, level) + config.motion.tower_clearance)


def lift_pose(grip: Grip, config: StackingConfig, level: int) -> Pose:
    """straight above the pick spot, where the block is carried from at the travel height of that level"""
    return gripper_down_pose(grip.x, grip.y, travel_height(config, level), grip.closing_angle, grip.flip)


def above_tower_pose(grip: Grip, config: StackingConfig, level: int) -> Pose:
    """straight above the tower, where the carried block goes straight down from"""
    return gripper_down_pose(grip.x, grip.y, travel_height(config, level), grip.closing_angle, grip.flip)


def place_pose(grip: Grip, config: StackingConfig, level: int) -> Pose:
    return gripper_down_pose(grip.x, grip.y, _place_height(config, level), grip.closing_angle, grip.flip)


# the rest area and the heights of a move


def in_rest_area(block: BlockPosition, config: StackingConfig, rest_point: tuple[float, float]) -> bool:
    """whether the block is closer than rest_clear_width to the line on the table from the base to the rest point"""
    end_x, end_y = rest_point
    length_squared = end_x * end_x + end_y * end_y
    # the foot of the perpendicular on the segment, with the base itself as the segment when it has no length
    along = 0.0 if length_squared == 0 else min(max((block.x * end_x + block.y * end_y) / length_squared, 0.0), 1.0)
    distance = math.hypot(block.x - along * end_x, block.y - along * end_y)
    return distance < config.motion.rest_clear_width


def min_tip_height(move: PlannedMove, robot_config: RobotConfig) -> float:
    """the lowest z of the tip over all steps of the move"""
    return float(min(forward_position_kinematics(angles, robot_config)[-1][2] for angles in move.angles))


# choosing a grip


def find_pick(block: BlockPosition, others: Sequence[BlockPosition], config: StackingConfig,
              robot: AbstractRobotController, flip: bool, level: int = 0) -> Grip | str:
    """
    how the gripper takes the block, one of the four directions along its edges: the one where the open fingers are
    clear of the other blocks, the arm can reach the poses above, at and above again the block at the travel height of
    the level the block is carried to, with every joint at least joint_margin from the ends of its range, and the
    straight moves down and up plan. Of these the one whose joints are furthest from their ends at the grip.
    The tower is in the way of the fingers too, as one more of the others, when blocks stand in it (level above 0).
    :param others: the other blocks on the table
    :return: the grip, or why there is none: the reason of the direction that got furthest
    """
    obstacles: list[BlockPosition] = list(others)
    if level > 0:
        obstacles.append(_TowerObstacle(config))
    candidates = [Grip(block.x, block.y, _normalise_angle(block.yaw + k * math.pi / 2), flip) for k in range(4)]

    def poses(grip: Grip) -> list[tuple[str, Pose]]:
        return [('above the block', hover_pose(grip, config)), ('at the block', pick_pose(grip, config)),
                ('above the block when it is lifted', lift_pose(grip, config, level))]

    def moves(grip: Grip) -> list[tuple[str, Pose, Pose]]:
        hover, pick, lift = (pose for _, pose in poses(grip))
        return [('down to the block', hover, pick), ('up with the block', pick, lift)]

    return _choose_grip(config, robot, candidates, poses, 1, lambda grip: _blockers(grip, obstacles, config), moves)


def find_place(config: StackingConfig, robot: AbstractRobotController, flip: bool, level: int) -> Grip | str:
    """
    how the gripper puts a block on the tower at that level, one of the four directions along the edges of the tower;
    the same rule as find_pick, with the poses above the tower and at the place
    :return: the grip, or why there is none
    """
    tower = config.tower
    turn = math.radians(tower.rotation)
    candidates = [Grip(tower.x, tower.y, _normalise_angle(turn + k * math.pi / 2), flip) for k in range(4)]

    def poses(grip: Grip) -> list[tuple[str, Pose]]:
        return [('above the tower', above_tower_pose(grip, config, level)),
                ('at the top of the tower', place_pose(grip, config, level))]

    def moves(grip: Grip) -> list[tuple[str, Pose, Pose]]:
        above, place = (pose for _, pose in poses(grip))
        return [('down onto the tower', above, place)]

    return _choose_grip(config, robot, candidates, poses, 1, lambda grip: [], moves)


def _choose_grip(config: StackingConfig, robot: AbstractRobotController, candidates: Sequence[Grip],
                 poses: Callable[[Grip], list[tuple[str, Pose]]], margin_index: int,
                 blockers: Callable[[Grip], list[BlockPosition]],
                 moves: Callable[[Grip], list[tuple[str, Pose, Pose]]]) -> Grip | str:
    """
    the good candidate whose joints are furthest from their limits at poses[margin_index], the first of equals, or the
    reason of the candidate that got furthest
    :param poses: the poses the arm has to reach with the margin, with a text where they are
    :param blockers: the blocks the open fingers would hit
    :param moves: the straight moves that have to plan, with a text what they do
    """
    limits = robot.get_angle_limits()
    min_margin = math.radians(config.motion.joint_margin)
    good: list[tuple[float, Grip]] = []
    failures: list[_Failure] = []
    for grip in candidates:
        try:
            margins = [_reach_margin(robot, pose, where, limits, min_margin) for where, pose in poses(grip)]
            in_the_way = blockers(grip)
            if in_the_way:
                raise _Failure(_FINGERS, _fingers_reason(in_the_way), -len(in_the_way))
        except _Failure as failure:
            failures.append(failure)
            continue
        good.append((margins[margin_index], grip))

    # the moves are the expensive part, so only the best candidates get them planned, in the order of their margin
    good.sort(key=lambda candidate: -candidate[0])
    for _, grip in good:
        move_failure = _move_failure(robot, config, moves(grip))
        if move_failure is None:
            return grip
        failures.append(move_failure)
    return max(failures, key=lambda failure: (failure.progress, failure.detail)).reason


def _move_failure(robot: AbstractRobotController, config: StackingConfig,
                  moves: list[tuple[str, Pose, Pose]]) -> _Failure | None:
    for where, start, stop in moves:
        try:
            plan_straight_line(robot, start, stop, config.motion.linear_speed)
        except MovementException as e:
            return _Failure(_MOVE, 'the move {} is not possible: {}'.format(where, e))
    return None


def _reach_margin(robot: AbstractRobotController, pose: Pose, where: str,
                  limits: dict[int, tuple[float, float]] | None, min_margin: float) -> float:
    """
    the smallest distance in radians of a joint to the end of its range at the pose, infinite when the ranges are
    not known
    :raises _Failure: if the arm can not reach the pose, or only with a joint closer than min_margin to the end of
        its range
    """
    angles = inverse_kinematics(pose, robot.robot_config)
    reached = forward_position_kinematics(angles, robot.robot_config)[-1]
    if np.linalg.norm(reached - [pose.x, pose.y, pose.z]) > MAX_POSITION_ERROR:
        raise _Failure(_OUT_OF_REACH, 'out of reach: the arm can not get to x={:.1f} y={:.1f} z={:.1f} ({})'.format(
            pose.x, pose.y, pose.z, where))
    violation = robot.workspace_violation(angles)
    if violation is not None:
        raise _Failure(_OUTSIDE_WORKSPACE, 'out of reach: {} ({})'.format(violation, where))
    margin = math.inf
    closest_joint = 0
    if limits is not None:
        for joint, (low, high) in limits.items():
            joint_margin = min(angles[joint] - low, high - angles[joint])
            if joint_margin < margin:
                margin, closest_joint = joint_margin, joint
    if margin < min_margin - ANGLE_LIMIT_TOLERANCE:
        raise _Failure(_JOINT_LIMIT, 'out of reach: joint {} would be only {:.1f} degrees from the end of its range, '
                                     '{:.1f} are needed ({})'.format(closest_joint, math.degrees(margin),
                                                                     math.degrees(min_margin), where))
    return margin


class _TowerObstacle:
    """the tower as one of the blocks the fingers have to stay clear of: its edges turn like the blocks in it"""

    marker_id = -1  # no block has it, _fingers_reason names the tower instead

    def __init__(self, config: StackingConfig) -> None:
        self.x = config.tower.x
        self.y = config.tower.y
        self.yaw = math.radians(config.tower.rotation)


def _blockers(grip: Grip, others: Sequence[BlockPosition], config: StackingConfig) -> list[BlockPosition]:
    """
    the blocks the open fingers would hit: those that come closer than block_clearance to a finger.

    In the frame of the grip (u along the closing direction, v across, origin under the tip) each open finger is a
    rectangle that spans from finger_gap / 2 to open_width / 2 along u on its side of the block, finger_depth wide
    across. A block is a square of its size, turned by its own yaw. The rectangle is grown by the clearance and the two
    shapes are tested for overlap on the four axes their edges give (the separating axis test).
    """
    clearance = config.motion.block_clearance
    gap, width = config.gripper.finger_gap, config.gripper.open_width
    finger_centre = (gap + width) / 4
    finger_half_length = (width - gap) / 4 + clearance
    finger_half_depth = config.gripper.finger_depth / 2 + clearance
    half_size = config.blocks.size / 2
    cos, sin = math.cos(grip.closing_angle), math.sin(grip.closing_angle)
    in_the_way = []
    for other in others:
        dx, dy = other.x - grip.x, other.y - grip.y
        along = dx * cos + dy * sin
        across = -dx * sin + dy * cos
        turn = other.yaw - grip.closing_angle
        if any(_square_overlaps_rectangle(along - side * finger_centre, across, turn, half_size,
                                          finger_half_length, finger_half_depth) for side in (-1, 1)):
            in_the_way.append(other)
    return in_the_way


def _square_overlaps_rectangle(u: float, v: float, turn: float, half_size: float, half_u: float,
                               half_v: float) -> bool:
    """
    whether a square overlaps a rectangle that is aligned with the axes and centred at the origin
    :param u: the centre of the square relative to the centre of the rectangle
    :param turn: the angle of the edges of the square to the axes
    :param half_size: half the edge of the square
    :param half_u: half the size of the rectangle along the u axis
    """
    cos, sin = abs(math.cos(turn)), abs(math.sin(turn))
    reach = half_size * (cos + sin)  # how far the square reaches along an axis of the rectangle
    if abs(u) >= half_u + reach or abs(v) >= half_v + reach:
        return False
    # the axes of the square: the rectangle's reach along them is its half sizes projected on them
    edge_u, edge_v = math.cos(turn), math.sin(turn)
    if abs(u * edge_u + v * edge_v) >= half_u * cos + half_v * sin + half_size:
        return False
    return abs(-u * edge_v + v * edge_u) < half_u * sin + half_v * cos + half_size


def _fingers_reason(blockers: Sequence[BlockPosition]) -> str:
    ids = [str(blocker.marker_id) for blocker in blockers if not isinstance(blocker, _TowerObstacle)]
    parts = []
    if len(ids) == 1:
        parts.append('block {}'.format(ids[0]))
    elif ids:
        parts.append('blocks {} and {}'.format(', '.join(ids[:-1]), ids[-1]))
    if len(ids) < len(blockers):
        parts.append('the tower')
    return '{} {} in the way of the fingers'.format(' and '.join(parts), 'is' if len(blockers) == 1 else 'are')


def _normalise_angle(angle: float) -> float:
    """the angle in (-pi, pi]"""
    angle = math.remainder(angle, 2 * math.pi)
    return math.pi if angle <= -math.pi else angle


# the plan


def plan_stack(blocks: Sequence[BlockPosition], config: StackingConfig, robot: AbstractRobotController,
               start_level: int = 0) -> StackingPlan:
    """
    plan the tower: which block goes when, how it is taken and put down, and which blocks can not be done.

    The whole plan uses one flip of the wrist: it is planned for both and the plan that stacks more blocks is kept,
    for equal numbers the one whose joints are further from their limits at the park pose. A flip that can not reach
    the park pose gives no plan.

    The order is greedy: the block nearest to the tower of those that can be taken goes next. Taking a block away can
    free another one, so a block that can not be taken is looked at again in the next round, and the reasons are those
    of the round in which the planning stops.
    :param start_level: the level of the first block, above the blocks that stand in the tower already
    :raises StackingError: if a block lies where the tower goes, or the arm can not reach the park pose with either flip
    """
    if start_level == 0:
        in_the_tower_spot = _block_on_the_tower_spot(blocks, config)
        if in_the_tower_spot is not None:
            raise StackingError(in_the_tower_spot)

    min_margin = math.radians(config.motion.joint_margin)
    limits = robot.get_angle_limits()
    best: StackingPlan | None = None
    best_margin = -math.inf
    why_not: list[str] = []
    # flip=True comes first: when nothing tells the flips apart (no limits known) joint 4 stays away from -180 degrees
    for flip in (True, False):
        try:
            park_margin = _reach_margin(robot, park_pose(config, flip), 'at the park pose', limits, min_margin)
        except _Failure as failure:
            why_not.append('flip {}: {}'.format(flip, failure.reason))
            continue
        plan = _plan_with_flip(blocks, config, robot, flip, start_level)
        if best is None or (len(plan.steps), park_margin) > (len(best.steps), best_margin):
            best, best_margin = plan, park_margin
    if best is None:
        raise StackingError('the arm can not reach the park pose: {}'.format('; '.join(why_not)))
    return best


def _block_on_the_tower_spot(blocks: Sequence[BlockPosition], config: StackingConfig) -> str | None:
    """why the tower can not be started, when a block lies within clear_radius of its centre; None when none does"""
    for block in blocks:
        distance = math.hypot(block.x - config.tower.x, block.y - config.tower.y)
        if distance < config.tower.clear_radius:
            return 'block {} lies where the tower goes, it is {:.1f}cm from its centre and the tower needs {:.1f}cm '                    'around it'.format(block.marker_id, distance, config.tower.clear_radius)
    return None


def plan_one(block: BlockPosition, others: Sequence[BlockPosition], config: StackingConfig,
             robot: AbstractRobotController, level: int, preferred_flip: bool = True) -> tuple[BlockStep, bool] | str:
    """
    the step that puts this one block on the tower at that level, and the flip of the wrist it uses; or why it can not
    be done, as a sentence for the user. It never raises for a block that can not be done.

    The same rules as plan_stack, for the one block the user chose: a flip counts when the arm reaches the park pose
    with it, finds a place on the tower at that level and a pick of the block. The preferred flip is tried first (the
    arm is waiting with it already, so keeping it saves a turn of the wrist), the other one when it does not work.
    :param others: the other blocks on the table, the fingers have to stay clear of them
    :param level: the level the block goes to, 0 when the tower has no blocks yet. The tower is an obstacle to the
        fingers when level is above 0, and at level 0 no block may lie where the tower goes.
    :return: the step and its flip, or when neither flip works the reason of the preferred one
    """
    if level == 0:
        in_the_tower_spot = _block_on_the_tower_spot((block, *others), config)
        if in_the_tower_spot is not None:
            return in_the_tower_spot

    min_margin = math.radians(config.motion.joint_margin)
    limits = robot.get_angle_limits()
    first_reason: str | None = None
    for flip in (preferred_flip, not preferred_flip):
        try:
            _reach_margin(robot, park_pose(config, flip), 'at the park pose', limits, min_margin)
        except _Failure as failure:
            first_reason = first_reason or 'the arm can not reach the park pose: {}'.format(failure.reason)
            continue
        place = find_place(config, robot, flip, level)
        if isinstance(place, str):
            first_reason = first_reason or 'the tower can not get higher than {} blocks here ({})'.format(level, place)
            continue
        pick = find_pick(block, others, config, robot, flip, level)
        if isinstance(pick, str):
            first_reason = first_reason or pick
            continue
        return BlockStep(block.marker_id, level, pick, place), flip
    assert first_reason is not None  # the loop ran twice and every run that did not return has set it
    return first_reason


def _plan_with_flip(blocks: Sequence[BlockPosition], config: StackingConfig, robot: AbstractRobotController,
                    flip: bool, start_level: int) -> StackingPlan:
    tower = (config.tower.x, config.tower.y)
    remaining = sorted(blocks, key=lambda block: math.hypot(block.x - tower[0], block.y - tower[1]))
    steps: list[BlockStep] = []
    skipped: dict[int, str] = {}
    level = start_level
    while remaining:
        place = find_place(config, robot, flip, level)
        if isinstance(place, str):
            reason = 'the tower can not get higher than {} blocks here ({})'.format(level, place)
            skipped.update({block.marker_id: reason for block in remaining})
            break
        # nearest to the tower first, the farther blocks are only looked at when the nearer ones can not be taken
        reasons: dict[int, str] = {}
        for block in remaining:
            others = [other for other in remaining if other is not block]
            pick = find_pick(block, others, config, robot, flip, level)
            if isinstance(pick, Grip):
                steps.append(BlockStep(block.marker_id, level, pick, place))
                remaining.remove(block)
                level += 1
                break
            reasons[block.marker_id] = pick
        else:
            skipped.update(reasons)
            break
    return StackingPlan(tuple(steps), skipped, flip)
