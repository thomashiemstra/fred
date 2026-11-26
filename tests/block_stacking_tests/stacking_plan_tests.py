import dataclasses
import math
import unittest
from dataclasses import dataclass
from typing import Any
from unittest import mock

import numpy as np
from numpy import pi

from src.block_stacking import stacking_plan
from tests.block_stacking_tests.frozen_config import load_frozen_config
from src.block_stacking.stacking_plan import StackingError, Grip, BlockStep, StackingPlan, gripper_down_pose, \
    park_pose, hover_pose, pick_pose, travel_height, lift_pose, above_tower_pose, place_pose, find_pick, find_place, \
    plan_stack, plan_one, in_rest_area, min_tip_height
from src.global_constants import dynamixel_robot_config, WorkSpaceLimits
from src.kinematics.kinematics import inverse_kinematics, forward_position_kinematics, forward_orientation_kinematics
from src.utils import workspace
from src.utils.movement import PlannedMove, plan_straight_line, MAX_POSITION_ERROR
from src.utils.movement_exception import MovementException

# the ranges of the servos of the arm
ANGLE_LIMITS = {1: (0, pi), 2: (0, pi), 3: (-pi / 2, pi / 2), 4: (-pi, pi), 5: (-pi, pi), 6: (-pi, pi)}


class StackingTestRobot:
    """the parts of a robot the planner asks for: the real link lengths, workspace limits and ranges of the arm"""

    def __init__(self, angle_limits: dict[int, tuple[float, float]] | None = ANGLE_LIMITS) -> None:
        self.robot_config = dynamixel_robot_config
        self.workspace_limits = WorkSpaceLimits
        self.angle_limits = angle_limits

    def get_angle_limits(self) -> dict[int, tuple[float, float]] | None:
        return self.angle_limits

    def workspace_violation(self, angles: np.ndarray) -> str | None:
        return workspace.workspace_violation(angles, self.robot_config, self.workspace_limits)

    def check_workspace(self, angles: np.ndarray) -> None:
        workspace.check_workspace(angles, self.robot_config, self.workspace_limits)


@dataclass(frozen=True)
class Block:
    """a block like the DetectedBlock of the vision: a frozen dataclass"""
    marker_id: int
    x: float
    y: float
    yaw: float = 0.0


class PlainBlock:
    """the same with plain attributes, which fit the BlockPosition protocol as well"""

    def __init__(self, marker_id: int, x: float, y: float, yaw: float) -> None:
        self.marker_id = marker_id
        self.x = x
        self.y = y
        self.yaw = yaw


config = load_frozen_config()
robot: Any = StackingTestRobot()
robot_without_limits: Any = StackingTestRobot(angle_limits=None)
margin = math.radians(config.motion.joint_margin)


def joint_margin(pose):
    """the smallest distance in radians of a joint to the end of its range at the pose"""
    angles = inverse_kinematics(pose, robot.robot_config)
    return min(min(angles[joint] - low, high - angles[joint]) for joint, (low, high) in ANGLE_LIMITS.items())


def same_direction(angle, other):
    return abs(math.remainder(angle - other, 2 * pi)) < 1e-9


def along_the_edges(angle, yaw):
    """whether the angle is the yaw turned by a multiple of a quarter turn"""
    return abs(math.remainder(angle - yaw, pi / 2)) < 1e-9


class GripperDownTests(unittest.TestCase):

    def test_gripper_points_down_and_fingers_close_along_the_angle_for_both_flips(self):
        for flip in (False, True):
            for angle in (0.0, 0.5, pi / 2, 2.0, pi, -2.5, -pi / 2):
                with self.subTest(flip=flip, angle=angle):
                    pose = gripper_down_pose(10, 25, 5, angle, flip)

                    angles = inverse_kinematics(pose, robot.robot_config)
                    orientation = forward_orientation_kinematics(angles)

                    np.testing.assert_allclose(orientation[:, 2], [0, 0, -1], atol=1e-9)
                    np.testing.assert_allclose(orientation[:, 1], [math.cos(angle), math.sin(angle), 0], atol=1e-9)
                    reached = forward_position_kinematics(angles, robot.robot_config)[-1]
                    np.testing.assert_allclose(reached, [10, 25, 5], atol=1e-9)

    def test_an_angle_and_the_same_angle_turned_half_a_turn_are_the_same_grip_with_joint_6_turned(self):
        for flip in (False, True):
            pose = gripper_down_pose(10, 25, 5, 0.5, flip)
            turned = gripper_down_pose(10, 25, 5, 0.5 + pi, flip)

            angles = inverse_kinematics(pose, robot.robot_config)
            turned_angles = inverse_kinematics(turned, robot.robot_config)

            np.testing.assert_allclose(forward_orientation_kinematics(turned_angles)[:, 1],
                                       -forward_orientation_kinematics(angles)[:, 1], atol=1e-9)
            self.assertAlmostEqual(abs(math.remainder(turned_angles[6] - angles[6], 2 * pi)), pi)

    def test_flip_false_puts_joint_4_at_its_limit_and_flip_true_at_0_in_front_of_the_robot(self):
        angles = inverse_kinematics(gripper_down_pose(10, 25, 5, 0.5, False), robot.robot_config)
        flipped = inverse_kinematics(gripper_down_pose(10, 25, 5, 0.5, True), robot.robot_config)

        self.assertAlmostEqual(abs(math.degrees(angles[4])), 180)
        self.assertAlmostEqual(math.degrees(flipped[4]), 0)


class PoseHelperTests(unittest.TestCase):

    def test_heights_of_the_poses(self):
        grip = Grip(10.0, 25.0, 0.5, True)

        self.assertEqual(config.motion.hover_height, hover_pose(grip, config).z)
        self.assertEqual(config.motion.grip_height, pick_pose(grip, config).z)
        self.assertEqual(travel_height(config, 0), lift_pose(grip, config, 0).z)
        self.assertEqual(travel_height(config, 3), lift_pose(grip, config, 3).z)
        self.assertEqual(travel_height(config, 3), above_tower_pose(grip, config, 3).z)
        for pose in (hover_pose(grip, config), pick_pose(grip, config), lift_pose(grip, config, 2),
                     above_tower_pose(grip, config, 2), place_pose(grip, config, 2)):
            self.assertEqual((10.0, 25.0, 0.5, True), (pose.x, pose.y, pose.alpha, pose.flip))
            self.assertEqual((0.0, -pi / 2), (pose.beta, pose.gamma))

    def test_place_pose_height_per_level_is_above_the_block_below_by_the_release_height(self):
        grip = Grip(22.0, 18.0, 0.0, True)
        size, motion = config.blocks.size, config.motion

        for level in range(5):
            self.assertAlmostEqual(level * size + motion.grip_height + motion.release_height,
                                   place_pose(grip, config, level).z)

    def test_travel_height_is_the_hover_height_until_the_tower_is_higher_then_clear_of_it(self):
        motion = config.motion
        place_height = lambda level: place_pose(Grip(0, 0, 0, True), config, level).z

        self.assertEqual(motion.hover_height, travel_height(config, 0))
        for level in range(8):
            self.assertAlmostEqual(max(motion.hover_height, place_height(level) + motion.tower_clearance),
                                   travel_height(config, level))
        self.assertGreater(travel_height(config, 6), motion.hover_height)
        self.assertGreater(travel_height(config, 6), travel_height(config, 5))

    def test_park_pose_is_the_park_position_with_the_gripper_down_and_closing_angle_0(self):
        for flip in (False, True):
            pose = park_pose(config, flip)

            self.assertEqual(config.motion.park, (pose.x, pose.y, pose.z))
            self.assertEqual(flip, pose.flip)
            self.assertEqual(0.0, pose.alpha)
            orientation = pose.get_euler_matrix()
            np.testing.assert_allclose(orientation[:, 2], [0, 0, -1], atol=1e-9)
            np.testing.assert_allclose(orientation[:, 1], [1, 0, 0], atol=1e-9)


class RestAreaAndTipHeightTests(unittest.TestCase):
    rest_point = (0.0, 20.0)

    def test_block_near_the_line_to_the_rest_point_is_in_the_rest_area(self):
        width = config.motion.rest_clear_width

        self.assertTrue(in_rest_area(Block(1, 0, 10), config, self.rest_point))
        self.assertTrue(in_rest_area(Block(1, width - 0.1, 10), config, self.rest_point))
        self.assertTrue(in_rest_area(Block(1, -(width - 0.1), 10), config, self.rest_point))
        self.assertFalse(in_rest_area(Block(1, width + 0.1, 10), config, self.rest_point))
        self.assertFalse(in_rest_area(Block(1, width, 10), config, self.rest_point))

    def test_beyond_the_ends_of_the_line_the_distance_is_to_the_end(self):
        width = config.motion.rest_clear_width

        self.assertTrue(in_rest_area(Block(1, 0, 20 + width - 0.1), config, self.rest_point))
        self.assertFalse(in_rest_area(Block(1, 0, 20 + width + 0.1), config, self.rest_point))
        self.assertTrue(in_rest_area(Block(1, 0, -(width - 0.1)), config, self.rest_point))
        # at x = 5 the distance to the end at (0, 20) is more than 6, though the point is 5 from the line
        self.assertFalse(in_rest_area(Block(1, 5, 24), config, self.rest_point))
        self.assertTrue(in_rest_area(Block(1, 5, 20), config, self.rest_point))

    def test_slanted_line_and_a_rest_point_at_the_base(self):
        self.assertTrue(in_rest_area(Block(1, 10, 10), config, (20.0, 20.0)))
        self.assertFalse(in_rest_area(Block(1, 20, 10), config, (20.0, 20.0)))
        self.assertTrue(in_rest_area(Block(1, 1, 1), config, (0.0, 0.0)))
        self.assertFalse(in_rest_area(Block(1, 10, 10), config, (0.0, 0.0)))

    def test_min_tip_height_of_a_straight_move_down_is_the_height_it_ends_at(self):
        grip = Grip(10.0, 25.0, 0.5, True)
        move = plan_straight_line(robot, hover_pose(grip, config), pick_pose(grip, config), 5.0)

        self.assertAlmostEqual(config.motion.grip_height, min_tip_height(move, robot.robot_config), places=6)

    def test_min_tip_height_is_the_lowest_of_all_steps_not_the_last(self):
        grip = Grip(10.0, 25.0, 0.5, True)
        down = plan_straight_line(robot, hover_pose(grip, config), pick_pose(grip, config), 5.0)
        move = PlannedMove('down and up again', np.vstack([down.angles, down.angles[::-1]]), 1.0)
        higher = PlannedMove('only the top', down.angles[:3], 1.0)

        self.assertAlmostEqual(config.motion.grip_height, min_tip_height(move, robot.robot_config), places=6)
        self.assertGreater(min_tip_height(higher, robot.robot_config), config.motion.hover_height - 0.5)


class FindPickTests(unittest.TestCase):

    def test_pick_closes_along_an_edge_of_the_block(self):
        for yaw in (0.0, 0.3, -1.0, 2.0):
            with self.subTest(yaw=yaw):
                grip = find_pick(Block(1, 10, 25, yaw), [], config, robot, True)

                self.assertIsInstance(grip, Grip)
                self.assertEqual((10, 25, True), (grip.x, grip.y, grip.flip))
                self.assertTrue(along_the_edges(grip.closing_angle, yaw))
                self.assertGreater(grip.closing_angle, -pi)
                self.assertLessEqual(grip.closing_angle, pi)

    def test_pick_takes_the_direction_with_the_joints_furthest_from_their_limits(self):
        for yaw in (0.0, 0.3, 1.2, 2.5, -2.0):
            with self.subTest(yaw=yaw):
                grip = find_pick(Block(1, 10, 25, yaw), [], config, robot, True)

                margins = [joint_margin(gripper_down_pose(10, 25, config.motion.grip_height, yaw + k * pi / 2, True))
                           for k in range(4)]
                chosen = [k for k in range(4) if same_direction(yaw + k * pi / 2, grip.closing_angle)]
                self.assertEqual(1, len(chosen))
                self.assertAlmostEqual(max(margins), margins[chosen[0]])

    def test_without_limits_the_first_direction_is_taken_and_angles_are_normalised(self):
        for yaw, expected in ((0.4, 0.4), (-pi, pi), (3.5, 3.5 - 2 * pi), (-3.5, 2 * pi - 3.5), (pi, pi)):
            with self.subTest(yaw=yaw):
                grip = find_pick(Block(1, 10, 25, yaw), [], config, robot_without_limits, True)

                self.assertAlmostEqual(expected, grip.closing_angle)

    def test_a_plain_object_with_the_attributes_is_a_block_too(self):
        grip = find_pick(PlainBlock(3, 10, 25, 0.2), [PlainBlock(4, 30, 25, 0.0)], config, robot, True)

        self.assertIsInstance(grip, Grip)

    def test_block_too_close_to_the_base_is_out_of_reach(self):
        reason = find_pick(Block(1, 0, 10), [], config, robot, True)

        self.assertIsInstance(reason, str)
        self.assertTrue(reason.startswith('out of reach'), reason)

    def test_block_too_far_is_out_of_reach(self):
        reason = find_pick(Block(1, 0, 60), [], config, robot, True)

        self.assertIsInstance(reason, str)
        self.assertTrue(reason.startswith('out of reach'), reason)

    def test_flip_false_has_no_pick_in_front_of_the_robot_because_joint_4_is_at_its_limit(self):
        reason = find_pick(Block(1, 10, 25), [], config, robot, False)

        self.assertIsInstance(reason, str)
        self.assertIn('joint 4', reason)
        self.assertIn('end of its range', reason)

    def test_flip_false_picks_when_the_joints_have_no_limits(self):
        self.assertIsInstance(find_pick(Block(1, 10, 25), [], config, robot_without_limits, False), Grip)

    def test_neighbour_5_cm_beside_a_block_is_avoided_by_closing_across_the_line_between_them(self):
        grip = find_pick(Block(1, 0, 30, 0.0), [Block(2, 5, 30, 0.0)], config, robot, True)

        self.assertIsInstance(grip, Grip)
        self.assertAlmostEqual(pi / 2, abs(grip.closing_angle))

    def test_neighbour_4_cm_away_with_aligned_edges_is_avoided_by_closing_across_the_line_between_them(self):
        # along the line the finger 3 to 4 cm from the block would hit the neighbour that starts 2.5 cm from it, across
        # the line the fingers are 3 cm deep and clear of the neighbour by 1 cm, with the clearance of 0.5 left
        for angle in (0.0, pi / 2, pi, -pi / 2):
            with self.subTest(angle=angle):
                neighbour = Block(2, 10 + 4 * math.cos(angle), 28 + 4 * math.sin(angle), 0.0)

                grip = find_pick(Block(1, 10, 28, 0.0), [neighbour], config, robot, True)

                self.assertIsInstance(grip, Grip)
                self.assertAlmostEqual(0.0, math.cos(angle - grip.closing_angle), places=9)

    def test_neighbour_in_any_direction_at_5_cm_is_avoided(self):
        for angle in (0.0, pi / 2, 1.0, -2.0):
            with self.subTest(angle=angle):
                neighbour = Block(2, 10 + 5 * math.cos(angle), 28 + 5 * math.sin(angle), 0.0)

                self.assertIsInstance(find_pick(Block(1, 10, 28, 0.0), [neighbour], config, robot, True), Grip)

    def test_neighbour_is_judged_in_the_frame_of_the_grip_when_the_blocks_are_turned_together(self):
        grip = find_pick(Block(1, 0, 30, 0.5), [Block(2, 4 * math.cos(0.5), 30 + 4 * math.sin(0.5), 0.5)], config,
                         robot, True)

        self.assertIsInstance(grip, Grip)
        self.assertAlmostEqual(0.0, math.cos(0.5 - grip.closing_angle), places=9)

    def test_neighbour_that_really_is_in_the_way_blocks_every_direction(self):
        # 3.2 cm to the side and 3.2 cm ahead: it overlaps a finger however the gripper closes
        reason = find_pick(Block(1, 0, 30, 0.0), [Block(2, 3.2, 33.2, 0.0)], config, robot, True)

        self.assertEqual('block 2 is in the way of the fingers', reason)

    def test_neighbour_turned_45_degrees_reaches_further_with_its_corner(self):
        # 4 cm to the side and 3.6 ahead: with aligned edges it stays 0.1 cm outside the finger and its clearance,
        # turned by 45 degrees its corner reaches 2.12 cm from its centre, into the fingers
        aligned = find_pick(Block(1, 0, 30, 0.0), [Block(2, 4, 33.6, 0.0)], config, robot, True)
        turned = find_pick(Block(1, 0, 30, 0.0), [Block(2, 4, 33.6, pi / 4)], config, robot, True)

        self.assertIsInstance(aligned, Grip)
        self.assertEqual('block 2 is in the way of the fingers', turned)

    def test_neighbour_turned_by_a_quarter_turn_is_the_same_as_aligned(self):
        self.assertIsInstance(find_pick(Block(1, 0, 30, 0.0), [Block(2, 4, 33.6, pi / 2)], config, robot, True), Grip)
        self.assertIsInstance(find_pick(Block(1, 0, 30, 0.0), [Block(2, 4, 33.6, -pi)], config, robot, True), Grip)

    def test_neighbour_is_in_the_way_when_it_comes_closer_to_a_finger_than_the_clearance(self):
        # closing along x the finger ends 4 cm from the block and a neighbour starts at its distance minus 1.5 cm; the
        # block above blocks closing across, so the neighbour decides
        clearance = config.motion.block_clearance
        above = Block(3, 0, 35, 0.0)

        free = find_pick(Block(1, 0, 30, 0.0), [Block(2, 4 + 1.5 + clearance + 0.05, 30, 0.0), above], config, robot,
                         True)
        in_the_way = find_pick(Block(1, 0, 30, 0.0), [Block(2, 4 + 1.5 + clearance - 0.05, 30, 0.0), above], config,
                               robot, True)

        self.assertIsInstance(free, Grip)
        self.assertTrue(same_direction(free.closing_angle, 0.0) or same_direction(free.closing_angle, pi))
        self.assertEqual('block 2 is in the way of the fingers', in_the_way)

    def test_the_reason_names_the_blocks_in_the_way_of_the_direction_with_the_fewest(self):
        # closing along x hits the neighbours left and right, closing along y only the one above
        reason = find_pick(Block(1, 0, 30, 0.0), [Block(2, 5, 30, 0.0), Block(3, -5, 30, 0.0), Block(4, 0, 35, 0.0)],
                           config, robot, True)
        self.assertEqual('block 4 is in the way of the fingers', reason)

        # now every direction has two, the first one is named
        reason = find_pick(Block(1, 0, 30, 0.0), [Block(2, 5, 30, 0.0), Block(3, -5, 30, 0.0), Block(4, 0, 35, 0.0),
                                                  Block(5, 0, 25, 0.0)], config, robot, True)
        self.assertEqual('blocks 2 and 3 are in the way of the fingers', reason)

    def test_block_far_from_the_fingers_is_no_obstacle(self):
        grip = find_pick(Block(1, 0, 30, 0.0), [Block(2, 12, 30, 0.0), Block(3, 0, 41, 0.0)], config, robot, True)

        self.assertIsInstance(grip, Grip)

    def test_a_move_that_does_not_plan_gives_its_reason(self):
        real = plan_straight_line

        def refuse_the_move_down(robot_, start, stop, *args):
            if stop.z == config.motion.grip_height:
                raise MovementException('joint 2 would turn too fast')
            return real(robot_, start, stop, *args)

        with mock.patch.object(stacking_plan, 'plan_straight_line', refuse_the_move_down):
            reason = find_pick(Block(1, 10, 25), [], config, robot, True)

        self.assertEqual('the move down to the block is not possible: joint 2 would turn too fast', reason)

    def test_the_next_direction_is_taken_when_the_move_of_the_best_does_not_plan(self):
        real = plan_straight_line
        refused: list[float] = []

        def refuse_the_first_direction(robot_, start, stop, *args):
            if not refused:
                refused.append(start.alpha)
                raise MovementException('refused')
            return real(robot_, start, stop, *args)

        with mock.patch.object(stacking_plan, 'plan_straight_line', refuse_the_first_direction):
            grip = find_pick(Block(1, 10, 25, 0.3), [], config, robot, True)

        self.assertIsInstance(grip, Grip)
        self.assertFalse(same_direction(refused[0], grip.closing_angle))

    def test_the_lift_at_the_height_of_a_high_level_has_to_be_reachable(self):
        block = Block(1, config.tower.x, config.tower.y)
        self.assertIsInstance(find_pick(block, [], config, robot, True, 0), Grip)

        reason = find_pick(block, [], config, robot, True, highest_level())

        self.assertIsInstance(reason, str)
        self.assertTrue(reason.startswith('out of reach'), reason)
        self.assertIn('when it is lifted', reason)


class TowerInTheWayTests(unittest.TestCase):
    # a block 5.5 cm from the tower on the side of the base: the line to the tower is along x, like the edges
    block = Block(1, config.tower.x - 5.5, config.tower.y, 0.0)

    def test_with_blocks_in_the_tower_the_fingers_close_across_the_line_to_the_tower(self):
        grip = find_pick(self.block, [], config, robot, True, 1)

        self.assertIsInstance(grip, Grip)
        self.assertAlmostEqual(pi / 2, abs(grip.closing_angle))

    def test_without_blocks_in_the_tower_the_tower_spot_is_no_obstacle(self):
        grip = find_pick(self.block, [], config, robot, True, 0)

        self.assertIsInstance(grip, Grip)
        height = config.motion.grip_height
        margins = [joint_margin(gripper_down_pose(self.block.x, self.block.y, height, k * pi / 2, True))
                   for k in range(4)]
        self.assertAlmostEqual(max(margins), joint_margin(pick_pose(grip, config)))
        # the direction with the most room is the one along the line to the tower, which level 1 does not allow
        self.assertAlmostEqual(0.0, math.sin(grip.closing_angle), places=9)

    def test_the_reason_names_the_tower_when_it_is_the_only_one_in_the_way(self):
        # 3.2 cm to the side and 3.2 cm farther from the base, the tower overlaps a finger however the gripper closes
        # (3.2 cm nearer to the base the block is closer than y_min = d6 + 0.1 = 15.1 cm to the base, which the 15 cm
        # gripper does not reach)
        block = Block(1, config.tower.x - 3.2, config.tower.y + 3.2, 0.0)

        self.assertEqual('the tower is in the way of the fingers', find_pick(block, [], config, robot, True, 1))
        self.assertIsInstance(find_pick(block, [], config, robot, True, 0), Grip)

    def test_the_reason_names_blocks_and_the_tower_together(self):
        # closing along x hits a block on the far side and the tower, closing across the two blocks above and below
        x, y = self.block.x, self.block.y
        others = [Block(2, x - 5, y), Block(3, x, y + 5), Block(4, x, y - 5)]

        reason = find_pick(self.block, others, config, robot, True, 1)

        self.assertEqual('block 2 and the tower are in the way of the fingers', reason)

    def test_the_tower_is_turned_by_its_rotation(self):
        # turned by 45 degrees the corner of the tower reaches 2.12 cm from its centre, 4.38 cm from this block, which
        # is inside the finger and its clearance; the tower with aligned edges starts 5 cm away
        turned = dataclasses.replace(config, tower=dataclasses.replace(config.tower, rotation=45.0))
        block = Block(1, config.tower.x - 6.5, config.tower.y, 0.0)

        aligned_grip = find_pick(block, [], config, robot, True, 1)
        turned_grip = find_pick(block, [], turned, robot, True, 1)

        self.assertIsInstance(aligned_grip, Grip)
        self.assertAlmostEqual(0.0, math.sin(aligned_grip.closing_angle), places=9)
        self.assertIsInstance(turned_grip, Grip)
        self.assertAlmostEqual(pi / 2, abs(turned_grip.closing_angle))

    def test_a_tower_6_5_cm_away_is_not_in_the_way_of_fingers_closing_towards_it(self):
        # the finger ends 4 cm from the block, the tower starts 5 cm from it: 1 cm, more than the clearance
        grip = find_pick(Block(1, config.tower.x - 6.5, config.tower.y, 0.0), [], config, robot, True, 1)

        self.assertIsInstance(grip, Grip)
        self.assertAlmostEqual(0.0, math.sin(grip.closing_angle), places=9)

    def test_a_plan_that_starts_above_level_0_keeps_the_fingers_off_the_tower(self):
        plan = plan_stack([self.block], config, robot, start_level=1)

        self.assertEqual(1, len(plan.steps))
        self.assertAlmostEqual(pi / 2, abs(plan.steps[0].pick.closing_angle))


class FindPlaceTests(unittest.TestCase):

    def test_place_is_on_the_tower_along_its_edges_for_every_level_up_to_the_highest(self):
        tower = config.tower
        for level in range(highest_level()):
            grip = find_place(config, robot, True, level)

            self.assertIsInstance(grip, Grip)
            self.assertEqual((tower.x, tower.y, True), (grip.x, grip.y, grip.flip))
            self.assertTrue(along_the_edges(grip.closing_angle, math.radians(tower.rotation)))

    def test_place_follows_the_rotation_of_the_tower(self):
        turned = dataclasses.replace(config, tower=dataclasses.replace(config.tower, rotation=30.0))

        grip = find_place(turned, robot, True, 1)

        self.assertIsInstance(grip, Grip)
        self.assertTrue(along_the_edges(grip.closing_angle, math.radians(30)))
        self.assertFalse(along_the_edges(grip.closing_angle, 0.0))

    def test_place_gives_the_reason_when_the_tower_is_too_high(self):
        reason = find_place(config, robot, True, highest_level())

        self.assertIsInstance(reason, str)
        self.assertTrue(reason.startswith('out of reach'), reason)

    def test_place_with_flip_false_is_out_of_reach_because_of_joint_4(self):
        reason = find_place(config, robot, False, 0)

        self.assertIsInstance(reason, str)
        self.assertIn('joint 4', reason)

    def test_a_move_down_that_does_not_plan_gives_its_reason(self):
        with mock.patch.object(stacking_plan, 'plan_straight_line', side_effect=MovementException('no way')):
            reason = find_place(config, robot, True, 0)

        self.assertEqual('the move down onto the tower is not possible: no way', reason)


def highest_level():
    """the number of blocks the default tower can have: the first level that can not be placed"""
    level = 0
    while isinstance(find_place(config, robot, True, level), Grip):
        level += 1
    return level


class AssertsPlans(unittest.TestCase):

    def assert_steps_are_reachable(self, plan: StackingPlan, blocks):
        """every pose of every step passes the checks of the move planner and keeps the joints off their limits"""
        for step in plan.steps:
            for pose in (hover_pose(step.pick, config), pick_pose(step.pick, config),
                         lift_pose(step.pick, config, step.level), above_tower_pose(step.place, config, step.level),
                         place_pose(step.place, config, step.level)):
                angles = inverse_kinematics(pose, robot.robot_config)
                reached = forward_position_kinematics(angles, robot.robot_config)[-1]
                self.assertLessEqual(np.linalg.norm(reached - [pose.x, pose.y, pose.z]), MAX_POSITION_ERROR)
                self.assertIsNone(robot.workspace_violation(angles))
                self.assertGreaterEqual(joint_margin(pose), margin - 1e-6)
                self.assertEqual(plan.flip, pose.flip)
            plan_straight_line(robot, hover_pose(step.pick, config), pick_pose(step.pick, config), 5.0)
            plan_straight_line(robot, pick_pose(step.pick, config), lift_pose(step.pick, config, step.level), 5.0)
            plan_straight_line(robot, above_tower_pose(step.place, config, step.level),
                               place_pose(step.place, config, step.level), 5.0)


class PlanStackTests(AssertsPlans):
    blocks = [Block(1, 10, 25, 0.3), Block(2, -5, 30, 1.0), Block(3, 15, 32, -0.4)]

    @classmethod
    def setUpClass(cls):
        cls.plan = plan_stack(cls.blocks, config, robot)

    def test_three_blocks_make_three_steps_with_levels_0_1_2(self):
        self.assertEqual([0, 1, 2], [step.level for step in self.plan.steps])
        self.assertEqual({}, self.plan.skipped)
        self.assertEqual({1, 2, 3}, {step.marker_id for step in self.plan.steps})

    def test_the_block_nearest_to_the_tower_goes_first(self):
        tower = (config.tower.x, config.tower.y)
        distance = {block.marker_id: math.hypot(block.x - tower[0], block.y - tower[1]) for block in self.blocks}

        order = [step.marker_id for step in self.plan.steps]

        self.assertEqual(sorted(distance, key=distance.__getitem__), order)
        self.assertEqual([1, 3, 2], order)

    def test_plan_uses_flip_true_because_joint_4_stays_away_from_its_limit_with_the_default_config(self):
        self.assertTrue(self.plan.flip)
        for step in self.plan.steps:
            self.assertTrue(step.pick.flip)
            self.assertTrue(step.place.flip)

    def test_pick_angles_are_along_the_edges_of_the_blocks_and_place_angles_along_the_tower(self):
        yaws = {block.marker_id: block.yaw for block in self.blocks}
        for step in self.plan.steps:
            self.assertTrue(along_the_edges(step.pick.closing_angle, yaws[step.marker_id]))
            self.assertTrue(along_the_edges(step.place.closing_angle, math.radians(config.tower.rotation)))

    def test_picks_are_at_the_blocks_and_places_at_the_tower(self):
        positions = {block.marker_id: (block.x, block.y) for block in self.blocks}
        for step in self.plan.steps:
            self.assertEqual(positions[step.marker_id], (step.pick.x, step.pick.y))
            self.assertEqual((config.tower.x, config.tower.y), (step.place.x, step.place.y))

    def test_every_pose_of_every_step_passes_the_checks(self):
        self.assert_steps_are_reachable(self.plan, self.blocks)

    def test_no_blocks_give_an_empty_plan(self):
        plan = plan_stack([], config, robot)

        self.assertEqual((), plan.steps)
        self.assertEqual({}, plan.skipped)
        self.assertTrue(plan.flip)

    def test_the_blocks_may_be_any_sequence_of_objects_with_the_attributes(self):
        plan = plan_stack((PlainBlock(7, 10, 25, 0.1),), config, robot)

        self.assertEqual([7], [step.marker_id for step in plan.steps])

    def test_the_order_of_the_blocks_given_does_not_matter(self):
        plan = plan_stack(self.blocks[::-1], config, robot)

        self.assertEqual(self.plan, plan)


class SkippedBlockTests(AssertsPlans):

    def test_blocks_too_close_and_too_far_are_skipped_with_a_reason_and_the_others_stacked(self):
        blocks = [Block(1, 10, 25, 0.3), Block(2, 0, 10, 0.0), Block(3, 0, 60, 0.0), Block(4, -5, 30, 1.0)]

        plan = plan_stack(blocks, config, robot)

        self.assertEqual([1, 4], [step.marker_id for step in plan.steps])
        self.assertEqual([0, 1], [step.level for step in plan.steps])
        self.assertEqual({2, 3}, set(plan.skipped))
        for reason in plan.skipped.values():
            self.assertTrue(reason.startswith('out of reach'), reason)
        self.assert_steps_are_reachable(plan, blocks)

    def test_two_blocks_side_by_side_are_both_taken_each_closing_across_the_line_between_them(self):
        blocks = [Block(1, 0, 30, 0.0), Block(2, 5, 30, 0.0)]

        plan = plan_stack(blocks, config, robot)

        self.assertEqual({}, plan.skipped)
        self.assertEqual(2, len(plan.steps))
        # the first block taken has the other one beside it, the second has nothing left to avoid
        self.assertAlmostEqual(pi / 2, abs(plan.steps[0].pick.closing_angle))
        self.assert_steps_are_reachable(plan, blocks)

    def test_blocks_4_cm_apart_are_both_taken_closing_across_the_line_between_them(self):
        blocks = [Block(1, 0, 30, 0.0), Block(2, 4, 30, 0.0)]

        plan = plan_stack(blocks, config, robot)

        self.assertEqual({}, plan.skipped)
        self.assertEqual(2, len(plan.steps))
        self.assertAlmostEqual(pi / 2, abs(plan.steps[0].pick.closing_angle))
        self.assert_steps_are_reachable(plan, blocks)

    def test_blocks_touching_side_by_side_are_taken_closing_across_the_line_between_them(self):
        # the fingers are 3 to 4 cm from the centre, so across the line they stand beside the neighbour
        blocks = [Block(1, 0, 30, 0.0), Block(2, 3, 30, 0.0)]

        plan = plan_stack(blocks, config, robot)

        self.assertEqual({}, plan.skipped)
        self.assertEqual(2, len(plan.steps))
        self.assertAlmostEqual(pi / 2, abs(plan.steps[0].pick.closing_angle))

    def test_two_blocks_that_are_in_the_way_of_each_other_whichever_way_the_gripper_closes_are_both_skipped(self):
        plan = plan_stack([Block(1, 0, 30, 0.0), Block(2, 3.2, 33.2, 0.0)], config, robot)

        self.assertEqual((), plan.steps)
        self.assertEqual({1: 'block 2 is in the way of the fingers', 2: 'block 1 is in the way of the fingers'},
                         plan.skipped)

    def test_a_block_boxed_in_on_all_four_sides_is_taken_after_its_neighbours(self):
        centre = Block(5, 10, 25, 0.0)
        east, west, north, south = Block(1, 15, 25), Block(2, 5, 25), Block(3, 10, 30), Block(4, 10, 20)
        blocks = [centre, east, west, north, south]
        self.assertIsInstance(find_pick(east, [centre, west, north, south], config, robot, True), Grip)
        self.assertIsInstance(find_pick(south, [centre, east, west, north], config, robot, True), Grip)
        self.assertIsInstance(find_pick(centre, [], config, robot, True), Grip)
        self.assertEqual('blocks 1 and 2 are in the way of the fingers',
                         find_pick(centre, [east, west, north, south], config, robot, True))

        plan = plan_stack(blocks, config, robot)

        self.assertEqual({}, plan.skipped)
        self.assertEqual(5, len(plan.steps))
        level = {step.marker_id: step.level for step in plan.steps}
        # closing along x is free once both neighbours in x are gone, along y once both in y are gone
        self.assertGreater(level[5], min(max(level[1], level[2]), max(level[3], level[4])))
        self.assert_steps_are_reachable(plan, blocks)


class TowerTests(AssertsPlans):

    def test_block_within_the_clear_radius_of_the_tower_raises_stacking_error(self):
        tower = config.tower
        block = Block(9, tower.x + tower.clear_radius - 0.5, tower.y)

        with self.assertRaisesRegex(StackingError, 'block 9 lies where the tower goes'):
            plan_stack([Block(1, 10, 25), block], config, robot)

    def test_block_just_outside_the_clear_radius_is_fine(self):
        tower = config.tower
        plan = plan_stack([Block(9, tower.x + tower.clear_radius + 0.5, tower.y)], config, robot)

        self.assertEqual([9], [step.marker_id for step in plan.steps])

    def test_block_where_the_tower_goes_is_no_error_when_the_tower_stands_already(self):
        tower = config.tower
        block = Block(9, tower.x + 1, tower.y)

        plan = plan_stack([block], config, robot, start_level=2)

        self.assertIsInstance(plan, StackingPlan)

    def test_start_level_is_the_level_of_the_first_block(self):
        blocks = [Block(1, 10, 25, 0.3), Block(2, -5, 30, 1.0)]

        plan = plan_stack(blocks, config, robot, start_level=2)

        self.assertEqual([2, 3], [step.level for step in plan.steps])
        self.assert_steps_are_reachable(plan, blocks)

    def test_the_default_tower_reaches_eight_blocks(self):
        self.assertEqual(8, highest_level())

    def test_the_tower_stops_growing_where_the_place_pose_gets_out_of_reach(self):
        highest = highest_level()
        blocks = [Block(1, 10, 25, 0.3), Block(2, -5, 30, 1.0), Block(3, 15, 32, -0.4), Block(4, 0, 22, 0.8)]

        plan = plan_stack(blocks, config, robot, start_level=highest - 2)

        self.assertEqual([highest - 2, highest - 1], [step.level for step in plan.steps])
        self.assertEqual(2, len(plan.skipped))
        for reason in plan.skipped.values():
            self.assertIn('the tower can not get higher than {} blocks here'.format(highest), reason)
        self.assert_steps_are_reachable(plan, blocks)

    def test_a_tower_that_starts_too_high_skips_every_block(self):
        plan = plan_stack([Block(1, 10, 25), Block(2, -5, 30)], config, robot, start_level=20)

        self.assertEqual((), plan.steps)
        self.assertEqual({1, 2}, set(plan.skipped))
        for reason in plan.skipped.values():
            self.assertIn('can not get higher than 20 blocks', reason)


class PlanOneTests(AssertsPlans):
    block = Block(1, 10, 25, 0.3)

    def test_a_reachable_block_gives_a_step_with_the_level_a_pick_along_its_edges_and_a_place_along_the_tower(self):
        for level in (0, 2):
            with self.subTest(level=level):
                result = plan_one(self.block, [Block(2, -5, 30, 1.0)], config, robot, level)

                self.assertIsInstance(result, tuple)
                step, flip = result
                self.assertIsInstance(step, BlockStep)
                self.assertTrue(flip)
                self.assertEqual((1, level), (step.marker_id, step.level))
                self.assertEqual((10, 25, True), (step.pick.x, step.pick.y, step.pick.flip))
                self.assertTrue(along_the_edges(step.pick.closing_angle, 0.3))
                self.assertEqual((config.tower.x, config.tower.y, True), (step.place.x, step.place.y, step.place.flip))
                self.assertTrue(along_the_edges(step.place.closing_angle, math.radians(config.tower.rotation)))
                self.assert_steps_are_reachable(StackingPlan((step,), {}, flip), [self.block])

    def test_the_other_blocks_may_be_plain_objects_and_none_at_all(self):
        self.assertIsInstance(plan_one(self.block, [], config, robot, 0), tuple)
        self.assertIsInstance(plan_one(PlainBlock(3, 10, 25, 0.2), [PlainBlock(4, 30, 25, 0.0)], config, robot, 0),
                              tuple)

    def test_a_block_out_of_reach_gives_a_sentence(self):
        for block in (Block(2, 0, 10), Block(3, 0, 60)):
            with self.subTest(block=block):
                reason = plan_one(block, [], config, robot, 0)

                self.assertIsInstance(reason, str)
                self.assertTrue(reason.startswith('out of reach'), reason)

    def test_the_reason_of_a_block_out_of_reach_is_the_one_of_its_pick_with_the_preferred_flip(self):
        block = Block(3, 0, 60)

        self.assertEqual(find_pick(block, [], config, robot, True, 0), plan_one(block, [], config, robot, 0))
        # flip False does not even get to the park pose with the limits of this robot
        reason = plan_one(block, [], config, robot, 0, preferred_flip=False)
        self.assertTrue(reason.startswith('the arm can not reach the park pose: '), reason)
        self.assertIn('joint 4', reason)

    def test_a_block_boxed_in_by_others_gives_the_fingers_reason_and_is_fine_once_one_is_gone(self):
        centre = Block(5, 10, 25, 0.0)
        east, west, north, south = Block(1, 15, 25), Block(2, 5, 25), Block(3, 10, 30), Block(4, 10, 20)

        reason = plan_one(centre, [east, west, north, south], config, robot, 0)
        freed = plan_one(centre, [north, south], config, robot, 0)

        self.assertEqual('blocks 1 and 2 are in the way of the fingers', reason)
        self.assertIsInstance(freed, tuple)
        # closing along x is free once the neighbours in x are gone
        self.assertAlmostEqual(0.0, math.sin(freed[0].pick.closing_angle), places=9)

    def test_a_block_close_to_the_tower_spot_is_refused_at_level_0_and_fine_at_level_2(self):
        tower = config.tower
        block = Block(9, tower.x + tower.clear_radius - 0.5, tower.y)

        reason = plan_one(block, [], config, robot, 0)
        result = plan_one(block, [], config, robot, 2)

        self.assertIsInstance(reason, str)
        self.assertTrue(reason.startswith('block 9 lies where the tower goes'), reason)
        self.assertIn('needs {:.1f}cm around it'.format(tower.clear_radius), reason)
        self.assertIsInstance(result, tuple)
        self.assertEqual(2, result[0].level)

    def test_another_block_on_the_tower_spot_is_named_at_level_0(self):
        tower = config.tower
        on_the_spot = Block(8, tower.x + 1, tower.y)

        reason = plan_one(self.block, [on_the_spot], config, robot, 0)

        self.assertTrue(reason.startswith('block 8 lies where the tower goes'), reason)
        self.assertIsInstance(plan_one(self.block, [on_the_spot], config, robot, 1), tuple)

    def test_a_block_just_outside_the_clear_radius_is_fine(self):
        tower = config.tower

        self.assertIsInstance(plan_one(Block(9, tower.x + tower.clear_radius + 0.5, tower.y), [], config, robot, 0),
                              tuple)

    def test_the_tower_is_in_the_way_of_the_fingers_above_level_0(self):
        # 5.5 cm from the tower on the side of the base: within the clear radius at level 0, so no place for a block
        # there, but with blocks in the tower the fingers close across the line to it
        block = Block(1, config.tower.x - 5.5, config.tower.y, 0.0)

        on_the_table = plan_one(block, [], config, robot, 0)
        on_the_tower = plan_one(block, [], config, robot, 1)

        self.assertIsInstance(on_the_table, str)
        self.assertAlmostEqual(pi / 2, abs(on_the_tower[0].pick.closing_angle))

    def test_a_tower_that_can_not_get_higher_gives_a_sentence_like_the_plan(self):
        highest = highest_level()

        reason = plan_one(self.block, [], config, robot, highest)

        self.assertIsInstance(reason, str)
        self.assertTrue(reason.startswith('the tower can not get higher than {} blocks here (out of reach'.format(
            highest)), reason)
        self.assertIsInstance(plan_one(self.block, [], config, robot, highest - 1), tuple)
        skipped = plan_stack([self.block], config, robot, start_level=highest).skipped
        self.assertEqual({self.block.marker_id: reason}, skipped)

    def test_an_unreachable_park_pose_gives_a_sentence_and_does_not_raise(self):
        far = dataclasses.replace(config, motion=dataclasses.replace(config.motion, park=(0.0, 80.0, 30.0)))

        reason = plan_one(self.block, [], far, robot, 0)

        self.assertIsInstance(reason, str)
        self.assertIn('park pose', reason)

    def test_the_result_agrees_with_plan_stack_for_a_single_block(self):
        for rbt in (robot, robot_without_limits):
            for level in (0, 1, 3):
                with self.subTest(limits=rbt.angle_limits is not None, level=level):
                    plan = plan_stack([self.block], config, rbt, start_level=level)

                    result = plan_one(self.block, [], config, rbt, level)

                    self.assertEqual((plan.steps[0], plan.flip), result)

    def test_the_result_agrees_with_plan_stack_for_the_first_block_of_several(self):
        blocks = [Block(1, 10, 25, 0.3), Block(2, -5, 30, 1.0), Block(3, 15, 32, -0.4)]
        plan = plan_stack(blocks, config, robot)
        first = next(block for block in blocks if block.marker_id == plan.steps[0].marker_id)

        result = plan_one(first, [block for block in blocks if block is not first], config, robot, 0)

        self.assertEqual((plan.steps[0], plan.flip), result)

    def test_the_preferred_flip_is_tried_first(self):
        for preferred in (True, False):
            with self.subTest(preferred=preferred):
                step, flip = plan_one(self.block, [], config, robot_without_limits, 1, preferred_flip=preferred)

                self.assertEqual(preferred, flip)
                self.assertEqual(preferred, step.pick.flip)
                self.assertEqual(preferred, step.place.flip)

    def test_the_other_flip_is_used_when_the_preferred_one_does_not_work(self):
        # with the limits of the arm flip False has joint 4 at the end of its range in front of the robot
        step, flip = plan_one(self.block, [], config, robot, 1, preferred_flip=False)

        self.assertTrue(flip)
        self.assertTrue(step.pick.flip)
        self.assertTrue(step.place.flip)
        # and the other way round, when only flip False can park
        step, flip = plan_one(self.block, [], config, only_false_can_park, 0, preferred_flip=True)
        self.assertFalse(flip)
        self.assertFalse(step.pick.flip)

    def test_the_other_flip_is_used_when_the_pick_fails_with_the_preferred_one(self):
        real = find_pick

        def pick_only_with_flip_false(block, others, config_, robot_, flip, level=0):
            return real(block, others, config_, robot_, flip, level) if not flip else 'refused with flip True'

        with mock.patch.object(stacking_plan, 'find_pick', pick_only_with_flip_false):
            result = plan_one(self.block, [], config, robot_without_limits, 0, preferred_flip=True)
            reason = plan_one(self.block, [], config, robot, 0, preferred_flip=True)

        self.assertFalse(result[1])
        self.assertFalse(result[0].pick.flip)
        # flip False can not park with the limits of the arm, so the reason is the one of the preferred flip
        self.assertEqual('refused with flip True', reason)

    def test_the_reason_is_the_one_of_the_preferred_flip_when_neither_works(self):
        refuse = lambda config_, robot_, flip, level: 'no place with flip {}'.format(flip)

        with mock.patch.object(stacking_plan, 'find_place', refuse):
            for preferred in (True, False):
                with self.subTest(preferred=preferred):
                    reason = plan_one(self.block, [], config, robot_without_limits, 3, preferred_flip=preferred)

                    self.assertEqual('the tower can not get higher than 3 blocks here (no place with flip {})'.format(
                        preferred), reason)


def park_margin(flip, limits_robot):
    """the smallest distance in radians of a joint to the end of its range at the park pose"""
    angles = inverse_kinematics(park_pose(config, flip), robot.robot_config)
    limits = limits_robot.angle_limits
    return min(min(angles[joint] - low, high - angles[joint]) for joint, (low, high) in limits.items())


def robot_with(**changes):
    """the robot with other ranges for some joints, given as joint_5=(low, high) in degrees"""
    limits = dict(ANGLE_LIMITS)
    for name, (low, high) in changes.items():
        limits[int(name.removeprefix('joint_'))] = (math.radians(low), math.radians(high))
    return StackingTestRobot(limits)


# at the park pose flip=False has the joints 4 and 6 at +-180 degrees, the end of their range, and joint 5 at +114;
# flip=True has them at 0, 0 and -114. Wide ranges for 4 and 6 let both flips reach it, and the range of joint 5 then
# decides which flip has the larger margin there: 16 degrees for the one near the end of its range, 66 for the other
true_has_room: Any = robot_with(joint_4=(-360, 360), joint_5=(-180, 130), joint_6=(-360, 360))
false_has_room: Any = robot_with(joint_4=(-360, 360), joint_5=(-130, 180), joint_6=(-360, 360))
only_false_can_park: Any = robot_with(joint_4=(-360, 360), joint_5=(0, 180), joint_6=(-360, 360))


class FlipAndLimitsTests(AssertsPlans):

    def test_no_limits_still_plans(self):
        blocks = [Block(1, 10, 25, 0.3), Block(2, -5, 30, 1.0), Block(3, 15, 32, -0.4)]

        plan = plan_stack(blocks, config, robot_without_limits, )

        self.assertEqual([1, 3, 2], [step.marker_id for step in plan.steps])
        self.assertEqual({}, plan.skipped)
        self.assertTrue(plan.flip)
        # the first of the equal directions
        self.assertTrue(same_direction(plan.steps[0].pick.closing_angle, 0.3))

    def test_the_flip_the_limits_allow_is_used(self):
        plan = plan_stack([Block(1, 10, 25, 0.3)], config, only_false_can_park)

        self.assertFalse(plan.flip)
        self.assertEqual(1, len(plan.steps))
        self.assertFalse(plan.steps[0].pick.flip)
        self.assertFalse(plan.steps[0].place.flip)

    def test_the_flip_with_more_steps_wins_even_when_the_other_has_the_larger_margin_at_the_park_pose(self):
        more = StackingPlan((BlockStep(1, 0, Grip(1, 2, 0.0, False), Grip(3, 4, 0.0, False)),), {}, False)
        fewer = StackingPlan((), {}, True)
        plans = {False: more, True: fewer}

        with mock.patch.object(stacking_plan, '_plan_with_flip', lambda blocks, cfg, rbt, flip, level: plans[flip]):
            plan = plan_stack([Block(1, 10, 25)], config, true_has_room)

        self.assertIs(more, plan)
        self.assertLess(park_margin(False, true_has_room), park_margin(True, true_has_room))

    def test_with_equal_steps_the_flip_with_the_larger_margin_at_the_park_pose_wins(self):
        self.assertTrue(plan_stack([], config, true_has_room).flip)
        self.assertFalse(plan_stack([], config, false_has_room).flip)
        self.assertGreater(park_margin(False, false_has_room), park_margin(True, false_has_room))

    def test_unreachable_park_pose_is_a_stacking_error(self):
        far = dataclasses.replace(config, motion=dataclasses.replace(config.motion, park=(0.0, 80.0, 30.0)))

        with self.assertRaisesRegex(StackingError, 'park pose'):
            plan_stack([Block(1, 10, 25)], far, robot)

    def test_park_pose_without_room_for_the_joints_is_a_stacking_error_for_both_flips(self):
        # the margin is more than the range of joint 3 has room for
        narrow = dataclasses.replace(config, motion=dataclasses.replace(config.motion, joint_margin=100.0))

        with self.assertRaises(StackingError):
            plan_stack([], narrow, robot)


if __name__ == '__main__':
    unittest.main()
