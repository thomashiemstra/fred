import dataclasses
import math
import threading
import unittest
from collections import deque
from collections.abc import Callable
from typing import Any
from unittest import mock

import cv2
import numpy as np

from src.block_stacking.stacking_config import StackingConfig, aruco_dictionary
from src.camera.block_detection import BlockImageHandler, BlockObservation, DetectedBlock, ScanError, board_image, \
    board_to_robot, make_board, robot_to_board, _COLOR_MESSAGE_BAR, _COLOR_MESSAGE_TEXT
from src.camera.util import interpolate_charuco_corners
from tests.block_stacking_tests.frozen_config import load_frozen_config

WIDTH, HEIGHT = 1920, 1080
BOARD_PPC = 40  # pixels per cm of the board texture
MARKER_CELLS = 7  # a marker of 5 by 5 bits and its black border
MARKER_PIXELS = MARKER_CELLS * 16
TOP_PPC = MARKER_PIXELS / 2.4  # pixels per cm of the top of a block, the marker is 2.4 cm
POSITION_TOLERANCE = 0.2  # cm
YAW_TOLERANCE = math.radians(2)


def fold_difference(a: float, b: float) -> float:
    """the difference of two yaws, which are the same every quarter turn"""
    return (a - b + math.pi / 4) % (math.pi / 2) - math.pi / 4


class SyntheticCamera:
    """a pinhole camera with a pose, that renders the table and the marker on top of blocks as the picture it takes"""

    def __init__(self, centre: tuple[float, float, float], target: tuple[float, float, float],
                 dist_coeffs: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0), focal: float = 1400.0) -> None:
        self.centre = np.array(centre, dtype=np.float64)
        self.camera_matrix = np.array([[focal, 0, WIDTH / 2], [0, focal, HEIGHT / 2], [0, 0, 1]], dtype=np.float64)
        self.dist_coeffs = np.array(dist_coeffs, dtype=np.float64)
        forward = np.array(target, dtype=np.float64) - self.centre
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        self.world_to_camera = np.array([right, down, forward])
        # a ray of the camera in the world for every pixel, the pixel centres have whole coordinates
        v, u = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)
        pixels = np.stack([u, v], axis=-1).reshape(-1, 1, 2)
        normalized = cv2.undistortPoints(pixels, self.camera_matrix, self.dist_coeffs).reshape(HEIGHT, WIDTH, 2)
        rays = np.concatenate([normalized, np.ones((HEIGHT, WIDTH, 1))], axis=-1)
        self.rays = rays @ self.world_to_camera  # the same as world_to_camera.T applied to every ray
        self._planes: dict[float, np.ndarray] = {}

    @staticmethod
    def looking_down(target: tuple[float, float], distance: float = 70.0, off_vertical: float = 35.0,
                     azimuth: float = 0.0, **kwargs: object) -> 'SyntheticCamera':
        """at distance cm from the target on the table, off_vertical degrees from straight above, azimuth 0 is in
        front of the target (smaller y), counter clockwise seen from above"""
        tilt, direction = math.radians(off_vertical), math.radians(azimuth)
        offset = distance * np.array([math.sin(tilt) * math.sin(direction), -math.sin(tilt) * math.cos(direction),
                                      math.cos(tilt)])
        centre = (target[0] + offset[0], target[1] + offset[1], offset[2])
        return SyntheticCamera(centre, (target[0], target[1], 0.0), **kwargs)  # type: ignore[arg-type]

    def plane_points(self, z: float) -> np.ndarray:
        """x and y in the robot frame of what every pixel sees in the plane at height z, shape (HEIGHT, WIDTH, 2)"""
        if z not in self._planes:
            distance = (z - self.centre[2]) / self.rays[..., 2]
            self._planes[z] = self.centre[:2] + distance[..., np.newaxis] * self.rays[..., :2]
        return self._planes[z]

    def render(self, config: StackingConfig, blocks: list[tuple[int, float, float, float, float]],
               with_board: bool = True, extra: list[tuple[int, float, float, float, float]] | None = None
               ) -> np.ndarray:
        """
        the gray picture of the table, with the board when asked, and of the blocks. Each block is (marker id, x, y,
        yaw in degrees, z of the marker), a block of the dictionary of the config that is not in marker_ids can be
        added with extra.
        """
        board = config.board
        # white all over, so far from the board there is still something to see
        picture = np.full((HEIGHT, WIDTH), 255.0)
        if with_board:
            texture = cv2.GaussianBlur(board_image(board, BOARD_PPC).astype(np.float32), (0, 0), 1.0)
            width_cm = board.squares_x * board.square_length
            height_cm = board.squares_y * board.square_length
            # from the robot frame to the board frame and to the pixels of the picture of the board, written out here
            # from the definition of the frames and not with the functions that are tested
            turn = math.radians(board.rotation)
            points = self.plane_points(0.0)
            dx, dy = points[..., 0] - board.x, points[..., 1] - board.y
            board_x = math.cos(turn) * dx + math.sin(turn) * dy
            board_y = -math.sin(turn) * dx + math.cos(turn) * dy
            # the origin is the bottom right of the picture: the left of the picture is at x = -width
            map_x = ((board_x + width_cm) * BOARD_PPC - 0.5).astype(np.float32)
            map_y = ((height_cm - board_y) * BOARD_PPC - 0.5).astype(np.float32)
            picture = cv2.remap(texture, map_x, map_y, cv2.INTER_LINEAR, borderValue=255).astype(np.float64)
        dictionary = aruco_dictionary(config.blocks.dictionary)
        size = config.blocks.size
        top_pixels = round(size * TOP_PPC)
        for marker_id, x, y, yaw, z in blocks + (extra or []):
            marker = cv2.aruco.generateImageMarker(dictionary, marker_id, MARKER_PIXELS)
            texture = np.full((top_pixels, top_pixels), 255.0, dtype=np.float32)
            offset = (top_pixels - MARKER_PIXELS) // 2
            texture[offset:offset + MARKER_PIXELS, offset:offset + MARKER_PIXELS] = marker
            texture = cv2.GaussianBlur(texture, (0, 0), 1.0)
            turn = math.radians(yaw)
            points = self.plane_points(z)
            dx, dy = points[..., 0] - x, points[..., 1] - y
            # the marker is upright with its right along (cos, sin) and its down along (sin, -cos): seen from above
            right = math.cos(turn) * dx + math.sin(turn) * dy
            down = math.sin(turn) * dx - math.cos(turn) * dy
            map_x = ((right + size / 2) * TOP_PPC - 0.5).astype(np.float32)
            map_y = ((down + size / 2) * TOP_PPC - 0.5).astype(np.float32)
            top = cv2.remap(texture, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)
            coverage = cv2.remap(np.ones_like(texture), map_x, map_y, cv2.INTER_LINEAR, borderValue=0)
            picture = picture * (1 - coverage) + top * coverage
        picture = cv2.GaussianBlur(picture, (0, 0), 0.7)
        return np.clip(np.round(picture), 0, 255).astype(np.uint8)


def with_board(config: StackingConfig, x: float, y: float, rotation: float) -> StackingConfig:
    return dataclasses.replace(config, board=dataclasses.replace(config.board, x=x, y=y, rotation=rotation))


def with_scan(config: StackingConfig, **changes: float) -> StackingConfig:
    return dataclasses.replace(config, scan=dataclasses.replace(config.scan, **changes))


def feed(handler: BlockImageHandler, gray: np.ndarray) -> np.ndarray:
    """give the handler a picture as the camera thread does and return the frame it drew on"""
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    handler.handle_frame(frame, gray)
    return frame


class BlockDetectionTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.config = load_frozen_config()

    def make_handler(self, camera: SyntheticCamera, config: StackingConfig | None = None,
                     clock: Callable[[], float] | None = None) -> BlockImageHandler:
        if clock is None:
            return BlockImageHandler(config or self.config, camera.camera_matrix, camera.dist_coeffs)
        return BlockImageHandler(config or self.config, camera.camera_matrix, camera.dist_coeffs, clock=clock)

    def assert_block(self, block: DetectedBlock, x: float, y: float, yaw_degrees: float) -> None:
        self.assertAlmostEqual(block.x, x, delta=POSITION_TOLERANCE, msg='x of {}'.format(block))
        self.assertAlmostEqual(block.y, y, delta=POSITION_TOLERANCE, msg='y of {}'.format(block))
        self.assertLess(abs(fold_difference(block.yaw, math.radians(yaw_degrees))), YAW_TOLERANCE,
                        'yaw of {}'.format(block))

    def assert_off_board(self, config: StackingConfig, blocks: list[tuple[int, float, float, float, float]]) -> None:
        """the test itself puts no block on the board"""
        board = config.board
        for _, x, y, _, _ in blocks:
            bx, by = robot_to_board([x, y], board)
            # the board lies at x from -width to 0 and y from 0 to height
            inside = -board.squares_x * board.square_length - 3 < bx < 3 and \
                -3 < by < board.squares_y * board.square_length + 3
            self.assertFalse(inside, 'the block at {}, {} is on the board'.format(x, y))


class TestFindingBlocks(BlockDetectionTestCase):

    def check_scene(self, config: StackingConfig, camera: SyntheticCamera,
                    blocks: list[tuple[int, float, float, float, float]]) -> list[DetectedBlock]:
        self.assert_off_board(config, blocks)
        handler = self.make_handler(camera, config)
        feed(handler, camera.render(config, blocks))
        found = handler.get_blocks()
        self.assertEqual([block[0] for block in blocks], [block.marker_id for block in found])
        for (marker_id, x, y, yaw, z), block in zip(blocks, found):
            self.assert_block(block, x, y, yaw)
            self.assertEqual(config.blocks.size, block.height)
            self.assertEqual(1, block.frames)
            self.assertAlmostEqual(config.blocks.marker_length, block.side_length, delta=0.1)
        return found

    def test_blocks_on_the_table_with_the_default_board(self) -> None:
        camera = SyntheticCamera.looking_down((0.0, 25.0))
        blocks = [(0, 3.0, 20.0, 0.0, 3.0), (1, 12.0, 32.0, 25.0, 3.0), (2, -2.0, 40.0, -40.0, 3.0),
                  (3, 20.0, 12.0, 70.0, 3.0), (9, -4.0, 8.0, 12.5, 3.0)]
        self.check_scene(self.config, camera, blocks)

    def test_board_rotated_by_30_degrees(self) -> None:
        config = with_board(self.config, -10.0, 14.0, 30.0)
        camera = SyntheticCamera.looking_down((0.0, 25.0))
        blocks = [(0, 3.0, 20.0, 5.0, 3.0), (1, 15.0, 30.0, -30.0, 3.0), (4, 2.0, 38.0, 41.0, 3.0)]
        self.check_scene(config, camera, blocks)

    def test_board_rotated_by_90_degrees(self) -> None:
        config = with_board(self.config, -8.0, 30.0, 90.0)
        camera = SyntheticCamera.looking_down((0.0, 25.0))
        blocks = [(0, 5.0, 20.0, 15.0, 3.0), (5, 18.0, 33.0, 60.0, 3.0), (6, -38.0, 12.0, -20.0, 3.0)]
        self.check_scene(config, camera, blocks)

    def test_camera_looking_from_the_side(self) -> None:
        camera = SyntheticCamera.looking_down((0.0, 25.0), distance=65.0, off_vertical=30.0, azimuth=60.0)
        blocks = [(0, 3.0, 20.0, 0.0, 3.0), (1, 12.0, 32.0, 25.0, 3.0)]
        self.check_scene(self.config, camera, blocks)

    def test_mild_lens_distortion(self) -> None:
        camera = SyntheticCamera.looking_down((0.0, 25.0), dist_coeffs=(-0.08, 0.05, 0.0, 0.0, 0.0))
        blocks = [(0, 3.0, 20.0, 0.0, 3.0), (1, 12.0, 32.0, 25.0, 3.0), (2, -2.0, 40.0, -40.0, 3.0)]
        self.check_scene(self.config, camera, blocks)

    def test_marker_on_the_third_level_needs_its_height(self) -> None:
        camera = SyntheticCamera.looking_down((0.0, 25.0))
        # a block on level 2 of a tower has its marker at (2 + 1) * 3 = 9
        blocks = [(0, 4.0, 22.0, 20.0, 9.0), (1, 14.0, 30.0, 0.0, 3.0)]
        handler = self.make_handler(camera)
        feed(handler, camera.render(self.config, blocks))
        with_height = {block.marker_id: block for block in handler.get_blocks({0: 9.0})}
        self.assert_block(with_height[0], 4.0, 22.0, 20.0)
        self.assertEqual(9.0, with_height[0].height)
        self.assertAlmostEqual(self.config.blocks.marker_length, with_height[0].side_length, delta=0.1)
        self.assert_block(with_height[1], 14.0, 30.0, 0.0)  # the other keeps the default of the table
        without = {block.marker_id: block for block in handler.get_blocks()}
        self.assertEqual(3.0, without[0].height)
        # the line of sight is wrong by 6 cm of height, which moves the block visibly sideways and makes it bigger
        self.assertGreater(math.hypot(without[0].x - 4.0, without[0].y - 22.0), 2.0)
        self.assertGreater(abs(without[0].side_length - self.config.blocks.marker_length), 0.2)

    def test_yaw_is_folded_into_a_quarter_turn(self) -> None:
        camera = SyntheticCamera.looking_down((0.0, 25.0))
        blocks = [(0, 3.0, 20.0, 80.0, 3.0), (1, 12.0, 32.0, -80.0, 3.0), (2, -2.0, 40.0, 100.0, 3.0),
                  (3, 20.0, 12.0, 44.0, 3.0)]
        handler = self.make_handler(camera)
        feed(handler, camera.render(self.config, blocks))
        found = {block.marker_id: block for block in handler.get_blocks()}
        self.assertAlmostEqual(-10.0, math.degrees(found[0].yaw), delta=2.0)
        self.assertAlmostEqual(10.0, math.degrees(found[1].yaw), delta=2.0)
        self.assertAlmostEqual(10.0, math.degrees(found[2].yaw), delta=2.0)
        self.assertAlmostEqual(44.0, math.degrees(found[3].yaw), delta=2.0)
        for block in found.values():
            self.assertGreaterEqual(block.yaw, -math.pi / 4)
            self.assertLess(block.yaw, math.pi / 4)

    def test_markers_that_are_not_blocks_are_ignored(self) -> None:
        camera = SyntheticCamera.looking_down((0.0, 25.0))
        blocks = [(0, 3.0, 20.0, 0.0, 3.0)]
        # 43 is a marker of the same dictionary that is on the arm and the table already
        self.assertNotIn(43, self.config.blocks.marker_ids)
        handler = self.make_handler(camera)
        feed(handler, camera.render(self.config, blocks, extra=[(43, 12.0, 32.0, 0.0, 3.0)]))
        self.assertEqual([0], [block.marker_id for block in handler.get_blocks()])

    def test_no_blocks_when_the_board_is_not_found(self) -> None:
        camera = SyntheticCamera.looking_down((0.0, 25.0))
        handler = self.make_handler(camera)
        feed(handler, camera.render(self.config, [(0, 3.0, 20.0, 0.0, 3.0)], with_board=False))
        self.assertEqual([], handler.get_blocks())
        self.assertFalse(handler.board_visible())
        self.assertEqual(0, handler.frames_with_board())

    def test_a_picture_without_the_board_does_not_use_the_board_of_an_earlier_one(self) -> None:
        camera = SyntheticCamera.looking_down((0.0, 25.0))
        handler = self.make_handler(camera)
        blocks = [(0, 3.0, 20.0, 0.0, 3.0)]
        feed(handler, camera.render(self.config, blocks))
        feed(handler, camera.render(self.config, [(0, 12.0, 32.0, 0.0, 3.0)], with_board=False))
        found = handler.get_blocks()
        self.assertEqual(1, len(found))
        self.assertEqual(1, found[0].frames)
        self.assertEqual(1, handler.frames_with_board())
        self.assert_block(found[0], 3.0, 20.0, 0.0)

    def hide_right_of_board(self, picture: np.ndarray, camera: SyntheticCamera, visible: float) -> None:
        """paint the board white but for its left part of this many cm, the rest of the table stays as it is"""
        board = self.config.board
        width, height = board.squares_x * board.square_length, board.squares_y * board.square_length
        board_x, board_y = np.moveaxis(robot_to_board(camera.plane_points(0.0), board), -1, 0)
        picture[(board_x > visible - width) & (board_x < 0.5) & (board_y > -0.5) & (board_y < height + 0.5)] = 255

    def test_a_board_that_is_mostly_hidden_is_not_used(self) -> None:
        camera = SyntheticCamera.looking_down((0.0, 25.0))
        picture = camera.render(self.config, [(0, 3.0, 20.0, 0.0, 3.0)])
        # the left 11 cm of the 15 cm board stay in view: 4 of its 8 inner corners, too few for a pose
        self.hide_right_of_board(picture, camera, 11.0)
        handler = self.make_handler(camera)
        feed(handler, picture)
        self.assertEqual(0, handler.frames_with_board())
        self.assertEqual([], handler.get_blocks())

    def test_a_pose_from_six_corners_is_still_good_enough(self) -> None:
        camera = SyntheticCamera.looking_down((0.0, 25.0))
        blocks = [(0, 3.0, 20.0, 0.0, 3.0), (1, 12.0, 32.0, 25.0, 3.0)]
        picture = camera.render(self.config, blocks)
        # the left 13 cm of the board stay in view: 6 inner corners, the least that is used
        self.hide_right_of_board(picture, camera, 13.0)
        handler = self.make_handler(camera)
        feed(handler, picture)
        self.assertEqual(1, handler.frames_with_board())
        for (_, x, y, yaw, _), block in zip(blocks, handler.get_blocks()):
            self.assert_block(block, x, y, yaw)

    def far_scene(self) -> tuple[SyntheticCamera, list[tuple[int, float, float, float, float]]]:
        """blocks 30 and 45 cm from the origin of the board, and the camera 70 cm up in front of them all"""
        board = self.config.board
        blocks = []
        for marker_id, (distance, direction, yaw) in enumerate(((30.0, 10.0, 0.0), (45.0, 35.0, 25.0),
                                                                (45.0, -15.0, -35.0))):
            x = board.x + distance * math.cos(math.radians(direction))
            y = board.y + distance * math.sin(math.radians(direction))
            blocks.append((marker_id, x, y, yaw, 3.0))
        tilt = 35.0
        camera = SyntheticCamera.looking_down((4.0, 20.0), distance=70.0 / math.cos(math.radians(tilt)),
                                              off_vertical=tilt)
        self.assertAlmostEqual(70.0, camera.centre[2])
        return camera, blocks

    def test_blocks_far_from_the_board(self) -> None:
        camera, blocks = self.far_scene()
        found = self.check_scene(self.config, camera, blocks)
        for block in found:
            self.assertLess(block.spread, 1e-9)  # the same picture every time

    def test_noise_on_the_board_corners_moves_the_blocks_little(self) -> None:
        camera, blocks = self.far_scene()
        picture = camera.render(self.config, blocks)
        generator = np.random.default_rng(7)
        detect = interpolate_charuco_corners

        def noisy(*arguments: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
            corners, ids = detect(*arguments)
            if corners is not None:
                corners = corners + generator.normal(0.0, 0.2, corners.shape).astype(corners.dtype)  # pixels
            return corners, ids

        handler = self.make_handler(camera)
        with mock.patch('src.camera.block_detection.interpolate_charuco_corners', noisy):
            for _ in range(self.config.scan.frames):
                feed(handler, picture)
        self.assertEqual(self.config.scan.frames, handler.frames_with_board())
        for (marker_id, x, y, yaw, z), block in zip(blocks, handler.get_blocks()):
            # the median of the pictures of a scan is better than a single one, which is off by 0.2 cm or so
            self.assertLess(math.hypot(block.x - x, block.y - y), 0.25, 'median of {}'.format(marker_id))
            self.assertLess(block.spread, 1.0)
            for observation in handler._observations[marker_id]:
                ox, oy, _, _ = observation.at_height(z)
                self.assertLess(math.hypot(ox - x, oy - y), 1.0)


class TestObservations(BlockDetectionTestCase):

    @staticmethod
    def observation_of(marker_id: int, origin: tuple[float, float, float], corners: list[tuple[float, float]],
                       z: float) -> BlockObservation:
        """the observation of a marker whose corners are at these x, y at the height z"""
        camera = np.array(origin)
        directions = np.array([np.array([x, y, z]) - camera for x, y in corners])
        directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]
        return BlockObservation(marker_id, camera, directions)

    def square(self, x: float, y: float, yaw_degrees: float, side: float = 2.4) -> list[tuple[float, float]]:
        """the corners in the order of the detector: clockwise seen from above, the first one top left"""
        turn = math.radians(yaw_degrees)
        right = np.array([math.cos(turn), math.sin(turn)])
        down = np.array([math.sin(turn), -math.cos(turn)])
        centre = np.array([x, y])
        return [tuple(centre + side / 2 * (a * right + b * down)) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]

    def test_at_height_finds_the_centre_the_yaw_and_the_side(self) -> None:
        observation = self.observation_of(3, (5.0, -20.0, 60.0), self.square(10.0, 20.0, 30.0), 3.0)
        x, y, yaw, side = observation.at_height(3.0)
        self.assertAlmostEqual(10.0, x, places=6)
        self.assertAlmostEqual(20.0, y, places=6)
        self.assertAlmostEqual(math.radians(30.0), yaw, places=6)
        self.assertAlmostEqual(2.4, side, places=6)

    def test_at_height_with_the_wrong_height(self) -> None:
        observation = self.observation_of(3, (5.0, -20.0, 60.0), self.square(10.0, 20.0, 0.0), 3.0)
        x, y, _, side = observation.at_height(9.0)
        # the lines of sight meet a higher plane closer to the camera, so the marker comes out smaller
        self.assertLess(y, 20.0)
        self.assertLess(side, 2.4)

    def test_at_height_refuses_a_plane_above_the_camera(self) -> None:
        observation = self.observation_of(3, (5.0, -20.0, 60.0), self.square(10.0, 20.0, 0.0), 3.0)
        with self.assertRaises(ValueError):
            observation.at_height(70.0)

    def test_yaw_folding(self) -> None:
        origin = (0.0, -20.0, 60.0)
        for degrees, expected in ((80.0, -10.0), (-80.0, 10.0), (45.0, -45.0), (-45.0, -45.0), (44.0, 44.0),
                                  (135.0, -45.0), (0.0, 0.0), (190.0, 10.0)):
            with self.subTest(degrees=degrees):
                _, _, yaw, _ = self.observation_of(0, origin, self.square(0.0, 20.0, degrees), 3.0).at_height(3.0)
                self.assertAlmostEqual(math.radians(expected), yaw, places=6)

    def add(self, handler: BlockImageHandler, observation: BlockObservation) -> None:
        with handler._lock:
            handler._observations.setdefault(observation.marker_id, deque(maxlen=handler._history)).append(observation)

    def test_the_median_ignores_an_outlier_and_the_spread_shows_it(self) -> None:
        handler = BlockImageHandler(self.config, np.eye(3), np.zeros(5))
        origin = (0.0, -20.0, 60.0)
        for dx in (-0.05, 0.0, 0.05, 0.02, -0.02):
            self.add(handler, self.observation_of(0, origin, self.square(10.0 + dx, 20.0, 5.0), 3.0))
        self.add(handler, self.observation_of(0, origin, self.square(14.0, 20.0, 5.0), 3.0))  # 4 cm away
        block = handler.get_blocks()[0]
        self.assertEqual(6, block.frames)
        self.assertAlmostEqual(10.0, block.x, delta=0.05)
        self.assertAlmostEqual(20.0, block.y, delta=0.05)
        self.assertAlmostEqual(4.0, block.spread, delta=0.1)
        self.assertFalse(block.stable)

    def test_stable_follows_the_configured_spread(self) -> None:
        origin = (0.0, -20.0, 60.0)
        for max_spread, stable in ((0.5, True), (0.2, False)):
            with self.subTest(max_spread=max_spread):
                handler = BlockImageHandler(with_scan(self.config, max_spread=max_spread), np.eye(3), np.zeros(5))
                for dx in (-0.3, 0.0, 0.3):
                    self.add(handler, self.observation_of(0, origin, self.square(10.0 + dx, 20.0, 0.0), 3.0))
                block = handler.get_blocks()[0]
                self.assertAlmostEqual(0.3, block.spread, delta=0.01)
                self.assertEqual(stable, block.stable)

    def test_the_yaw_mean_survives_the_fold(self) -> None:
        handler = BlockImageHandler(self.config, np.eye(3), np.zeros(5))
        origin = (0.0, -20.0, 60.0)
        # 44 and -44 degrees are 2 degrees apart on the circle of quarter turns, a plain mean would say 0
        for degrees in (44.0, -44.0, 44.0, -44.0, 44.0):
            self.add(handler, self.observation_of(0, origin, self.square(10.0, 20.0, degrees), 3.0))
        yaw = math.degrees(handler.get_blocks()[0].yaw)
        self.assertLess(abs(abs(yaw) - 45.0), 2.0)

    def test_blocks_are_sorted_and_unseen_ids_are_absent(self) -> None:
        handler = BlockImageHandler(self.config, np.eye(3), np.zeros(5))
        origin = (0.0, -20.0, 60.0)
        for marker_id in (7, 2, 5):
            self.add(handler, self.observation_of(marker_id, origin, self.square(marker_id, 20.0, 0.0), 3.0))
        self.assertEqual([2, 5, 7], [block.marker_id for block in handler.get_blocks()])

    def test_heights_per_id(self) -> None:
        handler = BlockImageHandler(self.config, np.eye(3), np.zeros(5))
        origin = (0.0, -20.0, 60.0)
        self.add(handler, self.observation_of(1, origin, self.square(10.0, 20.0, 0.0), 9.0))
        self.add(handler, self.observation_of(2, origin, self.square(0.0, 30.0, 0.0), 3.0))
        blocks = {block.marker_id: block for block in handler.get_blocks({1: 9.0})}
        self.assertEqual(9.0, blocks[1].height)
        self.assertEqual(3.0, blocks[2].height)
        self.assertAlmostEqual(10.0, blocks[1].x, places=6)
        self.assertAlmostEqual(0.0, blocks[2].x, places=6)

    def test_history_keeps_the_last_observations(self) -> None:
        handler = BlockImageHandler(self.config, np.eye(3), np.zeros(5), history=4)
        origin = (0.0, -20.0, 60.0)
        for x in (0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0):
            self.add(handler, self.observation_of(0, origin, self.square(x, 20.0, 0.0), 3.0))
        block = handler.get_blocks()[0]
        self.assertEqual(4, block.frames)
        self.assertAlmostEqual(5.0, block.x, places=6)

    def test_clear_forgets_the_observations(self) -> None:
        handler = BlockImageHandler(self.config, np.eye(3), np.zeros(5))
        self.add(handler, self.observation_of(0, (0.0, -20.0, 60.0), self.square(10.0, 20.0, 0.0), 3.0))
        handler.clear()
        self.assertEqual([], handler.get_blocks())
        self.assertEqual(0, handler.frames_with_board())

    def test_a_plane_that_is_not_reached_leaves_the_block_out(self) -> None:
        handler = BlockImageHandler(self.config, np.eye(3), np.zeros(5))
        self.add(handler, self.observation_of(0, (0.0, -20.0, 60.0), self.square(10.0, 20.0, 0.0), 3.0))
        self.assertEqual([], handler.get_blocks({0: 80.0}))


class TestScan(BlockDetectionTestCase):

    class FakeTime:
        """a clock that only moves when the scan sleeps, and the camera that takes a picture every time it does"""

        def __init__(self, pictures: Callable[[int], np.ndarray | None]) -> None:
            self.now = 100.0
            self.sleeps = 0
            self.pictures = pictures
            self.handler: BlockImageHandler | None = None

        def clock(self) -> float:
            return self.now

        def sleep(self, seconds: float) -> None:
            assert self.handler is not None
            self.now += seconds
            picture = self.pictures(self.sleeps)
            self.sleeps += 1
            if picture is not None:
                feed(self.handler, picture)

    def setUp(self) -> None:
        super().setUp()
        self.camera = SyntheticCamera.looking_down((0.0, 25.0))

    def make_scan(self, config: StackingConfig, pictures: Callable[[int], np.ndarray | None]
                  ) -> tuple[BlockImageHandler, 'TestScan.FakeTime']:
        fake = TestScan.FakeTime(pictures)
        handler = self.make_handler(self.camera, config, clock=fake.clock)
        fake.handler = handler
        return handler, fake

    def test_scan_returns_after_the_configured_number_of_pictures(self) -> None:
        picture = self.camera.render(self.config, [(0, 3.0, 20.0, 0.0, 3.0), (1, 12.0, 32.0, 25.0, 3.0)])
        handler, fake = self.make_scan(self.config, lambda _: picture)
        blocks = handler.scan(sleep=fake.sleep)
        self.assertEqual(self.config.scan.frames, fake.sleeps)
        self.assertEqual(self.config.scan.frames, handler.frames_with_board())
        self.assertEqual([0, 1], [block.marker_id for block in blocks])
        self.assert_block(blocks[0], 3.0, 20.0, 0.0)
        self.assert_block(blocks[1], 12.0, 32.0, 25.0)
        self.assertTrue(all(block.stable and block.frames == self.config.scan.frames for block in blocks))

    def test_scan_forgets_what_was_seen_before(self) -> None:
        old = self.camera.render(self.config, [(5, 3.0, 20.0, 0.0, 3.0)])
        new = self.camera.render(self.config, [(0, 12.0, 32.0, 0.0, 3.0)])
        handler, fake = self.make_scan(self.config, lambda _: new)
        feed(handler, old)
        self.assertEqual([5], [block.marker_id for block in handler.get_blocks()])
        self.assertEqual([0], [block.marker_id for block in handler.scan(sleep=fake.sleep)])

    def test_scan_drops_a_block_that_was_seen_in_fewer_than_half_of_the_pictures(self) -> None:
        both = self.camera.render(self.config, [(0, 3.0, 20.0, 0.0, 3.0), (1, 12.0, 32.0, 25.0, 3.0)])
        one = self.camera.render(self.config, [(0, 3.0, 20.0, 0.0, 3.0)])
        # marker 1 is in 3 of 10 pictures, marker 2 in 5 of 10, which is half and is kept
        both_and_two = self.camera.render(self.config, [(0, 3.0, 20.0, 0.0, 3.0), (2, -2.0, 40.0, 0.0, 3.0)])
        pictures = [both, both, both, one, one] + [both_and_two] * 5

        def picture(index: int) -> np.ndarray:
            return pictures[index]

        handler, fake = self.make_scan(self.config, picture)
        blocks = handler.scan(sleep=fake.sleep)
        self.assertEqual(10, handler.frames_with_board())
        self.assertEqual([0, 2], [block.marker_id for block in blocks])
        self.assertEqual([10, 5], [block.frames for block in blocks])
        self.assertEqual([0, 1, 2], [block.marker_id for block in handler.get_blocks()])

    def test_scan_gives_the_heights_to_get_blocks(self) -> None:
        picture = self.camera.render(self.config, [(0, 4.0, 22.0, 20.0, 9.0)])
        handler, fake = self.make_scan(self.config, lambda _: picture)
        blocks = handler.scan({0: 9.0}, sleep=fake.sleep)
        self.assert_block(blocks[0], 4.0, 22.0, 20.0)

    def test_scan_raises_when_there_is_no_board(self) -> None:
        config = with_scan(self.config, timeout=1.0)
        picture = self.camera.render(config, [(0, 3.0, 20.0, 0.0, 3.0)], with_board=False)
        handler, fake = self.make_scan(config, lambda _: picture)
        with self.assertRaises(ScanError) as context:
            handler.scan(sleep=fake.sleep)
        self.assertIn('the board is not in the picture, or the camera gives no pictures', str(context.exception))
        self.assertGreaterEqual(fake.now - 100.0, 1.0)  # it waited for the timeout, not less
        self.assertLess(fake.now - 100.0, 1.2)

    def test_scan_raises_when_the_camera_gives_no_pictures(self) -> None:
        config = with_scan(self.config, timeout=0.5)
        handler, fake = self.make_scan(config, lambda _: None)
        with self.assertRaises(ScanError):
            handler.scan(sleep=fake.sleep)

    def test_scan_with_some_pictures_before_the_timeout_returns_what_there_is(self) -> None:
        config = with_scan(self.config, timeout=1.0)
        picture = self.camera.render(config, [(0, 3.0, 20.0, 0.0, 3.0)])
        # a picture on every fourth look only: 5 pictures in the 1 second that there is
        handler, fake = self.make_scan(config, lambda index: picture if index % 4 == 0 else None)
        blocks = handler.scan(sleep=fake.sleep)
        pictures = len([index for index in range(fake.sleeps) if index % 4 == 0])
        self.assertIn(pictures, (5, 6))  # a clock that adds 0.05 twenty times may be just short of 1.0
        self.assertEqual(pictures, handler.frames_with_board())
        self.assertEqual([0], [block.marker_id for block in blocks])

    def test_board_visible_follows_the_clock(self) -> None:
        picture = self.camera.render(self.config, [(0, 3.0, 20.0, 0.0, 3.0)])
        handler, fake = self.make_scan(self.config, lambda _: picture)
        self.assertFalse(handler.board_visible())
        feed(handler, picture)
        self.assertTrue(handler.board_visible())
        fake.now += 0.9
        self.assertTrue(handler.board_visible())
        fake.now += 0.2
        self.assertFalse(handler.board_visible())
        self.assertTrue(handler.board_visible(max_age=5.0))


class TestFrames(BlockDetectionTestCase):

    def test_board_to_robot_and_back(self) -> None:
        board = dataclasses.replace(self.config.board, x=10.0, y=-5.0, rotation=90.0)
        np.testing.assert_allclose([10.0, -4.0, 0.0], board_to_robot([1.0, 0.0, 0.0], board), atol=1e-9)
        np.testing.assert_allclose([9.0, -5.0, 2.0], board_to_robot([0.0, 1.0, 2.0], board), atol=1e-9)
        board = dataclasses.replace(board, rotation=30.0)
        expected = [10.0 + 4 * math.cos(math.radians(30)), -5.0 + 4 * math.sin(math.radians(30))]
        np.testing.assert_allclose(expected, board_to_robot([4.0, 0.0], board), atol=1e-9)
        points = np.array([[1.0, 2.0, 3.0], [-4.0, 7.5, 0.5], [0.0, 0.0, 0.0]])
        np.testing.assert_allclose(points, robot_to_board(board_to_robot(points, board), board), atol=1e-9)
        self.assertEqual((3, 3), board_to_robot(points, board).shape)
        self.assertEqual((2,), board_to_robot([1.0, 2.0], board).shape)

    def test_the_origin_of_the_board_is_its_position(self) -> None:
        board = dataclasses.replace(self.config.board, x=-12.0, y=12.0, rotation=17.0)
        np.testing.assert_allclose([-12.0, 12.0, 0.0], board_to_robot([0.0, 0.0, 0.0], board), atol=1e-12)

    def test_rotation_is_counter_clockwise_seen_from_above(self) -> None:
        board = dataclasses.replace(self.config.board, x=0.0, y=0.0, rotation=90.0)
        np.testing.assert_allclose([0.0, 1.0, 0.0], board_to_robot([1.0, 0.0, 0.0], board), atol=1e-9)

    def test_board_image_size(self) -> None:
        board = self.config.board
        image = board_image(board, 20)
        # 5 by 3 squares of 3 cm, without a margin
        self.assertEqual((3 * 3 * 20, 5 * 3 * 20), image.shape)
        self.assertEqual((board.squares_y, board.squares_x), (3, 5))
        self.assertEqual(np.uint8, image.dtype)

    def test_board_image_needs_whole_pixels(self) -> None:
        with self.assertRaises(ValueError):
            # 3.5 cm squares of 24.5 pixels
            board_image(dataclasses.replace(self.config.board, square_length=3.5, marker_length=2.6), 7)

    def test_the_detector_finds_the_board_in_its_image(self) -> None:
        board = self.config.board
        image = board_image(board, 40)
        parameters = cv2.aruco.DetectorParameters()
        corners, ids, _ = cv2.aruco.ArucoDetector(aruco_dictionary(board.dictionary), parameters).detectMarkers(image)
        self.assertEqual(board.squares_x * board.squares_y // 2, len(ids))
        charuco_corners, charuco_ids, _, _ = cv2.aruco.CharucoDetector(make_board(board)).detectBoard(image)
        self.assertEqual((board.squares_x - 1) * (board.squares_y - 1), len(charuco_ids))

    def test_the_opencv_board_frame_is_ours_turned_around_x(self) -> None:
        # the origin of OpenCV's board is the top left of the picture and its y points down in the picture, so
        # that our y (up in the picture) is the height of the picture minus the y of OpenCV, and our x (the origin
        # is at the bottom right) is the x of OpenCV minus the width of the picture
        board = self.config.board
        ppc = 40
        image = board_image(board, ppc)
        detected, ids, _, _ = cv2.aruco.CharucoDetector(make_board(board)).detectBoard(image)
        opencv_corners = make_board(board).getChessboardCorners()[ids.ravel()]
        width, height = board.squares_x * board.square_length, board.squares_y * board.square_length
        for pixel, point in zip(detected.reshape(-1, 2), opencv_corners):
            self.assertAlmostEqual(point[0], pixel[0] / ppc, delta=0.05)
            self.assertAlmostEqual(point[1], pixel[1] / ppc, delta=0.05)  # y down in the picture
            ours_x = pixel[0] / ppc - width  # the definition of our frame
            ours_y = height - pixel[1] / ppc
            self.assertAlmostEqual(point[0] - width, ours_x, delta=0.05)
            self.assertAlmostEqual(height - point[1], ours_y, delta=0.05)
            self.assertTrue(-width <= ours_x <= 0 and 0 <= ours_y <= height)  # the board is left of its origin

    def test_handle_frame_draws_without_changing_what_is_detected(self) -> None:
        camera = SyntheticCamera.looking_down((0.0, 25.0))
        gray = camera.render(self.config, [(0, 3.0, 20.0, 0.0, 3.0), (1, 12.0, 32.0, 25.0, 3.0)])
        original = gray.copy()
        handler = BlockImageHandler(self.config, camera.camera_matrix, camera.dist_coeffs)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        before = frame.copy()
        handler.handle_frame(frame, gray)
        self.assertFalse(np.array_equal(before, frame), 'nothing was drawn')
        np.testing.assert_array_equal(original, gray)
        # the green outline of a block is on the picture
        self.assertTrue(np.any(np.all(frame == (0, 255, 0), axis=-1)))
        # the yellow ring of the tower is on the picture, the tower is at 22, 18 and in view
        self.assertTrue(np.any(np.all(frame == (0, 255, 255), axis=-1)))
        # the same detection when the picture that is shown is drawn on again, as with several handlers
        first = handler.get_blocks()
        handler.clear()
        handler.handle_frame(frame, gray)
        self.assertEqual(first, handler.get_blocks())

    def test_a_picture_without_the_board_says_so(self) -> None:
        camera = SyntheticCamera.looking_down((0.0, 25.0))
        handler = BlockImageHandler(self.config, camera.camera_matrix, camera.dist_coeffs)
        gray = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        handler.handle_frame(frame, gray)
        self.assertTrue(np.any(frame[..., 2] == 255), 'the red text is not there')
        self.assertEqual([], handler.get_blocks())
        self.assertFalse(handler.board_visible())

    def test_bad_pictures_do_not_raise(self) -> None:
        handler = BlockImageHandler(self.config, np.eye(3) * 1000, np.zeros(5))
        for shape in ((0, 0), (1, 1), (5, 7), (100, 100)):
            with self.subTest(shape=shape):
                gray = np.zeros(shape, dtype=np.uint8)
                handler.handle_frame(np.zeros(shape + (3,), dtype=np.uint8), gray)
        noise = np.random.default_rng(1).integers(0, 255, (HEIGHT, WIDTH), dtype=np.uint8)
        handler.handle_frame(cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR), noise)
        self.assertEqual(0, handler.frames_with_board())
        handler.deactivate_handler()


class ClickTestCase(BlockDetectionTestCase):
    """the scenes are rendered once for all the tests, rendering takes a while"""

    camera: SyntheticCamera
    scenes: dict[str, np.ndarray]

    # two markers far apart, and two that are one block apart
    FAR = [(0, 3.0, 20.0, 0.0, 3.0), (1, 12.0, 32.0, 25.0, 3.0)]
    CLOSE = [(0, 3.0, 20.0, 0.0, 3.0), (1, 6.1, 20.0, 0.0, 3.0)]

    @classmethod
    def setUpClass(cls) -> None:
        config = load_frozen_config()
        cls.camera = SyntheticCamera.looking_down((0.0, 25.0))
        cls.scenes = {
            'far': cls.camera.render(config, cls.FAR),
            'far_without_board': cls.camera.render(config, cls.FAR, with_board=False),
            'close': cls.camera.render(config, cls.CLOSE),
            # 43 is a marker of the dictionary that is not a block
            'with_arm_marker': cls.camera.render(config, cls.FAR[:1], extra=[(43, 12.0, 32.0, 0.0, 3.0)]),
        }

    def handler_after(self, scene: str) -> BlockImageHandler:
        handler = self.make_handler(self.camera)
        feed(handler, self.scenes[scene])
        return handler

    def project(self, x: float, y: float, z: float = 3.0) -> tuple[float, float]:
        """the pixel of a point of the robot frame, from the pose of the synthetic camera alone"""
        in_camera = self.camera.world_to_camera @ (np.array([x, y, z]) - self.camera.centre)
        pixel = self.camera.camera_matrix @ (in_camera / in_camera[2])
        return float(pixel[0]), float(pixel[1])

    def pixel(self, x: float, y: float) -> tuple[int, int]:
        """the whole pixel of the middle of the marker of a block at x, y, which is what a click is"""
        pixel_x, pixel_y = self.project(x, y)
        return round(pixel_x), round(pixel_y)

    def outline(self, handler: BlockImageHandler, marker_id: int) -> np.ndarray:
        found = [corners for found_id, corners in handler._outlines if found_id == marker_id]
        self.assertEqual(1, len(found), 'marker {} is not in the picture'.format(marker_id))
        return found[0]

    @staticmethod
    def sides_outwards(corners: np.ndarray, extra: float) -> list[tuple[float, float]]:
        """the points extra pixels outside the middle of each side of the outline"""
        centre = corners.mean(axis=0)
        points = []
        for index in range(4):
            middle = (corners[index] + corners[(index + 1) % 4]) / 2
            outwards = (middle - centre) / np.linalg.norm(middle - centre)
            point = middle + extra * outwards
            points.append((float(point[0]), float(point[1])))
        return points


class TestBlockAt(ClickTestCase):

    def test_no_picture_yet(self) -> None:
        handler = self.make_handler(self.camera)
        self.assertIsNone(handler.block_at(960, 540))

    def test_the_centre_of_a_marker(self) -> None:
        handler = self.handler_after('far')
        for marker_id, x, y, _, z in self.FAR:
            with self.subTest(marker_id=marker_id):
                pixel_x, pixel_y = self.project(x, y, z)
                self.assertEqual(marker_id, handler.block_at(pixel_x, pixel_y))
                # the outline that was detected is where the camera puts the marker
                centre = self.outline(handler, marker_id).mean(axis=0)
                self.assertAlmostEqual(pixel_x, centre[0], delta=1.5)
                self.assertAlmostEqual(pixel_y, centre[1], delta=1.5)

    def test_a_point_inside_a_corner(self) -> None:
        handler = self.handler_after('far')
        corners = self.outline(handler, 1)
        centre = corners.mean(axis=0)
        for corner in corners:
            inside = corner + 0.1 * (centre - corner)
            self.assertEqual(1, handler.block_at(float(inside[0]), float(inside[1])))

    def test_a_point_just_outside_the_outline(self) -> None:
        handler = self.handler_after('far')
        for marker_id in (0, 1):
            corners = self.outline(handler, marker_id)
            for point in self.sides_outwards(corners, 4.0):
                with self.subTest(marker_id=marker_id, point=point):
                    self.assertEqual(marker_id, handler.block_at(*point))

    def test_a_point_farther_than_a_diagonal_from_the_marker_is_nothing(self) -> None:
        handler = self.handler_after('far')
        corners = self.outline(handler, 0)
        diagonal = float(np.linalg.norm(corners[0] - corners[2]))
        centre = corners.mean(axis=0)
        # away from the other marker, so that this one is the nearest
        away = centre - self.outline(handler, 1).mean(axis=0)
        away /= np.linalg.norm(away)
        near = centre + 0.9 * diagonal * away
        far = centre + 1.1 * diagonal * away
        self.assertEqual(0, handler.block_at(float(near[0]), float(near[1])))
        self.assertIsNone(handler.block_at(float(far[0]), float(far[1])))

    def test_the_reach_follows_the_size_of_the_marker_in_the_picture(self) -> None:
        camera = SyntheticCamera.looking_down((0.0, 25.0), distance=100.0, off_vertical=20.0)
        handler = self.make_handler(camera)
        feed(handler, camera.render(self.config, self.FAR[:1]))
        corners = self.outline(handler, 0)
        diagonal = float(np.linalg.norm(corners[0] - corners[2]))
        usual = self.outline(self.handler_after('far'), 0)
        self.assertLess(diagonal, 0.95 * float(np.linalg.norm(usual[0] - usual[2])))  # smaller than in the others
        centre = corners.mean(axis=0)
        self.assertEqual(0, handler.block_at(centre[0] + 0.9 * diagonal, centre[1]))
        self.assertIsNone(handler.block_at(centre[0] + 1.1 * diagonal, centre[1]))

    def test_a_point_far_from_every_marker(self) -> None:
        handler = self.handler_after('far')
        for point in ((960, 540), (5, 5), (1900, 1070), (-50, -50), (1e6, 1e6)):
            with self.subTest(point=point):
                self.assertIsNone(handler.block_at(*point))

    def test_two_markers_close_together(self) -> None:
        handler = self.handler_after('close')
        first, second = self.outline(handler, 0), self.outline(handler, 1)
        first_centre, second_centre = first.mean(axis=0), second.mean(axis=0)
        gap = float(np.linalg.norm(second_centre - first_centre))
        self.assertLess(gap, 1.5 * float(np.linalg.norm(first[0] - first[2])))  # they are close indeed
        self.assertEqual(0, handler.block_at(float(first_centre[0]), float(first_centre[1])))
        self.assertEqual(1, handler.block_at(float(second_centre[0]), float(second_centre[1])))
        for fraction, expected in ((0.2, 0), (0.4, 0), (0.6, 1), (0.8, 1)):
            with self.subTest(fraction=fraction):
                point = first_centre + fraction * (second_centre - first_centre)
                self.assertEqual(expected, handler.block_at(float(point[0]), float(point[1])))
        # a point in one outline close to the other marker belongs to the marker it is in, the corner of each
        # outline that is nearest to the other marker
        corner_first = min(first, key=lambda corner: float(np.linalg.norm(corner - second_centre)))
        corner_second = min(second, key=lambda corner: float(np.linalg.norm(corner - first_centre)))
        inside_first = corner_first + 0.15 * (first_centre - corner_first)
        inside_second = corner_second + 0.15 * (second_centre - corner_second)
        self.assertEqual(0, handler.block_at(float(inside_first[0]), float(inside_first[1])))
        self.assertEqual(1, handler.block_at(float(inside_second[0]), float(inside_second[1])))

    def test_a_marker_that_is_not_a_block_is_nothing(self) -> None:
        handler = self.handler_after('with_arm_marker')
        self.assertEqual([0], [marker_id for marker_id, _ in handler._outlines])
        self.assertIsNone(handler.block_at(*self.project(12.0, 32.0)))

    def test_a_marker_is_found_when_the_board_is_not(self) -> None:
        handler = self.handler_after('far_without_board')
        self.assertEqual([], handler.get_blocks())
        self.assertEqual(1, handler.block_at(*self.project(12.0, 32.0)))

    def test_only_the_last_picture_counts(self) -> None:
        handler = self.handler_after('far')
        pixel = self.project(3.0, 20.0)
        self.assertEqual(0, handler.block_at(*pixel))
        feed(handler, np.full((HEIGHT, WIDTH), 255, dtype=np.uint8))
        self.assertIsNone(handler.block_at(*pixel))

    def test_a_bad_picture_forgets_the_outlines(self) -> None:
        handler = self.handler_after('far')
        with mock.patch.object(handler, '_detect', side_effect=cv2.error('bad picture')):
            feed(handler, self.scenes['far'])
        self.assertIsNone(handler.block_at(*self.project(3.0, 20.0)))


class TestClicks(ClickTestCase):

    def test_the_listener_gets_the_id_of_the_block(self) -> None:
        handler = self.handler_after('far')
        heard: list[int] = []
        handler.set_click_listener(lambda marker_id: heard.append(marker_id))
        handler.handle_click(*self.pixel(12.0, 32.0))
        handler.handle_click(*self.pixel(3.0, 20.0))
        self.assertEqual([1, 0], heard)

    def test_a_click_on_nothing_is_not_heard(self) -> None:
        handler = self.handler_after('far')
        listener = mock.Mock()
        handler.set_click_listener(listener)
        handler.handle_click(960, 540)
        handler.handle_click(0, 0)
        listener.assert_not_called()

    def test_a_click_before_any_picture_is_not_heard(self) -> None:
        handler = self.make_handler(self.camera)
        listener = mock.Mock()
        handler.set_click_listener(listener)
        handler.handle_click(*self.pixel(3.0, 20.0))
        listener.assert_not_called()

    def test_a_listener_that_raises_is_logged_and_the_next_click_works(self) -> None:
        handler = self.handler_after('far')
        listener = mock.Mock(side_effect=RuntimeError('the arm is not ready'))
        handler.set_click_listener(listener)
        with self.assertLogs('src.camera.block_detection', level='ERROR') as logs:
            handler.handle_click(*self.pixel(3.0, 20.0))
        self.assertIn('the arm is not ready', '\n'.join(logs.output))
        with self.assertLogs('src.camera.block_detection', level='ERROR'):
            handler.handle_click(*self.pixel(3.0, 20.0))
        self.assertEqual(2, listener.call_count)

    def test_without_a_listener_a_click_does_nothing(self) -> None:
        handler = self.handler_after('far')
        handler.handle_click(*self.pixel(3.0, 20.0))
        listener = mock.Mock()
        handler.set_click_listener(listener)
        handler.set_click_listener(None)
        handler.handle_click(*self.pixel(3.0, 20.0))
        listener.assert_not_called()

    def test_the_listener_is_called_outside_the_lock_of_the_handler(self) -> None:
        handler = self.handler_after('far')
        outcome: list[str] = []

        def other_thread_uses_the_handler() -> None:
            handler.set_message('selected')
            outcome.append('done')

        def listener(marker_id: int) -> None:
            # the same thread may use the handler, and so may another one while this one waits for it
            handler.set_selected(marker_id)
            other = threading.Thread(target=other_thread_uses_the_handler)
            other.start()
            other.join(5.0)
            self.assertFalse(other.is_alive(), 'another thread could not use the handler during the click')

        handler.set_click_listener(listener)
        handler.handle_click(*self.pixel(3.0, 20.0))
        self.assertEqual(['done'], outcome)


class TestSelectionAndMessage(ClickTestCase):

    def frame_after(self, handler: BlockImageHandler, scene: str) -> np.ndarray:
        return feed(handler, self.scenes[scene])

    def test_the_selected_block_is_drawn_apart(self) -> None:
        plain, selecting = self.make_handler(self.camera), self.make_handler(self.camera)
        before = self.frame_after(plain, 'far')
        selecting.set_selected(0)
        after = self.frame_after(selecting, 'far')
        self.assertFalse(np.array_equal(before, after))
        self.assertFalse(np.any(np.all(before == (255, 0, 255), axis=-1)))
        self.assertTrue(np.any(np.all(after == (255, 0, 255), axis=-1)))
        # only around marker 0, the other block and the rest of the picture are the same
        rows, columns = np.nonzero(np.any(before != after, axis=-1))
        corners = self.outline(plain, 0)
        low, high = corners.min(axis=0) - 40, corners.max(axis=0) + 40  # with room for the id above the corner
        self.assertTrue(np.all((columns >= low[0]) & (columns <= high[0]) & (rows >= low[1]) & (rows <= high[1])))
        # what is detected does not change
        self.assertEqual(plain.get_blocks(), selecting.get_blocks())
        self.assertEqual(plain.frames_with_board(), selecting.frames_with_board())

    def test_the_other_block_stays_green(self) -> None:
        handler = self.make_handler(self.camera)
        handler.set_selected(0)
        frame = self.frame_after(handler, 'far')
        x, y = np.round(self.outline(handler, 1)[0]).astype(int)
        # the first corner of the outline is on the line that is drawn
        patch = frame[y - 2:y + 3, x - 2:x + 3]
        self.assertTrue(np.any(np.all(patch == (0, 255, 0), axis=-1)))
        self.assertFalse(np.any(np.all(patch == (255, 0, 255), axis=-1)))

    def test_clearing_the_selection_restores_the_picture(self) -> None:
        plain, cleared = self.make_handler(self.camera), self.make_handler(self.camera)
        cleared.set_selected(1)
        cleared.set_selected(None)
        np.testing.assert_array_equal(self.frame_after(plain, 'far'), self.frame_after(cleared, 'far'))

    def test_a_selection_of_a_block_that_is_not_in_the_picture_changes_nothing(self) -> None:
        plain, selecting = self.make_handler(self.camera), self.make_handler(self.camera)
        selecting.set_selected(5)
        np.testing.assert_array_equal(self.frame_after(plain, 'far'), self.frame_after(selecting, 'far'))

    def test_the_selection_stays_over_pictures(self) -> None:
        handler = self.make_handler(self.camera)
        handler.set_selected(1)
        for _ in range(2):
            frame = self.frame_after(handler, 'far')
            self.assertTrue(np.any(np.all(frame == (255, 0, 255), axis=-1)))

    def test_the_selected_block_is_drawn_when_the_board_is_not_found(self) -> None:
        plain, selecting = self.make_handler(self.camera), self.make_handler(self.camera)
        selecting.set_selected(1)
        before = self.frame_after(plain, 'far_without_board')
        after = self.frame_after(selecting, 'far_without_board')
        self.assertEqual(0, selecting.frames_with_board())
        self.assertFalse(np.any(np.all(before == (255, 0, 255), axis=-1)))
        self.assertTrue(np.any(np.all(after == (255, 0, 255), axis=-1)))

    def test_the_message_is_drawn_at_the_top(self) -> None:
        plain, talking = self.make_handler(self.camera), self.make_handler(self.camera)
        talking.set_message('taking block 3')
        before = self.frame_after(plain, 'far')
        after = self.frame_after(talking, 'far')
        self.assertFalse(np.array_equal(before, after))
        rows, _ = np.nonzero(np.any(before != after, axis=-1))
        self.assertLess(rows.max(), 80)  # only the bar at the top changed
        self.assertTrue(np.all(after[0] == _COLOR_MESSAGE_BAR))
        self.assertTrue(np.any(np.all(after[:60] == _COLOR_MESSAGE_TEXT, axis=-1)))
        self.assertEqual(plain.get_blocks(), talking.get_blocks())

    def test_the_message_can_be_read_on_a_light_and_on_a_dark_picture(self) -> None:
        for name, level in (('light', 255), ('dark', 0)):
            with self.subTest(name):
                handler = self.make_handler(self.camera)
                handler.set_message('taking block 3')
                frame = feed(handler, np.full((HEIGHT, WIDTH), level, dtype=np.uint8))
                self.assertTrue(np.all(frame[0] == _COLOR_MESSAGE_BAR))
                self.assertTrue(np.any(np.all(frame[:60] == _COLOR_MESSAGE_TEXT, axis=-1)))

    def test_the_message_is_drawn_when_the_board_is_not_found(self) -> None:
        plain, talking = self.make_handler(self.camera), self.make_handler(self.camera)
        talking.set_message('where is the board')
        before = self.frame_after(plain, 'far_without_board')
        after = self.frame_after(talking, 'far_without_board')
        self.assertEqual(0, talking.frames_with_board())
        self.assertFalse(np.array_equal(before, after))
        self.assertTrue(np.all(after[0] == _COLOR_MESSAGE_BAR))
        self.assertTrue(np.any(np.all(after == (0, 0, 255), axis=-1)), 'the board warning has to stay in the picture')

    def test_none_removes_the_message(self) -> None:
        plain, talking = self.make_handler(self.camera), self.make_handler(self.camera)
        talking.set_message('taking block 3')
        talking.set_message(None)
        np.testing.assert_array_equal(self.frame_after(plain, 'far'), self.frame_after(talking, 'far'))

    def test_the_message_is_replaced(self) -> None:
        handler = self.make_handler(self.camera)
        handler.set_message('one')
        first = self.frame_after(handler, 'far')
        handler.set_message('two')
        second = self.frame_after(handler, 'far')
        self.assertFalse(np.array_equal(first, second))

    def test_an_empty_message_draws_nothing(self) -> None:
        plain, talking = self.make_handler(self.camera), self.make_handler(self.camera)
        talking.set_message('  \n ')
        np.testing.assert_array_equal(self.frame_after(plain, 'far'), self.frame_after(talking, 'far'))

    def test_a_long_message_is_cut_to_the_width_of_the_picture(self) -> None:
        handler = self.make_handler(self.camera)
        handler.set_message('this is a very long message about the block that is taken ' * 10)
        frame = feed(handler, np.full((HEIGHT, WIDTH), 255, dtype=np.uint8))
        # the margin at the right of the bar has no text, which the edge of the picture would cut off otherwise
        self.assertTrue(np.all(frame[:20, -8:] == _COLOR_MESSAGE_BAR))
        # and the text goes on to near the edge: it is not cut shorter than it has to be
        dark_columns = np.nonzero(np.any(np.all(frame[:60] == _COLOR_MESSAGE_TEXT, axis=-1), axis=0))[0]
        self.assertGreater(dark_columns.max(), WIDTH - 60)

    def test_what_is_written_is_the_message_or_its_start_with_an_ellipsis(self) -> None:
        gray = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
        long_message = 'this is a very long message about the block that is taken ' * 10
        for message, fits in (('taking block 3', True), (long_message, False)):
            with self.subTest(fits=fits):
                handler = self.make_handler(self.camera)
                handler.set_message(message)
                with mock.patch('src.camera.block_detection.cv2.putText', wraps=cv2.putText) as put_text:
                    feed(handler, gray)
                written = [call.args[1] for call in put_text.call_args_list if call.args[1] != 'BOARD NOT FOUND']
                self.assertEqual(1, len(written))
                if fits:
                    self.assertEqual(message, written[0])
                else:
                    self.assertTrue(written[0].endswith('...'))
                    self.assertTrue(message.startswith(written[0][:-3]))
                    self.assertLess(len(written[0]), len(message))

    def test_a_message_on_a_picture_that_is_too_narrow(self) -> None:
        handler = self.make_handler(self.camera)
        handler.set_message('taking block 3')
        for shape in ((0, 0), (1, 1), (50, 5), (60, 200)):
            with self.subTest(shape=shape):
                handler.handle_frame(np.zeros(shape + (3,), dtype=np.uint8), np.zeros(shape, dtype=np.uint8))

    def test_the_gray_picture_is_not_drawn_on(self) -> None:
        handler = self.make_handler(self.camera)
        handler.set_message('taking block 3')
        handler.set_selected(0)
        gray = self.scenes['far'].copy()
        feed(handler, gray)
        np.testing.assert_array_equal(self.scenes['far'], gray)


class TestThreadSafety(ClickTestCase):

    def test_the_setters_can_be_used_from_another_thread_while_pictures_come_in(self) -> None:
        handler = self.make_handler(self.camera)
        errors: list[BaseException] = []
        stop = threading.Event()
        heard: list[int] = []
        pixel = self.pixel(3.0, 20.0)

        def use_the_handler() -> None:
            try:
                count = 0
                while not stop.is_set():
                    handler.set_selected(count % 2)
                    handler.set_message('message {}'.format(count) if count % 3 else None)
                    handler.set_click_listener(heard.append if count % 2 else None)
                    handler.block_at(*pixel)
                    count += 1
            except BaseException as error:  # the test reports whatever it was
                errors.append(error)

        thread = threading.Thread(target=use_the_handler)
        thread.start()
        try:
            for _ in range(4):
                frame = feed(handler, self.scenes['far'])
                handler.handle_click(*pixel)
                self.assertEqual((HEIGHT, WIDTH, 3), frame.shape)
        finally:
            stop.set()
            thread.join(10.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual([], errors)
        self.assertEqual(4, handler.frames_with_board())
        self.assertEqual([0, 1], [block.marker_id for block in handler.get_blocks()])


if __name__ == '__main__':
    unittest.main()
