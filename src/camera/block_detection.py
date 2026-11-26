"""
Finds the blocks on the table in the robot frame, with a charuco board on the table and an aruco marker on every block.

Frames, all lengths in cm:

* Board frame (ours): the origin is the corner of the board that is bottom right in the picture that board_image
  makes, x goes to the right in that picture (along squares_x, so away from the board: the board lies at x from -width
  to 0), y goes up in that picture along the right edge of the board (along squares_y, the board lies at y from 0 to
  height) and z comes out of the paper towards the viewer. It is right handed, and z points up from the table. The
  print sheet draws the origin and the two arrows from this definition. The board lies to the left of the robot, and
  the corner that is nearest the base is the easiest to measure: the position of the board is that of this corner.
* OpenCV 4.9 puts the origin of its board at the top left corner of the same picture, with y pointing down in the
  picture and z into the paper (found by test: marker 0 of the board is in the first row of squares, and its object
  points have y between 0 and the square length). A point of ours (x, y, z) is the point (x + width, height - y, -z)
  of OpenCV's frame, where width is squares_x * square_length and height is squares_y * square_length: a turn of
  180 degrees around the x axis and a shift.
* Robot frame: the origin is on the table under the robot, z is up. A point of the board frame is in the robot frame
  after a rotation around z by board.rotation degrees (counter clockwise seen from above), and a shift by
  (board.x, board.y, 0).

How a block is located: the pose of the single marker on a block is not used for its position, since a marker of
2.4 cm measures its own distance badly. The pose of the board comes from its many charuco corners and tells where the
camera is relative to the table. The four corners of a block marker are lines of sight from the camera, and the marker
lies in a plane that is known: horizontal, at the height of the top of the block. The corners are where the lines of
sight meet that plane.
"""
import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import cv2
import numpy as np

from src.block_stacking.stacking_config import BoardConfig, StackingConfig, aruco_dictionary
from src.camera.image_handlers import ImageHandler
from src.camera.util import detect_aruco_markers, interpolate_charuco_corners, estimate_pose_charuco_board, \
    get_calibrations, get_default_calibration_path

# a pose from fewer charuco corners is too poorly conditioned to place a block with
MIN_CHARUCO_CORNERS = 6
# the seconds between two looks at the number of pictures while a scan waits
_POLL_INTERVAL = 0.05
_TOWER_RING_POINTS = 48

logger = logging.getLogger(__name__)

_COLOR_BOARD_MISSING = (0, 0, 255)
_COLOR_BLOCK = (0, 255, 0)
_COLOR_SELECTED = (255, 0, 255)
_COLOR_TOWER = (0, 255, 255)
_THICKNESS_BLOCK = 2
_THICKNESS_SELECTED = 5
# the message is dark text on a light bar, so that it can be read on any picture
_COLOR_MESSAGE_BAR = (235, 235, 235)
_COLOR_MESSAGE_TEXT = (20, 20, 20)
_MESSAGE_SCALE = 0.8
_MESSAGE_THICKNESS = 2
_MESSAGE_PADDING = 8
_MESSAGE_ELLIPSIS = '...'
# a click just next to a marker counts for it when it is within this many diagonals of the marker, see block_at
_CLICK_REACH_DIAGONALS = 1.0


class ScanError(Exception):
    """the scan could not be made, the message says why"""


def make_board(board_config: BoardConfig) -> cv2.aruco.CharucoBoard:
    """the charuco board that is on the table"""
    return cv2.aruco.CharucoBoard((board_config.squares_x, board_config.squares_y), board_config.square_length,
                                  board_config.marker_length, aruco_dictionary(board_config.dictionary))


def board_image(board_config: BoardConfig, pixels_per_cm: int) -> np.ndarray:
    """
    the picture of the board without a margin, to print it at pixels_per_cm. Pixel (u, v) is the point
    (u / pixels_per_cm - width, height - v / pixels_per_cm, 0) of the board frame, where width and height are those
    of the picture in cm: the origin is at the bottom right of the picture.
    :raises ValueError: if the board is not a whole number of pixels wide and high at that resolution
    """
    width = _whole_pixels(board_config.squares_x * board_config.square_length * pixels_per_cm)
    height = _whole_pixels(board_config.squares_y * board_config.square_length * pixels_per_cm)
    image: np.ndarray = make_board(board_config).generateImage((width, height), marginSize=0, borderBits=1)
    return image


def _whole_pixels(pixels: float) -> int:
    if pixels < 1 or abs(pixels - round(pixels)) > 1e-6:
        raise ValueError('the board is {} pixels wide or high, which is not a whole number of pixels: choose pixels '
                         'per cm so that the squares have a whole number of pixels'.format(pixels))
    return round(pixels)


def _rotation_z(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def _as_points(points: np.ndarray | list[float] | tuple[float, ...]) -> tuple[np.ndarray, int]:
    """the points as floats with the 3 coordinates as the last axis, and the number of coordinates they came with"""
    array = np.asarray(points, dtype=np.float64)
    dimensions = array.shape[-1] if array.ndim else 0
    if dimensions not in (2, 3):
        raise ValueError('points should have 2 or 3 coordinates, not shape {}'.format(array.shape))
    if dimensions == 2:
        array = np.concatenate([array, np.zeros(array.shape[:-1] + (1,))], axis=-1)
    return array, dimensions


def board_to_robot(points: np.ndarray | list[float] | tuple[float, ...], board_config: BoardConfig) -> np.ndarray:
    """points of the board frame in the robot frame, the last axis of points has x, y and optionally z"""
    array, dimensions = _as_points(points)
    result = array @ _rotation_z(board_config.rotation).T + np.array([board_config.x, board_config.y, 0.0])
    return result[..., :dimensions]


def robot_to_board(points: np.ndarray | list[float] | tuple[float, ...], board_config: BoardConfig) -> np.ndarray:
    """points of the robot frame in the board frame, the inverse of board_to_robot"""
    array, dimensions = _as_points(points)
    result = (array - np.array([board_config.x, board_config.y, 0.0])) @ _rotation_z(board_config.rotation)
    return result[..., :dimensions]


def _origin_in_opencv_board(board_config: BoardConfig) -> np.ndarray:
    """the origin of our board frame in the frame OpenCV uses for the board: the bottom right of the picture"""
    return np.array([board_config.squares_x * board_config.square_length,
                     board_config.squares_y * board_config.square_length, 0.0])


# OpenCV's board frame is ours turned 180 degrees around the x axis, see the docstring of the module
_FLIP = np.diag([1.0, -1.0, -1.0])


def _robot_to_opencv_board(points: np.ndarray, board_config: BoardConfig) -> np.ndarray:
    """points of shape (n, 3) of the robot frame in the frame OpenCV uses for the board"""
    board_points = robot_to_board(points, board_config)
    return np.asarray(board_points @ _FLIP.T + _origin_in_opencv_board(board_config))


def _fold_yaw(yaw: float) -> float:
    """into [-pi/4, pi/4): a square block looks the same every 90 degrees"""
    return (yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4


@dataclass(frozen=True, eq=False)  # the fields are arrays, which can not be compared with ==
class BlockObservation:
    """one marker in one picture"""
    marker_id: int
    origin: np.ndarray  # camera centre in the robot frame, shape (3,)
    directions: np.ndarray  # lines of sight of the four corners in the robot frame, shape (4, 3), detector order

    def at_height(self, z: float) -> tuple[float, float, float, float]:
        """
        x, y of the centre of the marker, its yaw in radians and its mean side length in cm, when it lies in the
        plane at height z
        :raises ValueError: if a line of sight does not reach that plane in front of the camera
        """
        vertical = self.directions[:, 2]
        if np.any(vertical >= -1e-9):
            raise ValueError('a line of sight does not go down')
        distances = (z - self.origin[2]) / vertical
        if np.any(distances <= 0):
            raise ValueError('the plane at height {} is not in front of the camera'.format(z))
        corners = self.origin + distances[:, np.newaxis] * self.directions
        edges = np.roll(corners, -1, axis=0) - corners  # the corners go round the marker, clockwise from above
        x, y, _ = corners.mean(axis=0)
        # the edge i is turned by i quarter turns: 4 times its angle is the same for all four edges
        edge_angles = np.arctan2(edges[:, 1], edges[:, 0])
        quadruple = np.exp(4j * edge_angles).sum()
        yaw = _fold_yaw(math.atan2(quadruple.imag, quadruple.real) / 4)
        return float(x), float(y), yaw, float(np.linalg.norm(edges, axis=1).mean())


@dataclass(frozen=True)
class DetectedBlock:
    """a block over several pictures"""
    marker_id: int
    x: float
    y: float
    yaw: float  # radians, of the edges of the block around z, folded into [-pi/4, pi/4)
    height: float  # the z of the marker that was assumed
    frames: int  # in how many pictures it was seen
    spread: float  # cm, the largest distance of a single picture from the median
    side_length: float  # cm, median measured side of the marker at that height: far from marker_length = wrong height
    stable: bool  # spread <= scan.max_spread


@dataclass(frozen=True)
class _BoardPose:
    """where the camera is, from the board in one picture"""
    rvec: np.ndarray  # OpenCV's board frame in the camera frame
    tvec: np.ndarray
    origin: np.ndarray  # camera centre in the robot frame
    camera_to_robot: np.ndarray  # rotation of directions from the camera frame to the robot frame


@dataclass(frozen=True)
class _Detection:
    """what one picture showed"""
    block_corners: list[np.ndarray]  # as detected, shape (1, 4, 2)
    block_ids: list[int]
    pose: _BoardPose | None  # None when the board was not found
    observations: list[BlockObservation]


class BlockImageHandler(ImageHandler):
    """Finds the blocks in every picture of the camera and remembers the last observations of each of them."""

    def __init__(self, config: StackingConfig, camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                 clock: Callable[[], float] = time.monotonic, history: int = 30) -> None:
        if history < 1:
            raise ValueError('history should be at least 1, not {}'.format(history))
        self._config = config
        self._camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        self._dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)
        self._clock = clock
        self._history = history
        self._board = make_board(config.board)
        self._board_dictionary = aruco_dictionary(config.board.dictionary)
        self._block_dictionary = aruco_dictionary(config.blocks.dictionary)
        self._board_marker_ids = {int(marker_id) for marker_id in np.asarray(self._board.getIds()).ravel()}
        self._block_marker_ids = set(config.blocks.marker_ids)
        self._parameters = cv2.aruco.DetectorParameters()
        # the corners of the small markers have to be placed to a fraction of a pixel
        self._parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._lock = threading.RLock()
        self._observations: dict[int, deque[BlockObservation]] = {}
        self._frames_with_board = 0
        self._last_board_time: float | None = None
        # the outlines of the block markers of the last picture, as (marker id, corners of shape (4, 2)), in pixels.
        # The list is replaced as a whole and never changed, so a reader can use it after it let go of the lock.
        self._outlines: list[tuple[int, np.ndarray]] = []
        self._selected: int | None = None
        self._message: str | None = None
        self._click_listener: Callable[[int], object] | None = None

    def handle_frame(self, frame: np.ndarray, gray: np.ndarray) -> None:
        """detect the blocks in gray, remember them and draw what was found on frame, never raises"""
        detection: _Detection | None = None
        outlines: list[tuple[int, np.ndarray]] = []
        try:
            found = self._detect(gray)
            outlines = [(marker_id, corners.reshape(4, 2).astype(np.float64))
                        for corners, marker_id in zip(found.block_corners, found.block_ids)]
            detection = found
        except (cv2.error, ValueError, TypeError, np.linalg.LinAlgError):
            # a bad picture must not stop the camera thread, the next one may be fine
            pass
        with self._lock:
            self._outlines = outlines
            selected, message = self._selected, self._message
            if detection is not None and detection.pose is not None:
                self._frames_with_board += 1
                self._last_board_time = self._clock()
                for observation in detection.observations:
                    self._observations.setdefault(observation.marker_id, deque(maxlen=self._history)).append(
                        observation)
        try:
            self._draw(frame, detection, selected)
        except (cv2.error, ValueError, TypeError, IndexError):
            pass
        try:
            _draw_message(frame, message)
        except (cv2.error, ValueError, TypeError, IndexError):
            pass

    def handle_click(self, x: int, y: int) -> None:
        """tell the click listener which block was clicked, if one was, and do nothing otherwise"""
        marker_id = self.block_at(x, y)
        if marker_id is None:
            return
        with self._lock:
            listener = self._click_listener
        if listener is None:
            return
        # outside the lock: the listener may call back into this handler, from this thread or from one it starts
        try:
            listener(marker_id)
        except Exception:
            logger.exception('the listener of the click on block %d failed', marker_id)

    def block_at(self, x: float, y: float) -> int | None:
        """
        the id of the block marker at pixel x, y of the last picture, None if there is none. That is the marker whose
        outline contains the point, and otherwise the nearest marker whose centre is within one diagonal of that
        marker from the point: the markers are small in the picture, and a click just next to one is meant for it.
        """
        with self._lock:
            outlines = self._outlines
        point = (float(x), float(y))
        candidates: list[tuple[int, float, int]] = []  # (0 inside or 1 near, distance to the centre, id)
        for marker_id, corners in outlines:
            centre = corners.mean(axis=0)
            distance = float(np.hypot(centre[0] - point[0], centre[1] - point[1]))
            if cv2.pointPolygonTest(corners.astype(np.float32).reshape(-1, 1, 2), point, False) >= 0:
                candidates.append((0, distance, marker_id))
                continue
            diagonal = float(np.mean([np.linalg.norm(corners[0] - corners[2]),
                                      np.linalg.norm(corners[1] - corners[3])]))
            if distance <= _CLICK_REACH_DIAGONALS * diagonal:
                candidates.append((1, distance, marker_id))
        # a point inside an outline beats a point that is only near another marker, then the nearest centre wins
        return min(candidates)[2] if candidates else None

    def set_click_listener(self, listener: Callable[[int], object] | None) -> None:
        """call listener with the id of the block that is clicked on in the picture, None stops calling"""
        with self._lock:
            self._click_listener = listener

    def set_selected(self, marker_id: int | None) -> None:
        """draw the outline of this block thick and in another colour until it is cleared with None"""
        with self._lock:
            self._selected = marker_id

    def set_message(self, text: str | None) -> None:
        """show a line of text at the top of the picture, None removes it"""
        with self._lock:
            self._message = text

    def deactivate_handler(self) -> None:
        pass

    def clear(self) -> None:
        """forget every observation"""
        with self._lock:
            self._observations.clear()
            self._frames_with_board = 0

    def frames_with_board(self) -> int:
        """the number of pictures in which the board was found since the last clear"""
        with self._lock:
            return self._frames_with_board

    def board_visible(self, max_age: float = 1.0) -> bool:
        """whether the board was found in a picture of the last max_age seconds"""
        with self._lock:
            return self._last_board_time is not None and self._clock() - self._last_board_time <= max_age

    def get_blocks(self, heights: Mapping[int, float] | None = None) -> list[DetectedBlock]:
        """
        the blocks of the remembered observations, sorted by marker id. A block is assumed to have its marker at
        heights[marker_id], or on top of a block on the table (blocks.size) when that is not given.
        """
        with self._lock:
            return self._blocks_locked(heights)

    def scan(self, heights: Mapping[int, float] | None = None,
             sleep: Callable[[float], None] = time.sleep) -> list[DetectedBlock]:
        """
        forget what was seen, wait for scan.frames pictures with the board and return the blocks that are in at least
        half of them
        :raises ScanError: if no picture with the board came within scan.timeout seconds
        """
        scan_config = self._config.scan
        self.clear()
        started = self._clock()
        while self.frames_with_board() < scan_config.frames and self._clock() - started < scan_config.timeout:
            sleep(_POLL_INTERVAL)
        with self._lock:
            pictures = min(self._frames_with_board, self._history)
            if pictures == 0:
                raise ScanError('the board is not in the picture, or the camera gives no pictures: no picture with '
                                'the board came in {} seconds'.format(scan_config.timeout))
            return [block for block in self._blocks_locked(heights) if 2 * block.frames >= pictures]

    def _blocks_locked(self, heights: Mapping[int, float] | None) -> list[DetectedBlock]:
        blocks = []
        for marker_id in sorted(self._observations):
            height = self._config.blocks.size if heights is None else heights.get(marker_id, self._config.blocks.size)
            block = self._block(marker_id, height, list(self._observations[marker_id]))
            if block is not None:
                blocks.append(block)
        return blocks

    def _block(self, marker_id: int, height: float, observations: list[BlockObservation]) -> DetectedBlock | None:
        placed = []
        for observation in observations:
            try:
                placed.append(observation.at_height(height))
            except ValueError:
                continue  # that picture does not reach this height, the others may
        if not placed:
            return None
        values = np.array(placed)
        x, y = (float(np.median(values[:, column])) for column in (0, 1))
        distances = np.hypot(values[:, 0] - x, values[:, 1] - y)
        # the yaws are folded, so the mean has to be of the angles on the circle, where 4 yaws are a whole turn
        quadruple = np.exp(4j * values[:, 2]).sum()
        yaw = _fold_yaw(math.atan2(quadruple.imag, quadruple.real) / 4)
        spread = float(distances.max())
        return DetectedBlock(marker_id=marker_id, x=x, y=y, yaw=yaw, height=height, frames=len(placed), spread=spread,
                             side_length=float(np.median(values[:, 3])), stable=spread <= self._config.scan.max_spread)

    def _detect(self, gray: np.ndarray) -> _Detection:
        board_corners, board_ids = detect_aruco_markers(gray, self._board_dictionary, self._parameters)
        if self._config.board.dictionary == self._config.blocks.dictionary:
            block_corners, block_ids = board_corners, board_ids
        else:
            block_corners, block_ids = detect_aruco_markers(gray, self._block_dictionary, self._parameters)
        blocks = _select(block_corners, block_ids, self._block_marker_ids)
        pose = self._board_pose(gray, *_select(board_corners, board_ids, self._board_marker_ids))
        observations: list[BlockObservation] = []
        if pose is not None:
            counts = {marker_id: blocks[1].count(marker_id) for marker_id in blocks[1]}
            for corners, marker_id in zip(*blocks):
                # the same id twice in a picture is a misread, which of the two is right can not be told
                if counts[marker_id] == 1:
                    observations.append(self._observation(marker_id, corners, pose))
        return _Detection(blocks[0], blocks[1], pose, observations)

    def _board_pose(self, gray: np.ndarray, corners: list[np.ndarray], ids: list[int]) -> _BoardPose | None:
        if not ids:
            return None
        charuco_corners, charuco_ids = interpolate_charuco_corners(
            self._board, gray, corners, np.array(ids, dtype=np.int32).reshape(-1, 1))
        if charuco_corners is None or charuco_ids is None or len(charuco_ids) < MIN_CHARUCO_CORNERS:
            return None
        found, rvec, tvec = estimate_pose_charuco_board(charuco_corners, charuco_ids, self._board,
                                                        self._camera_matrix, self._dist_coeffs)
        if not found or rvec is None or tvec is None:
            return None
        rotation, _ = cv2.Rodrigues(rvec)
        # the camera centre is the origin of the camera frame, and directions only turn
        centre_opencv = -rotation.T @ tvec.reshape(3)
        centre_board = _FLIP @ (centre_opencv - _origin_in_opencv_board(self._config.board))
        origin = board_to_robot(centre_board, self._config.board)
        camera_to_robot = _rotation_z(self._config.board.rotation) @ _FLIP @ rotation.T
        if not np.all(np.isfinite(origin)) or origin[2] <= 0:
            return None  # the camera under the table: the pose flipped, do not use it
        return _BoardPose(rvec, tvec, origin, camera_to_robot)

    def _observation(self, marker_id: int, corners: np.ndarray, pose: _BoardPose) -> BlockObservation:
        normalized = cv2.undistortPoints(corners.reshape(-1, 1, 2).astype(np.float64), self._camera_matrix,
                                         self._dist_coeffs).reshape(4, 2)
        camera_directions = np.column_stack([normalized, np.ones(4)])
        directions = camera_directions @ pose.camera_to_robot.T
        directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]
        return BlockObservation(marker_id, pose.origin, directions)

    def _draw(self, frame: np.ndarray, detection: _Detection | None, selected: int | None = None) -> None:
        """draw on the picture that is shown, detection is done on the gray picture and does not see this"""
        if detection is None or detection.pose is None:
            cv2.putText(frame, 'BOARD NOT FOUND', (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, _COLOR_BOARD_MISSING, 4)
        if detection is None:
            return
        for corners, marker_id in zip(detection.block_corners, detection.block_ids):
            points = np.round(corners.reshape(4, 2)).astype(np.int32)
            is_selected = marker_id == selected
            color = _COLOR_SELECTED if is_selected else _COLOR_BLOCK
            cv2.polylines(frame, [points], True, color, _THICKNESS_SELECTED if is_selected else _THICKNESS_BLOCK)
            cv2.putText(frame, str(marker_id), (int(points[0][0]), int(points[0][1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        pose = detection.pose
        if pose is None:
            return
        board = self._config.board
        # the axes of our board frame: x red, y green, z blue, drawn 5 cm long
        axes_rotation, _ = cv2.Rodrigues(pose.rvec)
        our_rotation = axes_rotation @ _FLIP
        our_translation = pose.tvec.reshape(3) + axes_rotation @ _origin_in_opencv_board(board)
        our_rvec, _ = cv2.Rodrigues(our_rotation)
        cv2.drawFrameAxes(frame, self._camera_matrix, self._dist_coeffs, our_rvec, our_translation, 5)
        for observation in detection.observations:
            x, y, _, _ = observation.at_height(self._config.blocks.size)
            index = detection.block_ids.index(observation.marker_id)
            corner = detection.block_corners[index].reshape(4, 2)[2]
            cv2.putText(frame, '{:.1f}, {:.1f}'.format(x, y), (int(corner[0]), int(corner[1]) + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, _COLOR_BLOCK, 2)
        # the tower: the ring that has to be free of blocks, round its centre on the table
        tower = self._config.tower
        angles = np.linspace(0, 2 * math.pi, _TOWER_RING_POINTS, endpoint=False)
        ring = np.column_stack([tower.x + self._config.tower.clear_radius * np.cos(angles),
                                tower.y + self._config.tower.clear_radius * np.sin(angles), np.zeros_like(angles)])
        centre = np.array([[tower.x, tower.y, 0.0]])
        projected, _ = cv2.projectPoints(_robot_to_opencv_board(np.vstack([ring, centre]), board), pose.rvec,
                                         pose.tvec, self._camera_matrix, self._dist_coeffs)
        pixels = np.asarray(projected, dtype=np.float64).reshape(-1, 2).round().astype(np.int32)
        cv2.polylines(frame, [pixels[:-1]], True, _COLOR_TOWER, 2)
        cv2.circle(frame, (int(pixels[-1][0]), int(pixels[-1][1])), 4, _COLOR_TOWER, -1)


def _draw_message(frame: np.ndarray, message: str | None) -> None:
    """a line of text on a light bar at the top of the picture, cut with an ellipsis where it would not fit"""
    if message is None:
        return
    text = ' '.join(message.split())  # one line, whatever the message holds
    if not text:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    available = frame.shape[1] - 2 * _MESSAGE_PADDING

    def text_width(candidate: str) -> int:
        return int(cv2.getTextSize(candidate, font, _MESSAGE_SCALE, _MESSAGE_THICKNESS)[0][0])

    if text_width(text) > available:
        # the longest start of the text that fits with the ellipsis, the width grows with the length
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if text_width(text[:middle].rstrip() + _MESSAGE_ELLIPSIS) <= available:
                low = middle
            else:
                high = middle - 1
        text = text[:low].rstrip() + _MESSAGE_ELLIPSIS
    (_, text_height), baseline = cv2.getTextSize(text, font, _MESSAGE_SCALE, _MESSAGE_THICKNESS)
    bar_height = text_height + baseline + 2 * _MESSAGE_PADDING
    cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, bar_height), _COLOR_MESSAGE_BAR, -1)
    cv2.putText(frame, text, (_MESSAGE_PADDING, _MESSAGE_PADDING + text_height), font, _MESSAGE_SCALE,
                _COLOR_MESSAGE_TEXT, _MESSAGE_THICKNESS)


def _select(corners: object, ids: np.ndarray | None, wanted: set[int]) -> tuple[list[np.ndarray], list[int]]:
    """the detected markers with an id in wanted, as lists"""
    if ids is None or not isinstance(corners, (list, tuple)):
        return [], []
    selected_corners = []
    selected_ids = []
    for marker_corners, marker_id in zip(corners, ids.ravel()):
        if int(marker_id) in wanted:
            selected_corners.append(np.asarray(marker_corners))
            selected_ids.append(int(marker_id))
    return selected_corners, selected_ids


def get_default_block_image_handler(config: StackingConfig) -> BlockImageHandler:
    """the handler with the calibration of the camera that is in the repository"""
    camera_matrix, dist_coeffs = get_calibrations(get_default_calibration_path())
    return BlockImageHandler(config, camera_matrix, dist_coeffs)
