import threading
from abc import ABC, abstractmethod
from collections.abc import Sequence

import cv2
import numpy as np
from cv2 import aruco

from src.camera.util import CaptureConfig, get_default_charuco_board, get_calibrations, aruco_marker_dictionary, \
    charuco_board_dictionary, aruco_marker_length, charuco_base_board_square_length, charuco__baseboard_marker_length, \
    find_relative_vector_and_rotation, charuco_board_dictionary_2, \
    detect_aruco_markers, interpolate_charuco_corners, estimate_pose_charuco_board, estimate_pose_single_markers
from src.utils.decorators import synchronized_with_lock


class ImageHandler(ABC):
    @abstractmethod
    def handle_frame(self, frame: np.ndarray, gray: np.ndarray) -> None:
        pass

    @abstractmethod
    def deactivate_handler(self) -> None:
        pass


class ImageRecorder(ImageHandler):

    def __init__(self, filename: str) -> None:
        self.out = cv2.VideoWriter(filename, CaptureConfig.image_format,
                                   CaptureConfig.fps, (CaptureConfig.screen_width, CaptureConfig.screen_height))

    def handle_frame(self, frame: np.ndarray, gray: np.ndarray) -> None:
        self.out.write(frame)

    def deactivate_handler(self) -> None:
        self.out.release()


class CrossDrawer(ImageHandler):
    half__height = int(CaptureConfig.screen_height / 2)
    half_width = int(CaptureConfig.screen_width / 2)
    marker_x_left = int(half_width - CaptureConfig.marker_size)
    marker_x_right = int(half_width + CaptureConfig.marker_size)
    marker_y_low = int(half__height - 20)
    marker_y_high = int(half__height + 20)

    def __init__(self) -> None:
        pass

    def handle_frame(self, frame: np.ndarray, gray: np.ndarray) -> None:
        cv2.line(frame, (self.marker_x_left, self.half__height), (self.marker_x_right, self.half__height), (0, 255, 0))
        cv2.line(frame, (self.half_width, self.marker_y_low), (self.half_width, self.marker_y_high), (0, 255, 0))

    def deactivate_handler(self) -> None:
        pass


class DetectedMarker:
    def __init__(self, id: np.ndarray, relative_rotation_matrix: np.ndarray, tvec: np.ndarray) -> None:
        self.id = id
        self.relative_rotation_matrix = relative_rotation_matrix
        self.tvec = tvec

    def copy(self) -> 'DetectedMarker':
        res = DetectedMarker(self.id, self.relative_rotation_matrix, self.tvec)
        return res


def get_default_aurco_image_handler() -> 'ArucoImageHandler':
    board = get_default_charuco_board(square_length=charuco_base_board_square_length,
                                      marker_length=charuco__baseboard_marker_length)
    cameraMatrix, distCoeffs = get_calibrations('src/camera/calibration/calibration_data_charuco.json')
    handler = ArucoImageHandler(board, cameraMatrix, distCoeffs, aruco_marker_dictionary,
                                charuco_board_dictionary, aruco_marker_length, should_draw=True)
    return handler


class ArucoImageHandler(ImageHandler):

    def __init__(self, board: aruco.CharucoBoard, cameraMatrix: np.ndarray, distCoeffs: np.ndarray,
                 aruco_dictionary: aruco.Dictionary, charuco_board_dictionary: aruco.Dictionary,
                 aruco_marker_length: float,
                 should_draw: bool = False) -> None:
        self.board = board
        self.lock = threading.RLock()
        self.parameters: aruco.DetectorParameters = aruco.DetectorParameters()
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs
        self.aruco_dictionary = aruco_dictionary
        self.charuco_board_dictionary = charuco_board_dictionary
        self.aruco_marker_length = aruco_marker_length
        self.should_draw = should_draw
        self.detected_markers: list[DetectedMarker] = []
        self._previous_board_rvec: np.ndarray | None = None
        self._previous_board_tvec: np.ndarray | None = None

    def handle_frame(self, frame: np.ndarray, gray: np.ndarray) -> None:
        retval, board_rvec, board_tvec = self.detect_board(gray, self.parameters)
        if self.board_not_detected(board_rvec, board_tvec, retval):
            # did not detect the board and no board was detected in the past
            return
        assert board_rvec is not None and board_tvec is not None

        ids, marker_rvecs, marker_tvecs, corners = self.detect_markers(gray, frame, self.parameters)
        if ids is None:
            self.clear_detected_markers()
            return

        relative_tvecs, relative_rotation_matrices = self.find_relative_vectors_of_markers_with_respect_to_board(
            board_rvec,
            board_tvec, ids,
            marker_rvecs,
            marker_tvecs)

        self.populate_detected_makers(ids, relative_rotation_matrices, relative_tvecs)

        if self.should_draw_markers():
            cv2.drawFrameAxes(frame, self.cameraMatrix, self.distCoeffs, board_rvec, board_tvec, length=10)
            # self.draw_marker_axis(frame, marker_rvecs, marker_tvecs, ids)
            aruco.drawDetectedMarkers(frame, corners, ids)

    @synchronized_with_lock("lock")
    def should_draw_markers(self) -> bool:
        return self.should_draw

    @synchronized_with_lock("lock")
    def disable_draw(self) -> None:
        self.should_draw = False

    @synchronized_with_lock("lock")
    def enable_draw(self) -> None:
        self.should_draw = True

    def draw_marker_axis(self, frame: np.ndarray, rvec: np.ndarray, tvecs: np.ndarray,
                         marker_ids: np.ndarray) -> None:
        for i in range(len(marker_ids)):
            cv2.drawFrameAxes(frame, self.cameraMatrix, self.distCoeffs, rvec[i], tvecs[i], length=50)

    @staticmethod
    def board_not_detected(board_rvec: np.ndarray | None, board_tvec: np.ndarray | None, retval: bool) -> bool:
        if not retval:
            return True
        if board_rvec is None or board_tvec is None:
            return True
        if len(board_rvec) == 3 and len(board_tvec) == 3:
            return False
        return True

    def deactivate_handler(self) -> None:
        pass

    def get_previous_vecs_if_exist(self) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
        if self._previous_board_rvec is not None and self._previous_board_tvec is not None:
            return True, self._previous_board_rvec, self._previous_board_tvec
        else:
            return False, None, None

    def detect_board(self, gray_image: np.ndarray, detection_parameters: aruco.DetectorParameters
                     ) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
        corners, ids = detect_aruco_markers(gray_image, self.charuco_board_dictionary, detection_parameters)
        if ids is None:
            return self.get_previous_vecs_if_exist()

        charuco_corners, charuco_ids = interpolate_charuco_corners(self.board, gray_image, corners, ids)
        retval, rvec, tvec = estimate_pose_charuco_board(charuco_corners, charuco_ids, self.board, self.cameraMatrix,
                                                         self.distCoeffs)
        if self.board_not_detected(rvec, tvec, retval):
            return self.get_previous_vecs_if_exist()

        self._previous_board_rvec, self._previous_board_tvec = rvec, tvec
        return retval, rvec, tvec

    def detect_markers(self, gray_image: np.ndarray, captured_frame: np.ndarray,
                       detection_parameters: aruco.DetectorParameters
                       ) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, Sequence[np.ndarray]]:
        corners, ids = detect_aruco_markers(gray_image, self.aruco_dictionary, detection_parameters)
        rvecs, tvecs = estimate_pose_single_markers(corners, self.aruco_marker_length, self.cameraMatrix,
                                                    self.distCoeffs)

        return ids, rvecs, tvecs, corners

    def find_relative_vectors_of_markers_with_respect_to_board(self, board_rvec: np.ndarray, board_tvec: np.ndarray,
                                                               marker_ids: np.ndarray, markers_rvecs: np.ndarray,
                                                               markers_tvecs: np.ndarray
                                                               ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        board_rotation_matrix, _ = cv2.Rodrigues(board_rvec)
        relative_tvecs = []
        relative_rotation_matrices = []
        for i in range(len(marker_ids)):
            marker_rotation_matrix, _ = cv2.Rodrigues(markers_rvecs[i])
            relative_rotation_matrix, relative_tvec = find_relative_vector_and_rotation(board_rotation_matrix,
                                                                                        board_tvec,
                                                                                        marker_rotation_matrix,
                                                                                        markers_tvecs[i])
            relative_tvecs.append(relative_tvec)
            relative_rotation_matrices.append(relative_rotation_matrix)

        return relative_tvecs, relative_rotation_matrices

    @synchronized_with_lock("lock")
    def populate_detected_makers(self, ids: np.ndarray | None, relative_rotation_matrices: list[np.ndarray],
                                 tvecs: list[np.ndarray]) -> None:
        if ids is None:
            return
        self.detected_markers = []
        for id, relative_rotation_matrix, tvec in zip(ids, relative_rotation_matrices, tvecs):
            self.detected_markers.append(DetectedMarker(id, relative_rotation_matrix, tvec))

    @synchronized_with_lock("lock")
    def clear_detected_markers(self) -> None:
        self.detected_markers = []

    @synchronized_with_lock("lock")
    def get_detected_markers(self) -> list[DetectedMarker]:
        return [detected_marker.copy() for detected_marker in self.detected_markers]


def get_default_board_to_board_image_handler() -> 'BoardToBoardImageHandler':
    base_board = get_default_charuco_board(square_length=charuco_base_board_square_length,
                                      marker_length=charuco__baseboard_marker_length)
    target_board = get_default_charuco_board(square_length=charuco_base_board_square_length,
                                      marker_length=charuco__baseboard_marker_length,
                                             dictionary=charuco_board_dictionary_2)

    cameraMatrix, distCoeffs = get_calibrations('src/camera/calibration/calibration_data.json')
    handler = BoardToBoardImageHandler(base_board, target_board, cameraMatrix, distCoeffs, charuco_board_dictionary,
                                       charuco_board_dictionary_2, should_draw=True)
    return handler


class BoardToBoardImageHandler(ImageHandler):

    def __init__(self, board: aruco.CharucoBoard, target_board: aruco.CharucoBoard, cameraMatrix: np.ndarray,
                 distCoeffs: np.ndarray, charuco_board_dictionary: aruco.Dictionary,
                 target_board_dictionary: aruco.Dictionary,
                 should_draw: bool = False) -> None:
        self.board = board
        self.target_board = target_board
        self.lock = threading.RLock()
        self.parameters: aruco.DetectorParameters = aruco.DetectorParameters()
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs

        self.charuco_board_dictionary = charuco_board_dictionary
        self.target_board_dictionary = target_board_dictionary

        self.should_draw = should_draw

        self.relative_matrix: np.ndarray | None = None
        self.relative_tvec: np.ndarray | None = None

        self._previous_board_rvec: np.ndarray | None = None
        self._previous_board_tvec: np.ndarray | None = None

    def handle_frame(self, frame: np.ndarray, gray: np.ndarray) -> None:
        retval, board_rvec, board_tvec = self.detect_board(gray, self.board, self.charuco_board_dictionary)
        if not retval:
            retval, board_rvec, board_tvec = self.get_previous_vecs_if_exist()
        if self.board_not_detected(board_rvec, board_tvec, retval):
            # did not detect the board and no board was detected in the past
            return
        assert board_rvec is not None and board_tvec is not None

        retval, target_board_rvec, target_board_tvec = self.detect_board(gray, self.target_board,
                                                                    self.target_board_dictionary)
        if self.board_not_detected(target_board_rvec, target_board_tvec, retval):
            # did not detect the board
            return
        assert target_board_rvec is not None and target_board_tvec is not None

        relative_rotation_matrix, relative_tvec = self.find_relative_matrix_and_vector(board_rvec, board_tvec,
                                                                                       target_board_rvec,
                                                                                       target_board_tvec)
        self.populate_relative_vecs(relative_rotation_matrix, relative_tvec)

        if self.should_draw:
            cv2.drawFrameAxes(frame, self.cameraMatrix, self.distCoeffs, board_rvec, board_tvec, length=5)
            cv2.drawFrameAxes(frame, self.cameraMatrix, self.distCoeffs, target_board_rvec, target_board_tvec, length=5)

    @staticmethod
    def board_not_detected(board_rvec: np.ndarray | None, board_tvec: np.ndarray | None, retval: bool) -> bool:
        if not retval:
            return True
        if board_rvec is None or board_tvec is None:
            return True
        if len(board_rvec) == 3 and len(board_tvec) == 3:
            return False
        return True

    def deactivate_handler(self) -> None:
        pass

    def get_previous_vecs_if_exist(self) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
        if self._previous_board_rvec is not None and self._previous_board_tvec is not None:
            return True, self._previous_board_rvec, self._previous_board_tvec
        else:
            return False, None, None

    def detect_board(self, gray_image: np.ndarray, board: aruco.CharucoBoard, board_dictionary: aruco.Dictionary
                     ) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
        corners, ids = detect_aruco_markers(gray_image, board_dictionary, self.parameters)
        if ids is None:
            return False, None, None

        charuco_corners, charuco_ids = interpolate_charuco_corners(board, gray_image, corners, ids)
        retval, rvec, tvec = estimate_pose_charuco_board(charuco_corners, charuco_ids, board, self.cameraMatrix,
                                                         self.distCoeffs)
        if self.board_not_detected(rvec, tvec, retval):
            return False, None, None

        if board is self.board:
            self._previous_board_rvec, self._previous_board_tvec = rvec, tvec
        return retval, rvec, tvec

    @staticmethod
    def find_relative_matrix_and_vector(board_rvec: np.ndarray, board_tvec: np.ndarray, target_board_rvec: np.ndarray,
                                        target_board_tvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        board_rotation_matrix, _ = cv2.Rodrigues(board_rvec)
        target_board_rotation_matrix, _ = cv2.Rodrigues(target_board_rvec)
        relative_rotation_matrix, relative_tvec = find_relative_vector_and_rotation(board_rotation_matrix,
                                                                                    board_tvec,
                                                                                    target_board_rotation_matrix,
                                                                                    target_board_tvec)
        return relative_rotation_matrix, relative_tvec

    @synchronized_with_lock("lock")
    def populate_relative_vecs(self, relative_matrix: np.ndarray, tvec: np.ndarray) -> None:
        self.relative_matrix = relative_matrix
        self.relative_tvec = tvec

    @synchronized_with_lock("lock")
    def get_revlative_vecs(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self.relative_matrix, self.relative_tvec
