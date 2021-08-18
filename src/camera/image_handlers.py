import threading
from abc import ABC, abstractmethod

import cv2
import numpy as np
from cv2 import aruco

from src.camera.util import CaptureConfig, get_default_charuco_board, get_calibrations, aruco_marker_dictionary, \
    charuco_board_dictionary, aruco_marker_length, charuco_base_board_square_length, charuco__baseboard_marker_length, \
    find_relative_vector_and_rotation, charuco_board_dictionary_2
from src.utils.decorators import synchronized_with_lock


class ImageHandler(ABC):
    @abstractmethod
    def handle_frame(self, frame, gray):
        pass

    @abstractmethod
    def deactivate_handler(self):
        pass


class ImageRecorder(ImageHandler):

    def __init__(self, filename):
        self.out = cv2.VideoWriter(filename, CaptureConfig.image_format,
                                   CaptureConfig.fps, (CaptureConfig.screen_width, CaptureConfig.screen_height))

    def handle_frame(self, frame, gray):
        self.out.write(frame)

    def deactivate_handler(self):
        self.out.release()


class CrossDrawer(ImageHandler):
    half__height = int(CaptureConfig.screen_height / 2)
    half_width = int(CaptureConfig.screen_width / 2)
    marker_x_left = int(half_width - CaptureConfig.marker_size)
    marker_x_right = int(half_width + CaptureConfig.marker_size)
    marker_y_low = int(half__height - 20)
    marker_y_high = int(half__height + 20)

    def __init__(self):
        pass

    def handle_frame(self, frame, gray):
        cv2.line(frame, (self.marker_x_left, self.half__height), (self.marker_x_right, self.half__height), (0, 255, 0))
        cv2.line(frame, (self.half_width, self.marker_y_low), (self.half_width, self.marker_y_high), (0, 255, 0))

    def deactivate_handler(self):
        pass


class DetectedMarker:
    def __init__(self, id, relative_rotation_matrix, tvec):
        self.id = id
        self.relative_rotation_matrix = relative_rotation_matrix
        self.tvec = tvec

    def copy(self):
        res = DetectedMarker(self.id, self.relative_rotation_matrix, self.tvec)
        return res


def get_default_aurco_image_handler():
    board = get_default_charuco_board(square_length=charuco_base_board_square_length,
                                      marker_length=charuco__baseboard_marker_length)
    cameraMatrix, distCoeffs = get_calibrations('src/camera/calibration/calibration_data_charuco.json')
    handler = ArucoImageHandler(board, cameraMatrix, distCoeffs, aruco_marker_dictionary,
                                charuco_board_dictionary, aruco_marker_length, should_draw=True)
    return handler


class ArucoImageHandler(ImageHandler):

    def __init__(self, board, cameraMatrix, distCoeffs, aruco_dictionary, charuco_board_dictionary, aruco_marker_length,
                 should_draw=False):
        self.board = board
        self.lock = threading.RLock()
        self.parameters = aruco.DetectorParameters_create()
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs
        self.aruco_dictionary = aruco_dictionary
        self.charuco_board_dictionary = charuco_board_dictionary
        self.aruco_marker_length = aruco_marker_length
        self.should_draw = should_draw
        self.detected_markers = []
        self._previous_board_rvec = None
        self._previous_board_tvec = None

    def handle_frame(self, frame, gray):
        retval, board_rvec, board_tvec = self.detect_board(gray, self.parameters)
        if self.board_not_detected(board_rvec, board_tvec, retval):
            # did not detect the board and no board was detected in the past
            return

        ids, marker_rvecs, marker_tvecs, corners = self.detect_markers(gray, frame, self.parameters)
        if ids is None:
            return
        relative_tvecs, relative_rotation_matrices = self.find_relative_vectors_of_markers_with_respect_to_board(
            board_rvec,
            board_tvec, ids,
            marker_rvecs,
            marker_tvecs)

        self.populate_detected_makers(ids, relative_rotation_matrices, relative_tvecs)

        if self.should_draw_markers():
            aruco.drawAxis(frame, self.cameraMatrix, self.distCoeffs, board_rvec, board_tvec, length=50)
            # self.draw_marker_axis(frame, marker_rvecs, marker_tvecs, ids)
            aruco.drawDetectedMarkers(frame, corners, ids)

    @synchronized_with_lock("lock")
    def should_draw_markers(self):
        return self.should_draw

    @synchronized_with_lock("lock")
    def disable_draw(self):
        self.should_draw = False

    @synchronized_with_lock("lock")
    def enable_draw(self):
        self.should_draw = True

    def draw_marker_axis(self, frame, rvec, tvecs, marker_ids):
        for i in range(len(marker_ids)):
            aruco.drawAxis(frame, self.cameraMatrix, self.distCoeffs, rvec[i], tvecs[i], length=50)

    @staticmethod
    def board_not_detected(board_rvec, board_tvec, retval):
        if not retval:
            return True
        if board_rvec is None or board_tvec is None:
            return True
        if len(board_rvec) == 3 and len(board_tvec) == 3:
            return False

    def deactivate_handler(self):
        pass

    def get_previous_vecs_if_exist(self):
        if self._previous_board_rvec is not None and self._previous_board_tvec is not None:
            return True, self._previous_board_rvec, self._previous_board_tvec
        else:
            return False, None, None

    def detect_board(self, gray_image, detection_parameters):
        corners, ids, rejected_img_points = aruco.detectMarkers(gray_image, self.charuco_board_dictionary,
                                                                parameters=detection_parameters)
        aruco.refineDetectedMarkers(gray_image, self.board, corners, ids, rejected_img_points)

        if ids is None:
            return self.get_previous_vecs_if_exist()

        charucoretval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(corners, ids, gray_image,
                                                                                    self.board)

        empty_array = np.array([])
        retval, rvec, tvec = aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, self.board, self.cameraMatrix,
                                                            self.distCoeffs, empty_array, empty_array,
                                                            useExtrinsicGuess=False)
        if self.board_not_detected(rvec, tvec, retval):
            return self.get_previous_vecs_if_exist()

        self._previous_board_rvec, self._previous_board_tvec = rvec, tvec
        return retval, rvec, tvec

    def detect_markers(self, gray_image, captured_frame, detection_parameters):
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_image, self.aruco_dictionary,
                                                              parameters=detection_parameters)
        aruco.refineDetectedMarkers(gray_image, self.board, corners, ids, rejectedImgPoints)

        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, self.aruco_marker_length, self.cameraMatrix,
                                                          self.distCoeffs)

        return ids, rvecs, tvecs, corners

    def find_relative_vectors_of_markers_with_respect_to_board(self, board_rvec, board_tvec, marker_ids, markers_rvecs,
                                                               markers_tvecs):
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
    def populate_detected_makers(self, ids, relative_rotation_matrices, tvecs):
        if ids is None:
            return
        self.detected_markers = []
        for id, relative_rotation_matrix, tvec in zip(ids, relative_rotation_matrices, tvecs):
            self.detected_markers.append(DetectedMarker(id, relative_rotation_matrix, tvec))

    @synchronized_with_lock("lock")
    def get_detected_markers(self):
        return [detected_marker.copy() for detected_marker in self.detected_markers]


def get_default_board_to_board_image_handler():
    base_board = get_default_charuco_board(square_length=charuco_base_board_square_length,
                                      marker_length=charuco__baseboard_marker_length)
    target_board = get_default_charuco_board(square_length=charuco_base_board_square_length,
                                      marker_length=charuco__baseboard_marker_length,
                                             dictionary=charuco_board_dictionary_2)

    cameraMatrix, distCoeffs = get_calibrations('src/camera/calibration/calibration_data.json')
    handler = BoardToBoardImageHandler(base_board, target_board, cameraMatrix, distCoeffs, charuco_board_dictionary,
                                       charuco_board_dictionary_2, should_draw=False)
    return handler


class BoardToBoardImageHandler(ImageHandler):

    def __init__(self, board, target_board, cameraMatrix, distCoeffs, charuco_board_dictionary, target_board_dictionary,
                 should_draw=False):
        self.board = board
        self.target_board = target_board
        self.lock = threading.RLock()
        self.parameters = aruco.DetectorParameters_create()
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs

        self.charuco_board_dictionary = charuco_board_dictionary
        self.target_board_dictionary = target_board_dictionary

        self.should_draw = should_draw

        self.relative_matrix = None
        self.relative_tvec = None

        self._previous_board_rvec = None
        self._previous_board_tvec = None

    def handle_frame(self, frame, gray):
        retval, board_rvec, board_tvec = self.detect_board(gray, self.board, self.charuco_board_dictionary)
        if not retval:
            retval, board_rvec, board_tvec = self.get_previous_vecs_if_exist()
        if self.board_not_detected(board_rvec, board_tvec, retval):
            # did not detect the board and no board was detected in the past
            return

        retval, target_board_rvec, target_board_tvec = self.detect_board(gray, self.target_board,
                                                                    self.target_board_dictionary)
        if self.board_not_detected(target_board_rvec, target_board_tvec, retval):
            # did not detect the board
            return

        relative_rotation_matrix, relative_tvec = self.find_relative_matrix_and_vector(board_rvec, board_tvec,
                                                                                       target_board_rvec,
                                                                                       target_board_tvec)
        self.populate_relative_vecs(relative_rotation_matrix, relative_tvec)

        if self.should_draw:
            aruco.drawAxis(frame, self.cameraMatrix, self.distCoeffs, board_rvec, board_tvec, length=50)
            aruco.drawAxis(frame, self.cameraMatrix, self.distCoeffs, target_board_rvec, target_board_tvec, length=50)

    @staticmethod
    def board_not_detected(board_rvec, board_tvec, retval):
        if not retval:
            return True
        if board_rvec is None or board_tvec is None:
            return True
        if len(board_rvec) == 3 and len(board_tvec) == 3:
            return False

    def deactivate_handler(self):
        pass

    def get_previous_vecs_if_exist(self):
        if self._previous_board_rvec is not None and self._previous_board_tvec is not None:
            return True, self._previous_board_rvec, self._previous_board_tvec
        else:
            return False, None, None

    def detect_board(self, gray_image, board, board_dictionary):
        corners, ids, rejected_img_points = aruco.detectMarkers(gray_image, board_dictionary,
                                                                parameters=self.parameters)
        aruco.refineDetectedMarkers(gray_image, self.board, corners, ids, rejected_img_points)

        if ids is None:
            return False, None, None

        charucoretval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(corners, ids, gray_image,
                                                                                    self.board)

        empty_array = np.array([])
        retval, rvec, tvec = aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, self.cameraMatrix,
                                                            self.distCoeffs, empty_array, empty_array,
                                                            useExtrinsicGuess=False)
        if self.board_not_detected(rvec, tvec, retval):
            return False, None, None

        self._previous_board_rvec, self._previous_board_tvec = rvec, tvec
        return retval, rvec, tvec

    @staticmethod
    def find_relative_matrix_and_vector(board_rvec, board_tvec, target_board_rvec, target_board_tvec):
        board_rotation_matrix, _ = cv2.Rodrigues(board_rvec)
        target_board_rotation_matrix, _ = cv2.Rodrigues(target_board_rvec)
        relative_rotation_matrix, relative_tvec = find_relative_vector_and_rotation(board_rotation_matrix,
                                                                                    board_tvec,
                                                                                    target_board_rotation_matrix,
                                                                                    target_board_tvec)
        return relative_rotation_matrix, relative_tvec

    @synchronized_with_lock("lock")
    def populate_relative_vecs(self, relative_matrix, tvec):
        self.relative_matrix = relative_matrix
        self.relative_tvec = tvec

    @synchronized_with_lock("lock")
    def get_revlative_vecs(self):
        return self.relative_matrix, self.relative_tvec
