import threading
from abc import ABC, abstractmethod

import cv2
import numpy as np
from cv2 import aruco

from src.camera.util import CaptureConfig, get_default_charuco_board, get_calibrations, aruco_marker_dictionary, \
    charuco_board_dictionary, aruco_marker_length, charuco_base_board_square_length, charuco__baseboard_marker_length
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
    def __init__(self, id, rvec, tvec):
        self.id = id
        self.rvec = rvec
        self.tvec = tvec

    def copy(self):
        res = DetectedMarker(self.id, self.rvec, self.tvec)
        return res


def get_default_aurco_image_handler():

    board = get_default_charuco_board(square_length=charuco_base_board_square_length,
                                      marker_length=charuco__baseboard_marker_length)
    cameraMatrix, distCoeffs = get_calibrations('src/camera/calibration/calibration_data.json')
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

    def handle_frame(self, frame, gray):
        retval, board_rvec, board_tvec = self.detect_board(gray, frame, self.parameters)
        if self.board_not_detected(board_rvec, board_tvec, retval):
            # did not detect the board and no board was detected in the past
            return

        ids, marker_rvecs, marker_tvecs, corners = self.detect_markers(gray, frame, self.parameters)
        if ids is None:
            return
        relative_tvecs, relative_rvecs = self.find_relative_vectors_of_markers_with_respect_to_board(board_rvec,
                                                                                                     board_tvec, ids,
                                                                                                     marker_rvecs,
                                                                                                     marker_tvecs)
        self.populate_detected_makers(ids, relative_tvecs, relative_rvecs)

        if self.should_draw:
            aruco.drawAxis(frame, self.cameraMatrix, self.distCoeffs, board_rvec, board_tvec, length=50)
            aruco.drawDetectedMarkers(frame, corners, ids)

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

    def detect_and_draw_board(self, gray_image, captured_frame, detection_parameters):
        retval, rvec, tvec = self.detect_board(gray_image, captured_frame, detection_parameters)
        if retval:
            aruco.drawAxis(captured_frame, self.cameraMatrix, self.distCoeffs, rvec, tvec,
                           50)  # axis length 100 can be changed according to your requirement
            return retval, rvec, tvec

    def detect_board(self, gray_image, captured_frame, detection_parameters):
        corners, ids, rejected_img_points = aruco.detectMarkers(gray_image, self.charuco_board_dictionary,
                                                                parameters=detection_parameters)
        aruco.refineDetectedMarkers(gray_image, self.board, corners, ids, rejected_img_points)

        if ids is None:
            # nothing found
            return False, None, None

        # aruco.drawDetectedMarkers(frame, corners, ids)
        charucoretval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(corners, ids, gray_image,
                                                                                    self.board)
        # im_with_charuco_board = aruco.drawDetectedCornersCharuco(frame, charucoCorners, charucoIds, (0, 255, 0))

        empty_array = np.array([])
        retval, rvec, tvec = aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, self.board, self.cameraMatrix,
                                                            self.distCoeffs, empty_array, empty_array,
                                                            useExtrinsicGuess=False)  # posture estimation from a charuco board

        return retval, rvec, tvec

    def detect_and_draw_marker(self, gray_image, captured_frame, detection_parameters):
        ids, rvecs, tvecs, corners = self.detect_markers(gray_image, captured_frame, detection_parameters)
        aruco.drawDetectedMarkers(captured_frame, corners, ids)

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
        relative_rvecs = []
        for i in range(len(marker_ids)):
            relative_tvec = np.matmul(board_rotation_matrix.transpose(),
                                      markers_tvecs[i].reshape((3, 1)) - board_tvec)
            relative_tvecs.append(relative_tvec)
            # TODO, copy paste from old shit, matrix multiplication
            relative_rvec = None
            relative_rvecs.append(relative_rvec)

        return relative_tvecs, relative_rvecs

    @synchronized_with_lock("lock")
    def populate_detected_makers(self, ids, rvecs, tvecs):
        if ids is None:
            return
        self.detected_markers = []
        for id, rvec, tvec in zip(ids, rvecs, tvecs):
            self.detected_markers.append(DetectedMarker(id, rvec, tvec))

    @synchronized_with_lock("lock")
    def get_detected_markers(self):
        return [detected_marker.copy() for detected_marker in self.detected_markers]
