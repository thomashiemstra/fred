import threading
from abc import ABC, abstractmethod
import cv2
from cv2 import aruco
import numpy as np

from src.camera.capture_config import CaptureConfig
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


class ArucoImageHandler(ImageHandler):

    def __init__(self, board, cameraMatrix, distCoeffs, aruco_dictionary, charuco_board_dictionary, should_draw=False):
        self.board = board
        self.lock = threading.RLock()
        self.parameters = aruco.DetectorParameters_create()
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs
        self.aruco_dictionary = aruco_dictionary
        self.charuco_board_dictionary = charuco_board_dictionary
        self.should_draw = should_draw
        self.board_rvec = None
        self.board_tvec = None


        # @synchronized_with_lock("lock")

    def handle_frame(self, frame, gray):
        retval, self.board_rvec, self.board_tvec = self.detect_board(gray, frame, self.parameters)
        if not retval and self.board_tvec is None:
            # did not detect the board and no board was detected in the past
            return

        ids, rvecs, tvecs, corners = self.detect_markers(gray, frame, self.parameters)

        if self.should_draw:
            aruco.drawAxis(frame, self.cameraMatrix, self.distCoeffs, self.board_rvec, self.board_tvec, length=50)
            aruco.drawDetectedMarkers(frame, corners, ids)

    def deactivate_handler(self):
        pass

    def detect_board(self, gray_image, captured_frame, detection_parameters):
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_image, self.charuco_board_dictionary,
                                                              parameters=detection_parameters)
        aruco.refineDetectedMarkers(gray_image, self.board, corners, ids, rejectedImgPoints)

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

        return False, rvec, tvec

    def detect_markers(self, gray_image, captured_frame, detection_parameters):
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_image, self.aruco_dictionary,
                                                              parameters=detection_parameters)
        aruco.refineDetectedMarkers(gray_image, self.board, corners, ids, rejectedImgPoints)

        aruco_marker_length = 2.65
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, aruco_marker_length, self.cameraMatrix,
                                                          self.distCoeffs)

        return ids, rvecs, tvecs, corners
