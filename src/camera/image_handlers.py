import threading
from abc import ABC, abstractmethod
import cv2
from cv2 import aruco
import numpy as np


class ImageHandler(ABC):
    @abstractmethod
    def handle_frame(self, frame, gray):
        pass

    @abstractmethod
    def deactivate_handler(self):
        pass


class ImageRecorder(ImageHandler):

    def __init__(self, filename, fps, screen_width, screen_height):
        self.out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                   fps, (screen_width, screen_height))

    def handle_frame(self, frame, gray):
        self.out.write(frame)

    def deactivate_handler(self):
        self.out.release()


class ArucoImageHandler(ImageHandler):

    def __init__(self, board, cameraMatrix, distCoeffs, aruco_dictionary, charuco_board_dictionary):
        self.board = board
        self.lock = threading.RLock()
        self.parameters = aruco.DetectorParameters_create()
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs
        self.aruco_dictionary = aruco_dictionary
        self.charuco_board_dictionary = charuco_board_dictionary

    def detect_and_draw_board(self, gray_image, captured_frame, detection_parameters):
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_image, self.charuco_board_dictionary,
                                                              parameters=detection_parameters)
        aruco.refineDetectedMarkers(gray_image, self.board, corners, ids, rejectedImgPoints)

        if ids is None:
            # nothing found
            return False, None, None

        # aruco.drawDetectedMarkers(frame, corners, ids)
        charucoretval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(corners, ids, gray_image, self.board)
        # im_with_charuco_board = aruco.drawDetectedCornersCharuco(frame, charucoCorners, charucoIds, (0, 255, 0))

        empty_array = np.array([])
        retval, rvec, tvec = aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, self.board, self.cameraMatrix,
                                                            self.distCoeffs, empty_array, empty_array,
                                                            useExtrinsicGuess=False)  # posture estimation from a charuco board
        if retval == True:
            aruco.drawAxis(captured_frame, self.cameraMatrix, self.distCoeffs, rvec, tvec,
                           50)  # axis length 100 can be changed according to your requirement
            return retval, rvec, tvec

        return False, None, None

    def detect_and_draw_markers(self, gray_image, captured_frame, detection_parameters):
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_image, self.aruco_dictionary,
                                                              parameters=detection_parameters)
        aruco.refineDetectedMarkers(gray_image, self.board, corners, ids, rejectedImgPoints)

        aruco_marker_length = 2.65
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, aruco_marker_length, self.cameraMatrix, self.distCoeffs)
        aruco.drawDetectedMarkers(captured_frame, corners, ids)

        return ids, rvecs, tvecs

    def handle_frame(self, frame, gray):
        pass

    def deactivate_handler(self):
        pass