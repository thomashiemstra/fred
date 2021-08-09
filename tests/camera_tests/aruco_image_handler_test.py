import unittest

import cv2
import jsonpickle
import numpy as np

from src.camera.util import charuco_board_dictionary, aruco_marker_dictionary, get_default_charuco_board
from src.camera.image_handlers import ArucoImageHandler


class ArucoImageHandlerTest(unittest.TestCase):

    def setUp(self):
        try:
            with open('resources/calibration_data.json', 'r') as calibartion_file:
                string = calibartion_file.read()
        except FileNotFoundError:
            self.fail("calibration file not found")

        calibrations = jsonpickle.decode(string)

        self.cameraMatrix = np.array(calibrations['cameraMatrix'])
        self.distCoeffs = np.array(calibrations['distCoeffs'])

        self.assertIsNotNone(self.cameraMatrix, "should have a camera matrix")
        self.assertIsNotNone(self.distCoeffs, "should have distCoeffs")

    def test_board_and_markers(self):
        # given
        frame = self.get_image('resources/image_with_markers_and_board.png')
        aruco_image_handler = self.get_default_image_handler()

        # when
        self.handle_frame(aruco_image_handler, frame)

        # then
        detected_makers = aruco_image_handler.get_detected_markers()
        self.assertIsNotNone(detected_makers, "should have gotten detected makers")
        self.assertEqual(3, len(detected_makers), "should have gotten 3 detected markers")

        for marker in detected_makers:
            self.assertIsNotNone(marker, "marker should not be null")
            self.assertIsNotNone(marker.id, "maker should have it's id filled")
            self.assertIsNotNone(marker.relative_rotation_matrix, "Marker should have an rvec")
            self.assertIsNotNone(marker.tvec, "Marker should have a tvec")

    def test_board_no_markers(self):
        # given
        frame = self.get_image('resources/image_only_board.png')
        aruco_image_handler = self.get_default_image_handler()

        # when
        self.handle_frame(aruco_image_handler, frame)

        # then
        detected_makers = aruco_image_handler.get_detected_markers()
        self.assertEqual([], detected_makers, "should not have gotten detected markers")

    def test_no_board(self):
        # given
        frame = self.get_image('resources/image_no_board.png')
        aruco_image_handler = self.get_default_image_handler()

        # when
        self.handle_frame(aruco_image_handler, frame)

        # then
        detected_makers = aruco_image_handler.get_detected_markers()
        self.assertEqual([], detected_makers, "should not have gotten detected markers")

    def get_default_image_handler(self):
        aruco_marker_length = 2.65
        return ArucoImageHandler(get_default_charuco_board(), self.cameraMatrix, self.distCoeffs,
                                                aruco_marker_dictionary, charuco_board_dictionary, aruco_marker_length)

    @staticmethod
    def handle_frame(image_handler, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_handler.handle_frame(frame, gray)

    def get_image(self, image_path):
        image = cv2.imread(image_path)
        self.assertIsNotNone(image, "should have gotten an image")
        return image
