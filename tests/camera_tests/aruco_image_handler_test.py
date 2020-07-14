import unittest

import jsonpickle
import numpy as np
import cv2

from src.camera.capture import charuco_board_dictionary, aruco_dictionary, get_default_charuco_board
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
        aruco_image_handler = ArucoImageHandler(get_default_charuco_board(), self.cameraMatrix, self.distCoeffs, aruco_dictionary, charuco_board_dictionary)

        # when
        aruco_image_handler.handle_frame(frame)

        # then
        # todo

    def get_image(self, image_path):
        image = cv2.imread('resources/image_with_markers_and_board.png')
        self.assertIsNotNone(image, "should have gotten an image")
        return image
