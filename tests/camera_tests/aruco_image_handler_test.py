import unittest

import jsonpickle
import numpy as np
import cv2


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
        frame = cv2.imread('resources/image_with_markers_and_board.png')
        self.assertIsNotNone(frame, "should have gotten an image")
