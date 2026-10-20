import os
import unittest

import cv2
import jsonpickle
import numpy as np

from src.camera.util import charuco_board_dictionary, aruco_marker_dictionary, get_default_charuco_board, \
    charuco_base_board_square_length, charuco__baseboard_marker_length, charuco_board_dictionary_2
from src.camera.image_handlers import ArucoImageHandler, BoardToBoardImageHandler


RESOURCES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources')
CALIBRATION_FILE = os.path.join(RESOURCES, 'calibration_data.json')

class ArucoImageHandlerTest(unittest.TestCase):

    def setUp(self):
        try:
            with open(CALIBRATION_FILE, 'r') as calibartion_file:
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
        frame = self.get_image('image_with_markers_and_board.png')
        aruco_image_handler = self.get_default_aruco_image_handler()

        # when
        self.handle_frame(aruco_image_handler, frame)

        # then
        detected_makers = aruco_image_handler.get_detected_markers()
        self.assertIsNotNone(detected_makers, "should have gotten detected makers")
        self.assertEqual(3, len(detected_makers), "should have gotten 3 detected markers")
        self.assertEqual([17, 26, 29], sorted(int(marker.id[0]) for marker in detected_makers),
                         "should have gotten the markers 17, 26 and 29")

        for marker in detected_makers:
            self.assertIsNotNone(marker, "marker should not be null")
            self.assertIsNotNone(marker.id, "maker should have it's id filled")
            self.assertIsNotNone(marker.relative_rotation_matrix, "Marker should have an rvec")
            self.assertIsNotNone(marker.tvec, "Marker should have a tvec")

    def test_board_no_markers(self):
        # given
        frame = self.get_image('image_only_board.png')
        aruco_image_handler = self.get_default_aruco_image_handler()

        # when
        self.handle_frame(aruco_image_handler, frame)

        # then
        detected_makers = aruco_image_handler.get_detected_markers()
        self.assertEqual([], detected_makers, "should not have gotten detected markers")

    def test_no_board(self):
        # given
        frame = self.get_image('image_no_board.png')
        aruco_image_handler = self.get_default_aruco_image_handler()

        # when
        self.handle_frame(aruco_image_handler, frame)

        # then
        detected_makers = aruco_image_handler.get_detected_markers()
        self.assertEqual([], detected_makers, "should not have gotten detected markers")

    def test_B2BeHandler_2_boards(self):
        # given
        frame = self.get_image('2_boards.png')
        handler = self.get_default_board_to_board_image_handler()

        # when
        self.handle_frame(handler, frame)

        # then
        relative_matrix, relative_tvec = handler.get_revlative_vecs()
        self.assertIsNotNone(relative_matrix, "should have gotten a relative rotation matrix")
        self.assertIsNotNone(relative_tvec, "should have gotten a relative rotation vector")

    def test_B2BeHandler_1_boards(self):
        # given
        frame = self.get_image('image_only_board.png')
        handler = self.get_default_board_to_board_image_handler()

        # when
        self.handle_frame(handler, frame)

        # then
        relative_matrix, relative_tvec = handler.get_revlative_vecs()
        self.assertIsNone(relative_matrix, "should have not gotten a relative rotation matrix")
        self.assertIsNone(relative_tvec, "should have not gotten a relative rotation vector")

    def test_B2BeHandler_no_boards(self):
        # given
        frame = self.get_image('image_no_board.png')
        handler = self.get_default_board_to_board_image_handler()

        # when
        self.handle_frame(handler, frame)

        # then
        relative_matrix, relative_tvec = handler.get_revlative_vecs()
        self.assertIsNone(relative_matrix, "should have not gotten a relative rotation matrix")
        self.assertIsNone(relative_tvec, "should have not gotten a relative rotation vector")

    def test_board_not_detected(self):
        three = np.zeros((3, 1))
        for handler_class in (ArucoImageHandler, BoardToBoardImageHandler):
            check = handler_class.board_not_detected
            self.assertFalse(check(three, three, True), "vectors of length 3 are a detected board")
            self.assertTrue(check(three, three, False), "no retval is no board")
            self.assertTrue(check(None, three, True), "no rvec is no board")
            self.assertTrue(check(three, None, True), "no tvec is no board")
            self.assertTrue(check(np.zeros((2, 1)), three, True), "a vector of the wrong length is no board")
            self.assertTrue(check(three, np.zeros((4, 1)), True), "a vector of the wrong length is no board")

    def get_default_aruco_image_handler(self):
        aruco_marker_length = 2.65
        return ArucoImageHandler(get_default_charuco_board(), self.cameraMatrix, self.distCoeffs,
                                 aruco_marker_dictionary, charuco_board_dictionary, aruco_marker_length)

    def get_default_board_to_board_image_handler(self):
        base_board = get_default_charuco_board(square_length=charuco_base_board_square_length,
                                               marker_length=charuco__baseboard_marker_length)
        target_board = get_default_charuco_board(square_length=charuco_base_board_square_length,
                                                 marker_length=charuco__baseboard_marker_length,
                                                 dictionary=charuco_board_dictionary_2)

        handler = BoardToBoardImageHandler(base_board, target_board, self.cameraMatrix, self.distCoeffs,
                                           charuco_board_dictionary,
                                           charuco_board_dictionary_2, should_draw=False)
        return handler

    @staticmethod
    def handle_frame(image_handler, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_handler.handle_frame(frame, gray)

    def get_image(self, image_path):
        image = cv2.imread(os.path.join(RESOURCES, image_path))
        self.assertIsNotNone(image, "should have gotten an image")
        return image
