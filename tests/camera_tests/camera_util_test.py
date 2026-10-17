import os
import threading
import unittest

import cv2
import numpy as np

from src.camera.util import find_relative_vector_and_rotation, get_calibrations, get_default_charuco_board, \
    charuco_board_dictionary_2
from src.camera.image_handlers import ArucoImageHandler, BoardToBoardImageHandler, DetectedMarker

RESOURCES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources')
CALIBRATION_FILE = os.path.join(RESOURCES, 'calibration_data.json')


def rotation_matrix(rvec):
    matrix, _ = cv2.Rodrigues(np.array(rvec, dtype=np.float64).reshape((3, 1)))
    return matrix


def compose_in_camera_frame(base_rvec, base_tvec, relative_rvec, relative_tvec):
    """Pose of a target in the camera frame, given the base pose in the camera frame
    and the target pose expressed in the base frame."""
    base_matrix = rotation_matrix(base_rvec)
    relative_matrix = rotation_matrix(relative_rvec)
    target_matrix = base_matrix @ relative_matrix
    target_tvec = base_matrix @ np.array(relative_tvec, dtype=np.float64).reshape((3, 1)) + \
        np.array(base_tvec, dtype=np.float64).reshape((3, 1))
    target_rvec, _ = cv2.Rodrigues(target_matrix)
    return target_matrix, target_rvec, target_tvec


# A board seen tilted and rotated by the camera, and a target placed somewhere on/above it
BASE_RVEC = [0.3, -0.5, 1.2]
BASE_TVEC = [5.0, -3.0, 60.0]
RELATIVE_RVEC = [0.0, 0.0, np.pi / 3]
RELATIVE_TVEC = [12.0, 4.0, 2.5]


class FindRelativeVectorAndRotationTest(unittest.TestCase):

    def test_identity_base_gives_plain_difference(self):
        base_tvec = np.array([[1.0], [2.0], [3.0]])
        target_tvec = np.array([4.0, 6.0, 8.0])
        target_matrix = rotation_matrix([0, 0, np.pi / 2])

        relative_matrix, relative_tvec = find_relative_vector_and_rotation(np.eye(3), base_tvec,
                                                                           target_matrix, target_tvec)

        np.testing.assert_allclose(relative_matrix, target_matrix, atol=1e-9)
        np.testing.assert_allclose(relative_tvec, [[3.0], [4.0], [5.0]], atol=1e-9)

    def test_same_pose_gives_identity_and_zero(self):
        base_matrix = rotation_matrix(BASE_RVEC)
        base_tvec = np.array(BASE_TVEC).reshape((3, 1))

        relative_matrix, relative_tvec = find_relative_vector_and_rotation(base_matrix, base_tvec,
                                                                           base_matrix, base_tvec)

        np.testing.assert_allclose(relative_matrix, np.eye(3), atol=1e-9)
        np.testing.assert_allclose(relative_tvec, np.zeros((3, 1)), atol=1e-9)

    def test_recovers_pose_expressed_in_base_frame(self):
        target_matrix, _, target_tvec = compose_in_camera_frame(BASE_RVEC, BASE_TVEC, RELATIVE_RVEC, RELATIVE_TVEC)
        # tvecs from estimatePoseSingleMarkers have shape (1, 3), board tvecs have shape (3, 1)
        relative_matrix, relative_tvec = find_relative_vector_and_rotation(rotation_matrix(BASE_RVEC),
                                                                           np.array(BASE_TVEC).reshape((3, 1)),
                                                                           target_matrix,
                                                                           target_tvec.reshape((1, 3)))

        np.testing.assert_allclose(relative_matrix, rotation_matrix(RELATIVE_RVEC), atol=1e-9)
        np.testing.assert_allclose(relative_tvec, np.array(RELATIVE_TVEC).reshape((3, 1)), atol=1e-9)


class ImageHandlerGeometryTest(unittest.TestCase):
    """Pure geometry/bookkeeping parts of the image handlers, no detection (and no legacy aruco API) needed"""

    def test_board_to_board_relative_matrix_and_vector(self):
        _, target_rvec, target_tvec = compose_in_camera_frame(BASE_RVEC, BASE_TVEC, RELATIVE_RVEC, RELATIVE_TVEC)

        relative_matrix, relative_tvec = BoardToBoardImageHandler.find_relative_matrix_and_vector(
            np.array(BASE_RVEC).reshape((3, 1)), np.array(BASE_TVEC).reshape((3, 1)), target_rvec, target_tvec)

        np.testing.assert_allclose(relative_matrix, rotation_matrix(RELATIVE_RVEC), atol=1e-9)
        np.testing.assert_allclose(relative_tvec, np.array(RELATIVE_TVEC).reshape((3, 1)), atol=1e-9)

    def test_markers_relative_to_board(self):
        # marker vectors shaped like the output of estimatePoseSingleMarkers: (n, 1, 3)
        relative_poses = [([0, 0, 0], [1.0, 2.0, 0.0]), ([0, 0, np.pi / 2], [-3.0, 7.0, 1.0])]
        marker_rvecs, marker_tvecs = [], []
        for relative_rvec, relative_tvec in relative_poses:
            _, rvec, tvec = compose_in_camera_frame(BASE_RVEC, BASE_TVEC, relative_rvec, relative_tvec)
            marker_rvecs.append(rvec.reshape((1, 3)))
            marker_tvecs.append(tvec.reshape((1, 3)))
        handler = ArucoImageHandler.__new__(ArucoImageHandler)  # skip __init__, it needs the legacy aruco API

        relative_tvecs, relative_matrices = handler.find_relative_vectors_of_markers_with_respect_to_board(
            np.array(BASE_RVEC).reshape((3, 1)), np.array(BASE_TVEC).reshape((3, 1)), np.array([[3], [7]]),
            np.array(marker_rvecs), np.array(marker_tvecs))

        self.assertEqual(2, len(relative_tvecs))
        for i, (relative_rvec, relative_tvec) in enumerate(relative_poses):
            np.testing.assert_allclose(relative_matrices[i], rotation_matrix(relative_rvec), atol=1e-9)
            np.testing.assert_allclose(relative_tvecs[i], np.array(relative_tvec).reshape((3, 1)), atol=1e-9)

    def test_board_not_detected(self):
        rvec = np.zeros((3, 1))
        tvec = np.ones((3, 1))
        for handler_class in (ArucoImageHandler, BoardToBoardImageHandler):
            self.assertTrue(handler_class.board_not_detected(rvec, tvec, False), "retval false means not detected")
            self.assertTrue(handler_class.board_not_detected(None, tvec, True), "missing rvec means not detected")
            self.assertTrue(handler_class.board_not_detected(rvec, None, True), "missing tvec means not detected")
            self.assertFalse(handler_class.board_not_detected(rvec, tvec, True), "valid vectors mean detected")

    def test_detected_markers_are_returned_as_copies_and_cleared(self):
        handler = ArucoImageHandler.__new__(ArucoImageHandler)  # skip __init__, it needs the legacy aruco API
        handler.lock = threading.RLock()
        handler.detected_markers = []

        handler.populate_detected_makers(np.array([[1], [4]]), [np.eye(3), np.eye(3)],
                                         [np.zeros((3, 1)), np.ones((3, 1))])
        markers = handler.get_detected_markers()

        self.assertEqual([1, 4], [int(marker.id[0]) for marker in markers])
        self.assertTrue(all(isinstance(marker, DetectedMarker) for marker in markers))
        markers.pop()
        self.assertEqual(2, len(handler.get_detected_markers()), "callers should not be able to change the state")

        handler.populate_detected_makers(None, [], [])
        self.assertEqual(2, len(handler.get_detected_markers()), "ids None should keep the previous markers")

        handler.clear_detected_markers()
        self.assertEqual([], handler.get_detected_markers())


class CalibrationAndBoardTest(unittest.TestCase):

    def test_get_calibrations(self):
        camera_matrix, dist_coeffs = get_calibrations(CALIBRATION_FILE)

        self.assertEqual((3, 3), camera_matrix.shape)
        self.assertEqual((1, 5), dist_coeffs.shape)
        self.assertAlmostEqual(1430.49791192, camera_matrix[0][0])
        self.assertEqual(1.0, camera_matrix[2][2])

    def test_get_calibrations_missing_file_exits(self):
        with self.assertRaises(SystemExit):
            get_calibrations(os.path.join(RESOURCES, 'does_not_exist.json'))

    def test_default_charuco_board(self):
        board = get_default_charuco_board(square_length=3, marker_length=2.4, dictionary=charuco_board_dictionary_2)

        self.assertEqual((6, 3), tuple(board.getChessboardSize()))
        self.assertAlmostEqual(3, board.getSquareLength())
        self.assertAlmostEqual(2.4, board.getMarkerLength(), places=5)
        # a 6x3 charuco board has 9 markers and (6-1)*(3-1) inner corners
        self.assertEqual(9, len(board.getIds()))
        self.assertEqual(10, len(board.getChessboardCorners()))


if __name__ == '__main__':
    unittest.main()
