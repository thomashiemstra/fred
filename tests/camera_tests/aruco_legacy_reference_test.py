"""The poses of the image handlers must be the same as the ones the removed cv2.aruco functions gave.

The reference numbers come from the legacy code (opencv-contrib-python 4.6.0.66, commit 477cbaa) run on the recorded
images, see legacy_opencv_reference.py.

Tolerance: the board poses and the marker poses of 2_boards.png are identical to the last bit here (difference 0.0).
rtol=1e-6 only leaves room for floating point noise of other cpus and builds, a pose that really changed is far off.

One known difference: opencv 4.7+ detects the aruco markers with another candidate selection than 4.6. On
image_with_markers_and_board.png two of the three small markers (17 and 29) have a corner 1 or 2 pixels off, which
moves their poses (the pose of such a small marker is sensitive to its corners). That is the detector, not the port:
the pose calculation itself gives the legacy numbers when it gets the legacy corners (test_marker_poses_from_legacy_corners).
"""
import os
import unittest

import cv2
import jsonpickle
import numpy as np

from src.camera.image_handlers import ArucoImageHandler, BoardToBoardImageHandler
from src.camera.util import charuco_board_dictionary, aruco_marker_dictionary, get_default_charuco_board, \
    charuco_base_board_square_length, charuco__baseboard_marker_length, charuco_board_dictionary_2, \
    detect_aruco_markers, estimate_pose_single_markers, calibrate_camera_charuco
from tests.camera_tests.legacy_opencv_reference import LEGACY_OPENCV_4_6_REFERENCE as REFERENCE

RESOURCES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources')
CALIBRATION_FILE = os.path.join(RESOURCES, 'calibration_data.json')
IMAGES = ['2_boards.png', 'image_only_board.png', 'image_with_markers_and_board.png', 'image_no_board.png']
RTOL = 1e-6
ATOL = 1e-8
ARUCO_MARKER_LENGTH = 2.65


class LegacyReferenceTest(unittest.TestCase):

    def setUp(self) -> None:
        with open(CALIBRATION_FILE, 'r') as calibration_file:
            calibrations = jsonpickle.decode(calibration_file.read())
        self.cameraMatrix = np.array(calibrations['cameraMatrix'])
        self.distCoeffs = np.array(calibrations['distCoeffs'])

    def test_board_pose_matches_legacy(self) -> None:
        for image in IMAGES:
            with self.subTest(image=image):
                frame, gray = self.get_image(image)
                handler = self.get_aruco_image_handler()
                reference = REFERENCE[image]['aruco_board']

                retval, rvec, tvec = handler.detect_board(gray, handler.parameters)

                self.assertEqual(reference['retval'], retval)
                if reference['retval']:
                    np.testing.assert_allclose(rvec, reference['rvec'], rtol=RTOL, atol=ATOL)
                    np.testing.assert_allclose(tvec, reference['tvec'], rtol=RTOL, atol=ATOL)
                    assert tvec is not None
                    self.assertGreater(tvec[2, 0], 0, "the board should be in front of the camera")
                else:
                    self.assertIsNone(rvec)
                    self.assertIsNone(tvec)

    def test_marker_poses_from_legacy_corners(self) -> None:
        for image in ['2_boards.png', 'image_with_markers_and_board.png']:
            with self.subTest(image=image):
                reference = REFERENCE[image]['aruco_markers']
                corners = [np.array(c, dtype=np.float32) for c in reference['corners']]

                rvecs, tvecs = estimate_pose_single_markers(corners, ARUCO_MARKER_LENGTH, self.cameraMatrix,
                                                            self.distCoeffs)

                np.testing.assert_allclose(rvecs, reference['rvecs'], rtol=RTOL, atol=ATOL)
                np.testing.assert_allclose(tvecs, reference['tvecs'], rtol=RTOL, atol=ATOL)

    def test_relative_poses_from_legacy_vectors(self) -> None:
        handler = self.get_aruco_image_handler()
        for image in ['2_boards.png', 'image_with_markers_and_board.png']:
            with self.subTest(image=image):
                reference = REFERENCE[image]
                board = reference['aruco_board']
                markers = reference['aruco_markers']

                relative_tvecs, relative_matrices = handler.find_relative_vectors_of_markers_with_respect_to_board(
                    np.array(board['rvec']), np.array(board['tvec']), np.array(markers['ids']),
                    np.array(markers['rvecs']), np.array(markers['tvecs']))

                self.assertEqual(len(reference['aruco_handle_frame']), len(relative_tvecs))
                for expected, matrix, tvec in zip(reference['aruco_handle_frame'], relative_matrices, relative_tvecs):
                    np.testing.assert_allclose(matrix, expected['rot'], rtol=RTOL, atol=ATOL)
                    np.testing.assert_allclose(tvec, expected['tvec'], rtol=RTOL, atol=ATOL)

    def test_detected_markers_of_2_boards_match_legacy(self) -> None:
        frame, gray = self.get_image('2_boards.png')
        handler = self.get_aruco_image_handler()

        handler.handle_frame(frame, gray)

        # the order of the detected markers is not the same as in opencv 4.6, the ids are
        detected = {int(marker.id[0]): marker for marker in handler.get_detected_markers()}
        expected = {int(marker['id'][0]): marker for marker in REFERENCE['2_boards.png']['aruco_handle_frame']}
        self.assertEqual(sorted(expected), sorted(detected))
        for marker_id, marker in detected.items():
            np.testing.assert_allclose(marker.relative_rotation_matrix, expected[marker_id]['rot'], rtol=RTOL, atol=ATOL)
            np.testing.assert_allclose(marker.tvec, expected[marker_id]['tvec'], rtol=RTOL, atol=ATOL)

    def test_detected_markers_with_board_match_legacy_ids_and_corners(self) -> None:
        image = 'image_with_markers_and_board.png'
        frame, gray = self.get_image(image)
        reference = REFERENCE[image]['aruco_markers']

        corners, ids = detect_aruco_markers(gray, aruco_marker_dictionary)

        assert ids is not None
        self.assertEqual(sorted(np.ravel(reference['ids'])), sorted(np.ravel(ids)))
        reference_ids = list(np.ravel(reference['ids']))
        for marker_id, marker_corners in zip(np.ravel(ids), corners):
            legacy_corners = np.array(reference['corners'][reference_ids.index(marker_id)])
            # see the module docstring, the corners are at most 2 pixels off
            self.assertLessEqual(np.abs(marker_corners - legacy_corners).max(), 2.0, "marker {}".format(marker_id))

        handler = self.get_aruco_image_handler()
        handler.handle_frame(frame, gray)
        detected = {int(marker.id[0]): marker for marker in handler.get_detected_markers()}
        expected = {int(marker['id'][0]): marker for marker in REFERENCE[image]['aruco_handle_frame']}
        self.assertEqual(sorted(expected), sorted(detected))
        # marker 26 got the same corners as in opencv 4.6
        np.testing.assert_allclose(detected[26].relative_rotation_matrix, expected[26]['rot'], rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(detected[26].tvec, expected[26]['tvec'], rtol=RTOL, atol=ATOL)

    def test_board_to_board_matches_legacy(self) -> None:
        for image in IMAGES:
            with self.subTest(image=image):
                frame, gray = self.get_image(image)
                handler = self.get_board_to_board_image_handler()
                reference = REFERENCE[image]

                base = handler.detect_board(gray, handler.board, handler.charuco_board_dictionary)
                target = handler.detect_board(gray, handler.target_board, handler.target_board_dictionary)
                handler.handle_frame(frame, gray)
                relative_matrix, relative_tvec = handler.get_revlative_vecs()

                for detected, expected in ((base, reference['b2b_base_board']), (target, reference['b2b_target_board'])):
                    self.assertEqual(expected['retval'], detected[0])
                    if expected['retval']:
                        np.testing.assert_allclose(detected[1], expected['rvec'], rtol=RTOL, atol=ATOL)
                        np.testing.assert_allclose(detected[2], expected['tvec'], rtol=RTOL, atol=ATOL)
                    else:
                        self.assertIsNone(detected[1])
                        self.assertIsNone(detected[2])
                expected_relative = reference['b2b_handle_frame']
                if expected_relative['rot'] is None:
                    self.assertIsNone(relative_matrix)
                    self.assertIsNone(relative_tvec)
                else:
                    np.testing.assert_allclose(relative_matrix, expected_relative['rot'], rtol=RTOL, atol=ATOL)
                    np.testing.assert_allclose(relative_tvec, expected_relative['tvec'], rtol=RTOL, atol=ATOL)

    def test_board_to_board_distance_is_plausible(self) -> None:
        frame, gray = self.get_image('2_boards.png')
        handler = self.get_board_to_board_image_handler()

        handler.handle_frame(frame, gray)

        _, relative_tvec = handler.get_revlative_vecs()
        assert relative_tvec is not None
        # both boards are 18 units wide (6 squares of 3) and lie next to each other, legacy value 18.5
        self.assertTrue(5 < np.linalg.norm(relative_tvec) < 40, np.linalg.norm(relative_tvec))

    def test_calibrate_camera_charuco_recovers_known_camera(self) -> None:
        board = get_default_charuco_board(square_length=4.0, marker_length=3.2)
        true_camera_matrix = np.array([[1430., 0., 989.], [0., 1430., 505.], [0., 0., 1.]])
        no_distortion = np.zeros((5, 1))
        board_corners = board.getChessboardCorners()
        ids = np.arange(len(board_corners), dtype=np.int32).reshape(-1, 1)
        rng = np.random.default_rng(1)
        all_corners = []
        all_ids = []
        for _ in range(12):
            rvec = rng.uniform(-0.5, 0.5, (3, 1))
            tvec = np.array([[-6.0], [-3.0], [0.0]]) + rng.uniform(-3, 3, (3, 1)) + np.array([[0.0], [0.0], [45.0]])
            projected, _ = cv2.projectPoints(board_corners, rvec, tvec, true_camera_matrix, no_distortion)
            all_corners.append(projected.astype(np.float32))
            all_ids.append(ids)
        initial_camera_matrix = np.array([[1300., 0., 960.], [0., 1300., 540.], [0., 0., 1.]])

        error, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera_charuco(
            board, all_corners, all_ids, (1920, 1080), initial_camera_matrix, np.zeros((5, 1)),
            cv2.CALIB_USE_INTRINSIC_GUESS, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-12))

        self.assertLess(error, 0.01)
        np.testing.assert_allclose(camera_matrix, true_camera_matrix, rtol=1e-3)
        self.assertEqual(12, len(rvecs))
        self.assertEqual(12, len(tvecs))

    def get_aruco_image_handler(self) -> ArucoImageHandler:
        return ArucoImageHandler(get_default_charuco_board(), self.cameraMatrix, self.distCoeffs,
                                 aruco_marker_dictionary, charuco_board_dictionary, ARUCO_MARKER_LENGTH)

    def get_board_to_board_image_handler(self) -> BoardToBoardImageHandler:
        base_board = get_default_charuco_board(square_length=charuco_base_board_square_length,
                                               marker_length=charuco__baseboard_marker_length)
        target_board = get_default_charuco_board(square_length=charuco_base_board_square_length,
                                                 marker_length=charuco__baseboard_marker_length,
                                                 dictionary=charuco_board_dictionary_2)
        return BoardToBoardImageHandler(base_board, target_board, self.cameraMatrix, self.distCoeffs,
                                        charuco_board_dictionary, charuco_board_dictionary_2, should_draw=False)

    def get_image(self, image_name: str) -> tuple[np.ndarray, np.ndarray]:
        frame = cv2.imread(os.path.join(RESOURCES, image_name))
        self.assertIsNotNone(frame, "should have gotten an image")
        return frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
