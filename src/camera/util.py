import sys
from collections.abc import Sequence

import cv2
import jsonpickle
import numpy as np


class CaptureConfig:
    screen_width = 1920
    screen_height = 1080
    fps = 60
    marker_size = 20
    image_format = cv2.VideoWriter_fourcc(*'MJPG')  # type: ignore[attr-defined]  # missing from the opencv stubs, exists at runtime
    FOURCC =  cv2.VideoWriter.fourcc('m', 'j', 'p', 'g')
    FOURCC2 = cv2.VideoWriter.fourcc('M', 'J', 'P', 'G')


charuco_board_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
charuco_board_dictionary_2 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
aruco_marker_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
charuco_base_board_square_length = 3
charuco__baseboard_marker_length = 2.4
aruco_marker_length = 2.65
base_marker_offset = [-26.5, 8, 0]


def get_default_charuco_board(squares_x: int = 6,
                              squares_y: int = 3,
                              square_length: float = 1,
                              marker_length: float = 0.8,
                              dictionary: cv2.aruco.Dictionary = charuco_board_dictionary
                              ) -> cv2.aruco.CharucoBoard:

    return cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, dictionary)


def get_calibrations(path: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        with open(path, 'r') as calibartion_file:
            string = calibartion_file.read()
    except FileNotFoundError:
        print("calibration file not found, exiting")
        sys.exit()

    calibrations = jsonpickle.decode(string)

    cameraMatrix = np.array(calibrations['cameraMatrix'])
    distCoeffs = np.array(calibrations['distCoeffs'])

    return cameraMatrix, distCoeffs


def find_relative_vector_and_rotation(base_rotation_matrix: np.ndarray, base_tvec: np.ndarray,
                                      target_rotation_matrix: np.ndarray,
                                      target_tvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base_to_target_vec = target_tvec.reshape((3, 1)) - base_tvec

    relative_tvec = np.matmul(base_rotation_matrix.transpose(), base_to_target_vec)
    relative_rotation_matrix = np.matmul(base_rotation_matrix.transpose(), target_rotation_matrix)

    return relative_rotation_matrix, relative_tvec


def detect_aruco_markers(gray_image: np.ndarray, dictionary: cv2.aruco.Dictionary,
                   parameters: cv2.aruco.DetectorParameters | None = None
                   ) -> tuple[Sequence[np.ndarray], np.ndarray | None]:
    """Corners and ids of the markers in the image, ids is None when no marker is found."""
    if parameters is None:
        parameters = cv2.aruco.DetectorParameters()
    corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, parameters).detectMarkers(gray_image)
    return corners, ids


def interpolate_charuco_corners(board: cv2.aruco.CharucoBoard, gray_image: np.ndarray,
                                marker_corners: Sequence[np.ndarray], marker_ids: np.ndarray
                                ) -> tuple[np.ndarray | None, np.ndarray | None]:
    """The chessboard corners of the board between the already detected markers, None when there are none."""
    charuco_corners, charuco_ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(
        gray_image, markerCorners=marker_corners, markerIds=marker_ids)
    if charuco_corners is None or charuco_ids is None or len(charuco_ids) == 0:
        return None, None
    return charuco_corners, charuco_ids


def estimate_pose_charuco_board(charuco_corners: np.ndarray | None, charuco_ids: np.ndarray | None,
                                board: cv2.aruco.CharucoBoard, camera_matrix: np.ndarray, dist_coeffs: np.ndarray
                                ) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
    """Pose of the board in the camera frame, rvec and tvec have shape (3, 1)."""
    # like the removed estimatePoseCharucoBoard: at least 4 corners, not all on one line
    if charuco_corners is None or charuco_ids is None or len(charuco_ids) < 4:
        return False, None, None
    if board.checkCharucoCornersCollinear(charuco_ids):
        return False, None, None
    object_points, image_points = board.matchImagePoints(charuco_corners, charuco_ids)  # type: ignore[call-overload]  # stubs expect a list of marker corners, a charuco board takes one array of corners
    retval, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
    if not retval:
        return False, None, None
    return True, rvec, tvec


def estimate_pose_single_markers(marker_corners: Sequence[np.ndarray], marker_length: float,
                                 camera_matrix: np.ndarray, dist_coeffs: np.ndarray
                                 ) -> tuple[np.ndarray, np.ndarray]:
    """Pose of every marker in the camera frame, rvecs and tvecs have shape (n, 1, 3)."""
    half = marker_length / 2
    # centred on the marker, same corner order as the detected corners
    object_points = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
                             dtype=np.float32)
    rvecs = np.zeros((len(marker_corners), 1, 3))
    tvecs = np.zeros((len(marker_corners), 1, 3))
    for i, corners in enumerate(marker_corners):
        # the iterative solver is what the removed estimatePoseSingleMarkers used, IPPE_SQUARE gives other poses
        _, rvec, tvec = cv2.solvePnP(object_points, corners, camera_matrix, dist_coeffs)
        rvecs[i, 0] = rvec.ravel()
        tvecs[i, 0] = tvec.ravel()
    return rvecs, tvecs


def calibrate_camera_charuco(board: cv2.aruco.CharucoBoard, charuco_corners: Sequence[np.ndarray],
                             charuco_ids: Sequence[np.ndarray], image_size: tuple[int, ...],
                             camera_matrix: np.ndarray, dist_coeffs: np.ndarray, flags: int,
                             criteria: tuple[int, int, float]
                             ) -> tuple[float, np.ndarray, np.ndarray, Sequence[np.ndarray], Sequence[np.ndarray]]:
    """Calibrates the camera with the interpolated charuco corners of several images."""
    object_points = []
    image_points = []
    for corners, ids in zip(charuco_corners, charuco_ids):
        # a view needs at least 4 corners
        if len(ids) < 4:
            continue
        view_object_points, view_image_points = board.matchImagePoints(corners, ids)  # type: ignore[call-overload]  # stubs expect a list of marker corners, a charuco board takes one array of corners
        object_points.append(view_object_points)
        image_points.append(view_image_points)
    (error, camera_matrix, dist_coeffs, rvecs, tvecs,
     _std_deviations_intrinsics, _std_deviations_extrinsics,
     _per_view_errors) = cv2.calibrateCameraExtended(object_points, image_points, image_size, camera_matrix,
                                                     dist_coeffs, flags=flags, criteria=criteria)
    return error, camera_matrix, dist_coeffs, rvecs, tvecs
