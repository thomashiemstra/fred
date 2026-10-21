import os
import sys

import cv2
from cv2 import aruco
import jsonpickle
import numpy as np
from src.camera.util import CALIBRATION_DIR, CHARUCO_CALIBRATION_FILE, \
    charuco_board_dictionary, aruco_marker_dictionary, get_default_charuco_board, \
    detect_aruco_markers, interpolate_charuco_corners, estimate_pose_charuco_board, estimate_pose_single_markers

try:
    with open(os.path.join(CALIBRATION_DIR, CHARUCO_CALIBRATION_FILE), 'r') as calibartion_file:
        string = calibartion_file.read()
except FileNotFoundError:
    print("calibration file not found, exiting")
    sys.exit()

calibrations = jsonpickle.decode(string)

cameraMatrix = np.array(calibrations['cameraMatrix'])
distCoeffs = np.array(calibrations['distCoeffs'])


cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
if not cap.isOpened():
    print("Camera not connected, exiting!")
    sys.exit()

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # type: ignore[attr-defined]  # missing from the opencv stubs, exists at runtime
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


board = get_default_charuco_board(square_length=4, marker_length=3.2)


def detect_and_draw_board(gray_image: np.ndarray, captured_frame: np.ndarray,
                          detection_parameters: aruco.DetectorParameters
                          ) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
    corners, ids = detect_aruco_markers(gray_image, charuco_board_dictionary, detection_parameters)

    if ids is None:
        # nothing found
        return False, None, None

    # aruco.drawDetectedMarkers(frame, corners, ids)
    charucoCorners, charucoIds = interpolate_charuco_corners(board, gray_image, corners, ids)
    # im_with_charuco_board = aruco.drawDetectedCornersCharuco(frame, charucoCorners, charucoIds, (0, 255, 0))
    # posture estimation from a charuco board
    retval, rvec, tvec = estimate_pose_charuco_board(charucoCorners, charucoIds, board, cameraMatrix, distCoeffs)
    if retval:
        # axis length 50 can be changed according to your requirement
        assert rvec is not None and tvec is not None
        cv2.drawFrameAxes(captured_frame, cameraMatrix, distCoeffs, rvec, tvec, 50)
        return retval, rvec, tvec

    return False, None, None


def detect_and_draw_markers(gray_image: np.ndarray, captured_frame: np.ndarray,
                            detection_parameters: aruco.DetectorParameters
                            ) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    corners, ids = detect_aruco_markers(gray_image, aruco_marker_dictionary, detection_parameters)

    aruco_marker_length = 2.65
    rvecs, tvecs = estimate_pose_single_markers(corners, aruco_marker_length, cameraMatrix, distCoeffs)
    aruco.drawDetectedMarkers(captured_frame, corners, ids)

    return ids, rvecs, tvecs


def find_marker_indices(marker_ids: np.ndarray | None, to_find_marker_id: int) -> list[int]:
    res: list[int] = []
    if marker_ids is None:
        return res

    for i, marker_id in enumerate(marker_ids):
        if marker_id == to_find_marker_id:
            res.append(i)
    return res


while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("failed to grab frame")
        continue

    key_pressed = cv2.waitKey(1)
    if key_pressed % 256 == 27:
        print("Escape hit, closing...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    parameters = aruco.DetectorParameters()
    board_retval, board_rvec, board_tvec = detect_and_draw_board(gray, frame, parameters)

    marker_ids, markers_rvecs, markers_tvecs = detect_and_draw_markers(gray, frame, parameters)

    if board_retval:
        assert board_rvec is not None and board_tvec is not None
        board_rotation_matrix, _ = cv2.Rodrigues(board_rvec)
        all_ids_of_marker_1 = find_marker_indices(marker_ids, 1)
        if all_ids_of_marker_1:
            id_of_marker_1 = all_ids_of_marker_1[0]
            relative_vec_to_id_1 = np.matmul(board_rotation_matrix.transpose(), markers_tvecs[id_of_marker_1].reshape((3,1)) - board_tvec)
            print(relative_vec_to_id_1)

    cv2.imshow("test", frame)

