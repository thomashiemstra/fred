import sys

import cv2
from cv2 import aruco
import jsonpickle
import numpy as np
from src.camera.util import charuco_board_dictionary, aruco_marker_dictionary, get_default_charuco_board

try:
    with open('calibration/calibration_data_charuco.json', 'r') as calibartion_file:
        string = calibartion_file.read()
except FileNotFoundError:
    print("calibration file not found, exiting")
    sys.exit()

calibrations = jsonpickle.decode(string)

cameraMatrix = np.array(calibrations['cameraMatrix'])
distCoeffs = np.array(calibrations['distCoeffs'])
empty_array = np.array([])


cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
if not cap.isOpened():
    print("Camera not connected, exiting!")
    sys.exit()

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


board = get_default_charuco_board(square_length=4, marker_length=3.2)


def detect_and_draw_board(gray_image, captured_frame, detection_parameters):
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_image, charuco_board_dictionary, parameters=detection_parameters)
    aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedImgPoints)

    if ids is None:
        # nothing found
        return False, None, None

    # aruco.drawDetectedMarkers(frame, corners, ids)
    charucoretval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(corners, ids, gray, board)
    # im_with_charuco_board = aruco.drawDetectedCornersCharuco(frame, charucoCorners, charucoIds, (0, 255, 0))
    retval, rvec, tvec = aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, cameraMatrix,
                                                        distCoeffs, empty_array, empty_array,
                                                        useExtrinsicGuess=False)  # posture estimation from a charuco board
    if retval == True:
        aruco.drawAxis(captured_frame, cameraMatrix, distCoeffs, rvec, tvec,
                       50)  # axis length 100 can be changed according to your requirement
        return retval, rvec, tvec

    return False, None, None


def detect_and_draw_markers(gray_image, captured_frame, detection_parameters):
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_image, aruco_marker_dictionary, parameters=detection_parameters)
    aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedImgPoints)

    aruco_marker_length = 2.65
    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, aruco_marker_length, cameraMatrix, distCoeffs)
    aruco.drawDetectedMarkers(captured_frame, corners, ids)

    return ids, rvecs, tvecs


def find_marker_indices(marker_ids, to_find_marker_id):
    res = []
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
    parameters = aruco.DetectorParameters_create()
    board_retval, board_rvec, board_tvec = detect_and_draw_board(gray, frame, parameters)

    marker_ids, markers_rvecs, markers_tvecs = detect_and_draw_markers(gray, frame, parameters)

    if board_retval:
        board_rotation_matrix, _ = cv2.Rodrigues(board_rvec)
        all_ids_of_marker_1 = find_marker_indices(marker_ids, 1)
        if all_ids_of_marker_1:
            id_of_marker_1 = all_ids_of_marker_1[0]
            relative_vec_to_id_1 = np.matmul(board_rotation_matrix.transpose(), markers_tvecs[id_of_marker_1].reshape((3,1)) - board_tvec)
            print(relative_vec_to_id_1)

    cv2.imshow("test", frame)

