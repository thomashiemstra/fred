import glob
import os
import sys
from collections.abc import Sequence

import cv2
import numpy as np
import json


from src.camera.util import CALIBRATION_DIR, CHARUCO_CALIBRATION_FILE, \
    get_default_charuco_board, charuco_board_dictionary, detect_aruco_markers, \
    interpolate_charuco_corners, calibrate_camera_charuco


def calibrate_camera(allCorners: list[np.ndarray], allIds: list[np.ndarray], imsize: tuple[int, ...]
                     ) -> tuple[float, np.ndarray, np.ndarray, Sequence[np.ndarray], Sequence[np.ndarray]]:
    """
    Calibrates the camera using the dected corners.
    """
    print("CAMERA CALIBRATION")

    cameraMatrixInit = np.array([[ 1430.,    0., 989.],
                                 [    0., 1430., 505.],
                                 [    0.,    0.,   1.]])

    distCoeffsInit = np.zeros((5,1))
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    #flags = (cv2.CALIB_RATIONAL_MODEL)
    ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors = calibrate_camera_charuco(
        board, allCorners, allIds, imsize, cameraMatrixInit, distCoeffsInit, flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors


board = get_default_charuco_board(square_length=4.0, marker_length=3.2)

allCorners = []
allIds = []
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

img_counter = 0
last_frame = None
imsize = None

images = glob.glob('captures/*.png')

print("reading images")
for fname in images:
    frame = cv2.imread(fname)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imsize = gray.shape[::-1]  # (width, height)
    last_frame = gray

    corners, ids = detect_aruco_markers(gray, charuco_board_dictionary)
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    if ids is not None and len(corners) > 7:
        # SUB PIXEL DETECTION
        for corner in corners:
            cv2.cornerSubPix(gray, corner,
                             winSize=(3, 3),
                             zeroZone=(-1, -1),
                             criteria=criteria)
        charucoCorners, charucoIds = interpolate_charuco_corners(board, gray, corners, ids)
        if charucoCorners is not None and charucoIds is not None and len(
                charucoCorners) > 3 and 7 < len(ids) <= 9:
            print("found {} ids".format(len(ids)))
            allCorners.append(charucoCorners)
            allIds.append(charucoIds)
            img_counter += 1

if last_frame is None:
    print("no last frame, camera not connected? exiting.")
    sys.exit()

if img_counter < 10:
    print("not enough images, need at least 10, but got {} instead. exiting.".format(img_counter))
    sys.exit()

assert imsize is not None
ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors = calibrate_camera(allCorners, allIds, imsize)

print("Camera matrix : \n")
print(camera_matrix)
print("dist : \n")
print(distortion_coefficients0)
print("rvecs : \n")
print(rotation_vectors)
print("tvecs : \n")
print(translation_vectors)

calibration_data = {'cameraMatrix': camera_matrix.tolist(), 'distCoeffs': distortion_coefficients0.tolist()}

with open(os.path.join(CALIBRATION_DIR, CHARUCO_CALIBRATION_FILE), 'w') as outfile:
    json.dump(calibration_data, outfile, indent=4)