import json
import sys

import numpy as np
import cv2, PIL, os
from cv2 import aruco
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl

from src.camera.util import get_default_charuco_board, charuco_board_dictionary
from src.utils.decorators import timer

ESC_KEY = 27
SPACE_KEY = 32

board = get_default_charuco_board(square_length=4.0, marker_length=3.2)


def calibrate_camera(allCorners,allIds,imsize):
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
    (ret, camera_matrix, distortion_coefficients0,
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                      charucoCorners=allCorners,
                      charucoIds=allIds,
                      board=board,
                      imageSize=imsize,
                      cameraMatrix=cameraMatrixInit,
                      distCoeffs=distCoeffsInit,
                      flags=flags,
                      criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors


@timer
def open_cap():
    return cv2.VideoCapture(0, cv2.CAP_DSHOW)


cap = open_cap()


@timer
def set_cap_properties(cap):
    cap.set(cv2.CAP_PROP_FPS, 30.0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m', 'j', 'p', 'g'))
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


set_cap_properties(cap)

if not cap.isOpened():
    print("Camera not connected, exiting!")
    sys.exit()


fps = int(cap.get(cv2.CAP_PROP_FPS))
print("fps:", fps)

img_counter = 0
last_frame = None

allCorners = []
allIds = []
decimator = 0
# SUB PIXEL CORNER DETECTION CRITERION
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
imsize = None


while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        print("failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    last_frame = gray
    imsize = gray.shape
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, charuco_board_dictionary)
    cv2.aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedImgPoints)
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    key_pressed = cv2.waitKey(1)
    if key_pressed % 256 == ESC_KEY:
        print("Escape hit, closing...")
        break
    elif key_pressed % 256 == SPACE_KEY:  # SPACE pressed
        if ret:
            img_name = "captures/opencv_frame_{}.png".format(img_counter)
            print("Saving image as: {}".format(img_name))
            cv2.imwrite(img_name, gray)
            img_counter += 1

            if len(corners) > 0:
                # SUB PIXEL DETECTION
                for corner in corners:
                    cv2.cornerSubPix(gray, corner,
                                     winSize=(3, 3),
                                     zeroZone=(-1, -1),
                                     criteria=criteria)
                charucoretval, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(corners, ids, gray,
                                                                                                board)
                if charucoCorners is not None and charucoIds is not None and len(
                        charucoCorners) > 3 and decimator % 1 == 0 and 7 < len(ids) <= 9:
                    print("found {} ids".format(len(ids)))
                    allCorners.append(charucoCorners)
                    allIds.append(charucoIds)
                else:
                    print("nope, corners:{}, ids:{} decimator:{}, ids {}".format(charucoCorners, charucoIds, decimator, len(ids)))
            else:
                print("no corners")

            decimator += 1
        else:
            print("Not saving image, no chessboard found!")

    cv2.imshow("test", frame)

cap.release()
cv2.destroyAllWindows()


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

with open('calibration/calibration_data_charuco.json', 'w') as outfile:
    json.dump(calibration_data, outfile, indent=4)
