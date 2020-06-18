import glob
import sys

import cv2
import numpy as np
import json


CHECKERBOARD = (6, 9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)


# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []

img_counter = 0
last_frame = None

images = glob.glob('captures/*.png')

print("reading images")
for fname in images:
    frame = cv2.imread(fname)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    last_frame = gray

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK
                                             + cv2.CALIB_CB_NORMALIZE_IMAGE)


    if ret:

        img_counter += 1

        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(frame, CHECKERBOARD, corners2, ret)
    else:
        print("no chessboard found!")

if last_frame is None:
    print("no last frame, camera not connected? exiting.")
    sys.exit()

if img_counter < 10:
    print("not enough images, need at least 10, but got {} instead. exiting.".format(img_counter))
    sys.exit()

"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""

print("calculating calibration parameters for {} images".format(img_counter))
retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, last_frame.shape[::-1], None, None)

print("Camera matrix : \n")
print(cameraMatrix)
print("dist : \n")
print(distCoeffs)

calibration_data = {'cameraMatrix': cameraMatrix.tolist(), 'distCoeffs': distCoeffs.tolist()}

with open('calibration/calibration_data.json', 'w') as write_file:
    json.dump(calibration_data, write_file, indent=4)

# https://longervision.github.io/2017/03/13/ComputerVision/OpenCV/opencv-external-posture-estimation-ChArUco-board/
