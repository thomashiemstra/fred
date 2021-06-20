import sys

import cv2
import jsonpickle
import numpy as np


class CaptureConfig:
    screen_width = 1920
    screen_height = 1080
    fps = 30
    marker_size = 20
    image_format = cv2.VideoWriter_fourcc(*'MJPG')


charuco_board_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
aruco_marker_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
aruco_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
charuco_base_board_square_length = 2
charuco__baseboard_marker_length = 1.6
aruco_marker_length = 2.65


def get_default_charuco_board(squares_x=6,
                              squares_y=3,
                              square_length=1,
                              marker_length=0.8
                              ):

    return cv2.aruco.CharucoBoard_create(squares_x, squares_y, square_length, marker_length, charuco_board_dictionary)


def get_calibrations(path):
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