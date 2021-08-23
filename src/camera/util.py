import sys

import cv2
import jsonpickle
import numpy as np


class CaptureConfig:
    screen_width = 1920
    screen_height = 1080
    fps = 60
    marker_size = 20
    image_format = cv2.VideoWriter_fourcc(*'MJPG')
    FOURCC =  cv2.VideoWriter.fourcc('m', 'j', 'p', 'g')
    FOURCC2 = cv2.VideoWriter.fourcc('M', 'J', 'P', 'G')



charuco_board_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
charuco_board_dictionary_2 = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
aruco_marker_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
charuco_base_board_square_length = 3
charuco__baseboard_marker_length = 2.4
aruco_marker_length = 2.65
base_marker_offset = [-26.5, 8, 0]


def get_default_charuco_board(squares_x=6,
                              squares_y=3,
                              square_length=1,
                              marker_length=0.8,
                              dictionary=charuco_board_dictionary
                              ):

    return cv2.aruco.CharucoBoard_create(squares_x, squares_y, square_length, marker_length, dictionary)


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


def find_relative_vector_and_rotation(base_rotation_matrix, base_tvec, target_rotation_matrix, target_tvec):
    base_to_target_vec = target_tvec.reshape((3, 1)) - base_tvec

    relative_tvec = np.matmul(base_rotation_matrix.transpose(), base_to_target_vec)
    relative_rotation_matrix = np.matmul(base_rotation_matrix.transpose(), target_rotation_matrix)

    return relative_rotation_matrix, relative_tvec
