import cv2


class CaptureConfig:
    screen_width = 1920
    screen_height = 1080
    fps = 30
    marker_size = 20
    image_format = cv2.VideoWriter_fourcc(*'MJPG')


charuco_board_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
aruco_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)


def get_default_charuco_board():
    squares_x = 5
    squares_y = 3
    square_length = 3.18
    marker_length = 2.55

    return cv2.aruco.CharucoBoard_create(squares_x, squares_y, square_length, marker_length, charuco_board_dictionary)