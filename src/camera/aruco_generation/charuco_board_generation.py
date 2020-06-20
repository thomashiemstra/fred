import cv2 as cv
import numpy as np

# Load the predefined dictionary
from src.camera.capture import charuco_board_dictionary

dictionary = charuco_board_dictionary

# Generate the marker
markerImage = np.zeros((200, 200), dtype=np.uint8)
markerImage = cv.aruco.drawMarker(dictionary, 33, 200, markerImage, 1)

squares_x = 5
squares_y = 3
square_length = 0.01
marker_length = 0.008

board = cv.aruco.CharucoBoard_create(squares_x, squares_y, square_length, marker_length, dictionary)
imboard = board.draw((1000, 1000))

cv.imwrite("markers/board.png", imboard)
