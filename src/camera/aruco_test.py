import cv2 as cv
import numpy as np

# Load the predefined dictionary
dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)

# Generate the marker
markerImage = np.zeros((200, 200), dtype=np.uint8)
markerImage = cv.aruco.drawMarker(dictionary, 33, 200, markerImage, 1)

squares_x = 5
squares_y = 3
square_length = 0.0265
marker_length = 0.0198

board = cv.aruco.CharucoBoard_create(squares_x, squares_y, square_length, marker_length, dictionary)
imboard = board.draw((2000, 2000))


cv.imwrite("marker33.png", imboard)
