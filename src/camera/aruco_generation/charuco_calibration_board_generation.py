import cv2 as cv

# Load the predefined dictionary
from src.camera.util import charuco_board_dictionary, get_default_charuco_board, charuco_board_dictionary_2

dir = "markers/board.png"

dictionary = charuco_board_dictionary

board1 = get_default_charuco_board(dictionary=charuco_board_dictionary)
board2 = get_default_charuco_board(dictionary=charuco_board_dictionary_2)
imboard1 = board1.generateImage((1000, 500))
imboard2 = board2.generateImage((1000, 500))


success = cv.imwrite("markers/board1.png", imboard1) and cv.imwrite("markers/board2.png", imboard2)
print("saving success = {}".format(success))
