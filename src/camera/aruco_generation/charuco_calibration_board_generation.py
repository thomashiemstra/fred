import cv2 as cv

# Load the predefined dictionary
from src.camera.util import charuco_board_dictionary, get_default_charuco_board

dictionary = charuco_board_dictionary

board = get_default_charuco_board()
imboard = board.draw((1000, 500))

dir = "markers/board.png"

success = cv.imwrite(dir, imboard)
print("saving success = {}".format(success))
