import cv2 as cv

from src.camera.util import aruco_marker_dictionary, charuco_board_dictionary
from pathlib import Path

# create directory if not exists
Path("markers/").mkdir(parents=True, exist_ok=True)

# Generate the marker
# markerImage = np.zeros((200, 200), dtype=np.uint8)
for marker_id in range(50):
    markerImage = cv.aruco.generateImageMarker(aruco_marker_dictionary, marker_id, 100)

    cv.imwrite("markers/marker{}.png".format(marker_id), markerImage)
