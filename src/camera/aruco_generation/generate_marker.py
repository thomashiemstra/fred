import cv2 as cv

from src.camera.util import aruco_marker_dictionary, charuco_board_dictionary

# Load the predefined dictionary


# Generate the marker
# markerImage = np.zeros((200, 200), dtype=np.uint8)
for marker_id in range(50):
    markerImage = cv.aruco.drawMarker(aruco_marker_dictionary, marker_id, 100)

    cv.imwrite("markers/marker{}.png".format(marker_id), markerImage)
