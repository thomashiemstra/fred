import cv2 as cv

marker_id = 1

# Load the predefined dictionary
dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)

# Generate the marker
# markerImage = np.zeros((200, 200), dtype=np.uint8)
for marker_id in range(250):
    markerImage = cv.aruco.drawMarker(dictionary, marker_id, 100)

    cv.imwrite("markers/marker{}.png".format(marker_id), markerImage)