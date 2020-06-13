import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
fps = int(cap.get(cv2.CAP_PROP_FPS))
print("fps:", fps)

CHECKERBOARD = (6, 9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)


# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []

img_counter = 0
last_frame = None

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        print("failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    last_frame = gray

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK
                                             + cv2.CALIB_CB_NORMALIZE_IMAGE)

    key_pressed = cv2.waitKey(1)
    if key_pressed % 256 == 27:
        print("Escape hit, closing...")
        break
    elif key_pressed % 256 == 32:  # SPACE pressed
        img_name = "captures/opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        img_counter += 1

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(frame, CHECKERBOARD, corners2, ret)

    cv2.imshow("test", frame)

cap.release()
cv2.destroyAllWindows()

"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
if last_frame is not None:
    retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, last_frame.shape[::-1], None, None)

    # print("Camera matrix : \n")
    # print(mtx)
    # print("dist : \n")
    # print(dist)
    # print("rvecs : \n")
    # print(rvecs)
    # print("tvecs : \n")
    # print(tvecs)

    # https://longervision.github.io/2017/03/13/ComputerVision/OpenCV/opencv-external-posture-estimation-ChArUco-board/