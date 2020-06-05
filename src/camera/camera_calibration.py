import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
# cap.set(cv2.CAP_PROP_FPS, 60) does nothing on ubuntu

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

fps = int(cap.get(cv2.CAP_PROP_FPS))
print("fps:", fps)



print(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

while cap.isOpened():

    ret, frame = cap.read()

    cv2.imshow('frame', frame)

    cv2.waitKey(1)

