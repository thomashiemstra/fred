import cv2


class CaptureConfig:
    screen_width = 1920
    screen_height = 1080
    fps = 30
    marker_size = 20
    image_format = cv2.VideoWriter_fourcc(*'MJPG')
