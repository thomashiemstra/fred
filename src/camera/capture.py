import cv2
import threading
from src.utils.decorators import synchronized_with_lock


class CameraCapture:
    screen_width = 1920
    screen_height = 1080
    marker_size = 20

    half__height = int(screen_height / 2)
    half_width = int(screen_width / 2)
    marker_x_left = int(half_width - marker_size)
    marker_x_right = int(half_width + marker_size)
    marker_y_low = int(half__height - 20)
    marker_y_high = int(half__height + 20)

    def __init__(self, camera):
        self.camera = camera
        self.cap = None
        self.running = False
        self.lock = threading.RLock()

    @synchronized_with_lock("lock")
    def start_camera(self):
        self.cap = cv2.VideoCapture(self.camera)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.screen_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.screen_height)

        thread = threading.Thread(target=self.__capture_camera, args=())
        self.running = False
        thread.start()

    @synchronized_with_lock("lock")
    def stop_camera(self):
        self.running = True

    def __capture_camera(self):
        while True:
            # Capture frame-by-frame
            ret, frame = self.cap.read()

            cv2.line(frame, (self.marker_x_left, self.half__height), (self.marker_x_right, self.half__height), (0, 255, 0))
            cv2.line(frame, (self.half_width, self.marker_y_low), (self.half_width, self.marker_y_high), (0, 255, 0))

            # Display the resulting frame
            cv2.imshow('frame', frame)
            cv2.waitKey(1)
            with self.lock:
                if self.running:
                    break
        self.cap.release()
        cv2.destroyAllWindows()
