import cv2
import threading

from src.camera.capture_config import CaptureConfig
from src.camera.image_handlers import CrossDrawer
from src.utils.decorators import synchronized_with_lock

charuco_board_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
aruco_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)


def get_default_charuco_board():
    squares_x = 5
    squares_y = 3
    square_length = 3.18
    marker_length = 2.55

    return cv2.aruco.CharucoBoard_create(squares_x, squares_y, square_length, marker_length, charuco_board_dictionary)

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

    def __init__(self, camera, image_handlers=None):
        if image_handlers is None:
            image_handlers = []

        self._camera = camera
        self._cap = None
        self._running = False
        self.lock = threading.RLock()
        self.image_handlers_lock = threading.RLock()
        self._image_handlers = image_handlers

    @synchronized_with_lock("lock")
    def start_camera(self):
        if self._running:
            print("camera already running")
            return

        self._cap = cv2.VideoCapture(self._camera)

        if not self._cap.isOpened():
            print("no camera connected!")
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, CaptureConfig.screen_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CaptureConfig.screen_height)
        self._cap.set(cv2.CAP_PROP_FOURCC, CaptureConfig.image_format)

        thread = threading.Thread(target=self.__capture_camera)
        self._running = True
        thread.start()

    @synchronized_with_lock("lock")
    def start_camera_recording(self):
        if self._running:
            print("camera already running")
            return

        self._cap = cv2.VideoCapture(self._camera)
        if not self._cap.isOpened():
            print("no camera connected!")
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, CaptureConfig.screen_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CaptureConfig.screen_height)
        self._cap.set(cv2.CAP_PROP_FOURCC, CaptureConfig.image_format)

        thread = threading.Thread(target=self.__capture_camera)
        self._running = True
        thread.start()

    @synchronized_with_lock("lock")
    def stop_camera(self):
        self._running = False

    def __capture_camera(self):
        while True:
            # Capture frame-by-frame
            ret, frame = self._cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.use_image_handlers(frame, gray)

            # Display the resulting frame
            cv2.imshow('frame', frame)
            cv2.waitKey(1)
            with self.lock:
                if not self._running:
                    break

        for image_handler in self._image_handlers:
            image_handler.deactivate_handler()

        self._cap.release()
        cv2.destroyAllWindows()

    @synchronized_with_lock("image_handlers_lock")
    def use_image_handlers(self, frame, gray):
        for image_handler in self._image_handlers:
            image_handler.handle_frame(frame, gray)

    @synchronized_with_lock("image_handlers_lock")
    def add_image_handler(self, image_handler):
        self._image_handlers.appen(image_handler)


if __name__ == '__main__':
    cross_drawer = CrossDrawer()
    test_image_handlers = [cross_drawer]

    capture = CameraCapture(0, test_image_handlers)
    capture.start_camera()