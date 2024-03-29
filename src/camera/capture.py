import cv2
import threading

from src.camera.util import CaptureConfig
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

    def __init__(self, camera, image_handlers=None):
        if image_handlers is None:
            image_handlers = []

        self._camera = camera
        self._cap = None
        self._running = False
        self.lock = threading.RLock()
        self.run_lock = threading.RLock()
        self.image_handlers_lock = threading.RLock()
        self._image_handlers = image_handlers
        self._thread = None

    @synchronized_with_lock("lock")
    def start_camera(self):
        if self._running:
            print("camera already running")
            return

        self._cap = cv2.VideoCapture(self._camera, cv2.CAP_DSHOW)

        if not self._cap.isOpened():
            print("no camera connected!")
            return

        self._cap.set(cv2.CAP_PROP_FPS, CaptureConfig.fps)
        self._cap.set(cv2.CAP_PROP_FOURCC, CaptureConfig.FOURCC)
        self._cap.set(cv2.CAP_PROP_FOURCC, CaptureConfig.FOURCC2)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, CaptureConfig.screen_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CaptureConfig.screen_height)

        self._thread = threading.Thread(target=self.__capture_camera)
        self._running = True
        self._thread.start()

    @synchronized_with_lock("lock")
    def start_camera_recording(self):
        if self._running:
            print("camera already running")
            return

        self._cap = cv2.VideoCapture(self._camera, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            print("no camera connected!")
            return

        self._cap.set(cv2.CAP_PROP_FPS, CaptureConfig.fps)
        self._cap.set(cv2.CAP_PROP_FOURCC, CaptureConfig.FOURCC)
        self._cap.set(cv2.CAP_PROP_FOURCC, CaptureConfig.FOURCC2)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, CaptureConfig.screen_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CaptureConfig.screen_height)

        self._thread = threading.Thread(target=self.__capture_camera)
        with self.run_lock:
            self._running = True
        self._thread.start()

    @synchronized_with_lock("lock")
    def stop_camera(self):
        if self._thread is None:
            return
        with self.run_lock:
            self._running = False
        self._thread.join()

    def __capture_camera(self):
        while self.__is_running():
            # Capture frame-by-frame
            ret, frame = self._cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.use_image_handlers(frame, gray)

            # Display the resulting frame
            cv2.imshow('frame', frame)
            cv2.waitKey(1)

        for image_handler in self._image_handlers:
            image_handler.deactivate_handler()

        self._cap.release()
        cv2.destroyAllWindows()

    def __is_running(self):
        with self.run_lock:
            return self._running

    @synchronized_with_lock("image_handlers_lock")
    def use_image_handlers(self, frame, gray):
        for image_handler in self._image_handlers:
            image_handler.handle_frame(frame, gray)

    @synchronized_with_lock("image_handlers_lock")
    def add_image_handler(self, image_handler):
        self._image_handlers.append(image_handler)
