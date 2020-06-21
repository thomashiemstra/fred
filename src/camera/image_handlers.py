from abc import ABC, abstractmethod
import cv2


class ImageHandler(ABC):
    @abstractmethod
    def handle_frame(self, frame, gray):
        pass

    @abstractmethod
    def deactivate_handler(self):
        pass


class ImageRecorder(ImageHandler):

    def __init__(self, filename, fps, screen_width, screen_height):
        self.out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                   fps, (screen_width, screen_height))

    def handle_frame(self, frame, gray):
        self.out.write(frame)

    def deactivate_handler(self):
        self.out.release()


class ArucoImageHandler(ImageHandler):

    def __init__(self, board):
        self.board = board

    def handle_frame(self, frame, gray):
        pass

    def deactivate_handler(self):
        pass