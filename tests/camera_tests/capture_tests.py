import threading
import unittest
from collections.abc import Callable
from typing import Any
from unittest import mock

import cv2
import numpy as np

from src.camera.capture import WINDOW_NAME, CameraCapture
from src.camera.image_handlers import ImageHandler

FRAME_TIMEOUT = 10.0  # seconds that a test waits for the capture thread


class FakeVideoCapture:
    """a camera that gives a small black picture, slowly enough not to spin the capture thread"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.released = False
        self._pause = threading.Event()

    def isOpened(self) -> bool:
        return True

    def set(self, *args: Any) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        self._pause.wait(0.001)
        return True, np.zeros((4, 6, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True


class RecordingHandler(ImageHandler):
    def __init__(self) -> None:
        self.clicks: list[tuple[int, int]] = []
        self.deactivated = False

    def handle_frame(self, frame: np.ndarray, gray: np.ndarray) -> None:
        pass

    def deactivate_handler(self) -> None:
        self.deactivated = True

    def handle_click(self, x: int, y: int) -> None:
        self.clicks.append((x, y))


class FailingHandler(RecordingHandler):
    def handle_click(self, x: int, y: int) -> None:
        super().handle_click(x, y)
        raise RuntimeError('this handler can not take a click')


class CameraCaptureClickTests(unittest.TestCase):

    def setUp(self) -> None:
        self.calls: list[str] = []
        self.shown = threading.Event()
        self.mouse_callbacks: list[tuple[str, Callable[..., None]]] = []
        self.captures: list[FakeVideoCapture] = []

        def make_capture(*args: Any, **kwargs: Any) -> FakeVideoCapture:
            capture = FakeVideoCapture()
            self.captures.append(capture)
            return capture

        def named_window(name: str, *args: Any) -> None:
            self.calls.append('namedWindow')

        def set_mouse_callback(name: str, callback: Callable[..., None], *args: Any) -> None:
            self.calls.append('setMouseCallback')
            self.mouse_callbacks.append((name, callback))

        def show(name: str, frame: np.ndarray) -> None:
            self.calls.append('imshow')
            self.shown.set()

        for target, replacement in (('VideoCapture', make_capture), ('namedWindow', named_window),
                                    ('setMouseCallback', set_mouse_callback), ('imshow', show),
                                    ('waitKey', lambda delay: -1), ('destroyAllWindows', mock.Mock())):
            patcher = mock.patch('cv2.' + target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def start(self, handlers: list[ImageHandler], recording: bool = False) -> CameraCapture:
        camera = CameraCapture(0, handlers)
        self.addCleanup(camera.stop_camera)  # the thread ends, whatever the test does
        if recording:
            camera.start_camera_recording()
        else:
            camera.start_camera()
        self.assertTrue(self.shown.wait(FRAME_TIMEOUT), 'the capture thread showed no picture')
        return camera

    def callback(self) -> Callable[..., None]:
        self.assertEqual(1, len(self.mouse_callbacks))
        name, callback = self.mouse_callbacks[0]
        self.assertEqual(WINDOW_NAME, name)
        return callback

    def test_starting_the_camera_registers_a_mouse_callback_before_the_first_picture(self) -> None:
        self.start([RecordingHandler()])
        self.callback()
        self.assertEqual(['namedWindow', 'setMouseCallback'], self.calls[:2])
        self.assertIn('imshow', self.calls)
        self.assertLess(self.calls.index('setMouseCallback'), self.calls.index('imshow'))

    def test_starting_a_recording_registers_a_mouse_callback_too(self) -> None:
        handler = RecordingHandler()
        self.start([handler], recording=True)
        self.callback()(cv2.EVENT_LBUTTONDOWN, 3, 2, 0, None)
        self.assertEqual([(3, 2)], handler.clicks)

    def test_a_left_click_reaches_every_handler(self) -> None:
        first, second = RecordingHandler(), RecordingHandler()
        self.start([first, second])
        self.callback()(cv2.EVENT_LBUTTONDOWN, 120, 45, cv2.EVENT_FLAG_LBUTTON, None)
        self.assertEqual([(120, 45)], first.clicks)
        self.assertEqual([(120, 45)], second.clicks)

    def test_a_handler_that_is_added_later_gets_the_clicks(self) -> None:
        camera = self.start([])
        handler = RecordingHandler()
        camera.add_image_handler(handler)
        self.callback()(cv2.EVENT_LBUTTONDOWN, 7, 8, 0, None)
        self.assertEqual([(7, 8)], handler.clicks)

    def test_other_mouse_events_are_not_clicks(self) -> None:
        handler = RecordingHandler()
        self.start([handler])
        callback = self.callback()
        for event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN,
                      cv2.EVENT_LBUTTONDBLCLK):
            with self.subTest(event=event):
                callback(event, 10, 10, 0, None)
        self.assertEqual([], handler.clicks)

    def test_a_handler_that_raises_does_not_stop_the_others_or_the_thread(self) -> None:
        failing, after = FailingHandler(), RecordingHandler()
        camera = self.start([failing, after])
        callback = self.callback()
        with self.assertLogs('src.camera.capture', level='ERROR'):
            callback(cv2.EVENT_LBUTTONDOWN, 1, 2, 0, None)
        self.assertEqual([(1, 2)], failing.clicks)
        self.assertEqual([(1, 2)], after.clicks)
        # the next click works the same, and the thread still shows pictures
        with self.assertLogs('src.camera.capture', level='ERROR'):
            callback(cv2.EVENT_LBUTTONDOWN, 3, 4, 0, None)
        self.assertEqual([(1, 2), (3, 4)], after.clicks)
        self.shown.clear()
        self.assertTrue(self.shown.wait(FRAME_TIMEOUT), 'the capture thread stopped showing pictures')
        camera.stop_camera()
        self.assertTrue(after.deactivated)

    def test_stopping_the_camera_ends_the_thread_and_closes_the_window(self) -> None:
        handler = RecordingHandler()
        camera = self.start([handler])
        camera.stop_camera()
        self.assertTrue(handler.deactivated)
        self.assertTrue(self.captures[0].released)
        cv2.destroyAllWindows.assert_called_once()  # type: ignore[attr-defined]


if __name__ == '__main__':
    unittest.main()
