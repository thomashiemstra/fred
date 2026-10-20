import threading

import serial
from time import sleep

from src.utils.decorators import synchronized_with_lock


class ArduinoAngleReader:

    def __init__(self, port: str = 'COM5', rate: int = 9600) -> None:
        self._port = port
        self._rate = rate
        self._serial = serial.Serial(self._port, self._rate)
        self.lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._val: int | None = None

    @synchronized_with_lock("lock")
    def _is_running(self) -> bool:
        return self._running

    @synchronized_with_lock("lock")
    def start(self) -> None:
        self._thread = threading.Thread(target=self._capture_arduino_output)
        self._running = True
        self._thread.start()

    @synchronized_with_lock("lock")
    def stop(self) -> None:
        if self._thread is None:
            return
        self._running = True
        self._thread.join()

    def _read_angle(self) -> int | None:
        raw_bytes = self._serial.readline()
        if raw_bytes is None:
            return None
        string = raw_bytes.decode().rstrip()
        try:
            value = int(string)
        except ValueError:
            value = None
        return value

    @synchronized_with_lock("lock")
    def get_val(self) -> int | None:
        return self._val

    @synchronized_with_lock("lock")
    def _set_val(self, val: int) -> None:
        self._val = val

    def _capture_arduino_output(self) -> None:

        while self._is_running():
            val = self._read_angle()
            if val is not None:
                self._set_val(val)


if __name__ == '__main__':

    reader = ArduinoAngleReader()
    reader.start()
    while True:
        print(reader.get_val())
        sleep(0.1)
