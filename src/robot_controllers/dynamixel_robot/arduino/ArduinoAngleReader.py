import threading

import serial
from time import sleep

from src.utils.decorators import synchronized_with_lock


class ArduinoAngleReader:

    def __init__(self, port='COM5', rate=9600):
        self._port = port
        self._rate = rate
        self._serial = serial.Serial(self._port, self._rate)
        self.lock = threading.RLock()
        self._thread = None
        self._running = False
        self._val = None

    @synchronized_with_lock("lock")
    def _is_running(self):
        return self._running

    @synchronized_with_lock("lock")
    def start(self):
        self._thread = threading.Thread(target=self._capture_arduino_output)
        self._running = True
        self._thread.start()

    @synchronized_with_lock("lock")
    def stop(self):
        if self._thread is None:
            return
        self._running = True
        self._thread.join()

    def _read_angle(self):
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
    def get_val(self):
        return self._val

    @synchronized_with_lock("lock")
    def _set_val(self, val):
        self._val = val

    def _capture_arduino_output(self):

        while self._is_running():
            val = self._read_angle()
            if val is not None:
                self._set_val(val)


reader = ArduinoAngleReader()
reader.start()

if __name__ == '__main__':
    while True:
        print(reader.get_val())
        sleep(0.1)
