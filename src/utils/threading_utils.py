import threading


class CountDownLatch(object):
    def __init__(self, count: int = 1) -> None:
        self.count = count
        self.lock = threading.Condition()

    def count_down(self) -> None:
        with self.lock:
            self.count -= 1
            if self.count <= 0:
                self.lock.notifyAll()

    def await_latch(self) -> None:
        with self.lock:
            while self.count > 0:
                self.lock.wait()
