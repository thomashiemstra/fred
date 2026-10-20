from __future__ import annotations

from multiprocessing import Process, Lock, Array
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.sharedctypes import SynchronizedArray
    from multiprocessing.synchronize import Lock as LockType


def f(lock: LockType, array: SynchronizedArray[float]) -> None:
    while True:
        with lock:
            print("content = {}, {}, {}".format(array[0], array[1], array[2]))
        time.sleep(0.75)


if __name__ == '__main__':
    lock = Lock()

    arr = Array('f', [1.1, 2.1, 3.1])

    p = Process(target=f, args=(lock, arr))
    p.start()

    while True:
        with lock:
            arr[0] += 1
        time.sleep(1)
