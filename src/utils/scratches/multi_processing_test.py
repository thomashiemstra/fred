from multiprocessing import Process, Lock, Array
import time


def f(lock, array):
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
