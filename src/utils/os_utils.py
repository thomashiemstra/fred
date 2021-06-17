import os
import inspect


def is_linux():
    return os.name == 'posix'


def get_current_dir():
    return os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
