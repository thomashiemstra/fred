import os


def is_linux():
    return os.name != 'nt'