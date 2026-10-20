import functools
import time
from collections.abc import Callable
from typing import Any, cast


def timer[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper_timer(*args: P.args, **kwargs: P.kwargs) -> R:
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print("Finished {} in {} milliseconds".format(func.__name__, run_time*1000))
        return value

    return wrapper_timer


def synchronized_with_lock[F: Callable[..., Any]](lock_name: str) -> Callable[[F], F]:
    def decorator(method: F) -> F:
        @functools.wraps(method)
        def synced_method(self: Any, *args: Any, **kws: Any) -> Any:
            lock = getattr(self, lock_name)
            with lock:
                return method(self, *args, **kws)
        return cast(F, synced_method)
    return decorator
