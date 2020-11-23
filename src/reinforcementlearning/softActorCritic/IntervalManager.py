import math


class IntervalManager:

    def __init__(self, log_interval):
        self._log_interval = log_interval
        self._last_log_interval = 0

    def should_trigger(self, total_step):
        log_intervals_passed = math.floor(total_step / self._log_interval)

        if log_intervals_passed > self._last_log_interval:
            self._last_log_interval = log_intervals_passed
            return True
        return False
