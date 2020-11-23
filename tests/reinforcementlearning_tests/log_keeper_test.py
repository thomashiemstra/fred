import unittest
from src.reinforcementlearning.softActorCritic.IntervalManager import IntervalManager


class TestPointInObstacle(unittest.TestCase):

    def test_interval_manager_1_pass_positive(self):
        interval = 10
        interval_manager = IntervalManager(interval)

        should_log = interval_manager.should_trigger(11)

        self.assertTrue(should_log, "should log after 11 steps with an interval of 10")

    def test_interval_manager_1_pass_negative(self):
        interval = 10
        interval_manager = IntervalManager(interval)

        should_log = interval_manager.should_trigger(9)

        self.assertFalse(should_log, "should not log after 9 steps with an interval of 10")

    def test_interval_manager_huge_step(self):
        interval = 10
        interval_manager = IntervalManager(interval)

        should_log = interval_manager.should_trigger(1337)

        self.assertTrue(should_log, "should log after 1337 steps with an interval of 10")

    def test_interval_manager_multiple_passes(self):
        interval = 10
        interval_manager = IntervalManager(interval)

        should_log_1 = interval_manager.should_trigger(11)
        self.assertTrue(should_log_1, "should log after 11 steps with an interval of 10")

        should_log_2 = interval_manager.should_trigger(19)
        self.assertFalse(should_log_2, "should not log after 19 steps with an interval of 10 if we logged at 11")

        should_log_3 = interval_manager.should_trigger(21)
        self.assertTrue(should_log_3, "should log after 21 steps with an interval of 10")

        should_log_3 = interval_manager.should_trigger(22)
        self.assertFalse(should_log_3, "should not log after 22 steps with an interval of 10 if we logged at 21")
