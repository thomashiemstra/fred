"""The stacking configuration the tests use: a frozen copy of the live stacking_config.yaml, which the user edits when
he measures the table (board position, tower, speeds). The tests that need a configuration load this copy, so that a
measured value never breaks them. Only the tests of the loader itself touch the live file, and only to see that it
loads."""
import os

from src.block_stacking.stacking_config import StackingConfig, load_stacking_config

FROZEN_STACKING_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_stacking_config.yaml')


def load_frozen_config() -> StackingConfig:
    return load_stacking_config(FROZEN_STACKING_CONFIG_PATH)
